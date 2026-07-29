"""Fourier Neural Operator backbone, so the conditioning comparison runs on the architecture the
PDE-surrogate field actually uses.

The convolutional result showed Complex FiLM beating the standard conditioning baselines by 29-45%
on held-out parameter combinations. The obvious objection is that the backbone was a small conv
encoder-decoder, and nobody solves PDEs with one. FNO is the standard, and it is not a neutral
swap: a spectral layer mixes information globally in Fourier space, and the conditioning is applied
to channels at every spatial location rather than to a single pooled latent. Whether a
magnitude-and-phase operator still helps there is a real question, not a formality.

Everything except the backbone is held fixed: same data, same splits, same conditioning arms, same
shared budget counter, same gate.

    .venv/bin/python -m conditional_operators.fno --smoke
    .venv/bin/python -m conditional_operators.fno --screen
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
from torch import nn

from . import improved, stage8  # noqa: F401  (register cfilm_hyb and the relaxed arms)
from .pde import DC, Task, _throttle
from .stage4 import ARM_CLASSES, DEVICE, DZ, RESULTS_DIR, _mean, _std

MODES = 16              # retained Fourier modes per dimension
WIDTH = DZ              # 128: the conditioning arms and their budget counters are built for it
LAYERS = 4


class SpectralConv2d(nn.Module):
    """The FNO layer: truncate to the lowest modes, apply a learned complex matrix, transform back."""

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        xh = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = self.modes
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xh[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xh[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(h, w))


class FNO(nn.Module):
    """FNO with a conditioning arm applied channel-wise after every spectral block.

    Channel-wise is the honest placement: it is where FiLM and adaLN inject conditioning in these
    models, and it lets every arm keep the parameter and FLOP counts the shared budget rule scores.
    """

    def __init__(self, arm: str, dc: int = DC):
        super().__init__()
        self.lift = nn.Linear(3, WIDTH)                        # field + 2 coordinate channels
        self.spectral = nn.ModuleList(SpectralConv2d(WIDTH, WIDTH, MODES) for _ in range(LAYERS))
        self.pointwise = nn.ModuleList(nn.Conv2d(WIDTH, WIDTH, 1) for _ in range(LAYERS))
        self.cond = ARM_CLASSES[arm](dc=dc)                    # ONE arm, reused at every site
        self.head = nn.Sequential(nn.Linear(WIDTH, 128), nn.GELU(), nn.Linear(128, 1))

    def _grid(self, b, h, w, device):
        ys = torch.linspace(0, 1, h, device=device).view(1, h, 1).expand(b, h, w)
        xs = torch.linspace(0, 1, w, device=device).view(1, 1, w).expand(b, h, w)
        return torch.stack([xs, ys], dim=-1)

    def forward(self, x, c):
        b, _, h, w = x.shape
        grid = self._grid(b, h, w, x.device)
        z = self.lift(torch.cat([x.permute(0, 2, 3, 1), grid], dim=-1))     # [b,h,w,WIDTH]
        z = z.permute(0, 3, 1, 2)
        cc = c.repeat_interleave(h * w, dim=0)                              # condition per location
        for spec, pw in zip(self.spectral, self.pointwise):
            z = spec(z) + pw(z)
            flat = z.permute(0, 2, 3, 1).reshape(b * h * w, WIDTH)
            flat = self.cond(cc, flat)                                      # the arm under test
            z = torch.nn.functional.gelu(flat.view(b, h, w, WIDTH).permute(0, 3, 1, 2))
        return self.head(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


GATE = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")
REPORTED = ("additive", "cfilm_hyb")
ALL_ARMS = GATE + REPORTED


def train_one(arm, seed, task, steps, batch=16, lr=1e-3):
    torch.manual_seed(seed)
    m = FNO(arm).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    diverged = False
    for step in range(steps):
        _throttle(step)
        x, c, y = task.sample("train", batch, g)
        loss = torch.mean((m(x, c) - y) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()

    row = dict(arm=arm, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    if diverged:
        return row | {k: math.nan for k in ("indist", "val", "test")}

    @torch.no_grad()
    def ev(kind):
        eg = torch.Generator().manual_seed(31_337)
        tot = 0.0
        for _ in range(8):
            x, c, y = task.sample(kind, 32, eg)
            tot += torch.mean((m(x, c) - y) ** 2).item()
        return tot / 8

    return row | dict(indist=ev("train"), val=ev("val"), test=ev("test"))


def smoke(steps=300):
    """Is it learning, and how fast does it run? Sizing information before any sweep."""
    task = Task()
    for arm in ("film", "cfilm_hyb"):
        t0 = time.time()
        r = train_one(arm, 0, task, steps)
        print(f"  {arm:12} indist={r['indist']:.5f} val={r['val']:.5f} "
              f"[{time.time()-t0:.0f}s for {steps} steps]", flush=True)


def screen(steps=1500):
    """Validation only. Does the advantage survive the spectral backbone at all?"""
    task = Task()
    rows = {}
    for arm in ("film", "concat_mlp", "hypernet", "proposed", "cfilm_hyb"):
        t0 = time.time()
        r = train_one(arm, 0, task, steps)
        rows[arm] = r
        print(f"  {arm:16} indist={r['indist']:.5f} val={r['val']:.5f} [{time.time()-t0:.0f}s]",
              flush=True)
    base = rows["concat_mlp"]["val"]          # what PDE surrogates actually do
    print(f"\n  {'arm':16} {'val':>9} {'vs concat_mlp':>14}")
    for a, r in rows.items():
        print(f"  {a:16} {r['val']:9.5f} {1 - r['val'] / base:+13.1%}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "fno_screen.json").write_text(json.dumps(
        {"note": "validation only; no test split read; no verdict claimed",
         "steps": steps, "rows": rows}, indent=2))
    return rows


def main():
    if "--smoke" in sys.argv:
        smoke(int(os.environ.get("FNO_SMOKE_STEPS", "300")))
        return
    if "--screen" in sys.argv:
        screen(int(os.environ.get("FNO_SCREEN_STEPS", "1500")))
        return
    print(__doc__)


if __name__ == "__main__":
    main()
