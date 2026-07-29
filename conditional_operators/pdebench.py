"""Conditioning comparison on PDEBench data, with an FNO backbone.

This is the real-data version of the physics experiment. Everything before it used a solver we
wrote ourselves, which answers the mechanism question but leaves the obvious objection: our task,
our splits, our numbers. PDEBench ships 1D reaction-diffusion as one file per physical setting over
a 4x4 grid of (nu, rho), which is exactly the structure the compositional protocol needs and is not
a structure we chose.

    u_t = nu u_xx + rho u (1 - u)

The condition is the pair of physical parameters as log-deviations from the (nu=1, rho=1) cell, so
the baseline is the zero vector, single-parameter cells have one nonzero entry, and cells where both
parameters differ are held out. Same shape as every compositional split in this project.

Honest scope, stated once: PDEBench's own protocol trains a separate model per setting, so this is
their data under our protocol, not a number comparable to their published per-setting baselines.

    .venv/bin/python -m conditional_operators.pdebench --inspect
    .venv/bin/python -m conditional_operators.pdebench --screen
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

from . import improved, stage8  # noqa: F401  (register cfilm_hyb and the relaxed arms)
from .pdebench_fetch import NUS, RHOS, OUT, cell_path
from .stage4 import ARM_CLASSES, DEVICE, DZ, RESULTS_DIR, _mean, _std

MODES, WIDTH, LAYERS = 16, DZ, 4
DC = 2                                   # (log nu, log rho), as deviations from the baseline cell
BASE_NU, BASE_RHO = 1.0, 1.0

# Cells where at most one parameter leaves the baseline are trained on; cells where both leave it
# are held out. Val and test are fixed here, before any model runs.
TRAIN_CELLS = [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (5.0, 1.0),
               (1.0, 2.0), (1.0, 5.0), (1.0, 10.0)]
VAL_CELLS = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (0.5, 10.0)]
TEST_CELLS = [(0.5, 5.0), (2.0, 2.0), (2.0, 10.0), (5.0, 2.0), (5.0, 5.0)]


def cond_of(nu: float, rho: float) -> tuple[float, float]:
    """Log-deviation from the baseline cell, scaled so each axis spans roughly [-0.4, 1]."""
    return (math.log(nu / BASE_NU) / math.log(5.0),
            math.log(rho / BASE_RHO) / math.log(10.0))


class Data:
    """One tensor per parameter cell: [n_traj, n_t, n_x]."""

    def __init__(self, t_in: int = 0, t_out: int = -1):
        self.cells, missing = {}, []
        for nu in NUS:
            for rho in RHOS:
                p = cell_path(nu, rho)
                if not p.exists():
                    missing.append((nu, rho))
                    continue
                self.cells[(nu, rho)] = torch.load(p, map_location="cpu")["u"]
        if missing:
            print(f"  (missing {len(missing)} cells, still downloading: {missing[:4]}...)")
        any_u = next(iter(self.cells.values()))
        self.n_t, self.n_x = any_u.shape[1], any_u.shape[2]
        self.t_in = t_in
        self.t_out = self.n_t - 1 if t_out < 0 else t_out
        # one shared normalisation across every cell, so no arm sees a different scale
        allu = torch.cat([v[:, [self.t_in, self.t_out]].reshape(-1) for v in self.cells.values()])
        self.mu, self.sd = allu.mean().item(), allu.std().item()
        self.splits = {
            "train": [c for c in TRAIN_CELLS if c in self.cells],
            "val": [c for c in VAL_CELLS if c in self.cells],
            "test": [c for c in TEST_CELLS if c in self.cells],
        }
        self.gpu = {k: v.to(DEVICE) for k, v in self.cells.items()}

    def available(self) -> str:
        return " ".join(f"{k}:{len(v)}" for k, v in self.splits.items())

    def sample(self, kind: str, n: int, gen: torch.Generator):
        cells = self.splits[kind]
        ci = torch.randint(len(cells), (n,), generator=gen).tolist()
        xs, cs, ys = [], [], []
        for i in ci:
            nu, rho = cells[i]
            u = self.gpu[(nu, rho)]
            j = int(torch.randint(u.shape[0], (1,), generator=gen))
            xs.append(u[j, self.t_in])
            ys.append(u[j, self.t_out])
            cs.append(cond_of(nu, rho))
        x = (torch.stack(xs) - self.mu) / self.sd
        y = (torch.stack(ys) - self.mu) / self.sd
        return x.unsqueeze(1), torch.tensor(cs, dtype=torch.float32, device=DEVICE), y.unsqueeze(1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.w = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):
        b, c, n = x.shape
        xh = torch.fft.rfft(x)
        out = torch.zeros(b, self.w.shape[1], n // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = min(self.modes, xh.shape[-1])
        out[:, :, :m] = torch.einsum("bix,iox->box", xh[:, :, :m], self.w[:, :, :m])
        return torch.fft.irfft(out, n=n)


class FNO1d(nn.Module):
    """Same conditioning placement as the 2D version: channel-wise after every spectral block."""

    def __init__(self, arm: str, dc: int = DC):
        super().__init__()
        self.lift = nn.Linear(2, WIDTH)                     # field + coordinate
        self.spectral = nn.ModuleList(SpectralConv1d(WIDTH, WIDTH, MODES) for _ in range(LAYERS))
        self.pointwise = nn.ModuleList(nn.Conv1d(WIDTH, WIDTH, 1) for _ in range(LAYERS))
        self.cond = ARM_CLASSES[arm](dc=dc)
        self.head = nn.Sequential(nn.Linear(WIDTH, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x, c):
        b, _, n = x.shape
        grid = torch.linspace(0, 1, n, device=x.device).view(1, n, 1).expand(b, n, 1)
        z = self.lift(torch.cat([x.permute(0, 2, 1), grid], dim=-1)).permute(0, 2, 1)
        cc = c.repeat_interleave(n, dim=0)
        for spec, pw in zip(self.spectral, self.pointwise):
            z = spec(z) + pw(z)
            flat = z.permute(0, 2, 1).reshape(b * n, WIDTH)
            flat = self.cond(cc, flat)
            z = torch.nn.functional.gelu(flat.view(b, n, WIDTH).permute(0, 2, 1))
        return self.head(z.permute(0, 2, 1)).permute(0, 2, 1)


GATE = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")
REPORTED = ("additive", "cfilm_hyb", "proposed_scaled_conj")
ALL_ARMS = GATE + REPORTED


def _throttle(step):
    ms = int(os.environ.get("PDEBENCH_THROTTLE_MS", "0"))
    if ms and step % 5 == 0:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        time.sleep(ms / 1000.0)


def train_one(arm, seed, data, steps, batch=32, lr=1e-3):
    torch.manual_seed(seed)
    m = FNO1d(arm).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    diverged = False
    for step in range(steps):
        _throttle(step)
        x, c, y = data.sample("train", batch, g)
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
            x, c, y = data.sample(kind, 64, eg)
            tot += torch.mean((m(x, c) - y) ** 2).item()
        return tot / 8

    return row | dict(indist=ev("train"), val=ev("val"), test=ev("test"))


def inspect():
    d = Data()
    print(f"grid cells present : {len(d.cells)} of 16")
    print(f"trajectory shape   : n_t={d.n_t}, n_x={d.n_x}")
    print(f"prediction task    : u(t={d.t_in}) -> u(t={d.t_out}), conditioned on (nu, rho)")
    print(f"splits (cells)     : {d.available()}")
    print(f"normalisation      : mean {d.mu:.4f}, sd {d.sd:.4f}")
    for kind in ("train", "val", "test"):
        cells = d.splits[kind]
        print(f"  {kind:5}: {cells}")
    g = torch.Generator().manual_seed(0)
    x, c, y = d.sample("train", 4, g)
    print(f"batch shapes: x {tuple(x.shape)} c {tuple(c.shape)} y {tuple(y.shape)}")
    print(f"identity baseline (predict input unchanged): {torch.mean((x - y) ** 2).item():.5f}")


def screen(steps=2000):
    """Validation only, one seed. No test read, no verdict claimed."""
    data = Data()
    rows = {}
    for arm in ("film", "concat_mlp", "hypernet", "proposed", "cfilm_hyb"):
        t0 = time.time()
        r = train_one(arm, 0, data, steps)
        r.pop("test", None)
        rows[arm] = r
        print(f"  {arm:16} indist={r['indist']:.5f} val={r['val']:.5f} [{time.time()-t0:.0f}s]",
              flush=True)
    base = rows["concat_mlp"]["val"]
    print(f"\n  {'arm':16} {'val':>10} {'vs concat_mlp':>14}")
    for a, r in rows.items():
        print(f"  {a:16} {r['val']:10.5f} {1 - r['val'] / base:+13.1%}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "pdebench_screen.json").write_text(json.dumps(
        {"note": "validation only; no test split read; no verdict claimed",
         "steps": steps, "rows": rows}, indent=2))
    return rows


def main():
    if "--inspect" in sys.argv:
        inspect()
        return
    if "--screen" in sys.argv:
        screen(int(os.environ.get("PDEBENCH_SCREEN_STEPS", "2000")))
        return
    print(__doc__)


if __name__ == "__main__":
    main()
