"""Parametric audio effects: the best structural fit left, and a task built not to flatter us.

The boundary map says the operator needs four things, and every application we tested violated at
least one. Audio effect modelling violates none of them on paper: the controls are continuous and
parametric, gains in dB compose by addition so exact composition is the *actual semantics* of the
control rather than an approximation of it, the audio signal supplies all the content, and the
model trains from scratch.

**The trap this task is designed to avoid.** A linear gain's dB value is additive and a filter
cascade multiplies transfer functions, so both are exactly `exp(linear in the condition)` --- our
operator's form. A task built from those alone would be won by construction and would prove
nothing, which is exactly what happened on the synthetic Navier-Stokes task, where one of four
axes turned out to be affine in the condition and I did not notice until PDEBench forced the
issue. So the chain deliberately mixes two aligned axes with two that cannot be of that form:
saturation is a pointwise nonlinearity and compression makes the gain depend on the signal's own
envelope. `alignment()` measures which is which before any model is trained.

    .venv/bin/python -m conditional_operators.audiofx --alignment
    .venv/bin/python -m conditional_operators.audiofx --screen
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
from .stage4 import ARM_CLASSES, DEVICE, DZ, RESULTS_DIR, _mean, _std

SR = 16_000
NSAMP = 1024
AXES = ("gain", "cutoff", "drive", "threshold")
DC = len(AXES)

# base setting, and the step each axis moves by. gain and cutoff are the aligned pair (additive in
# dB, multiplicative in frequency); drive and threshold drive nonlinear stages.
BASE = {"gain_db": 0.0, "cutoff": 4000.0, "drive": 1.0, "thresh_db": -20.0}
STEP = {"gain_db": 6.0, "cutoff": 2.0, "drive": 3.0, "thresh_db": 10.0}


def params_from(delta) -> dict:
    d = delta.tolist() if torch.is_tensor(delta) else list(delta)
    return {
        "gain_db": BASE["gain_db"] + STEP["gain_db"] * d[0],
        "cutoff": BASE["cutoff"] * (STEP["cutoff"] ** d[1]),
        "drive": BASE["drive"] * (STEP["drive"] ** d[2]),
        "thresh_db": BASE["thresh_db"] + STEP["thresh_db"] * d[3],
    }


# ---------------------------------------------------------------- the effect chain

def _onepole(x: torch.Tensor, cutoff: float) -> torch.Tensor:
    """One-pole lowpass applied in the frequency domain: exact, and vectorised over the batch."""
    n = x.shape[-1]
    f = torch.fft.rfftfreq(n, d=1.0 / SR).to(x.device)
    h = 1.0 / (1.0 + 1j * (f / cutoff))
    return torch.fft.irfft(torch.fft.rfft(x) * h, n=n)


def _envelope(x: torch.Tensor, tau_ms: float = 5.0) -> torch.Tensor:
    return _onepole(x.abs(), cutoff=1000.0 / tau_ms).clamp_min(1e-6)


def process(x: torch.Tensor, p: dict) -> torch.Tensor:
    """gain -> lowpass -> saturation -> compression. Order is fixed; the last two are nonlinear."""
    y = x * (10.0 ** (p["gain_db"] / 20.0))                      # aligned: additive in dB
    y = _onepole(y, p["cutoff"])                                 # aligned: cascades multiply
    drive = p["drive"]
    y = torch.tanh(drive * y) / math.tanh(drive)                 # NOT aligned: pointwise nonlinear
    env_db = 20.0 * torch.log10(_envelope(y))                    # NOT aligned: gain follows the
    over = (env_db - p["thresh_db"]).clamp_min(0.0)              #   signal's own envelope
    return y * (10.0 ** (-0.5 * over / 20.0))                    # 2:1 above threshold


def signals(n: int, gen: torch.Generator) -> torch.Tensor:
    """Harmonic stacks with random pitch, spectral tilt and envelope, plus a little noise."""
    t = torch.arange(NSAMP).float() / SR
    f0 = 80.0 * (2.0 ** (torch.rand(n, 1, generator=gen) * 3.0))
    out = torch.zeros(n, NSAMP)
    for k in range(1, 9):
        amp = torch.rand(n, 1, generator=gen) / k
        phase = torch.rand(n, 1, generator=gen) * 2 * math.pi
        out += amp * torch.sin(2 * math.pi * f0 * k * t.view(1, -1) + phase)
    out += 0.05 * torch.randn(n, NSAMP, generator=gen)
    env = torch.linspace(0, 1, NSAMP).view(1, -1) ** torch.rand(n, 1, generator=gen)
    out = out * env
    return (out / out.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)).to(DEVICE)


# ---------------------------------------------------------------- is the task aligned with us?

def alignment() -> dict:
    """Which axes are exp(linear in the condition), i.e. exactly our operator's form?

    For a linear, time-invariant stage the log-magnitude spectrum is additive in the condition. The
    test asks whether combining two axes reproduces the sum of their separate log-spectrum effects.
    Near zero means that pair is our form by construction; large means it is not.
    """
    g = torch.Generator().manual_seed(0)
    x = signals(64, g)

    def logspec(delta):
        y = process(x, params_from(delta))
        return torch.log10(torch.fft.rfft(y).abs().clamp_min(1e-8))

    base = logspec([0, 0, 0, 0])
    out = {}
    print("  pair                        residual   verdict")
    for i in range(DC):
        for j in range(i + 1, DC):
            ci = [0] * DC; ci[i] = 1
            cj = [0] * DC; cj[j] = 1
            both = [0] * DC; both[i] = 1; both[j] = 1
            lhs = logspec(both) - base
            rhs = (logspec(ci) - base) + (logspec(cj) - base)
            res = ((lhs - rhs).norm() / lhs.norm().clamp_min(1e-8)).item()
            out[f"{AXES[i]}+{AXES[j]}"] = res
            print(f"  {AXES[i]:>9} + {AXES[j]:<12} {res:9.4f}   "
                  f"{'ALIGNED (our form)' if res < 0.05 else 'not our form'}")
    aligned = sum(1 for v in out.values() if v < 0.05)
    print(f"\n  {aligned} of {len(out)} pairs are exactly our operator's form.")
    if aligned == len(out):
        print("  REJECT: the whole task matches the inductive bias by construction.")
    return out


# ---------------------------------------------------------------- task and backbone

SINGLES = tuple((i, s) for i in range(DC) for s in (-1, 1))
TRAIN_PAIRS = ((0, 1), (2, 3))
VAL_PAIRS = ((0, 2), (1, 3))
TEST_PAIRS = ((0, 3), (1, 2))
SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _settings(kind):
    out = []
    if kind == "train":
        for ax, s in SINGLES:
            d = [0] * DC; d[ax] = s; out.append(tuple(d))
        pairs = TRAIN_PAIRS
    else:
        pairs = {"val": VAL_PAIRS, "test": TEST_PAIRS}[kind]
    for i, j in pairs:
        for si, sj in SIGNS:
            d = [0] * DC; d[i] = si; d[j] = sj; out.append(tuple(d))
    return out


class Task:
    def __init__(self):
        self.splits = {k: _settings(k) for k in ("train", "val", "test")}

    def sample(self, kind, n, gen):
        x = signals(n, gen)
        deltas = self.splits[kind]
        di = torch.randint(len(deltas), (n,), generator=gen).tolist()
        y = torch.empty_like(x)
        for k, d in enumerate(deltas):
            m = [i for i, v in enumerate(di) if v == k]
            if m:
                y[m] = process(x[m], params_from(d))
        c = torch.tensor([deltas[i] for i in di], dtype=torch.float32, device=DEVICE)
        return x.unsqueeze(1), c, y.unsqueeze(1)


class ConvFX(nn.Module):
    """1D encoder/decoder with the conditioning arm on the bottleneck. Identical for every arm."""

    def __init__(self, arm: str, dc: int = DC):
        super().__init__()
        ch = 64
        self.enc = nn.Sequential(
            nn.Conv1d(1, ch, 8, 4, 2), nn.GELU(),        # 256
            nn.Conv1d(ch, ch, 8, 4, 2), nn.GELU(),       # 64
            nn.Conv1d(ch, ch * 2, 8, 4, 2), nn.GELU(),   # 16
            nn.Flatten(), nn.Linear(ch * 2 * 16, DZ))
        self.fc = nn.Linear(DZ, ch * 2 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(ch * 2, ch, 8, 4, 2), nn.GELU(),
            nn.ConvTranspose1d(ch, ch, 8, 4, 2), nn.GELU(),
            nn.ConvTranspose1d(ch, 1, 8, 4, 2))
        self.cond = ARM_CLASSES[arm](dc=dc)

    def forward(self, x, c):
        z = self.cond(c, self.enc(x))
        return self.dec(self.fc(z).view(x.shape[0], -1, 16))


def _throttle(step):
    ms = int(os.environ.get("AUDIOFX_THROTTLE_MS", "0"))
    if ms and step % 5 == 0:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        time.sleep(ms / 1000.0)


def train_one(arm, seed, task, steps, batch=64, lr=1e-3):
    torch.manual_seed(seed)
    m = ConvFX(arm).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    diverged = False
    for step in range(steps):
        _throttle(step)
        x, c, y = task.sample("train", batch, g)
        loss = torch.mean((m(x, c) - y) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()

    row = dict(arm=arm, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    if diverged:
        return row | {k: math.nan for k in ("indist", "val", "test")}

    @torch.no_grad()
    def ev(kind):
        eg = torch.Generator().manual_seed(4242)
        tot = 0.0
        for _ in range(6):
            x, c, y = task.sample(kind, 128, eg)
            tot += torch.mean((m(x, c) - y) ** 2).item()
        return tot / 6

    return row | dict(indist=ev("train"), val=ev("val"), test=ev("test"))


def screen(steps=3000):
    """Validation only. The fit ratio decides whether a pre-registration is worth writing."""
    task = Task()
    g = torch.Generator().manual_seed(0)
    x, c, y = task.sample("train", 512, g)
    identity = torch.mean((x - y) ** 2).item()

    rows = {}
    for arm in ("film", "concat_mlp", "hypernet", "proposed", "cfilm_hyb"):
        t0 = time.time()
        r = train_one(arm, 0, task, steps)
        r.pop("test", None)
        rows[arm] = r
        print(f"  {arm:14} indist={r['indist']:.6f} val={r['val']:.6f} [{time.time()-t0:.0f}s]",
              flush=True)

    best_fit = min(r["indist"] for r in rows.values())
    best_arm = min(rows, key=lambda a: rows[a]["indist"])
    gap = rows[best_arm]["val"] / rows[best_arm]["indist"]
    share = max(0.0, 1 - best_fit / identity)
    bias = rows["cfilm_hyb"]["indist"] / best_fit
    bias_p = rows["proposed"]["indist"] / best_fit
    base = rows["concat_mlp"]["val"]

    print(f"\n  identity baseline        {identity:.6f}")
    print(f"  conditioning share       {share:.1%}  (want >= 60%)")
    print(f"  compositional gap        {gap:.2f}x   (want >= 1.50x)")
    print(f"  fit ratio, cfilm_hyb     {bias:.3f}x  (want <= 1.05x)")
    print(f"  fit ratio, proposed      {bias_p:.3f}x")
    print(f"\n  {'arm':14} {'val':>10} {'vs concat_mlp':>14}")
    for a, r in rows.items():
        print(f"  {a:14} {r['val']:10.6f} {1 - r['val']/base:+13.1%}")
    ok = share >= 0.60 and gap >= 1.50 and min(bias, bias_p) <= 1.05
    print(f"\n  VERDICT: {'WORTH A SWEEP' if ok else 'NOT WORTH A SWEEP'}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "audiofx_screen.json").write_text(json.dumps(
        {"note": "validation only; no test split read; no verdict claimed",
         "steps": steps, "identity": identity, "conditioning_share": share,
         "compositional_gap": gap, "fit_ratio_cfilm": bias, "fit_ratio_proposed": bias_p,
         "worth_a_sweep": ok, "rows": rows}, indent=2))
    return ok


def main():
    if "--alignment" in sys.argv:
        alignment()
        return
    if "--screen" in sys.argv:
        screen(int(os.environ.get("AUDIOFX_SCREEN_STEPS", "3000")))
        return
    print(__doc__)


if __name__ == "__main__":
    main()
