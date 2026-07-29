"""2D Navier-Stokes surrogate conditioned on physical parameters.

Why this task, given everything that has failed. The boundary map says CGA needs four things:
conditions that form a group, a representation trained alongside the operator, a real
compositional gap, and content supplied by something other than the conditioning. Physical
parameters satisfy all four without strain. The initial vorticity field carries the content; the
parameters (viscosity, drag, forcing, background advection) only modulate how it evolves. That is
exactly the division of labour where the operator is strong, and it sidesteps the content failure
that killed the diffusion suite.

The dynamics are nonlinear, which matters: if the parameter dependence were exactly an exponential
of something linear, CGA's inductive bias would match the truth by construction and the comparison
would be rigged. Advection of vorticity by its own induced velocity is not of that form.

    omega_t + (u . grad) omega = nu lap omega - alpha omega + f,   lap psi = -omega

Solved pseudo-spectrally on a periodic box with 2/3 dealiasing and an integrating-factor RK4, so
the linear part is exact and only the nonlinear term is approximated.

    .venv/bin/python -m conditional_operators.pde --check     # physics validation
    .venv/bin/python -m conditional_operators.pde --screen    # discriminability screen
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

from .stage4 import DEVICE, RESULTS_DIR, _mean, _std

N = 64                       # grid points per side
LBOX = 2.0 * math.pi
KF = 4                       # Kolmogorov forcing wavenumber

# Base physical parameters, and the step each conditioning axis moves them by.
# Viscosity and drag move multiplicatively (they span orders of magnitude); forcing amplitude and
# background advection move additively.
BASE = {"nu": 2.0e-3, "alpha": 0.10, "force": 1.0, "vx": 0.0}
AXES = ("viscosity", "drag", "forcing", "advection")
DC = len(AXES)
# Ratio for nu/alpha, additive delta for the rest. Calibrated on the physics alone, before any
# model was trained: each axis moves the evolved field by 32-46% relative, so no single parameter
# dominates the condition and the compositional structure is genuinely four-dimensional.
STEP = {"nu": 16.0, "alpha": 4.0, "force": 0.5, "vx": 0.15}


def _throttle(step: int) -> None:
    """Duty-cycle the card; this machine's PSU trips at sustained full draw."""
    ms = int(os.environ.get("PDE_THROTTLE_MS", "0"))
    if ms and step % 5 == 0:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        time.sleep(ms / 1000.0)


def params_from(delta: torch.Tensor) -> dict:
    """Condition vector -> physical parameters. delta[i] in {-1, 0, +1} per axis."""
    d = delta.tolist() if torch.is_tensor(delta) else list(delta)
    return {
        "nu": BASE["nu"] * (STEP["nu"] ** d[0]),
        "alpha": BASE["alpha"] * (STEP["alpha"] ** d[1]),
        "force": BASE["force"] + STEP["force"] * d[2],
        "vx": BASE["vx"] + STEP["vx"] * d[3],
    }


# ---------------------------------------------------------------- spectral machinery

def _wavenumbers(device):
    k = torch.fft.fftfreq(N, d=1.0 / N).to(device)             # integer wavenumbers on [0,2pi)
    kx = k.view(-1, 1).expand(N, N)
    ky = k.view(1, -1).expand(N, N)
    k2 = kx ** 2 + ky ** 2
    inv = torch.where(k2 > 0, 1.0 / k2, torch.zeros_like(k2))  # mean mode has no streamfunction
    mask = (k.abs() <= N // 3).float()                          # 2/3 dealiasing
    return kx, ky, k2, inv, mask.view(-1, 1) * mask.view(1, -1)


def _forcing_hat(kx, ky, amp, device):
    """Kolmogorov forcing f = amp * cos(KF * y), a standard driven-turbulence setup."""
    y = torch.arange(N, device=device).float() * (LBOX / N)
    f = amp * torch.cos(KF * y).view(1, -1).expand(N, N)
    return torch.fft.fft2(f)


class Solver:
    """Pseudo-spectral 2D Navier-Stokes in vorticity form.

    The linear operator L = -nu|k|^2 - alpha - i*vx*kx is applied exactly through an integrating
    factor, so viscosity and drag never limit the step size; RK4 handles only the advection term.
    """

    def __init__(self, device=DEVICE):
        self.device = device
        self.kx, self.ky, self.k2, self.inv_k2, self.mask = _wavenumbers(device)

    def _nonlinear(self, w_hat):
        psi_hat = w_hat * self.inv_k2
        u = torch.fft.ifft2(1j * self.ky * psi_hat).real          # u =  d(psi)/dy
        v = torch.fft.ifft2(-1j * self.kx * psi_hat).real         # v = -d(psi)/dx
        wx = torch.fft.ifft2(1j * self.kx * w_hat).real
        wy = torch.fft.ifft2(1j * self.ky * w_hat).real
        return torch.fft.fft2(u * wx + v * wy) * self.mask

    def step(self, w_hat, dt, p):
        """One integrating-factor RK4 step. `p` is a dict of physical parameters."""
        L = -(p["nu"] * self.k2 + p["alpha"]) - 1j * p["vx"] * self.kx
        f_hat = _forcing_hat(self.kx, self.ky, p["force"], self.device)

        def rhs(wh):
            return -self._nonlinear(wh) + f_hat

        e_half, e_full = torch.exp(L * dt / 2), torch.exp(L * dt)
        a = rhs(w_hat)
        b = rhs(e_half * (w_hat + dt / 2 * a))
        c = rhs(e_half * (w_hat + dt / 2 * b))
        d = rhs(e_full * w_hat + dt * e_half * c)
        return (e_full * w_hat
                + dt / 6 * (e_full * a + 2 * e_half * (b + c) + d))

    def run(self, w0, dt, n_steps, p):
        w_hat = torch.fft.fft2(w0)
        for _ in range(n_steps):
            w_hat = self.step(w_hat, dt, p)
        return torch.fft.ifft2(w_hat).real


def initial_field(batch, gen, device=DEVICE, seed_scale=1.0):
    """Random smooth vorticity: a decaying spectrum with random phases, mean zero."""
    kx, ky, k2, _, _ = _wavenumbers(device)
    amp = (1.0 + k2) ** (-1.5)
    real = torch.randn(batch, N, N, generator=gen).to(device)
    field_hat = torch.fft.fft2(real) * amp
    w = torch.fft.ifft2(field_hat).real
    w = w - w.mean(dim=(1, 2), keepdim=True)
    return seed_scale * w / w.flatten(1).std(1).view(-1, 1, 1)


# ---------------------------------------------------------------- physics validation

def check() -> bool:
    """Validate the solver against facts that do not depend on it being correct."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    s = Solver()
    ok = True

    # 1. Pure diffusion has an exact solution: each mode decays as exp(-nu k^2 t).
    w0 = initial_field(1, g)
    p = {"nu": 5e-3, "alpha": 0.0, "force": 0.0, "vx": 0.0}
    T, steps = 0.5, 200
    got = s.run(w0, T / steps, steps, p)
    kx, ky, k2, _, _ = _wavenumbers(DEVICE)
    want = torch.fft.ifft2(torch.fft.fft2(w0) * torch.exp(-p["nu"] * k2 * T)).real
    # the nonlinear term is switched off only if we linearise; instead compare the *decay of a
    # single low mode*, where advection is negligible for a small-amplitude field
    small = 1e-3 * w0
    got_s = s.run(small, T / steps, steps, p)
    want_s = torch.fft.ifft2(torch.fft.fft2(small) * torch.exp(-p["nu"] * k2 * T)).real
    err = ((got_s - want_s).norm() / want_s.norm()).item()
    print(f"  diffusion vs analytic (small amplitude): relative error {err:.2e}")
    ok &= err < 1e-3

    # 2. Unforced, undamped, inviscid flow conserves energy and enstrophy.
    p0 = {"nu": 0.0, "alpha": 0.0, "force": 0.0, "vx": 0.0}
    w = initial_field(1, g)
    e0 = (w ** 2).sum().item()
    wT = s.run(w, 1e-3, 300, p0)
    eT = (wT ** 2).sum().item()
    drift = abs(eT - e0) / e0
    print(f"  inviscid enstrophy drift over 300 steps: {drift:.2e}")
    ok &= drift < 0.05

    # 3. More viscosity must decay faster, monotonically.
    w = initial_field(1, g)
    decays = []
    for nu in (1e-3, 4e-3, 1.6e-2):
        wT = s.run(w, {"nu": nu, "alpha": 0.0, "force": 0.0, "vx": 0.0} and 2e-3, 200,
                   {"nu": nu, "alpha": 0.0, "force": 0.0, "vx": 0.0})
        decays.append((wT ** 2).sum().item())
    print(f"  enstrophy after fixed time, rising viscosity: "
          f"{decays[0]:.3f} > {decays[1]:.3f} > {decays[2]:.3f}")
    ok &= decays[0] > decays[1] > decays[2]

    # 4. Background advection translates the field, and translation is a rigid motion.
    w = initial_field(1, g)
    pv = {"nu": 0.0, "alpha": 0.0, "force": 0.0, "vx": 1.0}
    shifted = s.run(w * 1e-4, LBOX / N / 1.0 / 8, 8, pv)      # advect exactly one grid cell
    rolled = torch.roll(w * 1e-4, shifts=1, dims=1)
    rel = ((shifted - rolled).norm() / rolled.norm()).item()
    print(f"  pure advection matches a one-cell roll: relative error {rel:.2e}")
    ok &= rel < 0.05

    # 5. Determinism.
    a = s.run(w, 1e-3, 20, BASE)
    b = s.run(w, 1e-3, 20, BASE)
    ok &= torch.equal(a, b)
    print(f"  deterministic: {torch.equal(a, b)}")

    print("\nSOLVER:", "VALIDATED" if ok else "FAILED — do not build on this")
    return ok


# ---------------------------------------------------------------- the conditioning task

DT, HORIZON = 2.0e-3, 1000                    # evolve for T = 2.0, long enough for parameters to bite
N_INIT = 128                                  # initial conditions per parameter setting
CACHE = RESULTS_DIR / f"pde_data_n{N_INIT}.pt"

SINGLES = tuple((i, s) for i in range(DC) for s in (-1, 1))
ALL_PAIRS = tuple((i, j) for i in range(DC) for j in range(i + 1, DC))
TRAIN_PAIRS = ((0, 1), (2, 3))
VAL_PAIRS = ((0, 2), (1, 3))
TEST_PAIRS = ((0, 3), (1, 2))
SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _settings(kind):
    """Every delta vector belonging to a split, as a list of tuples."""
    out = []
    if kind == "train":
        for ax, s in SINGLES:
            d = [0] * DC; d[ax] = s; out.append(tuple(d))
        pairs = TRAIN_PAIRS
    else:
        pairs = {"val": VAL_PAIRS, "test": TEST_PAIRS}[kind]
    for (i, j) in pairs:
        for si, sj in SIGNS:
            d = [0] * DC; d[i] = si; d[j] = sj; out.append(tuple(d))
    return out


def build_dataset(force=False):
    """Evolve every parameter setting from a shared set of initial fields. Cached; ~2 min."""
    if CACHE.exists() and not force:
        return torch.load(CACHE, map_location=DEVICE)
    s = Solver()
    g = torch.Generator().manual_seed(20260728)
    w0 = initial_field(N_INIT, g)                       # SAME initial fields for every setting,
    data = {}                                           # so only the parameters differ
    for kind in ("train", "val", "test"):
        for delta in _settings(kind):
            p = params_from(torch.tensor(delta, dtype=torch.float32))
            wT = s.run(w0, DT, HORIZON, p)
            data[delta] = wT.cpu()
            print(f"  {kind:5} {delta}  |w_T| std {wT.std().item():.4f}", flush=True)
    blob = {"w0": w0.cpu(), "fields": data}
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(blob, CACHE)
    return blob


class Task:
    """Serves (initial field, parameter delta, evolved field) triples per split."""

    def __init__(self):
        blob = build_dataset()
        self.w0 = blob["w0"].to(DEVICE)
        self.fields = {k: v.to(DEVICE) for k, v in blob["fields"].items()}
        self.splits = {k: _settings(k) for k in ("train", "val", "test")}
        # one shared normalisation, so every arm sees the same scale
        self.scale = self.w0.std().item()

    def sample(self, kind, n, gen):
        deltas = self.splits[kind]
        di = torch.randint(len(deltas), (n,), generator=gen).tolist()
        ii = torch.randint(self.w0.shape[0], (n,), generator=gen).tolist()
        x = self.w0[ii].unsqueeze(1) / self.scale
        y = torch.stack([self.fields[deltas[d]][i] for d, i in zip(di, ii)]).unsqueeze(1)
        c = torch.tensor([deltas[d] for d in di], dtype=torch.float32, device=DEVICE)
        return x, c, y / self.scale


def _screen_task():
    from .discriminability import Task as STask
    from .stage4 import Model
    t = Task()

    def mk(kind):
        def f(n, g):
            return t.sample(kind, n, g)
        return f

    # Screening reads VALIDATION, never test. The test split is read once, by the decision run.
    return STask(name="PDE parameter conditioning (2D Navier-Stokes)",
                 sample_train=mk("train"), sample_heldout=mk("val"),
                 build=lambda arm: Model(arm, dc=DC).to(DEVICE),
                 arms=("film", "hypernet", "proposed"),
                 activation=None)          # vorticity takes both signs; no squashing


def screen(steps=2000):
    from .discriminability import screen as run_screen
    s = run_screen(_screen_task(), steps=steps)
    print()
    print(s.report())
    return s


# ---------------------------------------------------------------- the decision sweep

GATE = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")
REPORTED = ("additive", "cfilm_hyb", "proposed_mlp_gs")
ALL_ARMS = GATE + REPORTED


def train_one(arm, seed, task, steps, batch=128, lr=1e-3):
    from .stage4 import Model
    torch.manual_seed(seed)
    m = Model(arm, dc=DC).to(DEVICE)
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
    def ev2(kind):
        eg = torch.Generator().manual_seed(31_337)
        tot = 0.0
        for _ in range(4):
            x, c, y = task.sample(kind, 256, eg)
            tot += torch.mean((m(x, c) - y) ** 2).item()
        return tot / 4

    return row | dict(indist=ev2("train"), val=ev2("val"), test=ev2("test"))


def run(n_seeds=10, steps=8000):
    task = Task()
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "pde_log.jsonl"
    runs = {a: [] for a in ALL_ARMS}
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        print(f"resuming: {len(done)} runs already done", flush=True)
    with log_path.open("a") as log:
        for arm in ALL_ARMS:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t0 = time.time()
                r = train_one(arm, seed, task, steps)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:16} seed={seed} test={r['test']:.5f} indist={r['indist']:.5f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    return runs


def summarize(runs, n_seeds, steps):
    """Scored by the unchanged shared gate, exactly as stages 4, 5 and 7 were."""
    from .verdict import Arm, ArmResult, decide

    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])

    results = {Arm(a): ArmResult(Arm(a), arr(a, "test"), arr(a, "indist"),
                                 sum(1 for r in runs[a] if r["diverged"]),
                                 runs[a][0]["params"], runs[a][0]["flops"], 1)
               for a in GATE}
    gate = decide(results, n_required=n_seeds)
    summary = {
        "experiment": "pde-params", "spec": "docs/specs/PDE1_SPEC.md",
        "task": "2D Navier-Stokes surrogate conditioned on physical-parameter deltas "
                "(viscosity, drag, forcing, background advection); OOD = unseen parameter pairs",
        "config": {"n_seeds": n_seeds, "steps": steps, "grid": N, "horizon_T": DT * HORIZON,
                   "device": DEVICE},
        "final_verdict": gate.verdict.value, "reasons": list(gate.reasons),
        "gate_criteria": gate.criteria,
        "best_unstructured": gate.best_unstructured.value if gate.best_unstructured else None,
        "margin_observed": gate.margin_observed, "p_value": gate.p_value,
        "cliffs_delta": gate.cliffs_delta,
        "per_arm": {a: {k: _mean(arr(a, k)) for k in ("indist", "val", "test")}
                       | {"test_std": _std(arr(a, "test")),
                          "n_diverged": sum(1 for r in runs[a] if r["diverged"]),
                          "params": runs[a][0]["params"], "flops": runs[a][0]["flops"],
                          "flops_vs_film": runs[a][0]["flops"] / runs["film"][0]["flops"]}
                    for a in ALL_ARMS},
    }
    (RESULTS_DIR / "pde_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    if "--check" in sys.argv:
        raise SystemExit(0 if check() else 1)
    if "--run" in sys.argv:
        n = int(sys.argv[sys.argv.index("--run") + 1]) if len(sys.argv) > sys.argv.index("--run") + 1 else 10
        steps = int(os.environ.get("PDE_STEPS", "8000"))
        t0 = time.time()
        s = summarize(run(n, steps), n, steps)
        print("\n" + "=" * 64)
        print(f"PDE VERDICT: {s['final_verdict'].upper()}  [{(time.time()-t0)/60:.0f} min]")
        for k, v in s["gate_criteria"].items():
            print(f"  {k}: {'pass' if v else 'FAIL'}")
        print(f"  vs {s['best_unstructured']}: margin {s['margin_observed']:+.1%} "
              f"p={s['p_value']:.2g} delta={s['cliffs_delta']:.2f}")
        print(f"\n  {'arm':16} {'indist':>9} {'test':>9} {'params':>9} {'flops/film':>10}")
        for a, v in s["per_arm"].items():
            print(f"  {a:16} {v['indist']:9.5f} {v['test']:9.5f} {v['params']:9d} "
                  f"{v['flops_vs_film']:10.2f}")
        return
    if "--data" in sys.argv:
        build_dataset(force="--force" in sys.argv)
        return
    if "--screen" in sys.argv:
        screen()
        return
    print(__doc__)


if __name__ == "__main__":
    main()
