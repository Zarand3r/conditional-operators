"""Latent-space attribute editing on 3D Shapes.

Designed against the failure of the camera experiment. There, most of the error was the model
struggling to reconstruct a scene through a bottleneck, which no conditioning mechanism can help
with, so every arm scored the same. Two changes remove that floor:

* A plain autoencoder is trained once with no conditioning, then **frozen**. Every arm shares it,
  so reconstruction quality is a constant, not a variable.
* The metric is computed **in latent space**: how close the predicted latent is to the encoder's
  latent for the true target image. The decoder never enters the score, so its error cannot mask
  the conditioning's.

What remains is exactly the question of interest: given the latent of an image and an attribute
change, does the operator land on the right latent? This is the setting real latent editing uses
(a frozen generator, edits applied in its latent space), so passing here is evidence about a
deployed pattern rather than about a toy.

Screen it before pre-registering anything:
    .venv/bin/python -m conditional_operators.latent_edit --screen
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
import torch
from torch import nn

from . import stage8  # noqa: F401  (registers the Complex FiLM arms)
from .stage4 import ARM_CLASSES, DEVICE, DZ, RESULTS_DIR, _mean, _std
from .stage5 import Data6, SIZES6

# DZ (=128) is stage-4's latent width, reused unchanged: the orthogonal basis is built for it and
# every arm's parameter and FLOP count is calibrated against it, so the budget comparison carries
# over from the earlier experiments without recalibration.
AE_PATH = RESULTS_DIR / "latent_edit_ae.pt"

# 3D Shapes factors: floor hue, wall hue, object hue, scale, shape, orientation.
# An edit moves one or more attributes by a fixed step, and the condition says which way each
# moved. Shape is excluded: swapping cube for cylinder is a categorical jump with no step size,
# so it has no "move by one" to compose.
EDIT_AXES = ("floor hue", "wall hue", "object hue", "scale", "orientation")
DC = len(EDIT_AXES)
AX_COL = (0, 1, 2, 3, 5)                  # column of each axis in the factor vector
AX_MAX = tuple(SIZES6[c] for c in AX_COL)
STEP = (2, 2, 2, 1, 3)                    # increment size per axis, in grid steps
# Hues and orientation are circular, so a step off the end wraps and stays a genuine step. Scale
# is not: wrapping tiny back to huge would make one condition value mean two different things.
CIRCULAR = (True, True, True, False, True)

SINGLES = tuple((i,) for i in range(DC))
TRAIN_PAIRS = ((0, 1), (2, 3), (0, 4))
VAL_PAIRS = ((1, 2), (3, 4))
TEST_PAIRS = ((0, 2), (0, 3), (1, 3), (1, 4), (2, 4))
TRIPLES = tuple(itertools.combinations(range(DC), 3))[:6]
TRAIN_TYPES = SINGLES + TRAIN_PAIRS


def _throttle(step: int) -> None:
    """Duty-cycle the card. Sustained full draw has tripped this machine's power supply before."""
    ms = int(os.environ.get("LATENT_THROTTLE_MS", "0"))
    if ms and step % 5 == 0:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        time.sleep(ms / 1000.0)


# ---------------------------------------------------------------- the frozen autoencoder

class AE(nn.Module):
    def __init__(self, ch: int = 48):
        super().__init__()
        self.ch = ch
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.SiLU(),
            nn.Flatten(), nn.Linear(ch * 4 * 16, DZ))
        self.fc = nn.Linear(DZ, ch * 4 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch * 2, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1))

        # Filled in once after training. The raw latent's scale is an accident of how the encoder
        # happened to converge; standardizing it means every arm sees a unit-scale input and the
        # error numbers are comparable to the earlier experiments.
        self.register_buffer("mu", torch.zeros(DZ))
        self.register_buffer("sd", torch.ones(DZ))

    def encode(self, x):
        return (self.enc(x) - self.mu) / self.sd

    def decode(self, z):
        return self.dec(self.fc(z * self.sd + self.mu).view(-1, self.ch * 4, 4, 4))


def train_ae(data: Data6, steps: int = 6000, batch: int = 128, lr: float = 1e-3) -> AE:
    """Trained once, with no conditioning, then frozen and shared by every arm."""
    if AE_PATH.exists():
        ae = AE().to(DEVICE)
        ae.load_state_dict(torch.load(AE_PATH, map_location=DEVICE))
        return ae.eval().requires_grad_(False)
    torch.manual_seed(0)
    ae = AE().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for step in range(steps):
        _throttle(step)
        lat = torch.stack([torch.randint(int(s), (batch,), generator=g) for s in SIZES6], 1)
        x = data.fetch(lat)
        rec = torch.sigmoid(ae.decode(ae.encode(x)))
        loss = torch.mean((rec - x) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  ae step {step+1}: recon {loss.item():.5f}", flush=True)

    ae.eval().requires_grad_(False)
    with torch.no_grad():
        lat = torch.stack([torch.randint(int(s), (4096,), generator=g) for s in SIZES6], 1)
        raw = ae.enc(data.fetch(lat))
        ae.mu.copy_(raw.mean(0))
        ae.sd.copy_(raw.std(0).clamp_min(1e-3))
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(ae.state_dict(), AE_PATH)
    return ae


# ---------------------------------------------------------------- the editing task

def apply_edits(lat, types, which, gen):
    """Factor indices in, (edit vector, edited factor indices) out. Separated from image fetching
    so the index arithmetic can be checked directly against the factor grid."""
    n = lat.shape[0]
    edit = torch.zeros(n, DC)
    tgt = lat.clone()
    for ti, t in enumerate(types):
        m = which == ti
        k = int(m.sum())
        if k == 0:
            continue
        for ax in t:
            col, size, st = AX_COL[ax], AX_MAX[ax], STEP[ax]
            sign = torch.randint(0, 2, (k,), generator=gen) * 2 - 1
            here = lat[m, col]
            if CIRCULAR[ax]:
                moved = (here + sign * st) % size
            else:
                # Turn the step around rather than clamping, so the move is always a full step
                # and the condition always describes it exactly.
                sign = torch.where((here + sign * st).clamp(0, size - 1) == here + sign * st,
                                   sign, -sign)
                moved = here + sign * st
            edit[m, ax] = sign.float()
            tgt[m, col] = moved
    return edit, tgt


def sample_edits(data: Data6, types, n, gen):
    """(source image, edit vector, target image). One entry of the edit vector per attribute:
    +1 or -1 for an attribute that moves one step, 0 for one that is left alone."""
    lat = torch.stack([torch.randint(int(s), (n,), generator=gen) for s in SIZES6], 1)
    which = torch.randint(len(types), (n,), generator=gen)
    edit, tgt = apply_edits(lat, types, which, gen)
    return data.fetch(lat), edit.to(DEVICE), data.fetch(tgt)


class Editor(nn.Module):
    """Frozen encoder, a conditioning arm on the latent, and nothing else."""

    def __init__(self, arm: str, ae: AE):
        super().__init__()
        self.ae = ae
        self.cond = ARM_CLASSES[arm](dc=DC)

    def forward(self, x_src, edit):
        with torch.no_grad():
            z = self.ae.encode(x_src)
        return self.cond(edit, z)


GATE_ARMS = ("additive", "film", "concat_mlp", "cond_layernorm", "hypernet",
             "dynamic_linear", "proposed")
REPORTED = ("cfilm_hyb", "proposed_mlp_gs")
ALL_ARMS = GATE_ARMS + REPORTED


def _latent_mse(pred, target):
    return torch.mean((pred - target) ** 2)


def train_one(arm, seed, data, ae, steps, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    m = Editor(arm, ae).to(DEVICE)
    opt = torch.optim.Adam(m.cond.parameters(), lr=lr)      # only the conditioning arm trains
    g = torch.Generator().manual_seed(seed)
    diverged = False
    for step in range(steps):
        _throttle(step)
        xs, edit, xt = sample_edits(data, TRAIN_TYPES, batch, g)
        with torch.no_grad():
            zt = ae.encode(xt)
        loss = _latent_mse(m(xs, edit), zt)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()

    row = dict(arm=arm, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    if diverged:
        row |= {k: math.nan for k in ("indist", "val", "test", "triples")}
        return row

    @torch.no_grad()
    def ev(types):
        eg = torch.Generator().manual_seed(31_337)
        tot = 0.0
        for _ in range(4):
            xs, edit, xt = sample_edits(data, types, 256, eg)
            tot += _latent_mse(m(xs, edit), ae.encode(xt)).item()
        return tot / 4

    row |= dict(indist=ev(TRAIN_TYPES), val=ev(VAL_PAIRS), test=ev(TEST_PAIRS),
                triples=ev(TRIPLES))
    return row


# ---------------------------------------------------------------- screening

def screen(steps: int = 1500):
    """Run the discriminability screen before committing to anything."""
    from .discriminability import Screen
    data = Data6()
    ae = train_ae(data)
    g = torch.Generator().manual_seed(0)

    with torch.no_grad():
        xs, edit, xt = sample_edits(data, TRAIN_TYPES, 512, g)
        zs, zt = ae.encode(xs), ae.encode(xt)
        identity = _latent_mse(zs, zt).item()                    # ignore the edit entirely
        mean_lat = _latent_mse(zt.mean(0, keepdim=True), zt).item()

    # A high-capacity arm has to be in here: the fit check asks whether the task is learnable at
    # all, and the strongest available model is the instrument for that.
    fits = {}
    for arm in ("film", "hypernet", "proposed"):
        r = train_one(arm, 0, data, ae, steps)
        fits[arm] = (r["indist"], r["val"])
        print(f"  screen {arm:9} indist={r['indist']:.5f} heldout={r['val']:.5f}", flush=True)

    tr, ho = min(fits.values(), key=lambda p: p[0])
    held = [h for _, h in fits.values()]
    s = Screen(task="latent-edit (3D Shapes)", identity_mse=identity, mean_mse=mean_lat,
               fitted_mse=tr, heldout_mse=ho,
               conditioning_share=max(0.0, 1 - tr / identity),
               compositional_gap=ho / tr if tr else float("inf"),
               separation=(max(held) - min(held)) / max(held))
    print()
    print(s.report())
    return s


def run(n_seeds, steps):
    data = Data6()
    ae = train_ae(data)
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = {a: [] for a in ALL_ARMS}
    log_path = RESULTS_DIR / "latent_edit_log.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} runs already done", flush=True)
    with log_path.open("a") as log:
        for arm in ALL_ARMS:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                r = train_one(arm, seed, data, ae, steps)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:16} seed={seed} test={r['test']:.5f} indist={r['indist']:.5f} "
                      f"[{time.time()-t:.0f}s]", flush=True)
    return runs


def summarize(runs, n_seeds, steps):
    """Score the pre-registered criteria of docs/specs/LATENT1_SPEC.md. Nothing here is chosen
    after the fact: C1/C3/C4 are the shared gate's AC-2/AC-5/AC-4, C2 mirrors the camera spec."""
    from .verdict import Arm, ArmResult, cliffs_delta, decide, mann_whitney_u

    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])

    std = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")
    results = {Arm(a): ArmResult(Arm(a), arr(a, "test"), arr(a, "indist"),
                                 sum(1 for r in runs[a] if r["diverged"]),
                                 runs[a][0]["params"], runs[a][0]["flops"], 1)
               for a in std}
    gate = decide(results, n_required=n_seeds)

    prop, add = arr("proposed", "test"), arr("additive", "test")
    m2 = (_mean(add) - _mean(prop)) / _mean(add)
    _, p2 = mann_whitney_u(prop, add)
    d2 = cliffs_delta(prop, add)
    crit = {
        "C1": bool(gate.criteria.get("AC-2")),
        "C2": m2 >= 0.20 and p2 <= 0.01 and d2 <= -0.474,
        "C3": bool(gate.criteria.get("AC-5")),
        "C4": (runs["proposed"][0]["flops"] <= 1.20 * runs["film"][0]["flops"]
               and runs["proposed"][0]["params"]
               <= 1.05 * min(runs[a][0]["params"] for a in ("hypernet", "dynamic_linear"))),
    }
    summary = {
        "experiment": "latent-edit", "spec": "docs/specs/LATENT1_SPEC.md",
        "task": "3D Shapes attribute editing in a FROZEN autoencoder latent; scored in latent "
                "space; held-out attribute pairs. Companion to stage 5, which is the same task "
                "and splits with the representation co-trained.",
        "config": {"n_seeds": n_seeds, "steps": steps, "device": DEVICE, "dz": DZ},
        "final_verdict": "confirmed" if all(crit.values()) else "kill",
        "criteria": crit,
        "margin_vs_unstructured": gate.margin_observed,
        "best_unstructured": gate.best_unstructured.value if gate.best_unstructured else None,
        "p_vs_unstructured": gate.p_value, "cliffs_delta_vs_unstructured": gate.cliffs_delta,
        "margin_vs_additive": m2, "p_vs_additive": p2, "delta_vs_additive": d2,
        "per_arm": {a: {k: _mean(arr(a, k)) for k in ("indist", "val", "test", "triples")}
                       | {"test_std": _std(arr(a, "test")),
                          "n_diverged": sum(1 for r in runs[a] if r["diverged"]),
                          "params": runs[a][0]["params"], "flops": runs[a][0]["flops"],
                          "flops_vs_film": runs[a][0]["flops"] / runs["film"][0]["flops"]}
                    for a in ALL_ARMS},
    }
    (RESULTS_DIR / "latent1_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    if "--screen" in sys.argv:
        screen()
        return
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    t = time.time()
    s = summarize(run(n, steps), n, steps)
    print("\n" + "=" * 64)
    print(f"LATENT-EDIT VERDICT: {s['final_verdict'].upper()}   [{(time.time()-t)/60:.0f} min]")
    for k, v in s["criteria"].items():
        print(f"  {k}: {'pass' if v else 'FAIL'}")
    print(f"\n  vs best unstructured ({s['best_unstructured']}): "
          f"margin {s['margin_vs_unstructured']:+.1%}, p={s['p_vs_unstructured']:.2g}")
    print(f"  vs additive (the deployed mechanism): margin {s['margin_vs_additive']:+.1%}, "
          f"p={s['p_vs_additive']:.2g}")
    print(f"\n  {'arm':16} {'indist':>9} {'test':>9} {'triples':>9} {'params':>9}")
    for a, v in s["per_arm"].items():
        print(f"  {a:16} {v['indist']:9.5f} {v['test']:9.5f} {v['triples']:9.5f} "
              f"{v['params']:9d}")


if __name__ == "__main__":
    main()
