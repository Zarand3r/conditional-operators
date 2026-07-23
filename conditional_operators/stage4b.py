"""Stage-4b: dSprites transformation WITH a categorical (non-group) factor — shape changes.

Addendum to STAGE4_SPEC.md (pre-registered 2026-07-21 before any run; same AC-1..AC-6 margins).
Factors: 0=shape (categorical, 3 values), 1=scale, 2=orientation, 3=posX, 4=posY.
Delta encoding (dc=7, identical input for all arms): 4 normalized geometric deltas + 3 categorical
dims = onehot(new_shape) - onehot(old_shape) (additive: (C-B)+(B-A)=C-A; zero when unchanged).
This is the reviewer question "does the Lie story survive factors that do NOT form a
one-parameter group?" made into a pre-registered experiment.

Run:  .venv/bin/python -m conditional_operators.stage4b [N_SEEDS] [STEPS]
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time

import torch
from torch import nn

from . import verdict
from .stage4 import (ARM_CLASSES, GATE_ARMS, DEVICE, RESULTS_DIR, Backbone, Data,
                     _mean, _std)
from .verdict import Arm, ArmResult, decide

DC5 = 7                       # 4 geometric + 3 categorical (shape one-hot difference)
NF = 5                        # transformable factors incl. shape
# dsprites latents columns: [color, shape, scale, orient, posX, posY]
GEOM_COLS = (2, 3, 4, 5)
GEOM_SIZES = (6, 40, 32, 32)
GEOM_MAXSTEP = (2, 5, 4, 4)
GEOM_NORM = torch.tensor([2.0, 5.0, 4.0, 4.0])

SINGLES5 = tuple((i,) for i in range(NF))
TRAIN_PAIRS5 = ((0, 1), (1, 2), (3, 4))
VAL_PAIRS5 = ((0, 3), (2, 4), (1, 3))
TEST_PAIRS5 = ((0, 2), (0, 4), (1, 4), (2, 3))
TRIPLES5 = tuple(itertools.combinations(range(NF), 3))
TRAIN_TYPES5 = SINGLES5 + TRAIN_PAIRS5


def sample5(data: Data, types, n, gen):
    """Triplets with 5-factor delta types (factor 0 = categorical shape swap)."""
    lat = torch.stack([torch.randint(int(s), (n,), generator=gen)
                       for s in data.sizes.tolist()], dim=1)
    which = torch.randint(len(types), (n,), generator=gen)
    geo = torch.zeros(n, 4, dtype=torch.int64)
    new_shape = lat[:, 1].clone()
    for t_idx, t in enumerate(types):
        mask = which == t_idx
        m = int(mask.sum())
        for f in t:
            if f == 0:                                   # categorical: uniform different shape
                shift = torch.randint(1, 3, (m,), generator=gen)
                new_shape[mask] = (lat[mask, 1] + shift) % 3
            else:
                g = f - 1
                step = torch.randint(1, GEOM_MAXSTEP[g] + 1, (m,), generator=gen)
                sign = torch.randint(0, 2, (m,), generator=gen) * 2 - 1
                geo[mask, g] = step * sign
    tgt = lat.clone()
    tgt[:, 1] = new_shape
    for g, col in enumerate(GEOM_COLS):
        moved = lat[:, col] + geo[:, g]
        if g == 1:                                       # orientation wraps
            tgt[:, col] = moved % GEOM_SIZES[g]
        else:
            clamped = moved.clamp(0, GEOM_SIZES[g] - 1)
            geo[:, g] = clamped - lat[:, col]
            tgt[:, col] = clamped
    cat = (torch.nn.functional.one_hot(tgt[:, 1], 3) -
           torch.nn.functional.one_hot(lat[:, 1], 3)).float()
    d = torch.cat([geo.float() / GEOM_NORM, cat], dim=1).to(DEVICE)
    i1 = (lat * data.bases).sum(1)
    i2 = (tgt * data.bases).sum(1)
    x1 = data.imgs[i1.to(DEVICE)].float().unsqueeze(1)
    x2 = data.imgs[i2.to(DEVICE)].float().unsqueeze(1)
    return x1, d, x2


class Model5(nn.Module):
    def __init__(self, arm_name: str):
        super().__init__()
        self.backbone = Backbone()
        self.cond = ARM_CLASSES[arm_name](dc=DC5)

    def forward(self, x1, d):
        return self.backbone.decode(self.cond(d, self.backbone.encode(x1)))


def train_one(name, seed, data, steps, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    m = Model5(name).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    bce = nn.BCEWithLogitsLoss()
    diverged = False
    for _ in range(steps):
        x1, d, x2 = sample5(data, TRAIN_TYPES5, batch, gen)
        loss = bce(m(x1, d), x2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()
    row = dict(arm=name, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    eg = torch.Generator().manual_seed(80_000 + seed)
    if diverged:
        row |= dict(indist=math.nan, ood_val=math.nan, ood_test=math.nan, triples=math.nan)
        return row

    @torch.no_grad()
    def mse(types):
        tot = 0.0
        for _ in range(4):
            x1, d, x2 = sample5(data, types, 512, eg)
            tot += torch.mean((torch.sigmoid(m(x1, d)) - x2) ** 2).item()
        return tot / 4

    row |= dict(indist=mse(TRAIN_TYPES5), ood_val=mse(VAL_PAIRS5),
                ood_test=mse(TEST_PAIRS5), triples=mse(TRIPLES5))
    return row


def run(n_seeds, steps):
    data = Data()
    RESULTS_DIR.mkdir(exist_ok=True)
    all_arms = GATE_ARMS + ("proposed_mlp_gs",)
    runs = {a: [] for a in all_arms}
    with (RESULTS_DIR / "stage4b_log.jsonl").open("w") as log:
        for name in all_arms:
            for seed in range(n_seeds):
                t = time.time()
                r = train_one(name, seed, data, steps)
                runs[name].append(r)
                log.write(json.dumps(r) + "\n"); log.flush()
                print(f"{name:16} seed={seed} ood_test={r['ood_test']:.6f} "
                      f"triples={r['triples']:.6f} indist={r['indist']:.6f} "
                      f"[{time.time()-t:.1f}s]", flush=True)

    results = {}
    for name in GATE_ARMS:
        rr = runs[name]
        ok = [r for r in rr if not r["diverged"]]
        results[Arm(name)] = ArmResult(
            arm=Arm(name), ood_test_mse=tuple(r["ood_test"] for r in ok),
            indist_test_mse=tuple(r["indist"] for r in ok),
            n_diverged=sum(1 for r in rr if r["diverged"]),
            params=rr[0]["params"], flops=rr[0]["flops"], ood_test_reads=1)
    gate = decide(results, n_required=n_seeds)

    def stat(name, key):
        v = [r[key] for r in runs[name] if not r["diverged"]]
        return _mean(v), _std(v)

    summary = {
        "stage": "4b", "spec": "docs/specs/STAGE4_SPEC.md addendum (stage4b.py docstring)",
        "task": "dSprites transformation INCLUDING categorical shape factor (non-group)",
        "config": {"n_seeds": n_seeds, "steps": steps, "device": DEVICE},
        "final_verdict": gate.verdict.value, "reasons": list(gate.reasons),
        "gate_criteria": gate.criteria,
        "best_unstructured": gate.best_unstructured.value if gate.best_unstructured else None,
        "margin_observed": gate.margin_observed, "p_value": gate.p_value,
        "cliffs_delta": gate.cliffs_delta,
        "per_arm": {n: {"ood_test_mean": stat(n, "ood_test")[0],
                        "ood_test_std": stat(n, "ood_test")[1],
                        "triples_mean": stat(n, "triples")[0],
                        "triples_std": stat(n, "triples")[1],
                        "indist_mean": stat(n, "indist")[0],
                        "params": runs[n][0]["params"], "flops": runs[n][0]["flops"],
                        "flops_vs_film": runs[n][0]["flops"] / runs["film"][0]["flops"]}
                    for n in all_arms},
    }
    (RESULTS_DIR / "stage4b_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else verdict.N_REQUIRED
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-4B VERDICT: {s['final_verdict'].upper()}")
    for r in s["reasons"]:
        print(f"  - {r}")
    pa = s["per_arm"]
    print(f"  ood: proposed={pa['proposed']['ood_test_mean']:.6f} "
          f"{s['best_unstructured']}={pa[s['best_unstructured']]['ood_test_mean']:.6f} "
          f"film={pa['film']['ood_test_mean']:.6f}")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
