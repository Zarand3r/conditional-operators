"""Stage-5: second benchmark — 3D Shapes compositional conditional transformation (RGB, 6 factors).

Pre-registered addendum (same AC-1..AC-6 margins as Stage-4; spec note in STAGE4_SPEC.md family).
Factors: 0=floor_hue(10,cyclic) 1=wall_hue(10,cyclic) 2=object_hue(10,cyclic) 3=scale(8)
4=shape(4,CATEGORICAL) 5=orientation(15). Delta (dc=9, identical input for all arms):
5 normalized scalars (hues wrap) + 4 categorical dims (shape one-hot difference).

Run:  .venv/bin/python -m conditional_operators.stage5 [N_SEEDS] [STEPS]
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time

import h5py
import numpy as np
import torch
from torch import nn

from . import verdict
from .stage4 import (ARM_CLASSES, GATE_ARMS, DEVICE, RESULTS_DIR, ROOT, Backbone,
                     _mean, _std)
from .verdict import Arm, ArmResult, decide

DC6 = 9
NF = 6
SIZES6 = (10, 10, 10, 8, 4, 15)
CYCLIC = (True, True, True, False, False, False)   # hues wrap; scale/orientation clamp
MAXSTEP6 = (3, 3, 3, 2, 0, 4)                       # shape handled categorically
NORM6 = torch.tensor([3.0, 3.0, 3.0, 2.0, 4.0])     # for the 5 scalar delta dims (no shape)
SCALAR_F = (0, 1, 2, 3, 5)                          # factors carried as scalars

SINGLES6 = tuple((i,) for i in range(NF))
TRAIN_PAIRS6 = ((0, 1), (2, 3), (4, 5), (0, 2), (3, 5))
VAL_PAIRS6 = ((1, 2), (0, 3), (1, 5), (3, 4))
TEST_PAIRS6 = ((0, 4), (0, 5), (1, 3), (1, 4), (2, 4), (2, 5))
TRIPLES6 = tuple(itertools.combinations(range(NF), 3))[:10]
TRAIN_TYPES6 = SINGLES6 + TRAIN_PAIRS6


class Data6:
    """3D Shapes resident on GPU (uint8 NHWC); canonical row-major factor indexing (asserted)."""

    def __init__(self) -> None:
        with h5py.File(ROOT / "datasets" / "3dshapes.h5", "r") as f:
            self.imgs = torch.from_numpy(f["images"][:]).to(DEVICE)     # [480000,64,64,3] uint8
            labels = f["labels"][:]
        cls = np.stack([np.unique(labels[:, i], return_inverse=True)[1]
                        for i in range(NF)], axis=1)
        sizes = np.array(SIZES6)
        bases = np.concatenate([sizes[::-1].cumprod()[::-1][1:], [1]]).astype(np.int64)
        idx = (cls * bases).sum(1)
        if not np.array_equal(idx, np.arange(len(labels))):
            raise RuntimeError("3dshapes factor ordering assumption violated")
        self.bases = torch.from_numpy(bases)

    def fetch(self, cls: torch.Tensor) -> torch.Tensor:
        i = (cls * self.bases).sum(1).to(DEVICE)
        return self.imgs[i].permute(0, 3, 1, 2).float() / 255.0          # NCHW in [0,1]


def sample6(data: Data6, types, n, gen):
    lat = torch.stack([torch.randint(s, (n,), generator=gen) for s in SIZES6], dim=1)
    which = torch.randint(len(types), (n,), generator=gen)
    delta = torch.zeros(n, NF, dtype=torch.int64)
    new_shape = lat[:, 4].clone()
    for t_idx, t in enumerate(types):
        mask = which == t_idx
        m = int(mask.sum())
        for f in t:
            if f == 4:                                     # categorical shape swap
                shift = torch.randint(1, SIZES6[4], (m,), generator=gen)
                new_shape[mask] = (lat[mask, 4] + shift) % SIZES6[4]
            else:
                step = torch.randint(1, MAXSTEP6[f] + 1, (m,), generator=gen)
                sign = torch.randint(0, 2, (m,), generator=gen) * 2 - 1
                delta[mask, f] = step * sign
    tgt = lat.clone()
    tgt[:, 4] = new_shape
    for f in SCALAR_F:
        moved = lat[:, f] + delta[:, f]
        if CYCLIC[f]:
            tgt[:, f] = moved % SIZES6[f]
        else:
            clamped = moved.clamp(0, SIZES6[f] - 1)
            delta[:, f] = clamped - lat[:, f]
            tgt[:, f] = clamped
    scal = torch.stack([delta[:, f].float() for f in SCALAR_F], dim=1) / NORM6
    cat = (torch.nn.functional.one_hot(tgt[:, 4], SIZES6[4]) -
           torch.nn.functional.one_hot(lat[:, 4], SIZES6[4])).float()
    d = torch.cat([scal, cat], dim=1).to(DEVICE)
    return data.fetch(lat), d, data.fetch(tgt)


class Model6(nn.Module):
    def __init__(self, arm_name: str):
        super().__init__()
        self.backbone = Backbone(in_ch=3)
        self.cond = ARM_CLASSES[arm_name](dc=DC6)

    def forward(self, x1, d):
        return self.backbone.decode(self.cond(d, self.backbone.encode(x1)))


def train_one(name, seed, data, steps, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    m = Model6(name).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    bce = nn.BCEWithLogitsLoss()
    diverged = False
    for _ in range(steps):
        x1, d, x2 = sample6(data, TRAIN_TYPES6, batch, gen)
        loss = bce(m(x1, d), x2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()
    row = dict(arm=name, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    eg = torch.Generator().manual_seed(90_000 + seed)
    if diverged:
        row |= dict(indist=math.nan, ood_val=math.nan, ood_test=math.nan, triples=math.nan)
        return row

    @torch.no_grad()
    def mse(types):
        tot = 0.0
        for _ in range(4):
            x1, d, x2 = sample6(data, types, 512, eg)
            tot += torch.mean((torch.sigmoid(m(x1, d)) - x2) ** 2).item()
        return tot / 4

    row |= dict(indist=mse(TRAIN_TYPES6), ood_val=mse(VAL_PAIRS6),
                ood_test=mse(TEST_PAIRS6), triples=mse(TRIPLES6))
    return row


def run(n_seeds, steps):
    data = Data6()
    RESULTS_DIR.mkdir(exist_ok=True)
    all_arms = GATE_ARMS + ("proposed_mlp_gs",)
    runs = {a: [] for a in all_arms}
    with (RESULTS_DIR / "stage5_log.jsonl").open("w") as log:
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
        "stage": 5, "task": "3D Shapes (RGB) transformation; 3 cyclic hues + scale + orientation "
                            "+ CATEGORICAL shape; OOD = unseen two-factor delta types",
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
    (RESULTS_DIR / "stage5_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else verdict.N_REQUIRED
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-5 VERDICT: {s['final_verdict'].upper()}")
    for r in s["reasons"]:
        print(f"  - {r}")
    pa = s["per_arm"]
    print(f"  ood: proposed={pa['proposed']['ood_test_mean']:.6f} "
          f"{s['best_unstructured']}={pa[s['best_unstructured']]['ood_test_mean']:.6f} "
          f"film={pa['film']['ood_test_mean']:.6f}")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
