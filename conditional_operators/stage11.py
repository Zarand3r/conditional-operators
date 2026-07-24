"""Stage-11: contraction sweep — T(a) = (1-eps) R(Wa) at eps in {0,.003,.01,.03,.1}.

Isolates spectral radius as the rollout-stability knob (STAGE11_SPEC.md). Stage-7 harness,
no consistency loss. Deliverable: error-vs-eps curve at h3/h10/h20 + registered U-shape gates.

Run:  .venv/bin/python -m conditional_operators.stage11 [N] [STEPS]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
from torch import nn

from . import stage7
from .stage4 import ARM_CLASSES, Data, DEVICE, RESULTS_DIR, _mean, _std
from .stage7 import DC7, TRAIN_TYPES7, TEST_PAIRS7, WorldModel, sample_rollouts
from .verdict import cliffs_delta, mann_whitney_u

EPS_GRID = (0.0, 0.003, 0.01, 0.03, 0.1)


def make_eps_arm(eps):
    base = ARM_CLASSES["proposed"]

    class EpsLie(base):
        def op(self, h, d, z):
            return (1.0 - eps) * super().op(h, d, z)
    EpsLie.__name__ = f"EpsLie_{eps}"
    return EpsLie


for _e in EPS_GRID:
    ARM_CLASSES[f"eps_{_e}"] = make_eps_arm(_e)

ARMS_11 = tuple(f"eps_{e}" for e in EPS_GRID)


def run(n_seeds, steps):
    data = Data()
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = {a: [] for a in ARMS_11}
    log_path = RESULTS_DIR / "stage11_log.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} runs", flush=True)
    os.environ["STAGE7_THROTTLE_MS"] = os.environ.get("STAGE11_THROTTLE_MS", "0")
    with log_path.open("a") as log:
        for arm in ARMS_11:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                r = stage7.train_one(arm, seed, data, steps)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:10} seed={seed} h20={r['h20']:.5f} indist={r['indist']:.5f} "
                      f"[{time.time()-t:.0f}s]", flush=True)

    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])
    h20 = {a: _mean(arr(a, "h20")) for a in ARMS_11}
    best = min(ARMS_11, key=lambda a: h20[a])
    base = "eps_0.0"
    m = (h20[base] - h20[best]) / h20[base] if best != base else 0.0
    if best != base:
        _, p = mann_whitney_u(arr(best, "h20"), arr(base, "h20")); d = cliffs_delta(
            arr(best, "h20"), arr(base, "h20"))
    else:
        p, d = 1.0, 0.0
    crit = {
        "AC-11.1": best != base and m >= 0.30 and p <= 0.01 and d <= -0.474,
        "AC-11.2": best not in (ARMS_11[0], ARMS_11[-1]),
        "AC-11.3-reported": _mean(arr(best, "indist")) > _mean(arr(base, "indist")),
    }
    verdict = "confirmed" if crit["AC-11.1"] and crit["AC-11.2"] else "kill"
    summary = {
        "stage": 11, "spec": "docs/specs/STAGE11_SPEC.md",
        "config": {"n_seeds": n_seeds, "steps": steps, "eps_grid": EPS_GRID},
        "final_verdict": verdict, "criteria": crit,
        "best_eps": best, "margin_vs_eps0_h20": m, "p": p, "delta": d,
        "per_arm": {a: {k: _mean(arr(a, k)) for k in ("indist", "ood_pairs", "h10", "h20")}
                    for a in ARMS_11},
    }
    (RESULTS_DIR / "stage11_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-11 VERDICT: {s['final_verdict'].upper()}  criteria={s['criteria']}")
    for a in ARMS_11:
        p = s["per_arm"][a]
        print(f"  {a:10} indist={p['indist']:.5f} h10={p['h10']:.5f} h20={p['h20']:.5f}")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
