"""Complex FiLM on 3D Shapes: the one past kill worth revisiting (SHAPES3D_CFILM_SPEC.md).

`shapes3d` failed AC-5 alone, at a 1.46x in-distribution fit ratio against a 1.10x ceiling, while
beating the hypernetwork by 87.5% on unseen combinations. That is the same criterion and the same
mechanism just resolved on the physics task, where a magnitude channel moved the fit ratio from
1.118x to 0.938x. Complex FiLM postdates stage 5 and was never run here.

Gated arm on fresh seeds 10-19; baselines reused from the original run, unchanged.

    .venv/bin/python -m conditional_operators.shapes3d_cfilm
"""

from __future__ import annotations

import json
import os
import sys
import time

from . import stage5, stage8  # noqa: F401  (stage8 registers cfilm_hyb)
from . import improved  # noqa: F401  (registers proposed_scaled_conj)
from .stage4 import RESULTS_DIR, _mean, _std
from .verdict import Arm, ArmResult, decide

GATED = "cfilm_hyb"
REPORTED = ("proposed_scaled_conj",)
SEED0, N_SEEDS, STEPS = 10, 10, 12_000


def run(n_seeds=N_SEEDS, steps=STEPS, seed0=SEED0):
    data = stage5.Data6()
    RESULTS_DIR.mkdir(exist_ok=True)

    base = {}
    for line in (RESULTS_DIR / "stage5_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        base.setdefault(r["arm"], []).append(r)

    log_path = RESULTS_DIR / "shapes3d_cfilm_log.jsonl"
    runs, done = {}, set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs.setdefault(r["arm"], []).append(r); done.add((r["arm"], r["seed"]))
        print(f"resuming: {len(done)} runs already done", flush=True)

    throttle = int(os.environ.get("SHAPES3D_THROTTLE_MS", "0"))
    with log_path.open("a") as log:
        for arm in (GATED,) + REPORTED:
            for seed in range(seed0, seed0 + n_seeds):
                if (arm, seed) in done:
                    continue
                t0 = time.time()
                if throttle:
                    os.environ["STAGE5_THROTTLE_MS"] = str(throttle)
                r = stage5.train_one(arm, seed, data, steps)
                runs.setdefault(arm, []).append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:22} seed={seed} ood_test={r['ood_test']:.6f} "
                      f"indist={r['indist']:.6f} [{time.time()-t0:.0f}s]", flush=True)

    def arr(rows, k):
        return tuple(r[k] for r in rows if not r["diverged"])

    src = {a: base[a] for a in ("film", "concat_mlp", "cond_layernorm", "hypernet",
                                "dynamic_linear")}
    src["proposed"] = runs[GATED]                      # the candidate occupies the gated slot
    results = {Arm(a): ArmResult(Arm(a), arr(rows, "ood_test"), arr(rows, "indist"),
                                 sum(1 for r in rows if r["diverged"]),
                                 rows[0]["params"], rows[0]["flops"], 1)
               for a, rows in src.items()}
    gate = decide(results, n_required=n_seeds)

    summary = {
        "experiment": "shapes3d-cfilm", "spec": "docs/specs/SHAPES3D_CFILM_SPEC.md",
        "gated_arm": GATED, "seed_range": [seed0, seed0 + n_seeds - 1],
        "baselines_reused_from": "shapes3d (seeds 0-9, identical protocol)",
        "config": {"n_seeds": n_seeds, "steps": steps},
        "final_verdict": gate.verdict.value, "reasons": list(gate.reasons),
        "gate_criteria": gate.criteria,
        "best_unstructured": gate.best_unstructured.value if gate.best_unstructured else None,
        "margin_observed": gate.margin_observed, "p_value": gate.p_value,
        "cliffs_delta": gate.cliffs_delta,
        "per_arm": {a: {"indist": _mean(arr(rows, "indist")),
                        "ood_test": _mean(arr(rows, "ood_test")),
                        "ood_test_std": _std(arr(rows, "ood_test")),
                        "triples": _mean(arr(rows, "triples")),
                        "params": rows[0]["params"], "flops": rows[0]["flops"]}
                    for a, rows in (base | runs).items()},
    }
    (RESULTS_DIR / "shapes3d_cfilm_summary.json").write_text(json.dumps(summary, indent=2))

    pa = summary["per_arm"]
    bu = pa[summary["best_unstructured"]]
    print("\n" + "=" * 64)
    print(f"SHAPES3D-CFILM VERDICT ({GATED}): {summary['final_verdict'].upper()}")
    for k, v in gate.criteria.items():
        print(f"  {k}: {'pass' if v else 'FAIL'}")
    print(f"  vs {summary['best_unstructured']}: margin {gate.margin_observed:+.1%} "
          f"p={gate.p_value:.2g} delta={gate.cliffs_delta:.2f}")
    print(f"  fit ratio: {pa[GATED]['indist'] / bu['indist']:.4f}x (ceiling 1.10)")
    print(f"  original 'proposed' fit ratio was "
          f"{pa['proposed']['indist'] / bu['indist']:.4f}x")
    for a in (GATED,) + REPORTED:
        v = pa[a]
        print(f"  {a:22} indist {v['indist']:.6f}  ood_test {v['ood_test']:.6f}")
    return summary


if __name__ == "__main__":
    run(steps=int(os.environ.get("SHAPES3D_STEPS", str(STEPS))))
