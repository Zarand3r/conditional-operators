"""Stage-1 sweep: 6 arms x N seeds -> results log -> verdict.decide() (R9 + AC-1..AC-6).

Run:  .venv/bin/python -m conditional_operators.sweep [N_SEEDS] [STEPS]
Writes results/results_log.jsonl (one row per run) and results/summary.json, prints the verdict.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import verdict
from .arms import ARM_CLASSES
from .data import make_splits
from .train import RunResult, TrainConfig, train_one
from .verdict import Arm, ArmResult, decide

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_sweep(n_seeds: int, steps: int) -> dict:
    splits = make_splits()
    cfg = TrainConfig(steps=steps)
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "results_log.jsonl"

    runs: dict[str, list[RunResult]] = {name: [] for name in ARM_CLASSES}
    with log_path.open("w") as log:
        for name in ARM_CLASSES:
            for seed in range(n_seeds):
                t = time.time()
                r = train_one(name, seed, splits, cfg)
                runs[name].append(r)
                row = asdict(r) | {"elapsed_s": round(time.time() - t, 2)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"{name:16} seed={seed} ood_test={r.ood_test_mse:.5f} "
                      f"indist={r.indist_mse:.5f} diverged={r.diverged} "
                      f"[{row['elapsed_s']}s]", flush=True)

    results: dict[Arm, ArmResult] = {}
    for name, rr in runs.items():
        ok = [r for r in rr if not r.diverged]
        results[Arm(name)] = ArmResult(
            arm=Arm(name),
            ood_test_mse=tuple(r.ood_test_mse for r in ok),
            indist_test_mse=tuple(r.indist_mse for r in ok),
            n_diverged=sum(1 for r in rr if r.diverged),
            params=rr[0].n_params,
            flops=rr[0].flops,
            ood_test_reads=1,  # eval_set on OOD-TEST is called exactly once per run (train.py)
        )

    report = decide(results, n_required=n_seeds)
    summary = {
        "config": {"n_seeds": n_seeds, "steps": steps, **asdict(cfg)},
        "verdict": report.verdict.value,
        "reasons": list(report.reasons),
        "criteria": report.criteria,
        "best_unstructured": report.best_unstructured.value if report.best_unstructured else None,
        "margin_observed": report.margin_observed,
        "p_value": report.p_value,
        "cliffs_delta": report.cliffs_delta,
        "divergence_rates": report.divergence_rates,
        "per_arm": {
            name: {
                "ood_test_mean": _mean(results[Arm(name)].ood_test_mse),
                "ood_test_std": _std(results[Arm(name)].ood_test_mse),
                "indist_mean": _mean(results[Arm(name)].indist_test_mse),
                "params": results[Arm(name)].params,
                "flops": results[Arm(name)].flops,
                "flops_vs_film": results[Arm(name)].flops / results[Arm.FILM].flops,
            }
            for name in ARM_CLASSES
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _mean(xs: tuple[float, ...]) -> float:
    return math.fsum(xs) / len(xs) if xs else math.nan


def _std(xs: tuple[float, ...]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else verdict.N_REQUIRED
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else TrainConfig().steps
    t = time.time()
    summary = run_sweep(n_seeds, steps)
    print("\n" + "=" * 60)
    print(f"VERDICT: {summary['verdict'].upper()}")
    for reason in summary["reasons"]:
        print(f"  - {reason}")
    print(f"total wall: {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
