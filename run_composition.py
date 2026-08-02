"""Decision run for docs/specs/COMPOSITION_SPEC.md.

Sweeps the two axes independently: n (simultaneously changed factors) and s (per-factor
magnitude). Difficulty rises with both for every arm, but only the additive arms should suffer
the second-order term, so the prediction is a scaling shape rather than a direction.
"""

import itertools
import json
import math
import os
import time

import torch

from conditional_operators import improved, stage4  # noqa: F401  (registers the attention arms)
from conditional_operators.stage4 import DC, DEVICE, RESULTS_DIR, _mean, _std
from conditional_operators.verdict import cliffs_delta, mann_whitney_u

ARMS = ("film", "xattn", "xattn_linear", "proposed")
ADDITIVE = ("film", "xattn", "xattn_linear")
NS = (1, 2, 3, 4)
SS = (0.25, 0.5, 1.0, 2.0)
SEEDS, STEPS = 10, 12_000
LOG = RESULTS_DIR / "composition_log.jsonl"


def sample_ns(data, n, s, count, gen):
    """`count` triplets in which exactly `n` factors change, each scaled by `s`."""
    types = tuple(itertools.combinations(range(DC), n))
    x1, d, x2 = data.sample(types, count, gen)
    return x1, d * s, x2


def train_one(arm, seed, data, steps):
    torch.manual_seed(seed)
    m = stage4.Model(arm).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    bce = torch.nn.BCEWithLogitsLoss()
    ms = int(os.environ.get("STAGE4_THROTTLE_MS", "0"))
    for step in range(steps):
        if ms and step % 5 == 0:
            torch.cuda.synchronize(); time.sleep(ms / 1000.0)
        x1, d, x2 = data.sample(stage4.TRAIN_TYPES, 256, gen)
        loss = bce(m(x1, d), x2)
        if not math.isfinite(loss.item()):
            return dict(arm=arm, seed=seed, diverged=True)
        opt.zero_grad(); loss.backward(); opt.step()

    cells = {}
    with torch.no_grad():
        for n, s in itertools.product(NS, SS):
            eg = torch.Generator().manual_seed(77_000)
            tot = 0.0
            for _ in range(4):
                x1, d, x2 = sample_ns(data, n, s, 256, eg)
                tot += torch.mean((torch.sigmoid(m(x1, d)) - x2) ** 2).item()
            cells[f"n{n}_s{s}"] = tot / 4
    return dict(arm=arm, seed=seed, diverged=False, cells=cells,
                params=m.cond.n_params(), flops=m.cond.flops())


def main():
    data = stage4.Data()
    RESULTS_DIR.mkdir(exist_ok=True)
    runs, done = {a: [] for a in ARMS}, set()
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        print(f"resuming: {len(done)} runs done", flush=True)
    with LOG.open("a") as log:
        for arm in ARMS:
            for seed in range(SEEDS):
                if (arm, seed) in done:
                    continue
                t0 = time.time()
                r = train_one(arm, seed, data, STEPS)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:14} seed={seed} n2_s1.0={r['cells']['n2_s1.0']:.5f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    def cell(a, key):
        return tuple(r["cells"][key] for r in runs[a] if not r["diverged"])

    # advantage of `proposed` over the best additive arm, per cell
    adv, best = {}, {}
    for n, s in itertools.product(NS, SS):
        k = f"n{n}_s{s}"
        b = min(ADDITIVE, key=lambda a: _mean(cell(a, k)))
        best[k] = b
        adv[k] = 1 - _mean(cell("proposed", k)) / _mean(cell(b, k))

    mono_s = all(adv[f"n2_s{SS[i+1]}"] > adv[f"n2_s{SS[i]}"] for i in range(len(SS) - 1))
    mono_n = all(adv[f"n{NS[i+1]}_s1.0"] > adv[f"n{NS[i]}_s1.0"] for i in range(len(NS) - 1))
    k0 = "n1_s0.25"
    spread0 = (max(_mean(cell(a, k0)) for a in ARMS) / min(_mean(cell(a, k0)) for a in ARMS)) - 1

    _, p = mann_whitney_u(cell("proposed", "n2_s1.0"), cell(best["n2_s1.0"], "n2_s1.0"))
    d = cliffs_delta(cell("proposed", "n2_s1.0"), cell(best["n2_s1.0"], "n2_s1.0"))
    crit = {
        "AC-1_scales_in_s": mono_s, "AC-1_scales_in_n": mono_n,
        "AC-2_no_edge_at_weak_single": spread0 <= 0.10,
        "AC-3_divergence": all(sum(1 for r in runs[a] if r["diverged"]) <= 2 for a in ARMS),
        "AC-4_budget": all(runs[a][0]["flops"] <= 1.20 * runs["film"][0]["flops"] for a in ARMS),
    }
    verdict = "confirmed" if all(crit.values()) else "kill"
    summary = {"experiment": "composition-order", "spec": "docs/specs/COMPOSITION_SPEC.md",
               "final_verdict": verdict, "criteria": crit,
               "advantage_by_cell": adv, "strongest_additive_by_cell": best,
               "spread_at_n1_s0.25": spread0, "p_at_n2_s1.0": p, "delta_at_n2_s1.0": d,
               "per_arm_cells": {a: {k: _mean(cell(a, k)) for k in adv} for a in ARMS}}
    (RESULTS_DIR / "composition_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print(f"COMPOSITION VERDICT: {verdict.upper()}")
    for k, v in crit.items():
        print(f"  {k}: {'pass' if v else 'FAIL'}")
    print(f"\n  advantage of proposed over the best additive arm:")
    print(f"  {'':6}" + "".join(f"{'s=' + str(s):>10}" for s in SS))
    for n in NS:
        print(f"  n={n}  " + "".join(f"{adv[f'n{n}_s{s}']:+9.1%} " for s in SS))
    print(f"\n  spread at n=1, s=0.25: {spread0:.1%} (AC-2 wants <= 10%)")
    print(f"  p={p:.2g} delta={d:.2f} at n=2, s=1.0")


if __name__ == "__main__":
    main()
