"""Stage-10: Stage-7 world model + latent-consistency loss (STAGE10_SPEC.md).

Identical to stage7 except training adds lambda * ||arm(a, z_s) - enc(x_{s+1})||^2 / d on an
independent single-step batch, for every arm equally. lambda = 1.0 (pre-registered).

Run:  .venv/bin/python -m conditional_operators.stage10 [N] [STEPS]
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
from .stage4 import Data, DEVICE, RESULTS_DIR, _mean, _std
from .stage7 import (GATE_ARMS7, TRAIN_TYPES7, VAL_PAIRS7, TEST_PAIRS7, WorldModel,
                     decide7, sample_rollouts)

LAMBDA = 1.0


def train_one(name, seed, data, steps, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    m = WorldModel(name).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    bce = nn.BCEWithLogitsLoss()
    throttle_ms = int(os.environ.get("STAGE10_THROTTLE_MS", "0"))
    diverged = False
    for step in range(steps):
        if throttle_ms and step % 5 == 0:
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            time.sleep(throttle_ms / 1000.0)
        x0, acts, x1 = sample_rollouts(data, TRAIN_TYPES7, (1, stage7.H_TRAIN_MAX), batch, gen)
        loss = bce(m(x0, acts), x1)
        # latent-consistency term on an independent single-step batch (identical for all arms)
        s0, sa, s1 = sample_rollouts(data, TRAIN_TYPES7, (1, 1), batch, gen)
        z_next_hat = m.cond(sa[:, 0], m.backbone.encode(s0))
        z_next = m.backbone.encode(s1)
        loss = loss + LAMBDA * torch.mean((z_next_hat - z_next) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()

    row = dict(arm=name, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    if diverged:
        row |= {k: math.nan for k in ("indist", "ood_val", "ood_pairs", "h10", "h20")}
        return row

    @torch.no_grad()
    def mse(types, hr):
        eg2 = torch.Generator().manual_seed(140_000)
        tot = 0.0
        for _ in range(4):
            x0, acts, x1 = sample_rollouts(data, types, hr, 512, eg2)
            tot += torch.mean((torch.sigmoid(m(x0, acts)) - x1) ** 2).item()
        return tot / 4

    row |= dict(indist=mse(TRAIN_TYPES7, (1, stage7.H_TRAIN_MAX)),
                ood_val=mse(VAL_PAIRS7, (3, 3)),
                ood_pairs=mse(TEST_PAIRS7, (3, 3)),
                h10=mse(TRAIN_TYPES7, (10, 10)),
                h20=mse(TRAIN_TYPES7, (20, 20)))
    return row


def run(n_seeds, steps):
    data = Data()
    RESULTS_DIR.mkdir(exist_ok=True)
    all_arms = GATE_ARMS7 + ("proposed_mlp_gs",)
    runs = {a: [] for a in all_arms}
    log_path = RESULTS_DIR / "stage10_log.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} runs", flush=True)
    with log_path.open("a") as log:
        for arm in all_arms:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                r = train_one(arm, seed, data, steps)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:16} seed={seed} pairs={r['ood_pairs']:.5f} h20={r['h20']:.5f} "
                      f"indist={r['indist']:.5f} [{time.time()-t:.0f}s]", flush=True)

    verdict, crit, stats = decide7(runs)
    # Stage-10 margins (spec): AC-10.1 = stage7's AC-2; AC-10.2 = AC-1; AC-10.3 = AC-3; AC-10.4 = AC-4.
    crit = {"AC-10.1": crit["AC-2"], "AC-10.2": crit["AC-1"],
            "AC-10.3": crit["AC-3"], "AC-10.4": crit["AC-4"]}
    verdict = "confirmed" if all(crit.values()) else "kill"
    summary = {
        "stage": 10, "spec": "docs/specs/STAGE10_SPEC.md", "lambda": LAMBDA,
        "config": {"n_seeds": n_seeds, "steps": steps},
        "final_verdict": verdict, "criteria": crit, **stats,
        "per_arm": {a: {k: _mean([r[k] for r in runs[a] if not r["diverged"]])
                        for k in ("indist", "ood_pairs", "h10", "h20")}
                       | {"params": runs[a][0]["params"], "flops": runs[a][0]["flops"]}
                    for a in all_arms},
    }
    (RESULTS_DIR / "stage10_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-10 VERDICT: {s['final_verdict'].upper()}  criteria={s['criteria']}")
    pa = s["per_arm"]
    for a in ("film", s["best_unstructured"], "proposed"):
        print(f"  {a:16} pairs={pa[a]['ood_pairs']:.5f} h20={pa[a]['h20']:.5f} "
              f"indist={pa[a]['indist']:.5f}")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
