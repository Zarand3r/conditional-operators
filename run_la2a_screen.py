"""Screen: Complex FiLM against FiLM on the real LA-2A recordings.

Validation only, three seeds, no test split read. The question is this field's own: does the
candidate model the device better than the incumbent, at 0.85x the conditioning parameters?
"""

import json
import statistics as st
import time

from conditional_operators import neuralfx as N

ARMS = ("film", "cfilm", "concat", "hyper")
SEEDS = 3
STEPS = 4000


def main():
    data = N.build_cache()
    rows = []
    for arm in ARMS:
        for seed in range(SEEDS):
            t0 = time.time()
            r = N.train_one_fx(arm, seed, data, STEPS)
            rows.append(r)
            print(f"{arm:8} seed={seed} train={r.get('train_esr', float('nan')):.4f} "
                  f"val={r.get('val_esr', float('nan')):.4f} [{time.time()-t0:.0f}s]", flush=True)

    by = {}
    for a in ARMS:
        v = [r["val_esr"] for r in rows if r["arm"] == a and not r["diverged"]]
        by[a] = (st.mean(v), st.pstdev(v)) if v else (float("nan"), float("nan"))
    film = by["film"][0]
    print(f"\n{'arm':8} {'val ESR':>10} {'sd':>8} {'vs film':>10} {'cond params':>12}", flush=True)
    for a, (mu, sd) in by.items():
        cp = next(r["cond_params"] for r in rows if r["arm"] == a and not r["diverged"])
        print(f"{a:8} {mu:10.4f} {sd:8.4f} {1 - mu/film:+9.1%} {cp:12,}", flush=True)

    with open("results/la2a_screen.json", "w") as fh:
        json.dump({"note": "validation only; no test split read; no verdict claimed",
                   "steps": STEPS, "seeds": SEEDS, "rows": rows,
                   "summary": {k: list(v) for k, v in by.items()}}, fh, indent=2)
    print("\nwritten to results/la2a_screen.json", flush=True)


if __name__ == "__main__":
    main()
