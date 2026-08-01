"""Decision run for docs/specs/LA2A_SPEC.md. Reads the test split exactly once.

AC-2 names the strongest baseline in advance rather than a fixed reference arm -- the criterion
the earlier gates lacked, whose absence produced two false CONFIRMEDs on PDEBench.
"""

import json
import os
import statistics as st
import time

from conditional_operators import neuralfx as N
from conditional_operators.verdict import cliffs_delta, mann_whitney_u

ARMS = ("film", "cfilm", "concat", "hyper")
GATED, INCUMBENT = "cfilm", "film"
SEEDS, STEPS = 10, 4000
LOG = "results/la2a_log.jsonl"


def evaluate_test(arm, seed, data, steps):
    """Same training as the screen, but the test split is read for the decision."""
    import torch
    torch.manual_seed(seed)
    m = N.TCN(arm).to(N.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    tr = data["Train"]
    n, w = tr["x"].shape[0], m.receptive
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        N._throttle_fx(step)
        idx = torch.randint(n, (16,), generator=g)
        x = tr["x"][idx].unsqueeze(1).to(N.DEV)
        c = tr["c"][idx].to(N.DEV)
        y = tr["y"][idx].unsqueeze(1).to(N.DEV)
        loss = ((m(x, c)[..., w:] - y[..., w:]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()

    @torch.no_grad()
    def ev(split):
        d = data[split]
        tot = k = 0
        for i in range(0, d["x"].shape[0], 32):
            tot += N.esr(m(d["x"][i:i + 32].unsqueeze(1).to(N.DEV), d["c"][i:i + 32].to(N.DEV)),
                         d["y"][i:i + 32].unsqueeze(1).to(N.DEV), w)
            k += 1
        return tot / k

    return dict(arm=arm, seed=seed, diverged=False, cond_params=m.cond_params(),
                train_esr=ev("Train"), val_esr=ev("Val"), test_esr=ev("Test"))


def main():
    data = N.build_cache()
    runs, done = {a: [] for a in ARMS}, set()
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        print(f"resuming: {len(done)} runs done", flush=True)
    with open(LOG, "a") as log:
        for arm in ARMS:
            for seed in range(SEEDS):
                if (arm, seed) in done:
                    continue
                t0 = time.time()
                r = evaluate_test(arm, seed, data, STEPS)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:8} seed={seed} test={r['test_esr']:.4f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    def arr(a):
        return tuple(r["test_esr"] for r in runs[a] if not r["diverged"])

    means = {a: st.mean(arr(a)) for a in ARMS}
    best_base = min((a for a in ARMS if a != GATED), key=lambda a: means[a])
    cand = arr(GATED)

    def beats(other):
        m = (means[other] - means[GATED]) / means[other]
        _, p = mann_whitney_u(cand, arr(other))
        d = cliffs_delta(cand, arr(other))
        return dict(margin=m, p=p, delta=d, pass_=bool(m > 0 and p <= 0.01 and d <= -0.474))

    vs_inc, vs_best = beats(INCUMBENT), beats(best_base)
    cp = {a: next(r["cond_params"] for r in runs[a]) for a in ARMS}
    crit = {"AC-1": vs_inc["pass_"], "AC-2": vs_best["pass_"],
            "AC-3": all(sum(1 for r in runs[a] if r["diverged"]) <= 2 for a in ARMS),
            "AC-4": cp[GATED] <= cp[INCUMBENT]}
    verdict = "confirmed" if all(crit.values()) else "kill"

    summary = {"experiment": "la2a-cfilm", "spec": "docs/specs/LA2A_SPEC.md",
               "gated_arm": GATED, "strongest_baseline": best_base,
               "final_verdict": verdict, "criteria": crit,
               "vs_incumbent": vs_inc, "vs_strongest_baseline": vs_best,
               "per_arm": {a: {"test_esr": means[a], "test_sd": st.pstdev(arr(a)),
                               "val_esr": st.mean([r["val_esr"] for r in runs[a]]),
                               "cond_params": cp[a]} for a in ARMS}}
    json.dump(summary, open("results/la2a_summary.json", "w"), indent=2)

    print("\n" + "=" * 60)
    print(f"LA2A VERDICT ({GATED}): {verdict.upper()}")
    for k, v in crit.items():
        print(f"  {k}: {'pass' if v else 'FAIL'}")
    print(f"  strongest baseline was: {best_base}")
    print(f"  vs {INCUMBENT}: {vs_inc['margin']:+.1%} p={vs_inc['p']:.3g} d={vs_inc['delta']:.2f}")
    print(f"  vs {best_base}: {vs_best['margin']:+.1%} p={vs_best['p']:.3g}")
    print(f"\n  {'arm':8} {'test ESR':>10} {'sd':>8} {'cond params':>12}")
    for a in ARMS:
        s = summary["per_arm"][a]
        print(f"  {a:8} {s['test_esr']:10.4f} {s['test_sd']:8.4f} {s['cond_params']:12,}")


if __name__ == "__main__":
    main()
