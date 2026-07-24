"""Stage-9: guidance as group power vs classifier-free guidance (STAGE9_SPEC.md).

Both arms train with 10% condition dropout (null condition = zero vector, which no real
condition equals since shape one-hots always carry a 1). Sampling:
  film_cfg  — standard adaLN DiT; CFG: eps = eps(0) + w*(eps(c) - eps(0)).
  cfilm_gp  — Complex-FiLM hybrid DiT (stage8.CFiLMDiT('hyb')); strength alpha applied by
              feeding alpha*c: exact group power on the linear phase channel, linear scaling of
              the magnitude head's input as the spec's disclosed approximation.
Metric: DDIM-50 generation MSE vs unique ground truth on 256 fixed OOD-TEST combos, per
strength in {1, 1.5, 2, 3, 5, 8}; in-dist at strength 1. One evaluation pass per trained model.

Run:  .venv/bin/python -m conditional_operators.stage9 [N] [STEPS]
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time

import torch

from . import stage6
from .stage4 import Data, DEVICE, RESULTS_DIR, _mean, _std
from .stage8 import CFiLMDiT
from .verdict import cliffs_delta, mann_whitney_u

STRENGTHS = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
DROPOUT = 0.1
ARMS_9 = ("film_cfg", "cfilm_gp")
PARITY, DEGRADE, BEST_PARITY, ALPHA9, CLIFF9 = 1.10, 0.5, 1.10, 0.01, 0.474


def _build(arm, seed):
    torch.manual_seed(seed)
    return (stage6.MiniDiT("film") if arm == "film_cfg" else CFiLMDiT("hyb")).to(DEVICE)


def ddim_cfg(model, c, noise, w):
    """DDIM-50 with classifier-free guidance at scale w (w=1 -> plain conditional)."""
    x = noise
    null = torch.zeros_like(c)
    ts = torch.linspace(stage6.T_STEPS, 0, stage6.DDIM_STEPS + 1, device=DEVICE).long()
    for i in range(stage6.DDIM_STEPS):
        t, t2 = ts[i], ts[i + 1]
        ab, ab2 = stage6.ABAR[t], stage6.ABAR[t2]
        tt = t.expand(x.shape[0])
        eps_c = model(x, tt, c)
        eps = eps_c if w == 1.0 else (model(x, tt, null) * (1 - w) + eps_c * w)
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        x = ab2.sqrt() * x0 + (1 - ab2).sqrt() * eps
    return x0


def train_one(arm, seed, data, splits, steps, batch=256, lr=1e-3):
    train_c, val_c, test_c = splits
    m = _build(arm, seed)
    ema = copy.deepcopy(m)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(seed)
    tc_all = stage6.cond_vec(train_c)
    throttle_ms = int(os.environ.get("STAGE9_THROTTLE_MS", "0"))
    diverged = False
    for step in range(steps):
        if throttle_ms and step % 5 == 0:
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            time.sleep(throttle_ms / 1000.0)
        idx = torch.randint(len(train_c), (batch,), generator=gen)
        x0 = stage6.combo_images(data, [train_c[i] for i in idx.tolist()])
        c = tc_all[idx.to(DEVICE)].clone()
        drop = (torch.rand(batch, generator=gen) < DROPOUT).to(DEVICE)
        c[drop] = 0.0                                        # null condition for CFG training
        t = torch.randint(1, stage6.T_STEPS, (batch,), generator=gen).to(DEVICE)
        ab = stage6.ABAR[t].view(-1, 1, 1, 1)
        eps = torch.randn(x0.shape, generator=gen).to(DEVICE)
        loss = torch.mean((m(ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t, c) - eps) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), m.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)

    row = dict(arm=arm, seed=seed, diverged=diverged)
    if diverged:
        row |= {f"ood_w{w}": math.nan for w in STRENGTHS} | {"indist": math.nan}
        return row

    @torch.no_grad()
    def gen_mse(combos, strength):
        eg = torch.Generator().manual_seed(123_000)          # identical combos+noise everywhere
        pick = torch.randperm(len(combos), generator=eg)[:stage6.EVAL_COMBOS].tolist()
        sel = [combos[i] for i in pick]
        gt = stage6.combo_images(data, sel)
        cv = stage6.cond_vec(sel)
        noise = torch.randn(gt.shape, generator=eg).to(DEVICE)
        outs = []
        for i in range(0, len(sel), 128):
            if arm == "film_cfg":
                outs.append(ddim_cfg(ema, cv[i:i + 128], noise[i:i + 128], strength))
            else:
                outs.append(stage6.ddim_sample(ema, cv[i:i + 128] * strength, noise[i:i + 128]))
        return torch.mean((torch.cat(outs) - gt) ** 2).item()

    for w in STRENGTHS:
        row[f"ood_w{w}"] = gen_mse(test_c, w)                # one registered eval pass
    row["indist"] = gen_mse(train_c, 1.0)
    return row


def decide9(runs):
    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])
    f1, g1 = _mean(arr("film_cfg", "ood_w1.0")), _mean(arr("cfilm_gp", "ood_w1.0"))
    # per-seed degradation ratios MSE(8)/MSE(1)
    fr = tuple(r["ood_w8.0"] / r["ood_w1.0"] for r in runs["film_cfg"] if not r["diverged"])
    gr = tuple(r["ood_w8.0"] / r["ood_w1.0"] for r in runs["cfilm_gp"] if not r["diverged"])
    _, p = mann_whitney_u(gr, fr); d = cliffs_delta(gr, fr)
    best_f = min(_mean(arr("film_cfg", f"ood_w{w}")) for w in STRENGTHS)
    best_g = min(_mean(arr("cfilm_gp", f"ood_w{w}")) for w in STRENGTHS)
    crit = {
        "AC-9.1": g1 <= PARITY * f1,
        "AC-9.2": _mean(gr) <= DEGRADE * _mean(fr) and p <= ALPHA9 and d <= -CLIFF9,
        "AC-9.3": best_g <= BEST_PARITY * best_f,
    }
    verdict = "confirmed" if all(crit.values()) else "kill"
    return verdict, crit, dict(parity_ratio=g1 / f1, degrade_film=_mean(fr),
                               degrade_gp=_mean(gr), p=p, delta=d,
                               best_film=best_f, best_gp=best_g)


def run(n_seeds, steps):
    data = Data()
    splits = stage6.make_combos()
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = {a: [] for a in ARMS_9}
    log_path = RESULTS_DIR / "stage9_log.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} runs", flush=True)
    with log_path.open("a") as log:
        for arm in ARMS_9:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                r = train_one(arm, seed, data, splits, steps)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:9} seed={seed} w1={r['ood_w1.0']:.4f} w3={r['ood_w3.0']:.4f} "
                      f"w8={r['ood_w8.0']:.4f} [{time.time()-t:.0f}s]", flush=True)

    verdict, crit, stats = decide9(runs)
    summary = {
        "stage": 9, "spec": "docs/specs/STAGE9_SPEC.md",
        "config": {"n_seeds": n_seeds, "steps": steps, "strengths": STRENGTHS,
                   "dropout": DROPOUT, "gp_arm": "cfilm_hyb (per pre-fixed selection rule; "
                   "cfilm_lin failed 8b)"},
        "final_verdict": verdict, "criteria": crit, **stats,
        "per_arm": {a: {f"ood_w{w}": _mean([r[f"ood_w{w}"] for r in runs[a] if not r["diverged"]])
                        for w in STRENGTHS}
                       | {"indist": _mean([r["indist"] for r in runs[a] if not r["diverged"]])}
                    for a in ARMS_9},
    }
    (RESULTS_DIR / "stage9_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-9 VERDICT: {s['final_verdict'].upper()}  criteria={s['criteria']}")
    for a in ARMS_9:
        p = s["per_arm"][a]
        print(f"  {a:9} " + " ".join(f"w{w}={p[f'ood_w{w}']:.4f}" for w in STRENGTHS))
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
