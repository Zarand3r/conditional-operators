"""Stage-7: action-conditioned world model on sprite-world (STAGE7_SPEC.md).

State = dSprites factor spec; 8 geometric actions (+/- posX, posY, scale, orient; 1 grid step,
orientation wraps, others clamp). Model: shared conv Backbone; the transition is PURELY the
conditioning arm applied recurrently in latent space: z_{t+1} = arm(a_t, z_t); decode at the end.
OOD = unseen action-TYPE pairs (length 3) and never-trained horizons (10, 20).

Run:  .venv/bin/python -m conditional_operators.stage7 [N_SEEDS] [STEPS]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
from torch import nn

from .stage4 import ARM_CLASSES, Backbone, Data, DEVICE, RESULTS_DIR, _mean, _std
from .verdict import cliffs_delta, mann_whitney_u

DC7 = 8                                  # signed one-hot over 8 actions
# factor axes (dsprites latents cols): posX=4, posY=5, scale=2, orient=3
AXES = ("posX", "posY", "scale", "orient")
AX_COL = {"posX": 4, "posY": 5, "scale": 2, "orient": 3}
AX_SIZE = {"posX": 32, "posY": 32, "scale": 6, "orient": 40}
AX_WRAP = {"posX": False, "posY": False, "scale": False, "orient": True}
# action index = 2*axis + (0 for +, 1 for -)

TRAIN_PAIRS7 = (("posX", "posY"), ("scale", "orient"), ("posX", "scale"))
VAL_PAIRS7 = (("posY", "orient"),)
TEST_PAIRS7 = (("posX", "orient"), ("posY", "scale"))
SINGLES7 = tuple((a,) for a in AXES)
TRAIN_TYPES7 = SINGLES7 + TRAIN_PAIRS7
H_TRAIN_MAX = 3
H_LONG = (10, 20)

MARGIN_A, MARGIN_H, ALPHA7, CLIFF7, INDIST7, GROWTH7 = 0.20, 0.30, 0.01, 0.474, 1.10, 0.5


def sample_rollouts(data: Data, types, horizon_range, n, gen):
    """n rollouts. Returns (x0 [n,1,64,64], actions [n,H,8], x_target, H) with H = max sampled
    horizon (sequences shorter than H are padded with zero-actions = identity steps)."""
    lo, hi = horizon_range
    H = hi
    lat = torch.stack([torch.randint(int(s), (n,), generator=gen)
                       for s in data.sizes.tolist()], dim=1)              # [n, 6]
    lengths = torch.randint(lo, hi + 1, (n,), generator=gen)
    which = torch.randint(len(types), (n,), generator=gen)
    acts = torch.zeros(n, H, DC7)
    cur = lat.clone()
    for t in range(H):
        active = (lengths > t)
        # choose an axis for each rollout from its type set (uniform each step -> interleaving)
        for ti, tp in enumerate(types):
            mask = active & (which == ti)
            m = int(mask.sum())
            if m == 0:
                continue
            axi = torch.randint(len(tp), (m,), generator=gen)
            sign = torch.randint(0, 2, (m,), generator=gen)               # 0:+  1:-
            rows = mask.nonzero(as_tuple=True)[0]
            for j in range(m):
                ax = tp[axi[j].item()]
                col, size = AX_COL[ax], AX_SIZE[ax]
                s = 1 - 2 * sign[j].item()
                v = cur[rows[j], col].item() + s
                if AX_WRAP[ax]:
                    v %= size
                else:
                    v = max(0, min(size - 1, v))
                s_eff = v - cur[rows[j], col].item()                       # clamped => may be 0
                if AX_WRAP[ax] and abs(s_eff) > 1:                         # wrapped
                    s_eff = s
                cur[rows[j], col] = v
                a_idx = 2 * AXES.index(ax) + (0 if s_eff >= 0 else 1)
                acts[rows[j], t, a_idx] = float(abs(s_eff))                # 0 if clamped no-op
    i0 = (lat * data.bases).sum(1).to(DEVICE)
    i1 = (cur * data.bases).sum(1).to(DEVICE)
    x0 = data.imgs[i0].float().unsqueeze(1)
    x1 = data.imgs[i1].float().unsqueeze(1)
    return x0, acts.to(DEVICE), x1


class WorldModel(nn.Module):
    def __init__(self, arm_name: str):
        super().__init__()
        self.backbone = Backbone()
        self.cond = ARM_CLASSES[arm_name](dc=DC7)

    def forward(self, x0, actions):
        z = self.backbone.encode(x0)
        for t in range(actions.shape[1]):
            z = self.cond(actions[:, t], z)
        return self.backbone.decode(z)


def train_one(name, seed, data, steps, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    m = WorldModel(name).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    bce = nn.BCEWithLogitsLoss()
    throttle_ms = int(os.environ.get("STAGE7_THROTTLE_MS", "0"))
    diverged = False
    for step in range(steps):
        if throttle_ms and step % 5 == 0:
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            time.sleep(throttle_ms / 1000.0)
        x0, acts, x1 = sample_rollouts(data, TRAIN_TYPES7, (1, H_TRAIN_MAX), batch, gen)
        loss = bce(m(x0, acts), x1)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()

    row = dict(arm=name, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    if diverged:
        row |= {k: math.nan for k in
                ("indist", "ood_val", "ood_pairs", "h10", "h20")}
        return row

    @torch.no_grad()
    def mse(types, hr):
        eg2 = torch.Generator().manual_seed(140_000)      # identical eval rollouts across arms
        tot = 0.0
        for _ in range(4):
            x0, acts, x1 = sample_rollouts(data, types, hr, 512, eg2)
            tot += torch.mean((torch.sigmoid(m(x0, acts)) - x1) ** 2).item()
        return tot / 4

    row |= dict(indist=mse(TRAIN_TYPES7, (1, H_TRAIN_MAX)),
                ood_val=mse(VAL_PAIRS7, (3, 3)),
                ood_pairs=mse(TEST_PAIRS7, (3, 3)),        # OOD-TEST(a): single read
                h10=mse(TRAIN_TYPES7, (10, 10)),           # HORIZON: single read
                h20=mse(TRAIN_TYPES7, (20, 20)))
    return row


GATE_ARMS7 = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")
UNSTRUCT7 = ("hypernet", "dynamic_linear")


def decide7(runs):
    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])
    bu = min(UNSTRUCT7, key=lambda a: _mean(arr(a, "ood_pairs")))
    prop_p, bu_p = arr("proposed", "ood_pairs"), arr(bu, "ood_pairs")
    prop_h, bu_h = arr("proposed", "h20"), arr(bu, "h20")
    m_a = (_mean(bu_p) - _mean(prop_p)) / _mean(bu_p)
    m_h = (_mean(bu_h) - _mean(prop_h)) / _mean(bu_h)
    _, p_a = mann_whitney_u(prop_p, bu_p); d_a = cliffs_delta(prop_p, bu_p)
    _, p_h = mann_whitney_u(prop_h, bu_h); d_h = cliffs_delta(prop_h, bu_h)
    growth_prop = _mean(prop_h) / _mean(arr("proposed", "indist"))
    growth_bu = _mean(bu_h) / _mean(arr(bu, "indist"))
    film_flops = runs["film"][0]["flops"]
    crit = {
        "AC-1": m_a >= MARGIN_A and p_a <= ALPHA7 and d_a <= -CLIFF7,
        "AC-2": (m_h >= MARGIN_H and p_h <= ALPHA7 and d_h <= -CLIFF7
                 and growth_prop <= GROWTH7 * growth_bu),
        "AC-3": _mean(arr("proposed", "indist")) <= INDIST7 * _mean(arr(bu, "indist")),
        "AC-4": runs["proposed"][0]["flops"] <= 1.20 * film_flops,
    }
    verdict = "confirmed" if all(crit.values()) else "kill"
    return verdict, crit, dict(best_unstructured=bu, margin_pairs=m_a, margin_h20=m_h,
                               p_pairs=p_a, p_h20=p_h, delta_pairs=d_a, delta_h20=d_h,
                               growth_proposed=growth_prop, growth_best_unstructured=growth_bu)


def run(n_seeds, steps):
    data = Data()
    RESULTS_DIR.mkdir(exist_ok=True)
    all_arms = GATE_ARMS7 + ("proposed_mlp_gs",)
    runs = {a: [] for a in all_arms}
    log_path = RESULTS_DIR / "stage7_log.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r)
            done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} completed runs found", flush=True)
    with log_path.open("a") as log:
        for arm in all_arms:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                r = train_one(arm, seed, data, steps)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n")
                log.flush(); os.fsync(log.fileno())
                print(f"{arm:16} seed={seed} pairs={r['ood_pairs']:.5f} h20={r['h20']:.5f} "
                      f"indist={r['indist']:.5f} [{time.time()-t:.0f}s]", flush=True)

    verdict, crit, stats = decide7(runs)
    summary = {
        "stage": 7, "spec": "docs/specs/STAGE7_SPEC.md (pre-registered 2026-07-22)",
        "task": "sprite-world action-conditioned world model; recurrent latent transitions; "
                "OOD = unseen action-type pairs + horizons 10/20 (trained <=3)",
        "config": {"n_seeds": n_seeds, "steps": steps},
        "final_verdict": verdict, "criteria": crit, **stats,
        "per_arm": {a: {k: _mean([r[k] for r in runs[a] if not r["diverged"]])
                        for k in ("indist", "ood_pairs", "h10", "h20")}
                       | {k + "_std": _std([r[k] for r in runs[a] if not r["diverged"]])
                          for k in ("ood_pairs", "h20")}
                       | {"params": runs[a][0]["params"], "flops": runs[a][0]["flops"],
                          "flops_vs_film": runs[a][0]["flops"] / runs["film"][0]["flops"]}
                    for a in all_arms},
    }
    (RESULTS_DIR / "stage7_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-7 VERDICT: {s['final_verdict'].upper()}  criteria={s['criteria']}")
    pa = s["per_arm"]
    for a in ("film", s["best_unstructured"], "proposed", "proposed_mlp_gs"):
        print(f"  {a:16} pairs={pa[a]['ood_pairs']:.5f} h10={pa[a]['h10']:.5f} "
              f"h20={pa[a]['h20']:.5f} indist={pa[a]['indist']:.5f}")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
