"""Stage-6: the adaLN swap — Lie conditioning inside a real (mini) diffusion transformer.

Class/factor-conditional DDPM on dSprites 64x64 with a width-128, 6-block DiT. The ONLY thing
varied is the class-conditioning path of each block (STAGE6_SPEC.md):
  film     — deployed DiT standard: adaLN modulation from (t_emb + c_emb)
  hypernet — adaLN(t) + per-sample dense (I + dW(c)) on block-entry tokens (shared head)
  proposed — adaLN(t) + R(W_l c) canonical-pair rotation of block-entry tokens (W_l linear,
             bias-free, zero-init). No explicit P at this site: the surrounding dense QKV/MLP
             projections learn the basis, as RoPE deploys. Composition in c exact per site.
  mlp_gs   — ablation: same site, angles from an MLP head on c_emb.

Full factor conditioning makes the target image UNIQUE, so OOD generation quality is the pixel
MSE of the DDIM-50 sample (fixed per-combo init noise) vs. the ground-truth image, on held-out
factor combinations. Run:  .venv/bin/python -m conditional_operators.stage6 [N_SEEDS] [STEPS]
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time

import torch
from torch import nn

from .stage3 import _rotate
from .stage4 import Data, DEVICE, RESULTS_DIR, _mean, _std
from .verdict import cliffs_delta, mann_whitney_u

DIM, BLOCKS, HEADS, PATCH = 128, 6, 4, 8
TOKENS = (64 // PATCH) ** 2
CDIM = 7                      # shape one-hot(3) + 4 normalized scalars
T_STEPS = 1000
DDIM_STEPS = 50
SPLIT_SEED = 5150
EVAL_COMBOS = 256

ARMS6 = ("film", "hypernet", "proposed", "mlp_gs")

# pre-registered margins (STAGE6_SPEC.md)
MARGIN6, ALPHA6, CLIFF6, INDIST6 = 0.20, 0.01, 0.474, 1.10


# ---------------------------------------------------------------- combos, conditions, splits

def make_combos():
    """9,216-combo grid (shape, scale, orient/5, posX/4, posY/4) and 70/15/15 split."""
    combos = []
    for sh in range(3):
        for sc in range(6):
            for o in range(8):
                for x in range(8):
                    for y in range(8):
                        combos.append((sh, sc, o * 5, x * 4, y * 4))
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(combos), generator=g).tolist()
    combos = [combos[i] for i in perm]
    n = len(combos)
    tr, va = int(0.7 * n), int(0.85 * n)
    return combos[:tr], combos[tr:va], combos[va:]


def cond_vec(combos: list[tuple]) -> torch.Tensor:
    c = torch.zeros(len(combos), CDIM)
    for i, (sh, sc, o, x, y) in enumerate(combos):
        c[i, sh] = 1.0
        c[i, 3] = sc / 5.0
        c[i, 4] = o / 35.0
        c[i, 5] = x / 28.0
        c[i, 6] = y / 28.0
    return c.to(DEVICE)


def combo_images(data: Data, combos: list[tuple]) -> torch.Tensor:
    lat = torch.zeros(len(combos), 6, dtype=torch.int64)
    for i, (sh, sc, o, x, y) in enumerate(combos):
        lat[i, 1], lat[i, 2], lat[i, 3], lat[i, 4], lat[i, 5] = sh, sc, o, x, y
    idx = (lat * data.bases).sum(1).to(DEVICE)
    return data.imgs[idx].float().unsqueeze(1) * 2.0 - 1.0          # [-1, 1]


# ---------------------------------------------------------------- diffusion schedule

def cosine_alphabar(t: torch.Tensor) -> torch.Tensor:
    s = 0.008
    return torch.cos((t / T_STEPS + s) / (1 + s) * math.pi / 2) ** 2


ABAR = cosine_alphabar(torch.arange(T_STEPS + 1).float()).to(DEVICE)


# ---------------------------------------------------------------- mini-DiT

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.n2 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))
        self.mod = nn.Linear(DIM, 6 * DIM)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)

    def forward(self, x, m):
        s1, sc1, g1, s2, sc2, g2 = self.mod(m).chunk(6, dim=1)
        h = modulate(self.n1(x), s1, sc1)
        x = x + g1.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.n2(x), s2, sc2))
        return x


class MiniDiT(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        self.arm = arm
        self.patch = nn.Conv2d(1, DIM, PATCH, PATCH)
        self.pos = nn.Parameter(torch.zeros(1, TOKENS, DIM))
        self.t_mlp = nn.Sequential(nn.Linear(DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.c_mlp = nn.Sequential(nn.Linear(CDIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.blocks = nn.ModuleList(Block() for _ in range(BLOCKS))
        self.final_n = nn.LayerNorm(DIM, elementwise_affine=False)
        self.final_mod = nn.Linear(DIM, 2 * DIM)
        nn.init.zeros_(self.final_mod.weight); nn.init.zeros_(self.final_mod.bias)
        self.final = nn.Linear(DIM, PATCH * PATCH)
        nn.init.zeros_(self.final.weight); nn.init.zeros_(self.final.bias)
        if arm == "hypernet":
            self.dw = nn.Linear(DIM, DIM * DIM)          # shared across blocks, zero-init
            nn.init.zeros_(self.dw.weight); nn.init.zeros_(self.dw.bias)
        elif arm == "proposed":
            # SINGLE rotation site at transformer entry (pre-run amendment: per-block rotation
            # costs 4.4x film's marginal class path, violating registered AC-4; the entry
            # rotation is bijective and persists through all blocks). W acts on RAW c.
            self.W = nn.Linear(CDIM, DIM // 2, bias=False)
            nn.init.zeros_(self.W.weight)
        elif arm == "mlp_gs":
            self.head = nn.Linear(DIM, DIM // 2)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    @staticmethod
    def t_embed(t):
        half = DIM // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    def forward(self, x, t, c):
        x = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        te = self.t_mlp(self.t_embed(t))
        ce = self.c_mlp(c)
        m = te + ce if self.arm == "film" else te        # class path via mod only for film
        if self.arm == "hypernet":
            dw = self.dw(ce).view(-1, DIM, DIM)
        if self.arm == "proposed":
            n, tok, d = x.shape
            x = _rotate(self.W(c).repeat_interleave(tok, 0), x.reshape(n * tok, d)).view(n, tok, d)
        elif self.arm == "mlp_gs":
            n, tok, d = x.shape
            x = _rotate(self.head(ce).repeat_interleave(tok, 0), x.reshape(n * tok, d)).view(n, tok, d)
        for blk in self.blocks:
            if self.arm == "hypernet":
                x = x + torch.einsum("nij,ntj->nti", dw, x)
            x = blk(x, m)
        s, sc = self.final_mod(m).chunk(2, dim=1)
        x = self.final(modulate(self.final_n(x), s, sc))
        n = x.shape[0]
        side = 64 // PATCH
        return x.view(n, side, side, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(n, 1, 64, 64)

    def class_path_flops(self) -> int:
        """Per-sample FLOPs of the class-conditioning path (encoder + heads + apply)."""
        enc = 2 * CDIM * DIM + 2 * DIM * DIM
        if self.arm == "film":
            return enc                                    # joins existing mod adds (~free)
        if self.arm == "hypernet":
            return enc + 2 * DIM * DIM * DIM // 1 + BLOCKS * TOKENS * 2 * DIM * DIM
        if self.arm == "proposed":
            return 2 * CDIM * (DIM // 2) + TOKENS * 3 * DIM              # single site, raw c
        return enc + 2 * DIM * (DIM // 2) + TOKENS * 3 * DIM             # mlp_gs, single site


# ---------------------------------------------------------------- train / sample / eval

def ddim_sample(model, c, noise):
    x = noise
    ts = torch.linspace(T_STEPS, 0, DDIM_STEPS + 1, device=DEVICE).long()
    for i in range(DDIM_STEPS):
        t, t2 = ts[i], ts[i + 1]
        ab, ab2 = ABAR[t], ABAR[t2]
        eps = model(x, t.expand(x.shape[0]), c)
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x0 = x0.clamp(-1, 1)
        x = ab2.sqrt() * x0 + (1 - ab2).sqrt() * eps
    return x0


def train_one(arm, seed, data, splits, steps, batch=256, lr=1e-3):
    train_c, val_c, test_c = splits
    torch.manual_seed(seed)
    model = MiniDiT(arm).to(DEVICE)
    ema = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(seed)
    tc_all = cond_vec(train_c)
    throttle_ms = int(os.environ.get("STAGE6_THROTTLE_MS", "0"))
    diverged = False
    for step in range(steps):
        if throttle_ms and step % 5 == 0:
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            time.sleep(throttle_ms / 1000.0)
        idx = torch.randint(len(train_c), (batch,), generator=gen)
        x0 = combo_images(data, [train_c[i] for i in idx.tolist()])
        c = tc_all[idx.to(DEVICE)]
        t = torch.randint(1, T_STEPS, (batch,), generator=gen).to(DEVICE)
        ab = ABAR[t].view(-1, 1, 1, 1)
        eps = torch.randn(x0.shape, generator=gen).to(DEVICE)
        loss = torch.mean((model(ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t, c) - eps) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)

    row = dict(arm=arm, seed=seed, diverged=diverged,
               class_path_flops=model.class_path_flops(),
               params=sum(p.numel() for p in model.parameters()))
    if diverged:
        row |= dict(indist=math.nan, ood_val=math.nan, ood_test=math.nan)
        return row

    @torch.no_grad()
    def gen_mse(combos, tag):
        eg = torch.Generator().manual_seed(123_000)       # same eval combos + noise for all arms
        pick = torch.randperm(len(combos), generator=eg)[:EVAL_COMBOS].tolist()
        sel = [combos[i] for i in pick]
        gt = combo_images(data, sel)
        c = cond_vec(sel)
        noise = torch.randn(gt.shape, generator=eg).to(DEVICE)
        outs = []
        for i in range(0, len(sel), 128):
            outs.append(ddim_sample(ema, c[i:i + 128], noise[i:i + 128]))
        x0 = torch.cat(outs)
        return torch.mean((x0 - gt) ** 2).item()

    row |= dict(indist=gen_mse(train_c, "train"), ood_val=gen_mse(val_c, "val"),
                ood_test=gen_mse(test_c, "test"))          # single OOD-TEST read
    return row


def decide6(runs):
    """Scoped pre-registered gate (STAGE6_SPEC.md)."""
    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])
    prop, film, hyp = arr("proposed", "ood_test"), arr("film", "ood_test"), arr("hypernet", "ood_test")
    m1 = (_mean(film) - _mean(prop)) / _mean(film)
    m2 = (_mean(hyp) - _mean(prop)) / _mean(hyp)
    _, p1 = mann_whitney_u(prop, film); d1 = cliffs_delta(prop, film)
    _, p2 = mann_whitney_u(prop, hyp); d2 = cliffs_delta(prop, hyp)
    crit = {
        "AC-1": m1 >= MARGIN6 and p1 <= ALPHA6 and d1 <= -CLIFF6,
        "AC-2": m2 >= MARGIN6 and p2 <= ALPHA6 and d2 <= -CLIFF6,
        "AC-3": _mean(arr("proposed", "indist")) <= INDIST6 * _mean(arr("film", "indist")),
        "AC-4": runs["proposed"][0]["class_path_flops"] <= 1.20 * max(
            runs["film"][0]["class_path_flops"], 1),
    }
    verdict = "confirmed" if all(crit.values()) else "kill"
    return verdict, crit, dict(margin_vs_film=m1, margin_vs_hypernet=m2, p_vs_film=p1,
                               p_vs_hypernet=p2, delta_vs_film=d1, delta_vs_hypernet=d2)


def run(n_seeds, steps):
    data = Data()
    splits = make_combos()
    runs = {a: [] for a in ARMS6}
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / "stage6_log.jsonl"
    done = set()
    if log_path.exists():                                   # resume after crash/power loss
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r)
            done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} completed runs found", flush=True)
    with log_path.open("a") as log:
        for arm in ARMS6:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                r = train_one(arm, seed, data, splits, steps)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n")
                log.flush(); os.fsync(log.fileno())          # survive hard power loss
                print(f"{arm:10} seed={seed} ood_test={r['ood_test']:.5f} "
                      f"indist={r['indist']:.5f} [{time.time()-t:.0f}s]", flush=True)
    verdict, crit, stats = decide6(runs)
    summary = {
        "stage": 6, "task": "mini-DiT conditional diffusion on dSprites; adaLN class-path swap; "
                            "OOD = generation MSE on held-out factor combinations",
        "config": {"n_seeds": n_seeds, "steps": steps, "dim": DIM, "blocks": BLOCKS,
                   "ddim_steps": DDIM_STEPS},
        "final_verdict": verdict, "criteria": crit, **stats,
        "per_arm": {a: {"ood_test_mean": _mean([r["ood_test"] for r in runs[a] if not r["diverged"]]),
                        "ood_test_std": _std([r["ood_test"] for r in runs[a] if not r["diverged"]]),
                        "indist_mean": _mean([r["indist"] for r in runs[a] if not r["diverged"]]),
                        "class_path_flops": runs[a][0]["class_path_flops"],
                        "params": runs[a][0]["params"]}
                    for a in ARMS6},
    }
    (RESULTS_DIR / "stage6_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-6 VERDICT: {s['final_verdict'].upper()}  criteria={s['criteria']}")
    pa = s["per_arm"]
    print("  ood:", {a: round(pa[a]["ood_test_mean"], 5) for a in ARMS6})
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
