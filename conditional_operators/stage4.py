"""Stage-4: real-image compositional conditional transformation on dSprites (STAGE4_SPEC.md).

Triplet task: encode real image x1 -> z, apply conditioning arm z' = T(delta) z + beta(delta) for a
factor-change vector delta, decode -> x2_hat, MSE against the GROUND-TRUTH image at z1+delta
(dSprites is deterministic). Compositional OOD = unseen combinations of change types.

Run:  .venv/bin/python -m conditional_operators.stage4 [N_SEEDS] [STEPS]
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import verdict
from .stage3 import GSOrthogonal, _rotate
from .verdict import Arm, ArmResult, decide

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DZ = 128          # latent width (GSOrthogonal is built for 128)
DC = 4            # delta over [scale, orientation, posX, posY]
H = 128           # condition-encoder hidden width
RANK = 4
LOWRANK = 8       # dynamic_linear rank on the latent

# factor columns in dsprites latents_classes: [color, shape, scale, orient, posX, posY]
FCOLS = (2, 3, 4, 5)
FSIZES = (6, 40, 32, 32)
MAXSTEP = (2, 5, 4, 4)         # per-factor |delta| caps (grid steps); orientation wraps
NORM = torch.tensor([2.0, 5.0, 4.0, 4.0])

SINGLES = ((0,), (1,), (2,), (3,))
TRAIN_PAIRS = ((0, 1), (2, 3))
VAL_PAIRS = ((0, 2), (1, 3))
TEST_PAIRS = ((0, 3), (1, 2))
TRIPLES = tuple(itertools.combinations(range(DC), 3))
TRAIN_TYPES = SINGLES + TRAIN_PAIRS


class Data:
    """Full dSprites grid resident on GPU (uint8), with vectorized triplet sampling."""

    def __init__(self) -> None:
        d = np.load(ROOT / "datasets" / "dsprites.npz", allow_pickle=True, encoding="latin1")
        self.imgs = torch.from_numpy(d["imgs"]).to(DEVICE)               # [737280, 64, 64] uint8
        sizes = d["metadata"][()]["latents_sizes"]                       # [1,3,6,40,32,32]
        bases = np.concatenate([sizes[::-1].cumprod()[::-1][1:], [1]])
        self.bases = torch.from_numpy(bases.astype(np.int64))
        self.classes = torch.from_numpy(d["latents_classes"].astype(np.int64))
        self.sizes = torch.from_numpy(sizes.astype(np.int64))

    def sample(self, types: tuple[tuple[int, ...], ...], n: int,
               gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """n triplets with delta types drawn uniformly from `types`. Returns (x1, dnorm, x2)."""
        lat = torch.stack([torch.randint(int(s), (n,), generator=gen)
                           for s in self.sizes.tolist()], dim=1)          # [n, 6]
        delta = torch.zeros(n, DC, dtype=torch.int64)
        which = torch.randint(len(types), (n,), generator=gen)
        for t_idx, t in enumerate(types):
            mask = which == t_idx
            for f in t:
                step = torch.randint(1, MAXSTEP[f] + 1, (int(mask.sum()),), generator=gen)
                sign = torch.randint(0, 2, (int(mask.sum()),), generator=gen) * 2 - 1
                delta[mask, f] = step * sign
        tgt = lat.clone()
        for f, col in enumerate(FCOLS):
            moved = lat[:, col] + delta[:, f]
            if f == 1:                                                    # orientation wraps
                tgt[:, col] = moved % FSIZES[f]
            else:                                                         # clamp + fix delta
                clamped = moved.clamp(0, FSIZES[f] - 1)
                delta[:, f] = clamped - lat[:, col]
                tgt[:, col] = clamped
        i1 = (lat * self.bases).sum(1)
        i2 = (tgt * self.bases).sum(1)
        x1 = self.imgs[i1.to(DEVICE)].float().unsqueeze(1)
        x2 = self.imgs[i2.to(DEVICE)].float().unsqueeze(1)
        return x1, (delta.float() / NORM).to(DEVICE), x2


# ---------------------------------------------------------------- backbone (identical everywhere)

class Backbone(nn.Module):
    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        ch = 64
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, ch, 4, 2, 1), nn.ReLU(),    # 32
            nn.Conv2d(ch, ch, 4, 2, 1), nn.ReLU(),       # 16
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.ReLU(),   # 8
            nn.Conv2d(ch * 2, ch * 2, 4, 2, 1), nn.ReLU(),  # 4
            nn.Flatten(), nn.Linear(ch * 2 * 16, DZ))
        self.fc = nn.Linear(DZ, ch * 2 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch * 2, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(ch, in_ch, 4, 2, 1))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        h = self.fc(z).view(-1, 128, 4, 4)
        return self.dec(h)


# ------------------------------------------------------- conditioning arms (the ONLY thing varied)

def _fl(i, o):  # per-sample FLOPs of a Linear
    return 2 * i * o


def _zero_(l: nn.Linear) -> nn.Linear:
    nn.init.zeros_(l.weight)
    if l.bias is not None:
        nn.init.zeros_(l.bias)
    return l


class CondArm(nn.Module):
    """Base: shared delta-encoder + zero-init beta. Subclass adds the operator on z.

    dc = condition input dimension (4 geometric deltas by default; Stage-4b appends 3
    categorical shape-delta dims).
    """

    def __init__(self, dc: int = DC) -> None:
        super().__init__()
        self.dc = dc
        self.e1, self.e2 = nn.Linear(dc, H), nn.Linear(H, H)
        self.beta = _zero_(nn.Linear(H, DZ))

    def enc(self, d):
        return torch.relu(self.e2(torch.relu(self.e1(d))))

    def forward(self, d, z):
        h = self.enc(d)
        return self.op(h, d, z) + self.beta(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops(self):
        return _fl(self.dc, H) + _fl(H, H) + _fl(H, DZ) + self.op_flops()


class FiLM4(CondArm):
    def __init__(self, dc=DC):
        super().__init__(dc)
        self.g = _zero_(nn.Linear(H, DZ))

    def op(self, h, d, z):
        return (1 + self.g(h)) * z

    def op_flops(self):
        return _fl(H, DZ) + DZ


class ConcatMLP4(CondArm):
    def __init__(self, dc=DC):
        super().__init__(dc)
        self.w1 = nn.Linear(DZ + H, H)
        self.w2 = _zero_(nn.Linear(H, DZ))

    def op(self, h, d, z):
        return z + self.w2(torch.relu(self.w1(torch.cat([z, h], 1))))

    def op_flops(self):
        return _fl(DZ + H, H) + _fl(H, DZ) + DZ


class CondLN4(CondArm):
    def __init__(self, dc=DC):
        super().__init__(dc)
        self.ln = nn.LayerNorm(DZ, elementwise_affine=False)
        self.g = _zero_(nn.Linear(H, DZ))

    def op(self, h, d, z):
        return (1 + self.g(h)) * self.ln(z)

    def op_flops(self):
        return _fl(H, DZ) + 5 * DZ


class Hypernet4(CondArm):
    def __init__(self, dc=DC):
        super().__init__(dc)
        self.w = _zero_(nn.Linear(H, DZ * DZ))

    def op(self, h, d, z):
        return z + torch.einsum("nij,nj->ni", self.w(h).view(-1, DZ, DZ), z)

    def op_flops(self):
        return _fl(H, DZ * DZ) + 2 * DZ * DZ


class DynLin4(CondArm):
    def __init__(self, dc=DC):
        super().__init__(dc)
        self.a = _zero_(nn.Linear(H, DZ * LOWRANK))
        self.b = nn.Linear(H, DZ * LOWRANK)

    def op(self, h, d, z):
        a = self.a(h).view(-1, DZ, LOWRANK)
        b = self.b(h).view(-1, DZ, LOWRANK)
        return z + torch.einsum("ndr,nr->nd", a, torch.einsum("ndr,nd->nr", b, z))

    def op_flops(self):
        return 2 * _fl(H, DZ * LOWRANK) + 4 * DZ * LOWRANK


class Lie4(CondArm):
    """T(d) = P R(W d) P^T: W linear bias-free (exact composition in d), GS-P from Stage-3."""

    def __init__(self, dc=DC):
        super().__init__(dc)
        self.W = nn.Linear(dc, DZ // 2, bias=False)
        nn.init.zeros_(self.W.weight)
        self.P = GSOrthogonal()

    def op(self, h, d, z):
        return self.P.apply_t(_rotate(self.W(d), self.P.apply(z)))

    def op_flops(self):
        return _fl(self.dc, DZ // 2) + 3 * DZ + 2 * GSOrthogonal.apply_flops()


class MLPGS4(Lie4):
    """Ablation: same GS-P, MLP angle head (reported, not gated; over FiLM ceiling)."""

    def __init__(self, dc=DC):
        super().__init__(dc)
        self.head = _zero_(nn.Linear(H, DZ // 2))

    def op(self, h, d, z):
        return self.P.apply_t(_rotate(self.head(h), self.P.apply(z)))

    def op_flops(self):
        return _fl(H, DZ // 2) + 3 * DZ + 2 * GSOrthogonal.apply_flops()


ARM_CLASSES = {"film": FiLM4, "concat_mlp": ConcatMLP4, "cond_layernorm": CondLN4,
               "hypernet": Hypernet4, "dynamic_linear": DynLin4, "proposed": Lie4,
               "proposed_mlp_gs": MLPGS4}
GATE_ARMS = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")


class Model(nn.Module):
    def __init__(self, arm_name: str, dc: int = DC):
        super().__init__()
        self.backbone = Backbone()
        self.cond = ARM_CLASSES[arm_name](dc=dc)

    def forward(self, x1, d):
        z = self.backbone.encode(x1)
        return self.backbone.decode(self.cond(d, z))


def train_one(name, seed, data, steps, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    m = Model(name).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    # AMENDMENT (pre-run, 2026-07-21): train with BCEWithLogits — plain MSE stalls at the
    # mean-predictor plateau on sparse binary dSprites (calibrated on TRAIN/VAL only). The
    # GATED metric remains per-pixel MSE exactly as pre-registered.
    bce = nn.BCEWithLogitsLoss()
    diverged = False
    for _ in range(steps):
        x1, d, x2 = data.sample(TRAIN_TYPES, batch, gen)
        loss = bce(m(x1, d), x2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()

    row = dict(arm=name, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    eg = torch.Generator().manual_seed(70_000 + seed)     # identical eval triplets across arms
    if diverged:
        row |= dict(indist=math.nan, ood_val=math.nan, ood_test=math.nan, triples=math.nan)
        return row

    @torch.no_grad()
    def mse(types):
        tot = 0.0
        for _ in range(4):                                 # 4x512 = 2048 eval triplets per split
            x1, d, x2 = data.sample(types, 512, eg)
            tot += torch.mean((torch.sigmoid(m(x1, d)) - x2) ** 2).item()
        return tot / 4

    row |= dict(indist=mse(TRAIN_TYPES), ood_val=mse(VAL_PAIRS),
                ood_test=mse(TEST_PAIRS),                  # single OOD-TEST read
                triples=mse(TRIPLES))                      # diagnostic (spec: not gated)
    return row


def run(n_seeds, steps):
    data = Data()
    RESULTS_DIR.mkdir(exist_ok=True)
    all_arms = GATE_ARMS + ("proposed_mlp_gs",)
    runs = {a: [] for a in all_arms}
    with (RESULTS_DIR / "stage4_log.jsonl").open("w") as log:
        for name in all_arms:
            for seed in range(n_seeds):
                t = time.time()
                r = train_one(name, seed, data, steps)
                runs[name].append(r)
                log.write(json.dumps(r) + "\n"); log.flush()
                print(f"{name:16} seed={seed} ood_test={r['ood_test']:.6f} "
                      f"triples={r['triples']:.6f} indist={r['indist']:.6f} "
                      f"[{time.time()-t:.1f}s]", flush=True)

    results = {}
    for name in GATE_ARMS:
        rr = runs[name]
        ok = [r for r in rr if not r["diverged"]]
        results[Arm(name)] = ArmResult(
            arm=Arm(name), ood_test_mse=tuple(r["ood_test"] for r in ok),
            indist_test_mse=tuple(r["indist"] for r in ok),
            n_diverged=sum(1 for r in rr if r["diverged"]),
            params=rr[0]["params"], flops=rr[0]["flops"], ood_test_reads=1)
    gate = decide(results, n_required=n_seeds)

    def stat(name, key):
        v = [r[key] for r in runs[name] if not r["diverged"]]
        return _mean(v), _std(v)

    summary = {
        "stage": 4, "spec": "docs/specs/STAGE4_SPEC.md (pre-registered 2026-07-21)",
        "task": "dSprites conditional transformation; OOD = unseen two-factor delta types",
        "config": {"n_seeds": n_seeds, "steps": steps, "device": DEVICE},
        "final_verdict": gate.verdict.value, "reasons": list(gate.reasons),
        "gate_criteria": gate.criteria,
        "best_unstructured": gate.best_unstructured.value if gate.best_unstructured else None,
        "margin_observed": gate.margin_observed, "p_value": gate.p_value,
        "cliffs_delta": gate.cliffs_delta,
        "per_arm": {n: {"ood_test_mean": stat(n, "ood_test")[0],
                        "ood_test_std": stat(n, "ood_test")[1],
                        "triples_mean": stat(n, "triples")[0],
                        "triples_std": stat(n, "triples")[1],
                        "indist_mean": stat(n, "indist")[0],
                        "params": runs[n][0]["params"], "flops": runs[n][0]["flops"],
                        "flops_vs_film": runs[n][0]["flops"] / runs["film"][0]["flops"]}
                    for n in all_arms},
    }
    (RESULTS_DIR / "stage4_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _mean(xs): return math.fsum(xs) / len(xs) if xs else math.nan
def _std(xs):
    if len(xs) < 2: return 0.0
    mu = _mean(xs); return math.sqrt(math.fsum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else verdict.N_REQUIRED
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-4 VERDICT: {s['final_verdict'].upper()}")
    for r in s["reasons"]:
        print(f"  - {r}")
    pa = s["per_arm"]
    print(f"  ood: proposed={pa['proposed']['ood_test_mean']:.6f} "
          f"{s['best_unstructured']}={pa[s['best_unstructured']]['ood_test_mean']:.6f} "
          f"film={pa['film']['ood_test_mean']:.6f} mlp_gs={pa['proposed_mlp_gs']['ood_test_mean']:.6f}")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
