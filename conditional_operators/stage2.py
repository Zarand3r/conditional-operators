"""Stage-2: the de-alignment control (AC-7 of STAGE2_SPEC.md).

The generative factors are hidden behind a FIXED random orthonormal change of basis B, shared across
all conditions: M(c) = B R(c) B^T, where R(c) is the Stage-1 product of coordinate-plane rotations.
A coordinate-block operator now has NO free lunch -- it must LEARN the basis to win.

The fair proposed operator is T(c) = P Q(c) P^T with:
  - P: a SHARED learned orthogonal matrix (matrix_exp of a learned skew-symmetric), NOT input-
    conditioned and NOT handed the true B -- it must discover it;
  - Q(c): the Stage-1 input-conditioned coordinate-block rotations (+ bounded low-rank).
Identity-init holds (A=0 -> P=I; angles=0 -> Q=I -> T=I). We also run a no-basis ablation (the
Stage-1 operator, P fixed to I) to show that learning the basis is what carries the result.

Reuses the committed arms and the verdict gate unchanged. Run:
  .venv/bin/python -m conditional_operators.stage2 [N_SEEDS] [STEPS]
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from . import verdict
from .arms import (H, ConcatMLP, CondLayerNorm, DynamicLinear, FiLM, Hypernet, Proposed)
from .data import ANGLES, BLOCK, D, K, SPLIT_SEED, Splits, multihot
from .train import TrainConfig
from .verdict import Arm, ArmResult, decide

BASIS_SEED = 777  # fixed random orthonormal basis B (shared across all conditions)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _fixed_basis() -> torch.Tensor:
    g = torch.Generator().manual_seed(BASIS_SEED)
    a = torch.randn(D, D, generator=g)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r))  # proper orthonormal, deterministic


B = _fixed_basis()


def dealigned_matrix(active: frozenset[int]) -> torch.Tensor:
    """M(c) = B R(c) B^T with R(c) the coordinate-plane rotation product (Stage-1 primitives)."""
    r = torch.eye(D)
    for i in active:
        a = ANGLES[i]
        c, s = math.cos(a), math.sin(a)
        p = BLOCK * i
        r[p, p], r[p, p + 1] = c, -s
        r[p + 1, p], r[p + 1, p + 1] = s, c
    return B @ r @ B.T


_CACHE: dict[frozenset[int], torch.Tensor] = {}


def _m(active: frozenset[int]) -> torch.Tensor:
    if active not in _CACHE:
        _CACHE[active] = dealigned_matrix(active)
    return _CACHE[active]


def make_splits() -> Splits:
    singles = [frozenset({i}) for i in range(K)]
    pairs = [frozenset(p) for p in itertools.combinations(range(K), 2)]
    g = torch.Generator().manual_seed(SPLIT_SEED)
    pairs = [pairs[i] for i in torch.randperm(len(pairs), generator=g).tolist()]
    return Splits(train=tuple(singles + pairs[:8]), ood_val=tuple(pairs[8:18]),
                  ood_test=tuple(pairs[18:28]))


def sample_batch(conds, n, gen):
    idx = torch.randint(len(conds), (n,), generator=gen)
    x = torch.randn(n, D, generator=gen)
    c = torch.empty(n, K)
    y = torch.empty(n, D)
    for row in range(n):
        active = conds[idx[row].item()]
        c[row] = multihot(active)
        y[row] = _m(active) @ x[row]
    return c, x, y


def eval_set(conds, npc, gen):
    cs, xs, ys = [], [], []
    for active in conds:
        x = torch.randn(npc, D, generator=gen)
        cs.append(multihot(active).expand(npc, K))
        xs.append(x)
        ys.append(x @ _m(active).T)
    return torch.cat(cs), torch.cat(xs), torch.cat(ys)


class ProposedBasis(Proposed):
    """T(c) = P Q(c) P^T. P shared learned orthogonal (discovers the hidden basis); Q(c) input-conditioned.

    Subclasses Proposed so it reuses the SAME encoder, beta head, block rotations, and bounded low-rank
    (no duplicate params) and adds only the shared skew-symmetric generator for P.
    """

    def __init__(self, learn_basis: bool = True) -> None:
        super().__init__()
        self.learn_basis = learn_basis
        if learn_basis:
            self.skew = nn.Parameter(torch.zeros(D, D))  # A=0 -> P=I at init (identity-init preserved)

    def _P(self) -> torch.Tensor:
        if not self.learn_basis:
            return torch.eye(D)
        return torch.matrix_exp(self.skew - self.skew.T)

    def operator(self, h, x):
        P = self._P()
        qx = super().operator(h, x @ P)  # Q(c) applied in the learned basis (x @ P == P^T x per row)
        return qx @ P.T                  # P (.)

    def op_flops(self):
        return super().op_flops() + (2 * D * D if self.learn_basis else 0)

    def dense_operator(self, c):
        P = self._P()
        h = self.enc(c)
        qx = super().operator(h.expand(D, H), torch.eye(D) @ P)
        return (qx @ P.T).T


def build2(name: str, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if name == "proposed":
        return ProposedBasis(learn_basis=True)
    if name == "proposed_nobasis":
        return ProposedBasis(learn_basis=False)
    return {"film": FiLM, "concat_mlp": ConcatMLP, "cond_layernorm": CondLayerNorm,
            "hypernet": Hypernet, "dynamic_linear": DynamicLinear}[name]()


ARMS = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")


def train_one(name, seed, splits, cfg):
    m = build2(name, seed)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(seed)
    diverged = False
    for _ in range(cfg.steps):
        c, x, y = sample_batch(splits.train, cfg.batch, gen)
        loss = torch.mean((m(c, x) - y) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()
    eg = torch.Generator().manual_seed(10_000 + seed)
    if diverged:
        return dict(arm=name, seed=seed, indist=math.nan, ood_val=math.nan, ood_test=math.nan,
                    diverged=True, params=m.n_params(), flops=m.flops())
    def mse(conds):
        c, x, y = eval_set(conds, cfg.eval_per_condition, eg)
        with torch.no_grad():
            return torch.mean((m(c, x) - y) ** 2).item()
    return dict(arm=name, seed=seed, indist=mse(splits.train), ood_val=mse(splits.ood_val),
                ood_test=mse(splits.ood_test), diverged=False, params=m.n_params(), flops=m.flops())


def run(n_seeds, steps):
    splits = make_splits()
    cfg = TrainConfig(steps=steps)
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = {a: [] for a in ARMS}
    ablation = []
    log = (RESULTS_DIR / "stage2_log.jsonl").open("w")
    for name in ARMS:
        for seed in range(n_seeds):
            t = time.time()
            r = train_one(name, seed, splits, cfg)
            runs[name].append(r)
            log.write(json.dumps(r) + "\n"); log.flush()
            print(f"{name:16} seed={seed} ood_test={r['ood_test']:.5f} indist={r['indist']:.5f} "
                  f"[{time.time()-t:.1f}s]", flush=True)
    # no-basis ablation: does the coordinate-block operator (P=I) fail on de-aligned data?
    for seed in range(n_seeds):
        r = train_one("proposed_nobasis", seed, splits, cfg)
        ablation.append(r["ood_test"])
        print(f"{'proposed_nobasis':16} seed={seed} ood_test={r['ood_test']:.5f}", flush=True)
    log.close()

    results = {}
    for name, rr in runs.items():
        ok = [r for r in rr if not r["diverged"]]
        results[Arm(name)] = ArmResult(
            arm=Arm(name),
            ood_test_mse=tuple(r["ood_test"] for r in ok),
            indist_test_mse=tuple(r["indist"] for r in ok),
            n_diverged=sum(1 for r in rr if r["diverged"]),
            params=rr[0]["params"], flops=rr[0]["flops"], ood_test_reads=1)
    report = decide(results, n_required=n_seeds)
    summary = {
        "stage": 2, "control": "de-aligned basis (AC-7)",
        "config": {"n_seeds": n_seeds, "steps": steps},
        "verdict": report.verdict.value, "reasons": list(report.reasons),
        "criteria": report.criteria,
        "best_unstructured": report.best_unstructured.value if report.best_unstructured else None,
        "margin_observed": report.margin_observed, "p_value": report.p_value,
        "cliffs_delta": report.cliffs_delta,
        "ablation_nobasis_ood_mean": sum(ablation) / len(ablation),
        "per_arm": {n: {"ood_test_mean": _mean(results[Arm(n)].ood_test_mse),
                        "ood_test_std": _std(results[Arm(n)].ood_test_mse),
                        "indist_mean": _mean(results[Arm(n)].indist_test_mse),
                        "params": results[Arm(n)].params, "flops": results[Arm(n)].flops,
                        "flops_vs_film": results[Arm(n)].flops / results[Arm.FILM].flops}
                    for n in ARMS},
    }
    (RESULTS_DIR / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _mean(xs): return math.fsum(xs) / len(xs) if xs else math.nan
def _std(xs):
    if len(xs) < 2: return 0.0
    m = _mean(xs); return math.sqrt(math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else verdict.N_REQUIRED
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else TrainConfig().steps
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-2 VERDICT: {s['verdict'].upper()}")
    for r in s["reasons"]:
        print(f"  - {r}")
    print(f"  no-basis ablation OOD mean: {s['ablation_nobasis_ood_mean']:.5f} "
          f"(vs proposed {s['per_arm']['proposed']['ood_test_mean']:.5f})")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
