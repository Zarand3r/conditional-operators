"""Stage-3: GRAPE-inspired Lie-algebra conditioning within the TRUE budget (STAGE3_SPEC.md).

Operator: T(c) = P · R(W·c) · Pᵀ
  - W: linear, bias-free K -> D/2 angle map. Condition enters LINEARLY in the Lie algebra
    (GRAPE, arXiv:2512.07805), so T(c1+c2) = T(c1)·T(c2) EXACTLY for any P (same-plane rotations
    commute). Composition is structural, not learned. W=0 at init -> T=I.
  - P: Group-and-Shuffle structured orthogonal (5 layers x 8 Cayley-orthogonal 16x16 blocks with
    fixed shuffle permutations between). Dense P costs 4*D*D/sample = 1.52x FiLM (the Stage-2
    erratum); GS-P applies at 2*(2*D*B_BLK) per layer and fits the 1.20x ceiling. Skew=0 -> P=I.

Sweep adds the TRIPLES split: all C(8,3)=56 three-hot conditions, never trained (max trained
composition length is 2), evaluated exactly once per arm at decision time (AC-8).

Run:  .venv/bin/python -m conditional_operators.stage3 [N_SEEDS] [STEPS]
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

from . import verdict
from .arms import Arm as ArmBase, ConcatMLP, CondLayerNorm, DynamicLinear, FiLM, Hypernet, H, _zero_
from .data import D, K, multihot
from .stage2 import eval_set, make_splits, sample_batch
from .train import TrainConfig
from .verdict import Arm, ArmResult, decide

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GS_LAYERS = 5
B_BLK = 16                  # block size
N_BLK = D // B_BLK          # 8 blocks per layer
SHUFFLE_SEED = 4242         # fixed inter-layer permutations (not learned, not data-derived)

# Pre-registered AC-8 thresholds (STAGE3_SPEC.md, 2026-07-21).
AC7_TOL = 1e-4
AC8_VS_UNSTRUCT = 0.5       # proposed triples MSE < 0.5x best-unstructured triples MSE
AC8_SYSTEMATICITY = 2.0     # proposed triples MSE <= 2x its own OOD-pairs MSE


def _shuffles() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(SHUFFLE_SEED)
    return [torch.randperm(D, generator=g) for _ in range(GS_LAYERS - 1)]


class GSOrthogonal(nn.Module):
    """Group-and-Shuffle orthogonal: block-diagonal Cayley layers with fixed shuffles between.

    Cayley: Q_blk = (I+A)^{-1}(I-A) with A skew -> exactly orthogonal, Q=I at A=0 (identity-init).
    """

    def __init__(self) -> None:
        super().__init__()
        self.skew = nn.Parameter(torch.zeros(GS_LAYERS, N_BLK, B_BLK, B_BLK))
        perms = _shuffles()
        self.register_buffer("perms", torch.stack(perms) if perms else torch.empty(0, D, dtype=torch.long))
        inv = [torch.argsort(p) for p in perms]
        self.register_buffer("inv_perms", torch.stack(inv) if inv else torch.empty(0, D, dtype=torch.long))

    def _blocks(self) -> torch.Tensor:
        a = self.skew - self.skew.transpose(-1, -2)           # enforce skew
        eye = torch.eye(B_BLK).expand_as(a)
        return torch.linalg.solve(eye + a, eye - a)           # [L, N_BLK, b, b], orthogonal

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x @ P (rows transformed by P^T ... i.e. this is the 'into learned basis' map)."""
        q = self._blocks()
        n = x.shape[0]
        for l in range(GS_LAYERS):
            x = torch.einsum("nbi,bij->nbj", x.view(n, N_BLK, B_BLK), q[l]).reshape(n, D)
            if l < GS_LAYERS - 1:
                x = x[:, self.perms[l]]
        return x

    def apply_t(self, x: torch.Tensor) -> torch.Tensor:
        """x @ P^T (inverse map: reversed layers, inverse shuffles, transposed blocks)."""
        q = self._blocks()
        n = x.shape[0]
        for l in reversed(range(GS_LAYERS)):
            if l < GS_LAYERS - 1:
                x = x[:, self.inv_perms[l]]
            x = torch.einsum("nbi,bji->nbj", x.view(n, N_BLK, B_BLK), q[l]).reshape(n, D)
        return x

    def dense(self) -> torch.Tensor:
        """Materialize P as a D x D matrix (tests/diagnostics only)."""
        return self.apply(torch.eye(D)).T

    @staticmethod
    def apply_flops() -> int:
        # per-sample, per direction: L block-matmuls of [1,b]@[b,b] per block -> 2*D*b each layer.
        per_layer = 2 * D * B_BLK
        # Cayley parametrization is shared across the batch; amortize at the registered batch size.
        cayley_per_batch = GS_LAYERS * N_BLK * (2 * B_BLK**3 + (2 * B_BLK**3) // 3)
        return GS_LAYERS * per_layer + cayley_per_batch // (2 * 256)  # split across both directions


def _rotate(angles: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply block-diagonal 2x2 rotations R(angles) to x. angles [N, D/2], x [N, D]."""
    c, s = torch.cos(angles), torch.sin(angles)
    x0, x1 = x[:, 0::2], x[:, 1::2]
    y = torch.empty_like(x)
    y[:, 0::2] = c * x0 - s * x1
    y[:, 1::2] = s * x0 + c * x1
    return y


class ProposedLie(ArmBase):
    """T(c) = P R(Wc) P^T + beta. W linear bias-free (exact compositionality); P = GSOrthogonal."""

    def __init__(self) -> None:
        super().__init__()
        self.W = nn.Linear(K, D // 2, bias=False)
        nn.init.zeros_(self.W.weight)                          # T(c)=I at init (INV-1)
        self.P = GSOrthogonal()

    def forward(self, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # NOTE: angles come from RAW c (linear in the Lie algebra), not from the encoder — the
        # encoder/beta path is kept identical to every other arm for structural fairness.
        xp = self.P.apply(x)
        y = self.P.apply_t(_rotate(self.W(c), xp))
        return y + self.beta(self.enc(c))

    def operator(self, h, x):  # pragma: no cover - forward() is overridden; kept for ABC shape
        raise NotImplementedError("ProposedLie overrides forward()")

    def op_flops(self) -> int:
        return 2 * K * (D // 2) + 3 * D + 2 * GSOrthogonal.apply_flops()

    def dense_operator(self, c: torch.Tensor) -> torch.Tensor:
        """T(c) as a dense D x D matrix (no beta), for AC-7 and invariants."""
        xp = self.P.apply(torch.eye(D))
        return self.P.apply_t(_rotate(self.W(c).expand(D, D // 2), xp)).T


class ProposedMLPGS(ProposedLie):
    """Ablation: same GS-P, but angles from the shared MLP encoder (Stage-1/2 style head).

    Isolates the Lie-linearity contribution. Over the 1.20x FLOP ceiling (~1.26x) — reported
    transparently, never gated (STAGE3_SPEC.md).
    """

    def __init__(self) -> None:
        super().__init__()
        self.head = _zero_(nn.Linear(H, D // 2))

    def forward(self, c, x):
        h = self.enc(c)
        xp = self.P.apply(x)
        y = self.P.apply_t(_rotate(self.head(h), xp))
        return y + self.beta(h)

    def op_flops(self) -> int:
        return 2 * H * (D // 2) + 3 * D + 2 * GSOrthogonal.apply_flops()


def build3(name: str, seed: int) -> ArmBase:
    torch.manual_seed(seed)
    return {"film": FiLM, "concat_mlp": ConcatMLP, "cond_layernorm": CondLayerNorm,
            "hypernet": Hypernet, "dynamic_linear": DynamicLinear,
            "proposed": ProposedLie, "proposed_mlp_gs": ProposedMLPGS}[name]()


GATE_ARMS = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")
ABLATION_ARMS = ("proposed_mlp_gs",)


def triples() -> tuple[frozenset[int], ...]:
    return tuple(frozenset(t) for t in itertools.combinations(range(K), 3))


def composition_error(m: ProposedLie, n_pairs: int = 64) -> float:
    """max ||T(c1+c2) - T(c2) T(c1)||_F over sampled disjoint condition pairs (AC-7)."""
    g = torch.Generator().manual_seed(99)
    worst = 0.0
    with torch.no_grad():
        for _ in range(n_pairs):
            picks = torch.randperm(K, generator=g)[:4].tolist()
            a, b = frozenset(picks[:2]), frozenset(picks[2:])
            ca, cb = multihot(a).unsqueeze(0), multihot(b).unsqueeze(0)
            cab = multihot(a | b).unsqueeze(0)
            err = (m.dense_operator(cab) - m.dense_operator(cb) @ m.dense_operator(ca)).norm().item()
            worst = max(worst, err)
    return worst


def stage3_verdict(gate_verdict: str, ac7_max_err: float, prop_triples: float,
                   unstruct_triples: float, prop_pairs: float) -> tuple[str, dict]:
    """STAGE-3 CONFIRMED <=> gate CONFIRMED AND AC-7 AND AC-8 (STAGE3_SPEC.md). Pure logic."""
    ac7 = ac7_max_err < AC7_TOL
    ac8a = prop_triples < AC8_VS_UNSTRUCT * unstruct_triples
    ac8b = prop_triples <= AC8_SYSTEMATICITY * prop_pairs
    crit = {"AC-7": ac7, "AC-8a": ac8a, "AC-8b": ac8b}
    if gate_verdict != "confirmed":
        return gate_verdict, crit           # UNFAIR/BLOCKED/INVALID/KILL pass through unchanged
    return ("confirmed" if all(crit.values()) else "kill"), crit


def train_one(name, seed, splits, cfg):
    m = build3(name, seed)
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
    row = dict(arm=name, seed=seed, diverged=diverged, params=m.n_params(), flops=m.flops())
    if diverged:
        row |= dict(indist=math.nan, ood_val=math.nan, ood_test=math.nan, triples=math.nan)
        return row, m

    def mse(conds):
        c, x, y = eval_set(conds, cfg.eval_per_condition, eg)
        with torch.no_grad():
            return torch.mean((m(c, x) - y) ** 2).item()

    row |= dict(indist=mse(splits.train), ood_val=mse(splits.ood_val),
                ood_test=mse(splits.ood_test), triples=mse(triples()))  # single triples read
    return row, m


def run(n_seeds, steps):
    splits = make_splits()
    cfg = TrainConfig(steps=steps)
    RESULTS_DIR.mkdir(exist_ok=True)
    all_arms = GATE_ARMS + ABLATION_ARMS
    runs = {a: [] for a in all_arms}
    ac7_errs = []
    with (RESULTS_DIR / "stage3_log.jsonl").open("w") as log:
        for name in all_arms:
            for seed in range(n_seeds):
                t = time.time()
                row, model = train_one(name, seed, splits, cfg)
                if name == "proposed" and not row["diverged"]:
                    row["ac7_composition_err"] = composition_error(model)
                    ac7_errs.append(row["ac7_composition_err"])
                runs[name].append(row)
                log.write(json.dumps(row) + "\n"); log.flush()
                print(f"{name:16} seed={seed} ood_test={row['ood_test']:.5f} "
                      f"triples={row['triples']:.5f} indist={row['indist']:.5f} "
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
        vals = [r[key] for r in runs[name] if not r["diverged"]]
        return _mean(vals), _std(vals)

    bu = gate.best_unstructured.value if gate.best_unstructured else "hypernet"
    prop_tri, _ = stat("proposed", "triples")
    bu_tri, _ = stat(bu, "triples")
    prop_pairs, _ = stat("proposed", "ood_test")
    ac7_max = max(ac7_errs) if ac7_errs else math.inf
    final, crit = stage3_verdict(gate.verdict.value, ac7_max, prop_tri, bu_tri, prop_pairs)

    summary = {
        "stage": 3, "spec": "docs/specs/STAGE3_SPEC.md (pre-registered 2026-07-21)",
        "config": {"n_seeds": n_seeds, "steps": steps},
        "gate_verdict": gate.verdict.value, "final_verdict": final,
        "reasons": list(gate.reasons), "gate_criteria": gate.criteria,
        "stage3_criteria": crit, "ac7_max_composition_err": ac7_max,
        "best_unstructured": bu, "margin_observed": gate.margin_observed,
        "p_value": gate.p_value, "cliffs_delta": gate.cliffs_delta,
        "per_arm": {n: {"ood_test_mean": stat(n, "ood_test")[0],
                        "ood_test_std": stat(n, "ood_test")[1],
                        "triples_mean": stat(n, "triples")[0],
                        "triples_std": stat(n, "triples")[1],
                        "indist_mean": stat(n, "indist")[0],
                        "params": runs[n][0]["params"], "flops": runs[n][0]["flops"],
                        "flops_vs_film": runs[n][0]["flops"] / runs["film"][0]["flops"]}
                    for n in all_arms},
    }
    (RESULTS_DIR / "stage3_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _mean(xs): return math.fsum(xs) / len(xs) if xs else math.nan
def _std(xs):
    if len(xs) < 2: return 0.0
    mu = _mean(xs); return math.sqrt(math.fsum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else verdict.N_REQUIRED
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else TrainConfig().steps
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-3 FINAL VERDICT: {s['final_verdict'].upper()}  (gate: {s['gate_verdict']})")
    print(f"  stage3 criteria: {s['stage3_criteria']}")
    print(f"  AC-7 max composition err: {s['ac7_max_composition_err']:.2e}")
    p = s["per_arm"]
    print(f"  triples: proposed={p['proposed']['triples_mean']:.5f} "
          f"{s['best_unstructured']}={p[s['best_unstructured']]['triples_mean']:.5f} "
          f"mlp_gs={p['proposed_mlp_gs']['triples_mean']:.5f}")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
