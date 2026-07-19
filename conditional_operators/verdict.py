"""Stage-1 Pursue/Kill decision logic — the pre-registered gate.

Turns per-arm, per-seed results into a single verdict by applying acceptance criteria
AC-1..AC-6 from docs/specs/STAGE1_SPEC.md. This module is deliberately dependency-free
(stdlib only) and fail-fast: a bug here would defeat the whole pre-registration discipline,
so it is the most-tested surface in the project.

Statistics note: mann_whitney_u uses the normal approximation with tie + continuity
correction. For N≈10 per arm this is close to exact; the α=0.01 threshold has margin. If the
PI wants exact tails, swap in scipy.stats.mannwhitneyu(method="exact") — the decision code
below is agnostic to how (U, p) is produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum


class Arm(StrEnum):
    FILM = "film"                    # diagonal floor (AC-1)
    CONCAT_MLP = "concat_mlp"
    COND_LAYERNORM = "cond_layernorm"
    HYPERNET = "hypernet"            # unstructured W(c) — a candidate "best unstructured"
    DYNAMIC_LINEAR = "dynamic_linear"  # unstructured W(c) — a candidate "best unstructured"
    PROPOSED = "proposed"           # Q(c) + U(c)V(c)^T


UNSTRUCTURED_ARMS: frozenset[Arm] = frozenset({Arm.HYPERNET, Arm.DYNAMIC_LINEAR})


class Verdict(StrEnum):
    CONFIRMED = "confirmed"   # AC-1..AC-5 pass, fair, enough seeds -> proceed to Stage 2
    KILL = "kill"             # fair, enough seeds, evaluated once, but a criterion failed
    UNFAIR = "unfair"         # AC-4 violated -> comparison inconclusive, fix budget and rerun
    BLOCKED = "blocked"       # fewer than N non-diverged seeds -> rerun to restore N
    INVALID = "invalid"       # leakage (R7) or hygiene (AC-6) breach -> no scientific verdict


# Pre-registered thresholds (STAGE1_SPEC.md, PI sign-off 2026-07-19).
N_REQUIRED = 10
MARGIN = 0.20            # AC-2 relative OOD-MSE improvement over best unstructured
ALPHA = 0.01            # AC-3 one-sided significance
CLIFF_THRESHOLD = 0.474  # AC-3 large effect size (magnitude)
FILM_FLOP_OVERHEAD = 0.20  # AC-4 proposed FLOPs <= 1.20x FiLM
PARAM_TOLERANCE = 0.05     # AC-4 proposed params <= 1.05x smallest unstructured
INDIST_REGRESSION = 0.10   # AC-5 proposed in-dist MSE <= 1.10x best unstructured


@dataclass(frozen=True)
class ArmResult:
    """Per-arm results. MSE lists hold non-diverged seeds only; n_diverged tracks the rest."""
    arm: Arm
    ood_test_mse: tuple[float, ...]    # compositional OOD-TEST MSE, one per non-diverged seed
    indist_test_mse: tuple[float, ...]  # in-distribution test MSE, one per non-diverged seed
    n_diverged: int                     # seeds excluded for NaN/Inf loss (reported, not dropped silently)
    params: int                         # trainable params of the whole conditioning module (R5)
    flops: int                          # forward-pass FLOPs from the shared counter (R5)
    ood_test_reads: int                 # times OOD-TEST was evaluated (AC-6 requires exactly 1)

    @property
    def n_seeds(self) -> int:
        return len(self.ood_test_mse)

    @property
    def divergence_rate(self) -> float:
        total = self.n_seeds + self.n_diverged
        if total == 0:
            raise ValueError(f"arm {self.arm}: zero total seeds")
        return self.n_diverged / total


@dataclass(frozen=True)
class VerdictReport:
    verdict: Verdict
    reasons: tuple[str, ...]
    criteria: dict[str, bool] = field(default_factory=dict)  # AC-1..AC-5 booleans (when computed)
    best_unstructured: Arm | None = None
    margin_observed: float | None = None
    p_value: float | None = None
    cliffs_delta: float | None = None
    divergence_rates: dict[str, float] = field(default_factory=dict)


def _mean(xs: tuple[float, ...]) -> float:
    if not xs:
        raise ValueError("mean of empty sample")
    return math.fsum(xs) / len(xs)


def _rank_average(values: list[float]) -> tuple[list[float], list[int]]:
    """Return average ranks (1-based) and tie-group sizes, for tie-corrected statistics."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    ties: list[int] = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average of 1-based ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u(a: tuple[float, ...], b: tuple[float, ...], *, alternative: str = "less") -> tuple[float, float]:
    """Mann-Whitney U for sample `a` vs `b`. alternative="less" tests H1: a stochastically < b.

    Returns (U_a, p_value) via the normal approximation with tie and continuity correction.
    """
    if alternative != "less":
        raise ValueError(f"only alternative='less' is supported, got {alternative!r}")
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        raise ValueError("empty sample in mann_whitney_u")
    ranks, ties = _rank_average(list(a) + list(b))
    r_a = math.fsum(ranks[:na])
    u_a = r_a - na * (na + 1) / 2
    n = na + nb
    mean_u = na * nb / 2
    tie_term = math.fsum(t**3 - t for t in ties) / (n * (n - 1))
    var_u = (na * nb / 12) * ((n + 1) - tie_term)
    if var_u <= 0:
        # All values identical -> no evidence of difference.
        return u_a, 1.0
    # Left tail (a smaller -> smaller ranks -> smaller U_a) with +0.5 continuity correction.
    z = (u_a + 0.5 - mean_u) / math.sqrt(var_u)
    return u_a, _phi(z)


def cliffs_delta(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cliff's delta in [-1, 1]. Negative => `a` tends to be smaller than `b` (better for MSE)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        raise ValueError("empty sample in cliffs_delta")
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (greater - less) / (na * nb)


def decide(results: dict[Arm, ArmResult], *, leakage: bool = False,
           n_required: int = N_REQUIRED) -> VerdictReport:
    """Apply AC-1..AC-6 and the failure semantics to per-arm results and return a verdict.

    Fail-fast on structural problems (missing arms); scientific-invalidations (leakage, hygiene)
    return INVALID rather than raising, because they are legitimate — if disappointing — outcomes.
    """
    missing = set(Arm) - results.keys()
    if missing:
        raise ValueError(f"results missing arms: {sorted(m.value for m in missing)}")

    div_rates = {a.value: r.divergence_rate for a, r in results.items()}

    # R7 — leakage from OOD-VAL selection into OOD-TEST invalidates the run.
    if leakage:
        return VerdictReport(Verdict.INVALID, ("OOD-VAL leaked into OOD-TEST (R7)",),
                             divergence_rates=div_rates)

    # AC-6 — OOD-TEST must be read exactly once per arm.
    bad_reads = [a.value for a, r in results.items() if r.ood_test_reads != 1]
    if bad_reads:
        return VerdictReport(Verdict.INVALID,
                             (f"OOD-TEST not read exactly once for: {sorted(bad_reads)} (AC-6)",),
                             divergence_rates=div_rates)

    # Failure semantics — need N non-diverged seeds for every arm before a verdict.
    short = [f"{a.value}={r.n_seeds}" for a, r in results.items() if r.n_seeds < n_required]
    if short:
        return VerdictReport(Verdict.BLOCKED,
                             (f"fewer than {n_required} non-diverged seeds: {sorted(short)}",),
                             divergence_rates=div_rates)

    proposed = results[Arm.PROPOSED]
    film = results[Arm.FILM]
    best_unstruct = min(UNSTRUCTURED_ARMS, key=lambda a: _mean(results[a].ood_test_mse))
    bu = results[best_unstruct]

    prop_mean = _mean(proposed.ood_test_mse)
    bu_mean = _mean(bu.ood_test_mse)
    margin = (bu_mean - prop_mean) / bu_mean
    u_stat, p = mann_whitney_u(proposed.ood_test_mse, bu.ood_test_mse, alternative="less")
    delta = cliffs_delta(proposed.ood_test_mse, bu.ood_test_mse)

    min_unstruct_params = min(results[a].params for a in UNSTRUCTURED_ARMS)

    criteria = {
        "AC-1": prop_mean < _mean(film.ood_test_mse),
        "AC-2": margin >= MARGIN,
        "AC-3": p <= ALPHA and delta <= -CLIFF_THRESHOLD,
        "AC-5": _mean(proposed.indist_test_mse) <= (1 + INDIST_REGRESSION) * _mean(bu.indist_test_mse),
    }
    fair = (proposed.params <= (1 + PARAM_TOLERANCE) * min_unstruct_params
            and proposed.flops <= (1 + FILM_FLOP_OVERHEAD) * film.flops)

    common = dict(criteria=criteria, best_unstructured=best_unstruct, margin_observed=margin,
                  p_value=p, cliffs_delta=delta, divergence_rates=div_rates)

    # AC-4 — an unfair comparison cannot confirm; it is inconclusive, not a kill.
    if not fair:
        return VerdictReport(Verdict.UNFAIR,
                             (f"AC-4 fairness violated: params={proposed.params} vs "
                              f"1.05x{min_unstruct_params}, flops={proposed.flops} vs "
                              f"1.20x{film.flops}",), **common)

    if all(criteria.values()):
        return VerdictReport(Verdict.CONFIRMED,
                             (f"AC-1..AC-5 pass; proposed beats {best_unstruct.value} by "
                              f"{margin:.1%} (p={p:.2g}, delta={delta:.2f})",), **common)

    failed = tuple(f"{k} failed" for k, v in criteria.items() if not v)
    return VerdictReport(Verdict.KILL, failed, **common)
