"""Synthetic analytic-transformation benchmark (R1/R2 of STAGE1_SPEC.md).

Primitives are rotations in DISJOINT 2-planes of R^d, so they commute and M(c) is well-defined
for any subset of primitives regardless of order. A condition c is a subset of primitives; the
ground truth is y = M(c) x with M(c) = product of the active block rotations.

Splits (compositions of length <= 2):
  TRAIN     = all K singletons + a fixed subset of pairs
  OOD_VAL   = held-out pairs (model/hparam selection, the auto-research sweep)
  OOD_TEST  = held-out pairs disjoint from OOD_VAL (read exactly once, AC-6)
The three condition sets are disjoint by construction (test_data asserts it).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

D = 128           # activation dimension
K = 8             # number of primitive transforms
BLOCK = 2         # each primitive rotates a disjoint 2-plane
SPLIT_SEED = 1234  # fixed: splits are stable across training seeds

# Distinct, non-trivial rotation angles (avoid 0 and multiples of pi).
ANGLES: tuple[float, ...] = tuple(0.3 + 0.9 * i / (K - 1) for i in range(K))  # 0.3 .. 1.2 rad


def transform_matrix(active: frozenset[int]) -> torch.Tensor:
    """d x d matrix M(c): identity except a 2x2 rotation in plane i for each active primitive i."""
    m = torch.eye(D)
    for i in active:
        if not 0 <= i < K:
            raise ValueError(f"primitive index {i} out of range [0,{K})")
        a = ANGLES[i]
        c, s = math.cos(a), math.sin(a)
        p = BLOCK * i
        m[p, p], m[p, p + 1] = c, -s
        m[p + 1, p], m[p + 1, p + 1] = s, c
    return m


def multihot(active: frozenset[int]) -> torch.Tensor:
    """K-dim multi-hot conditioning input for a subset of primitives."""
    v = torch.zeros(K)
    for i in active:
        v[i] = 1.0
    return v


@dataclass(frozen=True)
class Splits:
    train: tuple[frozenset[int], ...]
    ood_val: tuple[frozenset[int], ...]
    ood_test: tuple[frozenset[int], ...]

    def all_conditions(self) -> tuple[frozenset[int], ...]:
        return self.train + self.ood_val + self.ood_test


def make_splits() -> Splits:
    """Deterministic partition: 8 singletons + 8 pairs -> TRAIN; 10 pairs -> VAL; 10 pairs -> TEST."""
    singletons = [frozenset({i}) for i in range(K)]
    pairs = [frozenset(p) for p in itertools.combinations(range(K), 2)]  # 28 pairs
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    pairs = [pairs[i] for i in perm]
    train_pairs, val_pairs, test_pairs = pairs[:8], pairs[8:18], pairs[18:28]
    return Splits(
        train=tuple(singletons + train_pairs),
        ood_val=tuple(val_pairs),
        ood_test=tuple(test_pairs),
    )


# Precompute transform matrices once per condition (cached by identity of the frozenset contents).
_MATRIX_CACHE: dict[frozenset[int], torch.Tensor] = {}


def _matrix(active: frozenset[int]) -> torch.Tensor:
    if active not in _MATRIX_CACHE:
        _MATRIX_CACHE[active] = transform_matrix(active)
    return _MATRIX_CACHE[active]


def sample_batch(conditions: tuple[frozenset[int], ...], n: int,
                 generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw n examples, each with a uniformly-random condition from `conditions`.

    Returns (c_multihot [n,K], x [n,D], y [n,D]) with y = M(c) x per row.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    idx = torch.randint(len(conditions), (n,), generator=generator)
    x = torch.randn(n, D, generator=generator)
    c = torch.empty(n, K)
    y = torch.empty(n, D)
    for row in range(n):
        active = conditions[idx[row].item()]
        c[row] = multihot(active)
        y[row] = _matrix(active) @ x[row]
    return c, x, y


def eval_set(conditions: tuple[frozenset[int], ...], n_per_condition: int,
             generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced evaluation set: n_per_condition examples for EACH condition in the split."""
    cs, xs, ys = [], [], []
    for active in conditions:
        x = torch.randn(n_per_condition, D, generator=generator)
        m = _matrix(active)
        cs.append(multihot(active).expand(n_per_condition, K))
        xs.append(x)
        ys.append(x @ m.T)  # (n,D) @ (D,D)^T == M x per row
    return torch.cat(cs), torch.cat(xs), torch.cat(ys)
