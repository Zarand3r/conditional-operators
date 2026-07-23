"""Train/eval one arm at one seed under an IDENTICAL fixed config for every arm (R3/R4).

Equal-budget discipline: a single shared TrainConfig is used for all arms and seeds — no per-arm
hyperparameter tuning, the strictest form of "equal tuning budget" (R4). OOD-VAL is available for
monitoring/selection; OOD-TEST is evaluated exactly once at the end (AC-6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import arms
from .data import Splits, eval_set, make_splits, sample_batch


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 4000
    batch: int = 256
    lr: float = 1e-3
    eval_per_condition: int = 64


@dataclass(frozen=True)
class RunResult:
    arm: str
    seed: int
    indist_mse: float
    ood_val_mse: float
    ood_test_mse: float
    diverged: bool
    n_params: int
    flops: int


def _mse(model: arms.Arm, c: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(c, x)
    return torch.mean((pred - y) ** 2).item()


def train_one(arm_name: str, seed: int, splits: Splits, cfg: TrainConfig) -> RunResult:
    model = arms.build(arm_name, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(seed)  # data stream seed (shared scheme across arms)

    diverged = False
    for _ in range(cfg.steps):
        c, x, y = sample_batch(splits.train, cfg.batch, gen)
        pred = model(c, x)
        loss = torch.mean((pred - y) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Deterministic eval sets (separate generator so evaluation is identical across arms).
    egen = torch.Generator().manual_seed(10_000 + seed)
    if diverged:
        return RunResult(arm_name, seed, math.nan, math.nan, math.nan, True,
                         model.n_params(), model.flops())

    ci, xi, yi = eval_set(splits.train, cfg.eval_per_condition, egen)
    cv, xv, yv = eval_set(splits.ood_val, cfg.eval_per_condition, egen)
    ct, xt, yt = eval_set(splits.ood_test, cfg.eval_per_condition, egen)  # OOD-TEST: single read
    return RunResult(
        arm=arm_name, seed=seed,
        indist_mse=_mse(model, ci, xi, yi),
        ood_val_mse=_mse(model, cv, xv, yv),
        ood_test_mse=_mse(model, ct, xt, yt),
        diverged=False,
        n_params=model.n_params(),
        flops=model.flops(),
    )
