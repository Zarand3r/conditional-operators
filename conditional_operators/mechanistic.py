"""Mechanistic-interpretability probe (the CLAUDE.md differentiator).

Trains one proposed operator and asks: do the learned rotations correspond to interpretable,
compositional subspace transforms of the primitives? Writes results/mechanistic.json.

Run:  .venv/bin/python -m conditional_operators.mechanistic
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import data
from .arms import build
from .train import TrainConfig

OUT = Path(__file__).resolve().parent.parent / "results" / "mechanistic.json"


def probe(seed: int = 0, cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    splits = data.make_splits()
    m = build("proposed", seed)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(cfg.steps):
        c, x, y = data.sample_batch(splits.train, cfg.batch, gen)
        loss = torch.mean((m(c, x) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 1. Do singleton conditions recover the true primitive rotation angle in the right subspace?
    learned = []
    for i in range(data.K):
        c = data.multihot(frozenset({i})).unsqueeze(0)
        learned.append(m.angles(m.enc(c))[0, i].item())  # block i == primitive i's plane
    angle_abs_err = [abs(l - t) for l, t in zip(learned, data.ANGLES)]

    # 2. On every HELD-OUT test pair: composition error and recovery error (INV-4 diagnostic).
    comp_err, rec_err = [], []
    for pair in splits.ood_test:
        i, j = sorted(pair)
        ci = data.multihot(frozenset({i})).unsqueeze(0)
        cj = data.multihot(frozenset({j})).unsqueeze(0)
        cij = data.multihot(pair).unsqueeze(0)
        Ti, Tj, Tij = m.dense_operator(ci), m.dense_operator(cj), m.dense_operator(cij)
        Mtrue = data.transform_matrix(pair)
        comp_err.append((Tij - Ti @ Tj).norm().item())
        rec_err.append((Tij - Mtrue).norm().item())

    result = {
        "true_angles": [round(a, 4) for a in data.ANGLES],
        "learned_singleton_angles": [round(a, 4) for a in learned],
        "max_singleton_angle_abs_err": max(angle_abs_err),
        "mean_ood_composition_err_fro": sum(comp_err) / len(comp_err),
        "mean_ood_recovery_err_fro": sum(rec_err) / len(rec_err),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = probe()
    print(json.dumps(r, indent=2))
