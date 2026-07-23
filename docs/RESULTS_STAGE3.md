# Stage-3 Results — Lie-Algebra Conditioning (GRAPE-inspired), Within the True Budget

*Auto-generated from `results/stage3_summary.json` by `render_stage3.py`.*
Criteria pre-registered in [`specs/STAGE3_SPEC.md`](specs/STAGE3_SPEC.md) (2026-07-21, before any triples evaluation). Mechanism per GRAPE ([arXiv:2512.07805](https://arxiv.org/abs/2512.07805)): the condition enters **linearly in the Lie algebra** — `T(c) = P·R(W·c)·Pᵀ` — so `T(c₁+c₂) = T(c₁)T(c₂)` exactly, by construction. `P` is a Group-and-Shuffle structured orthogonal that fits the corrected ≤1.20× FiLM FLOP ceiling (the dense-P Stage-2 arm did not: see the Stage-2 erratum).
Config: 10 seeds, 4000 steps.

## Final verdict: **CONFIRMED**  (gate: confirmed)

> CONFIRMED — Lie-algebra conditioning wins within the TRUE budget, with exact compositionality and systematic length extrapolation.

- gate: AC-1..AC-5 pass; proposed beats hypernet by 100.0% (p=9.1e-05, delta=-1.00)
- AC-7 exact compositionality: max ‖T(c₁+c₂)−T(c₂)T(c₁)‖_F = 6.67e-06 (gate < 1e-4) → ✅
- AC-8a triples vs best unstructured: 5.33e-08 vs 0.5×6.84e-03 → ✅
- AC-8b systematicity (triples ≤ 2× own pairs): 5.33e-08 vs 2×3.31e-08 → ✅

## Composition-length extrapolation (never-trained 3-hot conditions)

| Arm | OOD pairs MSE | **Triples MSE** (all 56, unseen) | FLOPs/FiLM |
|---|---|---|---|
| film | 1.55e-02 | 2.77e-02 ± 3.33e-04 | 1.00× |
| concat_mlp | 1.12e-02 | 2.01e-02 ± 3.45e-03 | 1.65× |
| cond_layernorm | 2.71e-02 | 3.91e-02 ± 4.48e-04 | 1.01× |
| hypernet | 3.44e-03 | 6.84e-03 ± 9.21e-04 | 42.74× |
| dynamic_linear | 6.08e-03 | 1.23e-02 ± 4.91e-04 | 3.30× |
| **proposed** | 3.31e-08 | 5.33e-08 ± 2.41e-08 | 1.11× |
| proposed_mlp_gs *(ablation, over budget)* | 1.07e-03 | 2.31e-03 ± 3.92e-04 | 1.26× |

**Gate margin (pairs, vs `hypernet`):** 100.0% · p=9.1e-05 · Cliff's δ=-1

## Per-arm detail

| Arm | OOD-TEST pairs (mean±sd) | in-dist | params | FLOPs/FiLM |
|---|---|---|---|---|
| film | 1.55e-02 ± 4.32e-04 | 1.57e-02 | 50,688 | 1.00× |
| concat_mlp | 1.12e-02 ± 3.65e-03 | 1.42e-05 | 83,584 | 1.65× |
| cond_layernorm | 2.71e-02 ± 4.90e-04 | 2.72e-02 | 50,688 | 1.01× |
| hypernet | 3.44e-03 ± 8.70e-04 | 5.23e-05 | 2,147,712 | 42.74× |
| dynamic_linear | 6.08e-03 ± 4.42e-04 | 2.13e-05 | 166,272 | 3.30× |
| **proposed** | 3.31e-08 ± 2.19e-08 | 9.38e-13 | 44,928 | 1.11× |
| proposed_mlp_gs | 1.07e-03 ± 1.95e-04 | 7.19e-06 | 53,184 | 1.26× |

## Reading

- **The MLP-head ablation isolates the mechanism**: same structured `P`, same task — the only difference is whether the condition enters the Lie algebra *linearly* (exact composition) or through an MLP (learned composition). Compare their triples rows.
- The Lie arm is **exactly orthogonal** (singular values ≡ 1: INV-3 for free) and satisfies AC-7 *for any weights* — composition is a property of the parametrization, not of training.
- `P` is undercomplete (4,800 params vs dim SO(128)=8,128) yet suffices — it only needs the K active eigenplanes of the hidden basis, not all of it. This is the within-budget answer to the Stage-2 erratum and the OFT→BOFT boundary (structured orthogonal, Group-and-Shuffle style).

## Honest scope

- The data-generating process is *literally a commuting one-parameter group* — the exact match for a Lie parametrization. This stage proves the mechanism decisively **when the conditioning factors form a group**; it does not show real-world conditions behave this way. That is the Stage-4 (real conditional generation, GPU) question.
- Mechanism credit: linear-in-Lie-algebra conditioning is GRAPE's construction (for attention position); the contribution here is transplanting it to FiLM's role — per-sample activation conditioning on arbitrary multi-hot conditions — and the budget-fair, pre-registered demonstration of systematic compositional generalization in that role.
