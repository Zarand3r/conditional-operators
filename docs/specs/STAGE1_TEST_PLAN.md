# Test Plan: Stage-1 Kill-Test (derived from STAGE1_SPEC.md)

Every acceptance criterion and invariant in [`STAGE1_SPEC.md`](STAGE1_SPEC.md) maps to at least one
named test below (spec verification check #3). Tests are tagged by the **strongest local signal**
that proves the claim, cheapest-first.

## Merge gate (binary — stated before coding)

A Stage-1 verdict may be emitted **only** when:
1. `python -m unittest discover -s tests` is green (decision-logic + stats tests — runnable today, stdlib-only).
2. The invariant property tests (INV-1…4) are green under the torch stack.
3. The results log (R9) is attached and the `verdict.decide()` output is reproduced from it.
4. Coverage: every AC-N and INV-N ID below resolves to ≥1 passing test.

## Test pyramid allocation

| Layer | What | Runnable now? |
|---|---|---|
| **Pure-logic unit** | verdict decision, Mann-Whitney U, Cliff's δ, fairness/leakage/hygiene guards | **Yes** (stdlib) |
| **Property** | operator invariants INV-1…4 (identity-init, orthogonality, spectrum, composition error) | No — needs torch |
| **Integration** | end-to-end: data-gen → train one arm → eval → log row | No — needs torch |
| **Evidence artifact** | full six-arm sweep results log + reproduced verdict | No — needs GPU run |

## Traceability — AC/INV → test

| ID | Claim | Test(s) | Layer | Signal today |
|---|---|---|---|---|
| **AC-1** | proposed OOD-MSE < FiLM (floor) | `test_verdict::test_ac1_floor_required` | unit | ✅ green |
| **AC-2** | proposed beats best-unstructured by ≥20% rel | `test_verdict::test_ac2_margin_pass`, `::test_ac2_margin_fail_is_kill` | unit | ✅ green |
| **AC-3** | improvement significant (MWU p≤0.01 ∧ Cliff's δ≤−0.474) | `test_stats::test_mwu_*`, `test_stats::test_cliffs_delta_*`, `test_verdict::test_ac3_significance_fail_is_kill` | unit | ✅ green |
| **AC-4** | params ≤1.05× min-unstruct ∧ FLOPs ≤1.20× FiLM | `test_verdict::test_ac4_unfair_blocks_confirm` | unit | ✅ green |
| **AC-5** | no in-dist regression (≤1.10×) | `test_verdict::test_ac5_indist_regression_is_kill` | unit | ✅ green |
| **AC-6** | OOD-TEST read exactly once | `test_verdict::test_ac6_double_read_is_invalid` | unit | ✅ green |
| **R7** | OOD-VAL→OOD-TEST leakage invalidates | `test_verdict::test_leakage_is_invalid` | unit | ✅ green |
| **Failure: divergence** | diverged seeds excluded, rate reported; <N blocks verdict | `test_verdict::test_insufficient_seeds_blocks`, `::test_divergence_rate_reported` | unit | ✅ green |
| **INV-1** | `‖T(c)−I‖_F < 1e-5` at init, ∀c | `test_operators::test_identity_init` | property | ⏳ pending torch |
| **INV-2** | `‖Q(c)ᵀQ(c)−I‖_F < 1e-4`, ∀c, throughout | `test_operators::test_orthogonality` | property | ⏳ pending torch |
| **INV-3** | `σ_max(T(c)) ≤ 1+1e-2`, ∀c, throughout | `test_operators::test_bounded_spectrum` | property | ⏳ pending torch |
| **INV-4** | composition error logged (diagnostic) | `test_operators::test_composition_error_is_finite_and_logged` | property | ⏳ pending torch |
| **R5/R8** | shared param+FLOP counter agrees across arms | `test_counter::test_flops_params_counted_per_arm` | integration | ⏳ pending torch |
| **R2/R7** | TRAIN / OOD-VAL / OOD-TEST splits disjoint | `test_data::test_splits_disjoint`, `::test_ood_is_unseen_composition` | unit | ⏳ pending numpy |

## Property-test specifications (for `test-driven-verification` under torch)

- **INV-1:** sample 256 random `c`; assert `max_c ‖T(c)−I‖_F < 1e-5` on the *untrained* proposed layer. Red on any non-identity init.
- **INV-2:** after each of {0, 1, mid, final} training checkpoints, sample 256 `c`; assert `‖Q(c)ᵀQ(c)−I‖_F < 1e-4`. Parametrize the orthogonal parametrization (Cayley/matrix-exp) as the unit under test.
- **INV-3:** `σ_max` via power iteration or SVD on `T(c)`; assert `≤ 1.01` for 256 `c` at every checkpoint. This is the "no exploding activations" guard.
- **INV-4:** compute `‖T(c₂∘c₁) − T(c₂)T(c₁)‖_F` for sampled composed conditions; assert finite and log to the results row. **Not a gate** — it is the mechanistic-interpretability diagnostic (spec Out-of-scope for pass/fail).

## What is deliberately NOT tested

- Absolute MSE values (only relative margins are pre-registered — AC-2).
- Wall-clock (non-deterministic); FLOPs stands in for cost (AC-4).
- Interpretability *quality* (concepts↔rotations) — reported in Stage-1 writeup, not gated.
