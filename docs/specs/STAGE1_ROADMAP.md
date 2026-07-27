# Stage-1 Harness Roadmap (strategic-engineering-planner discipline)

Architecture for the harness that produces the `ArmResult` rows consumed by
[`conditional_operators/verdict.py`](../../conditional_operators/verdict.py). Grounded in
[`STAGE1_SPEC.md`](STAGE1_SPEC.md); every module ties to a requirement ID.

## Vertical slices (each independently testable)

1. **Data (R1/R2)** — `data.py`: `K=8` primitive block-rotations in disjoint 2-planes of ℝ¹²⁸; a
   condition is a **multi-hot subset** of primitives; `y = M(c)·x` where `M(c)=∏ Rᵢ` over active `i`.
   Splits: singletons + subset of pairs = **TRAIN**; disjoint held-out pairs = **OOD-VAL**, **OOD-TEST**.
2. **Arms + counter (R5)** — `arms.py`: shared condition-encoder MLP, six operator heads, `params()`
   and analytic `flops(batch)` via one shared MAC helper.
3. **Invariants (INV-1…4)** — `tests/test_operators.py`: property tests on the proposed operator.
4. **Train/eval + sweep (R3/R4/R9)** — `train.py`, `sweep.py`: identical optimizer/schedule/seeds;
   OOD-VAL for selection, single OOD-TEST read; append-only JSONL log → `verdict.decide()`.
5. **Run + report** — `results/summary.json`: real numbers + verdict.

## The two rigor-critical decisions

- **Compositionality construction.** Primitives are rotations in **disjoint** coordinate planes, so
  they **commute** and `M(c)` is well-defined for any subset regardless of order — this makes
  "unseen composition" clean (train singletons, test unseen pairs). The hypothesis under test:
  block-structure that *matches* the data's compositional structure generalizes to unseen
  compositions, where an unstructured `W(c)` must extrapolate a nonlinear product-of-primitives map
  from few examples. This is the honest core of the claim; **Stage-1 deliberately tests the
  favorable case** (structure matches world) — richer non-commuting/non-orthogonal transforms are a
  documented follow-up, not smuggled in.
- **Budget matching = proposed is the *cheap* arm (AC-4).** The unstructured arms emit a full/low-rank
  `W(c)` (large head, `O(d²)` apply); the proposed arm emits `K` block angles + a small low-rank term
  (`O(d·block + d·r)` apply). AC-4 requires **proposed params ≤ 1.05× the smallest unstructured** and
  **FLOPs ≤ 1.20× FiLM** — so the structured operator must win with **no more** params and near-FiLM
  cost. If it needs *more* compute to win, that is not a win (CLAUDE.md equal-budget rule).

## Identity-init recipe (INV-1)

Every arm's final head layer is zero-initialized so `T(c)=I`, `β(c)=0` at step 0: FiLM `γ=1+0`;
proposed rotation-generator `A(c)=0 → matrix_exp=I`, low-rank `U=V=0`; hypernet/dynamic `W=I+0`.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Rigged-in favor of structure (block bias = data process) | Documented as the favorable-case kill-test; unstructured arms get equal capacity + tuning; a *loss* here is a hard KILL |
| Unfair handicap to hypernet via param cap | AC-4 caps the *proposed* arm, not the unstructured ones — unstructured may be larger; proposed must still win |
| Overfitting the eval via the sweep | OOD-VAL drives all selection; OOD-TEST read once (AC-6, enforced in `verdict.py`) |
| CPU-only (no GPU here) | Task is ℝ¹²⁸ + tiny nets; 6 arms × 10 seeds fits in minutes on 32 cores — no GPU needed for Stage-1 |
| Wrong significance test laundering a win | `verdict.decide()` already tested; stdlib MWU cross-checked against scipy in `tests/` |

## Scope guard

Family 3 (block-orthogonal + low-rank) only. Lie/flow families, CIFAR/DiT/multimodal = Stages 2–4,
gated behind a CONFIRMED verdict here.
