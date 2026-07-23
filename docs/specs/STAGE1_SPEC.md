# Spec: Stage-1 Synthetic Kill-Test — Structured Conditional Operators vs. FiLM

**Status:** Signed off — Open Questions closed 2026-07-19; ready for test derivation + implementation
**Derived from:** [`../PROPOSAL.md`](../PROPOSAL.md), [`../RESEARCH_NOTES.md`](../RESEARCH_NOTES.md)
**Registered:** _fill on sign-off_ · **PI:** Richard Bao

> This spec is the pre-registration. Success margins, seed counts, and the held-out split are
> fixed **here, before running**, so a soft win cannot be laundered into a "Pursue" (CLAUDE.md
> non-negotiable). Code and tests are derived from the AC-N / INV-N IDs below.

---

## Goal

Determine, at equal parameters and ≤20% FLOP overhead, whether an input-conditioned structured
operator `T(c)=Q(c)+U(c)V(c)ᵀ` applied per-sample in activation space generalizes to **unseen
compositions** of analytic transforms better than the strongest *unstructured* conditioner — a
binary Pursue/Kill decision.

## Scope

**In:**
- One synthetic analytic-transformation benchmark (`x ∈ ℝ^d`, condition `c` requests a linear transform `y = M(c)·x`).
- Six conditioning arms at matched budget: **FiLM** (diagonal floor), **Concat+MLP**, **Conditional-LayerNorm**, **Hypernetwork→W(c)**, **Dynamic-Linear→W(c)**, **Proposed `Q(c)+U(c)V(c)ᵀ`**.
- Two evaluation regimes: **in-distribution** (seen conditions) and **compositional-OOD** (unseen compositions of seen primitives).
- Pre-registered statistics: multi-seed, effect size, one-sided significance test, locked test split.
- Fairness harness: param counter, FLOP counter, equal hyperparameter-tuning budget per arm.
- Operator invariant tests (identity-init, orthogonality, bounded spectrum, compositionality error).
- Interpretability measurements (spectrum, effective rank, composition error) — *reported*, not gating the kill-test.

**Out:**
- Stages 2–4 (CIFAR-100, DiT/adaLN, multimodal). Gated behind a Confirmed verdict here.
- Non-linear ground-truth transforms, Lie-group (Family 4) and Flow (Family 5) operators. Family 3 only.
- Any claim that the operator families are novel (they are PEFT-owned; see RESEARCH_NOTES.md).
- Interpretability as a *gate* — it is a differentiator to report, not a Stage-1 pass/fail condition.

---

## Key design correction (why the kill-test is not "beat FiLM")

FiLM is diagonal; it **cannot** represent a rotation/permutation/shear, so a structured `T(c)·x`
beating it on these tasks is near-tautological and a reviewer will discount it. The falsifiable
hypothesis is that **structure buys *compositional generalization at equal capacity***: the
unstructured baselines (`Hypernetwork→W(c)`, `Dynamic-Linear`) can *also* represent every primitive,
so the real bar is whether the structured operator generalizes to **unseen compositions** where an
equally-expressive unstructured `W(c)` memorizes and fails. FiLM is therefore a **floor** (AC-1),
not the bar; the bar is the best unstructured arm (AC-2).

---

## Requirements (EARS)

- **R1 (ubiquitous):** The system shall generate ground-truth pairs `(x, c) → y=M(c)·x` where `M(c)` is composed from a fixed library of `K` primitive analytic transforms `{M_1..M_K}` (rotations, permutations, shears, scalings) with analytically known matrices.
- **R2 (ubiquitous):** The system shall define three condition splits: **TRAIN** (single primitives + a fixed subset of primitive pairs), **OOD-VAL** (held-out compositions, used for model/hparam selection and the auto-research sweep), and **OOD-TEST** (held-out compositions disjoint from OOD-VAL, evaluated **exactly once** at decision time).
- **R3 (event):** WHEN an arm is trained, the system shall use identical optimizer, LR schedule, epoch budget, batch size, and the identical seed set across all six arms.
- **R4 (event):** WHEN an arm is tuned, the system shall grant every arm — including FiLM — the identical hyperparameter-search budget (same number of trials over an arm-appropriate space).
- **R5 (ubiquitous):** The system shall count trainable parameters and forward-pass FLOPs of the **entire conditioning module** (condition-encoder → operator-parameter generation → operator application) for every arm with one shared counter.
- **R6 (state):** WHILE reporting the compositional-OOD result, the system shall report per-arm mean ± 95% CI over `N` seeds, the effect size vs. the best unstructured arm, and the one-sided test p-value.
- **R7 (unwanted):** IF any arm's OOD-VAL metric is used to alter the OOD-TEST evaluation (model choice, early stop, threshold), THEN the run is invalid and the system shall refuse to emit a verdict.
- **R8 (unwanted):** IF the proposed arm's parameter count exceeds the smallest unstructured baseline by >5%, or its FLOP overhead vs. FiLM exceeds 20%, THEN the comparison is disqualified as unfair and reported as such.
- **R9 (ubiquitous):** The system shall emit an append-only results log (one row per arm×seed) sufficient to recompute every reported statistic.

---

## Acceptance Criteria (the pre-registered kill-test)

> Verdict = **CONFIRMED** (→ Stage 2) only if **AC-1 ∧ AC-2 ∧ AC-3 ∧ AC-4** all hold.
> Any one failing → **KILL** (diagonal-is-enough / structure-doesn't-help is a valid negative result).

- **AC-1 (floor):** Proposed mean compositional-OOD-TEST MSE < FiLM mean compositional-OOD-TEST MSE. *(Sanity that the setup isolates conditioning; expected to pass trivially.)*
- **AC-2 (the real bar):** Proposed mean compositional-OOD-TEST MSE is lower than the **best unstructured arm** (min over Hypernetwork, Dynamic-Linear) by a **relative margin ≥ δ = 20%**: `(MSE_best_unstruct − MSE_proposed) / MSE_best_unstruct ≥ 0.20`.
- **AC-3 (significance):** The AC-2 improvement is significant at **one-sided α = 0.01** (Mann-Whitney U over `N ≥ 10` paired seeds) **and** effect size **Cliff's δ ≥ 0.474** (large).
- **AC-4 (fairness):** Proposed trainable params ≤ 1.05× the smallest unstructured baseline, **and** proposed forward FLOPs ≤ 1.20× FiLM FLOPs, both from the shared counter (R5).
- **AC-5 (no in-dist regression):** Proposed in-distribution test MSE ≤ 1.10× the best unstructured arm's in-distribution MSE. *(Structure must not buy OOD by wrecking the fit.)*
- **AC-6 (sweep hygiene):** OOD-TEST is read exactly once per arm; the results log (R9) shows a single OOD-TEST evaluation timestamp per arm, after all OOD-VAL selection is frozen.

## Invariants (property tests, derived from the operator math)

- **INV-1 (identity init):** At initialization, `‖T(c) − I‖_F < 1e-5` for every sampled `c`; the untrained proposed arm reproduces the identity-transform baseline exactly.
- **INV-2 (orthogonality):** For the `Q(c)` component, `‖Q(c)ᵀQ(c) − I‖_F < 1e-4` for every sampled `c`, throughout training.
- **INV-3 (bounded spectrum):** `σ_max(T(c)) ≤ 1 + ε` (ε = 1e-2) for every sampled `c`, throughout training — no exploding activations.
- **INV-4 (compositionality error is measured):** The quantity `‖T(c₂∘c₁) − T(c₂)·T(c₁)‖_F` is logged per step; it is a reported diagnostic (the mechanistic story), **not** a pass/fail gate.

## Failure Semantics

- Training divergence (NaN/Inf loss) on an arm → record as `diverged` for that seed, exclude from MSE stats, **report the divergence rate** per arm (training-stability metric, not silently dropped).
- OOD-VAL leakage into OOD-TEST (R7) → hard invalidation, no verdict emitted.
- Fairness violation (R8/AC-4) → verdict may still compute but is stamped **UNFAIR**; a Confirmed verdict is not allowed while the UNFAIR stamp is set.
- Fewer than `N` non-diverged seeds for any arm → verdict blocked until reruns restore `N`.

## Non-Functional

- **Seeds:** `N = 10` non-diverged seeds per arm (see Open Question OQ-3).
- **Dimension:** `d = 128` (per PROPOSAL.md); primitive library `K = 8` (see OQ-2).
- **Compute:** full six-arm sweep completes on a single GPU within one overnight run (≤ ~12 h wall).
- **Determinism:** every run reproducible from (seed, config hash); config hash recorded per row.
- **Overhead ceiling:** proposed FLOP overhead ≤ 20% vs. FiLM (also AC-4).

## Open Questions — RESOLVED (PI sign-off 2026-07-19)

- [x] **OQ-1:** Kill-test margin fixed at **δ = 20%** relative OOD-MSE improvement over best unstructured arm (AC-2).
- [x] **OQ-2:** Benchmark shape fixed at **d = 128, K = 8** primitives, **length-2** compositions for the OOD splits.
- [x] **OQ-3:** **N = 10** non-diverged seeds per arm; test = **Mann-Whitney U + Cliff's δ ≥ 0.474** (AC-3).

## Dependencies / Assumptions

- Python + a tensor framework (PyTorch assumed); single GPU.
- Ground-truth transforms are **linear** in `x` by construction, so every arm *can* fit primitives in-distribution — the discriminator is compositional OOD, not raw capacity.
- `auto-research` sweep optimizes on **OOD-VAL only**; OOD-TEST is untouched until R7/AC-6 decision time.
- No large artifacts in git (CLAUDE.md); results log + configs only.
