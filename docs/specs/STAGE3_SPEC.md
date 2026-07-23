# Spec: Stage-3 — Lie-Algebra Conditioning (GRAPE-inspired), Within the True Budget

**Status:** Pre-registered 2026-07-21, **before** any triple-composition evaluation was run.
**Derived from:** [`STAGE1_SPEC.md`](STAGE1_SPEC.md) (CONFIRMED), [`STAGE2_SPEC.md`](STAGE2_SPEC.md)
(**UNFAIR after erratum** — dense-P basis cost 1.52× FiLM, over the 1.20× ceiling),
**GRAPE** ([arXiv:2512.07805](https://arxiv.org/abs/2512.07805), "Group Representational Position
Encoding", ICLR 2026).

## What GRAPE contributes, honestly positioned

GRAPE makes position enter **linearly in a Lie algebra**: `G(n) = exp(n·ω·L)`, so relativity/
compositionality `G(n+m)=G(n)G(m)` is **exact by construction**, with closed-form rank-2 rotations —
RoPE is the special case. GRAPE does this for *attention position*. We test the same mechanism in
**FiLM's role**: an arbitrary multi-hot *condition* (not a scalar position) entering a
per-sample activation operator linearly in the Lie algebra. Positioning: mechanism from GRAPE
(cited, not claimed); the *setting* (conditioning for compositional generation) is ours — same
positioning rule as with OFT/BOFT (CLAUDE.md).

## The Stage-3 operator

`T(c) = P · R(W·c) · Pᵀ` where:

- **`W` — linear, bias-free** `K → D/2` angle map. Condition enters linearly in so(D)'s maximal
  torus: angles(c₁+c₂) = angles(c₁)+angles(c₂), and same-plane rotations commute, so
  **`T(c₁+c₂) = T(c₁)·T(c₂)` exactly, for any `P`, by construction** — composition is structural,
  not learned by an MLP head (which is where Stage-1/2's residual OOD error came from).
  Identity-init: `W=0 → T=I`.
- **`P` — structured orthogonal within the true budget** (this is the erratum fix and the
  OFT→BOFT/Group-and-Shuffle boundary): 5 layers of block-diagonal orthogonal (8 Cayley blocks of
  16×16, skew-init 0 → `P=I`) with fixed shuffle permutations between layers. Dense `P` costs
  4D² per sample (1.52× FiLM — the Stage-2 violation); GS-P applies at 2·2·D·b per layer.
- **No low-rank term** — it would break exact compositionality; the arm is pure orthogonal-Lie.

**The open empirical question (the kill-test):** can a *structured, undercomplete* `P`
(5×8×120 = 4,800 params vs dim SO(128) = 8,128) still discover the hidden basis `B` well enough to
win — within the true FLOP ceiling? If not, that is an honest KILL for within-budget basis
discovery.

## Task (unchanged) + one new split

De-aligned data `M(c)=B·R(c)·Bᵀ` (Stage-2's control). TRAIN = 8 singletons + 8 pairs;
OOD-VAL / OOD-TEST = held-out pairs (same splits, same seeds). **New:** **TRIPLES** = all C(8,3)=56
three-hot conditions, none ever trained (max trained composition length is 2) — evaluated
**exactly once**, at decision time, for every arm (single-read discipline extends to it).

## Pre-registered acceptance criteria

Gate verdict = `verdict.decide()` unchanged (corrected FLOP counter), proposed slot = the Lie arm:

- **AC-1..AC-6:** identical to Stage-1/2 (floor vs FiLM; ≥20% vs best unstructured on OOD-TEST pairs;
  MWU p≤0.01 ∧ Cliff's δ≤−0.474 over N=10 seeds; params ≤1.05× min-unstructured ∧ FLOPs ≤1.20× FiLM
  under the **corrected** counter; no in-dist regression; single OOD-TEST read).
- **AC-7 (compositionality, now a GATE):** `max‖T(c₁+c₂) − T(c₂)·T(c₁)‖_F < 1e-4` over 64 sampled
  disjoint condition pairs, on trained models. (Was diagnostic INV-4; GRAPE parametrization makes it
  provable, so we gate on it.)
- **AC-8 (length extrapolation — the GRAPE payoff):** on the 56 never-seen triples,
  (a) proposed mean MSE < **0.5×** best-unstructured triples mean MSE, and
  (b) proposed triples MSE ≤ **2×** its own OOD-pairs MSE (systematicity: error must not blow up
  with composition length).

**STAGE-3 CONFIRMED ⇔ gate-CONFIRMED ∧ AC-7 ∧ AC-8.** Any failure = KILL (with the failing
criterion named); AC-4 failure = UNFAIR, as before.

## Ablation (reported, NOT gated)

Same GS-P, **MLP** angle head (Stage-1/2 style) instead of linear `W` — isolates the Lie-linearity
contribution. Its FLOPs run ≈1.27× FiLM (over ceiling): reported transparently; if it loses on
triples *despite more compute*, the linearity claim strengthens.

## Failure semantics / hygiene

Carried over verbatim from Stage-1/2 (divergence handling, N=10 non-diverged, VAL/TEST separation,
single-read log). TRIPLES inherits the OOD-TEST single-read rule.

## Open Questions

*(none — margins above fixed at pre-registration, 2026-07-21)*
