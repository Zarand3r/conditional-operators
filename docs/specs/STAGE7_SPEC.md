# Spec: Stage-7 — Action-Conditioned World Model (sprite-world rollouts)

**Status:** Pre-registered 2026-07-22, before any run.
**Hardware:** RTX PRO 6000 Blackwell; sweep runs ONLY after Stage-6 completes (sequential GPU use
after the PSU trip); throttled via `STAGE7_THROTTLE_MS` unless the 300 W cap has been set.

## Goal

Test the conditioning mechanism in its natural role — **relative conditions (actions)** applied
**recurrently**: does exact operator composition prevent the error compounding that learned
transitions suffer over rollouts, on unseen action combinations and never-trained horizons?

## Task

Sprite-world on the dSprites grid (state = full factor spec; deterministic frame per state).
**8 actions:** ±posX, ±posY, ±scale, ±orientation (1 grid step; orientation wraps, others clamp —
the *effective* action after clamping is what the model receives). A sample: initial frame `x₀`,
action sequence `a₁..a_H`, target = ground-truth frame of the final state.

**Model (identical across arms):** Stage-4 conv Backbone; transition is purely the conditioning
arm applied recurrently in latent space: `z_{t+1} = arm(a_t, z_t)`; decode once at the end;
BCE training loss (per the Stage-4 amendment), **pixel-MSE gated metric**. Arms =
`stage4.ARM_CLASSES` with `dc=8` (signed one-hot: +v at the action's dim, magnitudes 1).

## Splits (action-TYPE compositional, mirrors S1–S4 discipline)

Factor axes: {posX, posY, scale, orient} → C(4,2)=6 unordered type-pairs.
- **TRAIN:** rollouts length 1–3; sequences drawn from single types and pairs {posX+posY,
  scale+orient, posX+scale} (interleaved orders).
- **OOD-VAL:** pair {posY+orient}, length ≤3 (all selection).
- **OOD-TEST (read once):** (a) pairs {posX+orient, posY+scale}, length 3;
  (b) **HORIZON:** lengths 10 and 20 using TRAIN types only — never trained beyond 3.

## Pre-registered acceptance criteria (N=10 seeds; MWU one-sided; Cliff's δ)

- **AC-1 (unseen combinations):** proposed < best unstructured arm (hypernet/dyn-low-rank) on
  OOD-TEST(a) by **≥20%** rel., p≤0.01, |δ|≥0.474.
- **AC-2 (the rollout claim):** at H=20, proposed < best unstructured by **≥30%**, p≤0.01,
  |δ|≥0.474, **and** growth ratio `MSE(H=20)/MSE(H=3-indist)` for proposed ≤ **0.5×** the best
  unstructured arm's ratio.
- **AC-3 (fit parity):** in-dist (trained types, length ≤3) ≤ **1.10×** best unstructured.
- **AC-4 (budget):** conditioning-path FLOPs per applied step ≤ **1.20×** FiLM (shared counter).
- **AC-6:** OOD-TEST(a) and HORIZON sets evaluated once per trained arm, after VAL-only selection.

**CONFIRMED ⇔ AC-1 ∧ AC-2 ∧ AC-3 ∧ AC-4.** Any failure = KILL, reported as registered.
Ablation `proposed_mlp_gs` reported, never gated.

## Config

Adam 1e-3, batch 256, 12,000 steps, N=10; eval 2,048 rollouts per split (fixed generator,
identical across arms); fsync-per-row + resume logging (stage6 pattern); divergence semantics
verbatim from prior stages.

**Config amendment (pre-run, VAL-calibrated 2026-07-22):** film-arm VAL MSE at batch 256:
0.0049/0.0036/0.0025/0.0023 at 1k/2k/4k/6k steps — diminishing beyond 4k. Steps registered
12,000 → **6,000** before any decision run. No other change.
