# Spec: Stage-6 — The adaLN Swap: Lie Conditioning Inside a Diffusion Transformer

**Status:** Pre-registered 2026-07-21, before any training run.
**Hardware:** RTX PRO 6000 Blackwell (this box).

## Goal

Test the deployment claim: replacing the class-conditional path of **adaLN** (FiLM's role inside a
DiT block) with Lie conditioning improves **generation of unseen factor combinations** in a real
conditional diffusion transformer, at comparable conditioning cost.

## Task

Class/factor-conditional **diffusion** (DDPM, ε-prediction, cosine schedule, T=1000) on dSprites
at 64×64, conditioned on the full factor specification (shape one-hot + 4 normalized scalars).
Because dSprites is a deterministic factor→image map, the conditional target is **unique**:
generation quality on a held-out combination = pixel MSE between the DDIM-50 sample (fixed init
noise per eval combo) and the unique ground-truth image.

- **Grid:** 3×6×8×8×8 = 9,216 combos (as Stage-4 family). Fixed-seed shuffle → TRAIN 70% /
  OOD-VAL 15% / OOD-TEST 15% (unseen combinations; all marginal values in TRAIN).
- **Model (shared):** mini-DiT, width 128, 6 blocks, 4 heads, patch 8 (64 tokens), adaLN-zero
  final layer; identical everywhere except the class-conditioning path.

## Arms (the swap)

Timestep conditioning is standard adaLN(t) for **every** arm (timesteps are not compositional).
The class path varies:

1. **film** — the deployed DiT standard: adaLN(t + c): mod params from t-emb + c-emb jointly.
2. **hypernet** — unstructured: adaLN(t) + per-sample dense `(I+ΔW(c))` on block-entry features.
3. **proposed (Lie)** — adaLN(t) + `R(W_ℓ c)` canonical-pair rotation of block-entry features,
   `W_ℓ` linear bias-free per block. **No explicit `P`** at this site: the block's adjacent dense
   projections learn the basis, exactly as RoPE deploys (S2's coordinate-collapse does not apply
   when learned dense maps surround the rotation). Composition in `c` remains exact per site.
4. **mlp_gs (ablation, reported)** — same site, angles from an MLP head on c-emb.

## Pre-registered acceptance criteria (scoped gate; margins fixed now)

Metric: mean pixel MSE of DDIM-50 samples vs. ground truth over ≥256 OOD-TEST combos ×
10 seeds (model seeds; fixed eval noise). Statistics as always: one-sided Mann–Whitney,
Cliff's δ, N=10.

- **AC-1:** proposed < film-adaLN (the deployed standard) by **≥20%** relative, p≤0.01, |δ|≥0.474.
- **AC-2:** proposed < hypernet-adaLN by ≥20% relative, p≤0.01, |δ|≥0.474.
- **AC-3 (no in-dist regression):** proposed in-dist (TRAIN-combo) generation MSE ≤ **1.10×**
  film-adaLN's. (Kept at the S1–S5 value; S4 showed this gate can fire on us — it stays anyway.)
- **AC-4 (budget):** class-path conditioning FLOPs (encoder + heads + apply, per sample) within
  **1.20×** of film-adaLN's class path, by the shared counting convention.
- **AC-6:** OOD-TEST combos evaluated once per trained arm, after all selection on OOD-VAL.

**CONFIRMED ⇔ AC-1 ∧ AC-2 ∧ AC-3 ∧ AC-4.** Any failure = KILL/UNFAIR as before, reported as
registered.

## Config (identical across arms)

Adam 1e-3 (β2=0.99), batch 256, 20,000 steps, EMA 0.999 for sampling; N=10 seeds;
divergence/hygiene semantics verbatim from prior stages.

## Pre-run amendment (2026-07-21, before any decision run)

Per-block rotation costs 4.4× film's marginal class path (the rotation applies per token), which
would violate AC-4 by construction. Amended: the Lie arm applies **one** rotation `R(Wc)` at the
transformer entry (bijective; persists through all blocks), `W` on raw `c` (no c-MLP needed).
Class-path cost: 25,472 FLOPs = 0.74× film's marginal class path (34,560) — AC-4 satisfied with
margin. The MLP ablation uses the same single site (reported-only; ~2.2× film). Hypernet keeps
per-block dense mixing (the unstructured arm is allowed arbitrary size, as in every prior stage).

**Config amendment (pre-run, VAL-calibrated):** film-arm VAL generation MSE at 20k steps (0.063)
is still improving steeply (0.43→0.15→0.063 at 5k/10k/20k); steps raised 20,000 → 40,000 before
any decision run. No other change.
