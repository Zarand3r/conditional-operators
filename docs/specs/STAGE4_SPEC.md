# Spec: Stage-4 — Real-Image Compositional Conditional Transformation (dSprites, GPU)

## Outcome (added after the run)

> This section was written after the experiment. Everything below it is the
> pre-registration, unchanged from before the decision run.

- **dsprites** (Does the advantage survive a learned latent on real images?) → **confirmed**: every pre-registered criterion passed. Numbers in `results/stage4_summary.json`.
- **dsprites+shape** (Does it survive a categorical (non-group) factor?) → **confirmed**: every pre-registered criterion passed. Numbers in `results/stage4b_summary.json`.
- **shapes3d** (Does it hold on a second dataset (3D Shapes, RGB)?) → **kill**: at least one criterion failed, so the hypothesis is rejected for this setting. Numbers in `results/stage5_summary.json`.

---

**Status:** Pre-registered 2026-07-21, **before any training or OOD evaluation was run**.
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (96 GB), this box.
**Derived from:** Stages 1–3 ([`STAGE3_SPEC.md`](STAGE3_SPEC.md) CONFIRMED on group-constructed
synthetic data). This is the first **real-image, learned-representation** test — the escalation
Stage-3's "honest scope" demanded, and the world-model action-conditioning setting flagged in
[`../RESEARCH_NOTES.md`](../RESEARCH_NOTES.md) ("Fit to prior discussion").

> **Design note (hygiene):** a first draft of this spec used constant-latent conditional
> *generation* (`h = T(c)·x₀ + β(c)`, learned constant `x₀`). Discarded before any run: with a
> constant input, `β(c)` alone saturates every arm's expressivity, washing out the operator
> comparison. The transformation design below keeps a *varying* input (real image latents) in
> FiLM's role.

## Goal

Determine whether Lie-algebra conditioning (`T(Δ)=P·R(WΔ)·Pᵀ`, Stage-3's arm) beats FiLM and the
strongest unstructured conditioner at **transforming real images by unseen combinations of factor
changes**, on a learned latent it does not control, at equal backbone and ≤1.20× FiLM conditioning
cost.

## Task

dSprites (deterministic image ↔ factor mapping; full 737,280-image grid resident on GPU).

- A sample is a triplet `(x₁, Δ, x₂)`: image `x₁` with factors `z₁`, change vector `Δ` over the 4
  transformable factors **[scale, orientation, posX, posY]** (shape is categorical — never changed;
  it must flow through the encoder), and ground-truth `x₂` = the dataset image at `z₁+Δ`
  (orientation wraps; others sampled valid). `Δ` is given to all arms identically as 4 normalized
  scalars. Magnitudes: scale ±1–2, orientation ±1–5, posX/posY ±1–4 (grid steps).
- **Model:** shared conv encoder → `z ∈ ℝ¹²⁸` → conditioning arm: `ẑ = T(Δ)·z + β(Δ)` → shared
  conv decoder → `x̂₂`. Encoder/decoder architecture and training are **identical across arms**;
  only the conditioning module differs (FiLM's exact role). Loss/metric: per-pixel MSE vs `x₂`.
- **Compositional splits by Δ-type** (the conditioning analog of Stage-1's splits; C(4,2)=6
  two-factor types):
  - **TRAIN:** all 4 single-factor types + two-factor types {scale+orient, posX+posY}
  - **OOD-VAL:** two-factor types {scale+posX, orient+posY} (all selection/monitoring)
  - **OOD-TEST:** two-factor types {scale+posY, orient+posX} — **read exactly once**
  - **TRIPLES (diagnostic):** all four 3-factor types, never trained — reported, not gated
    (real-image analog of Stage-3's AC-8; not gated because the learned latent may not support
    exact group action, which is itself the finding).

## Arms

Gate arms (six, as always): `film`, `concat_mlp`, `cond_layernorm`, `hypernet`, `dynamic_linear`,
`proposed` = Lie (`angles = W·Δ` linear bias-free ⇒ `T(Δ₁+Δ₂)=T(Δ₁)T(Δ₂)` exactly; GS-P from
Stage-3). Reported ablation: `proposed_mlp_gs` (same GS-P, MLP angle head). All heads
zero-initialized (`T=I`, `β=0` at init). The Lie arm additionally satisfies `T(0)=I` structurally.

## Pre-registered acceptance criteria

Verdict via the **unchanged** `verdict.decide()` on OOD-TEST pixel-MSE, N=10 seeds:

- **AC-1..AC-6:** as Stages 1–3 (floor vs FiLM; **≥20%** vs best unstructured; MWU p≤0.01 ∧
  Cliff's δ≤−0.474; conditioning-module params ≤1.05× min-unstructured ∧ FLOPs ≤1.20× FiLM under
  the corrected counter; in-dist ≤1.10×; single OOD-TEST read).
- **AC-7 (composition):** diagnostic only on this stage (see TRIPLES above).

**STAGE-4 CONFIRMED ⇔ gate-CONFIRMED.** A KILL — including "orthogonal rigidity hurts on learned
real-image latents" — is a valid, reportable negative. Margins fixed now, before any run.

## Training config (identical across arms)

Adam 1e-3, batch 256, **12,000 steps**, N=10 seeds; fixed eval sets of 2,048 triplets per split
(deterministic generator); divergence/N/hygiene semantics verbatim from Stages 1–3.

**Pre-run amendments (2026-07-21, before any decision data):**
1. **Loss = BCEWithLogits** (plain MSE stalls at the mean-predictor plateau on sparse binary
   dSprites; diagnosed on TRAIN/VAL only). The **gated metric remains per-pixel MSE** as
   registered. Steps raised 6,000 → 12,000 from the same calibration.
2. **Hygiene disclosure:** three 600-step plumbing smoke runs (seed 0, discarded models)
   evaluated OOD-TEST-type triplets before this amendment; their values were identical across
   arms (conditioning untrained) and no design or selection decision used them. All subsequent
   calibration used TRAIN/VAL only. The decision sweep's OOD-TEST read remains single-read per
   trained arm.
