# Spec: Stage-10 — Consistency-Trained Latents (the Stage-7 redemption experiment)

## Outcome (added after the run)

> This section was written after the experiment. Everything below it is the
> pre-registration, unchanged from before the decision run.

- **S6'** (Does a latent-consistency loss unlock rollout guarantees?) → **kill**: at least one criterion failed, so the hypothesis is rejected for this setting. Numbers in `results/stage10_summary.json`.

---

**Status:** Pre-registered 2026-07-23, before any run. Queued after Stage-8b.

## Hypothesis

Stage-7 found rollout error is representation-consistency error: nothing forces
$\T(a)\,z(s) \approx z(s{+}1)$, and an isometric operator carries each step's mismatch forever.
Add that consistency as a training loss and the picture should invert: with the mismatch driven
to zero, the Lie arm's exact composition finally pays (flat rollout error), while MLP-map
transitions still drift because their per-step operator error remains.

## Design (Stage-7 harness + one loss term)

Same sprite-world, splits, arms (film, hypernet, dynamic\_linear, proposed, mlp ablation),
budget, and eval as Stage-7. Training loss becomes, for every arm equally:
pixel BCE on the rollout target (as Stage-7) **+ $\lambda\,\|\mathrm{arm}(a, z_s) -
\mathrm{enc}(x_{s+1})\|^2/d$** on an independent single-step batch. $\lambda = 1.0$, fixed here
(a VAL sensitivity check at \{0.1, 1.0\} may be reported, never used for gate selection).

## Pre-registered acceptance criteria (N=10; MWU p≤0.01; |δ|≥0.474)

- **AC-10.1 (the claim):** at horizon 20, proposed < best unstructured arm by **≥30%**, and
  proposed's growth ratio MSE(h20)/MSE(in-dist) ≤ **0.5×** the best unstructured arm's.
- **AC-10.2:** proposed < best unstructured on unseen action-type pairs by **≥20%**.
- **AC-10.3:** in-dist ≤ **1.10×** best unstructured. **AC-10.4:** budget as Stage-7.
- Also reported: Stage-7 vs Stage-10 deltas per arm (does consistency help everyone, and whom
  most?).

**CONFIRMED ⇔ all four.** A KILL here means consistency training alone does not unlock operator
guarantees, which bounds direction 1 and elevates direction 2 (contraction).
