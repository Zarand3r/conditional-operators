# Pre-registration: camera control, phase 1 (synthetic renders)

**Status:** pre-registered 2026-07-27, before the harness was written and before any run.
**Name:** `camera-sliders` · **Hardware:** one RTX PRO 6000.

## Why this experiment

The deployed method for camera control in video diffusion, CameraCtrl
([arXiv:2404.02101](https://arxiv.org/abs/2404.02101)), injects camera pose by **pixel-wise
addition**: a Plücker embedding goes through an encoder and is added to the latent before the
first temporal attention layer. Addition has no compositional structure at all, so combining two
camera moves means summing two independently learned feature maps. That is the failure mode this
project studies, at a site where a real system uses it.

Before touching a video model, phase 1 asks the mechanism question in a setting where the
trajectory distribution is under our control. Real capture datasets (RealEstate10K, DL3DV) cannot
answer it: their camera-move distribution is whatever the videographer did, so combinations
cannot be held out by construction, only hoped for. Here we choose exactly which combinations the
model sees.

## Scope, stated honestly

Camera pose lives in SE(3), which does not commute. This experiment uses the **slider
parametrization** a user actually manipulates: yaw, pitch, distance, roll, each a scalar, with a
move being a vector of increments. The parameter space is $(\mathbb{R}^4,+)$, which is abelian,
so the composition law of the method applies to it directly. Full 6-DoF trajectory composition is
**out of scope** and needs the non-abelian extension, which does not exist yet. A confirmation
here is evidence about slider-style camera control, not about SE(3) trajectories.

## Task

A deterministic renderer draws a fixed synthetic scene from a camera pose. A sample is a triplet
$(x_1, \Delta, x_2)$: the frame at pose $p$, a slider delta $\Delta \in \mathbb{R}^4$, and the
ground-truth frame at pose $p + \Delta$. The model encodes $x_1$, applies the conditioning arm to
the latent with $\Delta$ as the condition, decodes, and is scored by pixel MSE against $x_2$.

Renderer: coloured 3D points splatted through a pinhole camera to a 64×64 RGB image. Because it
is analytic, every held-out camera move has an exact ground-truth frame, so the metric needs no
perceptual proxy and no human rating.

## Splits (the point of the experiment)

Four axes give $\binom{4}{2}=6$ two-axis move types.

- **TRAIN:** all four single-axis moves, plus the pairs {yaw+pitch} and {distance+roll}.
- **VAL:** the pairs {yaw+distance} and {pitch+roll}. All selection happens here.
- **TEST (read once):** the pairs {yaw+roll} and {pitch+distance}.
- **TRIPLES (reported, not gated):** all four three-axis moves, never trained.

Scenes are split too: the test frames use scene seeds never trained on, so the model cannot win
by memorising a scene.

## Arms

Identical backbone, optimizer, schedule, seeds and data for every arm; only the conditioning
module differs.

- `additive` — $z + \mathrm{enc}(\Delta)$. **The mechanism CameraCtrl deploys**, and the
  baseline that matters most.
- `film`, `concat_mlp`, `cond_layernorm` — the standard conditioners.
- `hypernet`, `dynamic_linear` — the unstructured, higher-capacity conditioners.
- `proposed` — CGA: $T(\Delta) = P\,R(W\Delta)\,P^\top$, angles linear in the slider vector.
- `cfilm_hyb` — Complex FiLM, magnitude for content and phase for composition.
- `proposed_mlp_gs` — reported ablation: same basis, MLP angle head, no linearity.

## Pre-registered criteria

Scored by the unchanged `verdict.decide()` on TEST pixel MSE, $N=10$ seeds, one-sided
Mann-Whitney $p \le 0.01$, Cliff's $|\delta| \ge 0.474$.

- **C1 (against the deployed mechanism):** `proposed` beats `additive` on held-out move types by
  **≥25%** relative.
- **C2 (against the strongest unstructured arm):** `proposed` beats the better of `hypernet` and
  `dynamic_linear` by **≥20%** relative.
- **C3 (no fit regression):** `proposed` in-distribution MSE ≤ **1.10×** the best unstructured
  arm's.
- **C4 (budget):** `proposed` conditioning FLOPs ≤ **1.20×** `film`'s, parameters ≤ **1.05×** the
  smallest unstructured arm's, from the shared counter.
- **C5 (length extrapolation, reported not gated):** MSE on never-trained three-axis moves,
  and its ratio to the two-axis result.

**CONFIRMED ⇔ C1 ∧ C2 ∧ C3 ∧ C4.** Any failure is a KILL and is reported as one. A KILL on C1
with a pass on C2 would mean the win is over research baselines but not over what is actually
deployed, and would be reported in exactly those words.

## Config

Adam 1e-3, batch 128, 8000 steps, $N=10$ seeds; 2048 fixed evaluation triplets per split from a
deterministic generator, identical across arms; fsync-per-row logging with resume; divergence
handling as in every previous experiment. Throttled (`CAMERA_THROTTLE_MS=60`) unless the card is
power-capped.

Amendments after this point must be dated and recorded below.
