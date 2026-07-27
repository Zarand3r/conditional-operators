# Pre-registration: attribute editing in a frozen latent space

**Status:** pre-registered 2026-07-27, before the decision sweep.
**Name:** `latent-edit` · **Hardware:** one RTX PRO 6000.

## Disclosure, up front

This registration is **not blind**, and pretending otherwise would be worse than saying so. Before
writing it we ran a one-seed calibration across every arm and, in the first version of that run,
read the test split by mistake. It showed our operator **losing** to an unstructured hypernet on
held-out attribute pairs by roughly 60%. Two things follow.

First, nothing here was tuned to that observation. The success criteria below are copied unchanged
from the criteria this project has used since stage 4 — same margins, same statistics, same budget
ceilings. We are registering a criterion we expect to **fail**, which is the opposite of the
failure mode pre-registration exists to prevent.

Second, the sweep is still worth running. The one-seed number says which way the result points; it
does not say whether the effect is real, how large it is, or whether it survives ten seeds. And the
comparison against the co-trained run on this same dataset is the actual scientific content, so it
needs to rest on more than one seed.

## Why this experiment

Every result this project has confirmed so far shares a feature we never isolated: the
representation the operator acts on was **trained alongside the operator**. That leaves an obvious
question unanswered, and it is the question a practitioner would ask first.

The deployed pattern for attribute editing is a **frozen** generator with edits applied in its
latent space — InterFaceGAN and GANSpace move along fixed directions in a frozen GAN's latent,
diffusion editing methods apply edits in a frozen autoencoder's latent. Nobody retrains the
generator. If our operator only helps when it gets to shape the representation, that rules out the
single most common way anyone would try to use it, and saying so is more useful than another win on
a co-trained benchmark.

The task is a controlled version of exactly that pattern.

## Task

An autoencoder is trained once on 3D Shapes with no conditioning at all, then **frozen**. Every
arm shares it. A sample is a source image, an edit vector saying which attributes move and in which
direction, and the target image. The arm maps the source's latent to a predicted target latent.

Scoring happens **in latent space**, against the encoder's latent for the true target image. The
decoder is never in the loop, so reconstruction quality cannot mask or flatter any arm. This is the
fix for what sank the camera experiment, where reconstruction error dominated and every arm scored
alike.

Five attribute axes: floor hue, wall hue, object hue, scale, orientation. Object shape is excluded
because a categorical swap has no step size and so nothing to compose. Hues and orientation are
circular; a scale step that would leave the range is turned around rather than clamped, so the edit
vector always describes the move exactly.

## Splits

Ten attribute pairs, partitioned and never overlapping.

- **TRAIN:** all five single-attribute edits, plus the pairs {floor+wall hue}, {object hue+scale},
  {floor hue+orientation}.
- **VAL:** {wall+object hue}, {scale+orientation}. Screening and any calibration read this only.
- **TEST (read once for the decision):** the remaining five pairs.
- **TRIPLES (reported, not gated):** six three-attribute edits, never trained.

## Pre-registered criteria

Scored by the unchanged `verdict.decide()` on TEST latent MSE, N=10 seeds, one-sided Mann-Whitney
p ≤ 0.01, Cliff's |δ| ≥ 0.474. Margins identical to stages 4, 5 and 7.

- **C1:** `proposed` beats the better of `hypernet` and `dynamic_linear` on held-out pairs by
  **≥20%** relative.
- **C2:** `proposed` beats `additive` — the mechanism latent editing actually deploys — by
  **≥20%** relative.
- **C3:** `proposed` in-distribution MSE ≤ **1.10×** the best unstructured arm's.
- **C4:** `proposed` conditioning FLOPs ≤ **1.20×** `film`'s and parameters ≤ **1.05×** the
  smallest unstructured arm's, from the shared counter.
- **C5 (reported, not gated):** error on never-trained three-attribute edits.

**CONFIRMED ⇔ C1 ∧ C2 ∧ C3 ∧ C4.** Any failure is a KILL and will be reported as one.

## What each outcome would mean

This is written down now so that neither result can be reinterpreted later.

- **KILL,** with the co-trained run on this same dataset having confirmed: the operator's advantage
  requires co-training, and the frozen-generator deployment pattern is outside its range. That is a
  boundary condition on our own claim and belongs in the paper as one.
- **CONFIRMED:** the advantage is a property of the operator itself and transfers to frozen
  representations, which would considerably widen what the method is good for.

Either way the comparison of interest is against **stage 5**, which ran the same dataset, the same
attribute-pair splits and the same arms with the representation co-trained. The two runs differ in
one variable, so the pair is the evidence, not either run alone.

## Config

Adam 1e-3, batch 128, 4000 steps, N=10 seeds. The autoencoder is trained once (6000 steps, batch
128, reconstruction MSE 0.00044) and its weights are reused byte-identically by every arm and seed;
its latent is standardized by a fixed mean and scale recorded at freeze time. Evaluation uses 1024
fixed samples per split from a deterministic generator, identical across arms. fsync-per-row
logging with resume. Throttled (`LATENT_THROTTLE_MS=40`) per the standing power rule.

Amendments after this point must be dated and recorded below.
