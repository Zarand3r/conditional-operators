# Spec: Stage-9 — Guidance as Group Power (follow-up, pre-registered)

## Outcome (added after the run)

> This section was written after the experiment. Everything below it is the
> pre-registration, unchanged from before the decision run.

- **S8** (Does condition powering beat classifier-free guidance?) → **confirmed**: every pre-registered criterion passed. Numbers in `results/stage9_summary.json`.

---

**Status:** Pre-registered 2026-07-23 (margins fixed now, before any run). Implementation and
launch follow Stage-8b's verdict; the only 8b-dependence is the arm-selection rule below, which
is itself fixed here.

## Hypothesis

Classifier-free guidance (CFG) strengthens conditioning by linear extrapolation in noise space,
$\hat\varepsilon = \varepsilon_u + w(\varepsilon_c - \varepsilon_u)$, and distorts at large $w$
(oversaturation, mode collapse); the extrapolated prediction leaves the distribution the network
was trained on. In CGA, strengthening a condition is $T(\alpha c) = T(c)^\alpha$: an exact group
power that walks further along the same one-parameter subgroup, so the network receives a valid
operator at every $\alpha$ (rotation spectra stay at 1; exponential-FiLM magnitudes stay
exponential-family). Prediction: **distortion grows with $w$ under CFG and stays flat or grows
slowly with $\alpha$ under group power.**

## Why our harness is the right instrument

dSprites conditioning is deterministic, so "guidance distortion" is objectively measurable:
generation MSE against the unique ground-truth image, per guidance strength. No FID, no human
eval, no proxy.

## Design

Stage-6 mini-DiT harness. Both arms train with 10\% condition dropout (identical data,
optimizer, 40k steps, N=10 seeds; fsync+resume; throttled).

1. **film + CFG:** standard adaLN; sample with CFG at strength $w$.
2. **cfilm + group power:** Complex-FiLM class path; sample at strength $\alpha$ by scaling the
   condition's algebra coordinates. **Arm-selection rule (fixed now):** use cfilm\_lin if it
   passed Stage-8b's non-inferiority gate (both channels linear, power exact in both); otherwise
   cfilm\_hyb with the exact power applied to the phase channel and linear scaling of the
   magnitude head's input as a disclosed approximation.

Guidance sweep (single read, fixed eval combos/noise): $w,\alpha \in \{1, 1.5, 2, 3, 5, 8\}$;
metrics: OOD-combination generation MSE per strength (the distortion curve) and in-dist MSE at
strength 1.

## Pre-registered acceptance criteria

- **AC-9.1 (parity at native strength):** cfilm arm's OOD MSE at strength 1 $\le$ 1.10$\times$
  film's at $w{=}1$.
- **AC-9.2 (graceful scaling, the claim):** degradation ratio $\mathrm{MSE}(8)/\mathrm{MSE}(1)$
  for group power $\le$ **0.5$\times$** CFG's ratio, MWU $p\le0.01$, $|\delta|\ge0.474$ over
  seeds.
- **AC-9.3 (usefulness, not just flatness):** group power's best-over-strengths OOD MSE $\le$
  CFG's best-over-strengths OOD MSE $\times$ 1.10 (a flat curve that never improves anything is
  not guidance).
- Registered caveat: CFG sharpens the sampling distribution; group power amplifies the
  conditioning signal. These are different semantics; the deterministic-MSE metric measures
  distortion and favors neither a priori. A KILL on AC-9.3 with a PASS on AC-9.2 would mean
  "graceful but not useful," and will be reported exactly so.

**CONFIRMED $\Leftrightarrow$ AC-9.1 $\wedge$ AC-9.2 $\wedge$ AC-9.3.** If confirmed, the
deployment pitch writes itself: guidance without the second forward pass and without the
distortion, at any strength, with per-dimension control ($\alpha$ per condition component).

## Execution

Queued behind Stage-8b (sequential GPU; power rules as established). Estimated cost: 2 arms
$\times$ 10 seeds $\times$ 40k steps $\approx$ 5--6 h throttled, plus a cheap sampling sweep.
