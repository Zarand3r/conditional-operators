# Spec: Stage-8 — Complex FiLM (the candidate FiLM improvement)

**Status:** Pre-registered 2026-07-23, before any run. Runs automatically after Stage-7
(sequential GPU; throttled; fsync+resume).

## Hypothesis

FiLM multiplies each feature by a real number; our boundary map (S1–S6) shows the affine/real
channel injects content but cannot compose, while the rotation/phase channel composes but cannot
inject content. **Complex FiLM** multiplies feature *pairs* by a complex number
$m\,e^{i\theta}$: magnitude = content channel, phase = composition channel. One operator, both
roles. "FiLM is the real part; use the whole complex number."

## Arms (added to the established harnesses)

- **cfilm\_lin:** $s=W_s c$, $\theta=W_\theta c$, both linear bias-free (zero-init ⇒ identity);
  $y_{pair} = e^{s}\,R(\theta)\,x_{pair}$. Fully compositional in both channels
  ($s$ clamped to $|s|\le4$ for stability; clamp active only far off-distribution).
- **cfilm\_hyb (the candidate FiLM replacement):** magnitude from the standard MLP head (FiLM's
  content mechanism), phase linear $\theta=W_\theta c$ (exact phase composition). Trades
  magnitude-composition for content power; keeps the composition channel exact.

## Stage-8a — transformation role (dSprites delta task, Stage-4 harness)

Arms: film, hypernet, proposed (Lie), cfilm\_lin, cfilm\_hyb. N=10 seeds, 12k steps, identical
protocol/eval to Stage-4 (BCE train, pixel-MSE gate, single-read OOD).

- **AC-8a1:** cfilm\_hyb < film on OOD (unseen two-factor change types) by **≥20%**, MWU p≤0.01,
  |δ|≥0.474.
- **AC-8a2:** cfilm\_hyb in-dist ≤ **1.10×** film.
- **AC-8a3:** cfilm\_hyb conditioning FLOPs ≤ **1.20×** film (shared counter).
- Reported: cfilm vs Lie arm (does adding the magnitude channel cost composition?), cfilm\_lin
  (is full linearity viable here?).

## Stage-8b — content role (conditional diffusion, Stage-6 harness)

Arms: film (adaLN standard), cfilm\_hyb, cfilm\_lin. N=10 seeds, 40k steps, same task/eval as
Stage-6 (DDIM-50 generation MSE vs unique GT on held-out combos). Success here is
**non-inferiority**: the improved FiLM must not lose FiLM's content ability.

- **AC-8b1:** cfilm\_hyb OOD generation MSE ≤ **1.10×** film's.
- **AC-8b2:** cfilm\_hyb in-dist ≤ **1.10×** film's.
- **AC-8b3 (budget, registered honestly):** at the DiT site, film's modulation *apply* is shared
  with the timestep path, making its marginal class cost artificially small (34.6k FLOPs); any
  per-token channel operation costs ~25k alone. Registered ceiling for this suite:
  cfilm class path ≤ **2.0×** film's marginal class path **and** ≤0.1% of backbone forward
  FLOPs. Disclosed here, before the run, with this rationale.
- Reported: cfilm\_lin (expected to fail like the rotation arms; tests whether *linearity of the
  magnitude head* is the content bottleneck, completing the mechanism story).

## Combined verdict

**IMPROVED-FILM CONFIRMED ⇔ Stage-8a CONFIRMED ∧ Stage-8b CONFIRMED (both for cfilm\_hyb).**
Then one operator covers both conditioning roles at FiLM-class cost: the direct FiLM successor.
Any failure = KILL for that role, reported as registered.

## Execution

Chained automatically: Stage-7 exit → 8a → 8b (each fsync+resume, `STAGE*_THROTTLE_MS=60`
unless the 300 W cap is set). Statistics via the shared `verdict.py` primitives.
