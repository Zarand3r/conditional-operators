# Spec: Stage-11 — Contraction as a Design Principle

## Outcome (added after the run)

> This section was written after the experiment. Everything below it is the
> pre-registration, unchanged from before the decision run.

- **aux-contraction** (Is a fixed contraction rate the rollout-stability knob?) → **kill**: at least one criterion failed, so the hypothesis is rejected for this setting. Numbers in `results/stage11_summary.json`.

---

**Status:** Pre-registered 2026-07-23, before any run. Queued after Stage-10.

## Hypothesis

Stage-7's best long-horizon arm was the mildly contractive one. Claim: spectral radius, not
expressivity, controls rollout degradation. A small deliberate contraction re-projects latents
toward the decodable manifold each step; too much contraction destroys information. Error at
horizon should be U-shaped in the contraction rate, with the optimum strictly inside.

## Design (Stage-7 harness; one knob)

Arms are five copies of the Lie operator scaled by $(1-\varepsilon)$:
$\T(a) = (1-\varepsilon)\,R(Wa)$ with
$\varepsilon \in \{0, 0.003, 0.01, 0.03, 0.1\}$, all else identical (no consistency loss; this
suite isolates contraction). Stage-7 training, splits, eval, N=10 seeds, budget unchanged.

## Pre-registered acceptance criteria

- **AC-11.1 (contraction helps):** some $\varepsilon^\ast > 0$ beats $\varepsilon = 0$ at
  horizon 20 by **≥30%** (MWU p≤0.01, |δ|≥0.474).
- **AC-11.2 (interior optimum):** the best $\varepsilon$ at horizon 20 is neither endpoint of
  the grid.
- **AC-11.3 (the trade is real):** the best-at-h20 $\varepsilon$ is worse than $\varepsilon=0$
  on in-dist single steps (contraction costs fidelity; report the exchange rate).
- Deliverable regardless of verdict: the error-vs-$\varepsilon$ curve at h3/h10/h20 (the design
  chart practitioners would use).

**CONFIRMED ⇔ AC-11.1 ∧ AC-11.2.** AC-11.3 is reported, not gated.
