# Pre-registration: 2D Navier-Stokes surrogate, conditioned on physical parameters

**Status:** pre-registered 2026-07-28, before the decision run.
**Name:** `pde-params` · **Hardware:** one RTX PRO 6000.

## Disclosure

The first screening run read the **test** split by mistake (the harness passed `test` where it
should have passed `val`). It was fixed and re-screened on validation; the numbers below are the
validation ones. The earlier read is disclosed rather than hidden, and the test split has been read
exactly once since, by nothing. No margin here was chosen after seeing any number: every criterion
is the shared gate `verdict.decide()`, unchanged since stage 1.

## Why this task

Every failure this project has recorded violated one of four conditions the boundary map now
states. Camera had no compositional gap. `latent-edit` froze the representation. The content suite
asked an orthogonal operator to inject information. This task satisfies all four without strain:

- **Conditions form a group.** Physical parameters are continuous, and the deltas compose.
- **The representation is co-trained.** The surrogate is trained from scratch.
- **A real compositional gap exists.** Measured at $4.29\times$ on validation, twice dSprites'
  $2.1\times$.
- **Content comes from somewhere else.** The initial vorticity field carries the content; the
  parameters only modulate how it evolves. This is the division of labour where the operator is
  strong, and it sidesteps the $8\times$ content failure entirely.

The dynamics are nonlinear, which matters for fairness. If the parameter dependence were exactly an
exponential of something linear in the parameters, CGA's inductive bias would match the truth by
construction and the comparison would be rigged. Advection of vorticity by its own induced velocity
is not of that form.

## Task

$$\omega_t + (u\cdot\nabla)\omega = \nu\nabla^2\omega - \alpha\omega + f,\qquad \nabla^2\psi=-\omega$$

on a periodic $2\pi$ box at $64\times64$, solved pseudo-spectrally with 2/3 dealiasing and an
integrating-factor RK4 so the linear part is exact. The solver is validated against facts
independent of its own correctness: pure diffusion matches the analytic $e^{-\nu k^2 t}$ to
$2.4\times10^{-4}$; inviscid enstrophy drifts $8\times10^{-7}$ over 300 steps; enstrophy decay is
monotone in viscosity; pure advection reproduces an exact one-cell roll to $4.8\times10^{-6}$.

A sample is $(\omega_0, \Delta, \omega_T)$: an initial field, a parameter-delta vector, and the
field after $T=2.0$ under those parameters. **Every parameter setting is evolved from the same set
of initial fields**, so the only thing that differs between conditions is the physics.

Four conditioning axes, each moving one parameter by one step in one direction: viscosity
($\times16$), drag ($\times4$), forcing amplitude ($\pm0.5$), background advection ($\pm0.15$).

The step sizes were calibrated **on the physics alone, before any model was trained**: each axis
moves the evolved field by 32-46% in relative norm, so no single parameter dominates the condition
and the compositional structure is genuinely four-dimensional. An earlier setting left advection at
105% against viscosity's 3%, which would have made the task effectively one-dimensional.

## Splits

Six parameter pairs, partitioned and disjoint.

- **TRAIN:** all eight single-axis deltas, plus pairs {viscosity+drag} and {forcing+advection}.
- **VAL:** {viscosity+forcing} and {drag+advection}. Screening and any calibration read this only.
- **TEST (read once, for the decision):** {viscosity+advection} and {drag+forcing}.

## Arms

Identical backbone (the stage-4 convolutional encoder/decoder, unchanged), optimizer, schedule,
seeds and data for every arm; only the conditioning module differs, applied to the latent.

Gated: `film`, `concat_mlp`, `cond_layernorm`, `hypernet`, `dynamic_linear`, `proposed`.
Reported, not gated: `additive`, `cfilm_hyb`, `proposed_mlp_gs`.

## Pre-registered criteria

Scored by the **unchanged** `verdict.decide()` on TEST MSE, $N=10$ seeds, one-sided Mann-Whitney
$p\le0.01$, Cliff's $|\delta|\ge0.474$. Identical to stages 4, 5 and 7.

- **AC-1:** `proposed` beats `film` by $\ge20\%$ on held-out parameter pairs.
- **AC-2:** `proposed` beats the better of `hypernet` and `dynamic_linear` by $\ge20\%$.
- **AC-3:** at most 2 of 10 seeds diverge.
- **AC-5:** `proposed` in-distribution MSE $\le1.10\times$ the best unstructured arm's.
- **AC-4 (budget):** `proposed` FLOPs $\le1.20\times$ `film`'s, parameters $\le1.05\times$ the
  smallest unstructured arm's, from the shared counter.

**CONFIRMED $\Leftrightarrow$ all of the above.** Any failure is a KILL and will be reported as one.

## What each outcome means

**AC-5 is where I expect this to die, and saying so now is the point of writing it down.** The fit
penalty has appeared in every setting we have measured: $1.46\times$ on 3D Shapes, $2.43\times$ on
frozen latents, $33\%$ in diffusion. It is structural — the operator is a strict isometry, so it
can rotate features but never rescale them. A KILL on AC-5 with AC-1 and AC-2 passing would mean
the same story a fourth time, and would make the case for the scaled-rotation variant rather than
against the method.

A CONFIRMED here would be the first result on a task with genuine scientific standing rather than a
disentanglement benchmark built for studying factors.

## Config

Adam 1e-3, batch 128, 8000 steps, $N=10$ seeds. 128 initial fields shared across all parameter
settings; 1024 fixed evaluation samples per split from a deterministic generator, identical across
arms. fsync-per-row logging with resume. Throttled (`PDE_THROTTLE_MS=40`) per the standing power
rule.

Amendments after this point must be dated and recorded below.
