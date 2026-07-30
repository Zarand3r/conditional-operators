# Pre-registration: PDEBench 1D reaction-diffusion, FNO backbone

## Outcome: the gate says CONFIRMED. The experiment failed. (2026-07-29)

> Written after the run. Everything below is the pre-registration, unchanged. Numbers in
> `results/pdebench_summary.json`.

`verdict.decide()` returned **CONFIRMED**: AC-1, AC-2, AC-3 and AC-5 all passed, with `proposed`
$31.1\%$ below the best unstructured arm ($p=0.0011$, $\delta=-0.82$).

**It is a gate artifact, and this spec said so before the run.** The registration states that a
gate pass "while losing to concatenation would be worthless and will be reported as such." That is
what happened:

| arm | test MSE | vs `concat_mlp` |
|---|---|---|
| **`concat_mlp`** | **0.25669** | **best in the run** |
| `proposed_scaled_conj` | 0.28269 | $-10.1\%$ |
| `proposed` (the gated arm) | 0.29353 | $-14.4\%$ |
| `cond_layernorm` | 0.32515 | $-26.7\%$ |
| `additive` | 0.34120 | $-32.9\%$ |
| `cfilm_hyb` | 0.35956 | $-40.1\%$ |
| `film` | 0.36458 | $-42.0\%$ |
| `hypernet` | 0.42625 | $-66.1\%$ |
| `dynamic_linear` | 0.44897 | $-74.9\%$ |

**The verdict for the record is a KILL of the experiment's purpose.** On a benchmark we did not
construct, no arm of ours beats the mechanism the field already uses. Complex FiLM, which won the
synthetic physics task, is $40\%$ worse here than plain concatenation.

## Why the gate could return CONFIRMED while we lost

The shared gate was built in stage 1 around one question: does a structured operator beat
*unstructured, high-capacity* conditioners at equal budget? So AC-1 references `film` and AC-2
references the better of `hypernet` and `dynamic_linear`. On every earlier task those were the
strongest baselines and the gate was well aimed.

Here they are the 4th, 8th and 9th best arms of nine. The strongest baseline, `concat_mlp`, is
**referenced by no criterion at all** — it has been present in every sweep since stage 4 as a
reported arm, and no criterion was ever pointed at it. The gate answered its own question
correctly and that question stopped being the right one.

This is a flaw in the criteria, not in this run, and it is recorded rather than repaired
retroactively. Earlier verdicts are not revisited: on the tasks where `concat_mlp` was not the
strongest baseline, the gate's reference was appropriate and its verdicts stand. Any future
registration should reference **whichever baseline is strongest on that task**, named in advance.


**Status:** pre-registered 2026-07-29, before the decision run.
**Name:** `pdebench-reacdiff` · **Hardware:** one RTX PRO 6000.

## Disclosure

A one-seed validation screen has already been run and **points strongly against the method**:
against `concat_mlp`, the baseline PDE surrogates actually use, `proposed` is $133\%$ worse and
`cfilm_hyb` is $201\%$ worse. The test split has not been read.

This registration exists so the negative is recorded through the same gate as the positives, not
because there is any expectation of a win. Declining to run the gate once a screen turns against us
is exactly how a body of work quietly becomes one-sided. Every margin below is the shared gate,
unchanged.

## Why this experiment

Everything positive in this program was measured on data we generated. The physics result
(`pde-cfilm`, CONFIRMED, $+64.6\%$) used a solver we wrote, our own parameter grid, and our own
splits. PDEBench is a published benchmark; its 1D reaction-diffusion set ships one file per
physical setting across a $4\times4$ grid of $(\nu,\rho)$, which is a compositional structure we did
not choose.

$$u_t = \nu u_{xx} + \rho\,u(1-u)$$

**A caveat on the earlier result that this experiment prompted.** Our synthetic Navier-Stokes
operator is $\exp(TL(c))$ with $L=-(\nu k^2+\alpha)-i v_x k_x$, and the method's composition law
holds exactly when $L$ is affine in the condition. Checked directly: **advection is affine — exactly
our form — while viscosity and drag are not** (their steps are multiplicative, $\times16$ and
$\times4$, which places them exponentially in the condition), and forcing never enters $L$ at all.
So one of four axes matched the inductive bias by construction. That is a partial alignment, not a
rigged task, but it belongs in the record and it is a reason to want a benchmark we did not design.

## Task and data

The first 512 trajectories of each grid cell, read over HTTP Range directly from the remote HDF5
(212 MB per cell instead of 4.1 GB; the `tensor` dataset is contiguous). Predict $u$ at the final
stored timestep from $u$ at $t=0$, conditioned on the physical parameters, with one shared
normalisation across all cells.

The condition is $(\log\nu/\log 5,\ \log\rho/\log 10)$ as deviations from the $(\nu{=}1,\rho{=}1)$
cell, so the baseline is the zero vector and single-parameter cells have one nonzero entry.

## Splits (fixed in source before any data was downloaded)

- **TRAIN (7 cells):** every cell where at most one parameter leaves the baseline.
- **VAL (4 cells):** $(0.5,2)$, $(2,5)$, $(5,10)$, $(0.5,10)$.
- **TEST (5 cells, read once):** $(0.5,5)$, $(2,2)$, $(2,10)$, $(5,2)$, $(5,5)$.

## Backbone

FNO: 4 spectral blocks, 16 modes, width 128, with the conditioning arm applied channel-wise after
each block — where FiLM and adaLN inject in these models. Identical for every arm; only the
conditioning module differs.

## Instrument checks already passed

- **Conditioning is load-bearing.** A model that ignores the condition entirely reaches 0.424 on
  validation against `concat_mlp`'s 0.029 — conditioning is worth $14.6\times$, so the comparison
  measures something real.
- **The task separates arms.** Validation spread across arms is $2$–$12\times$, so a null here would
  not be an instrument failure.

## Pre-registered criteria

Unchanged `verdict.decide()` on TEST MSE, $N=10$ seeds, one-sided Mann-Whitney $p\le0.01$, Cliff's
$|\delta|\ge0.474$.

- **AC-1:** `proposed` beats `film` by $\ge20\%$.
- **AC-2:** `proposed` beats the better of `hypernet` and `dynamic_linear` by $\ge20\%$.
- **AC-3:** at most 2 of 10 seeds diverge.
- **AC-5:** `proposed` in-distribution MSE $\le1.10\times$ the best unstructured arm's.
- **AC-4:** budget, from the shared counter.

Reported, not gated: `additive`, `cfilm_hyb`, `proposed_scaled_conj`.

**Note on AC-5's reference.** On an FNO backbone `hypernet` fits by memorising — a $2815\times$
train/validation gap in the screen, against `cfilm_hyb`'s $601\times$ — so "best unstructured
in-distribution error" is a degenerate reference and penalises arms that do not overfit. The
criterion is left **unchanged** rather than adjusted mid-programme. The in-distribution ratio
against `film` is reported alongside so a reader can apply either, and neither is relabelled the
gate afterwards.

## What the outcomes mean

**KILL** is the expected result and would say the method's advantage does not transfer to a
benchmark we did not construct. Given that `pde-cfilm` confirmed on our own solver, the pair would
bound the earlier claim sharply: it holds on a task whose parameters partly match the operator's
form, and fails on real physics where the reaction term is nonlinear in a parameter.

**CONFIRMED** would overturn the screen and be the strongest result in the programme.

## Config

Adam 1e-3, cosine schedule, gradient clip 1.0, batch 32, 8000 steps, $N=10$ seeds. 512 fixed
evaluation samples per split from a deterministic generator, identical across arms. fsync-per-row
with resume. Throttled (`PDEBENCH_THROTTLE_MS=40`).

Amendments after this point must be dated and recorded below.
