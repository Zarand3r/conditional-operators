# Pre-registration: Complex FiLM as the gated arm on the physics task

**Status:** pre-registered 2026-07-29, before the confirmatory run.
**Name:** `pde-cfilm` · **Hardware:** one RTX PRO 6000.

## Disclosure, up front

`cfilm_hyb` ran in `pde-params` as a **reported, not gated** arm, and I have seen those numbers:
$46.6\%$ below FiLM and $65.3\%$ below the best unstructured arm on held-out parameter pairs, with
a fit ratio of $1.030\times$ and a cost of $0.84\times$ FiLM's FLOPs. Every gate criterion would
have passed.

That is not a verdict and it will not be turned into one by rescoring the data I already looked at.
This run uses **fresh seeds 10-19**, never trained before, against the identical protocol. The
criteria below are the shared gate, unchanged since stage 1, with no margin adjusted. What is being
tested is whether the observation replicates on new seeds, not whether it can be written up.

The baselines remain the seeds 0-9 results from `pde-params`. They are fixed, were generated before
any of this, and are not re-run.

## Why this arm, and why it matters

Two registered runs on this task have now failed AC-5 and nothing else. `proposed` fits at
$1.118\times$ against a $1.10\times$ ceiling; `proposed_conj` at $1.1007\times$. The penalty is
structural: a rotation is an isometry, so the operator can turn features but never rescale them,
and in-distribution accuracy pays for it.

Complex FiLM does not have that constraint. It multiplies feature pairs by $m\,e^{i\theta}$: the
phase channel composes exactly, and the magnitude channel — an expressive head — supplies precisely
the rescaling the pure rotation cannot. If the diagnosis is right, this arm should carry the
compositional advantage *without* the fit penalty, which is exactly what the reported numbers
suggested.

It also carries a cost worth stating: the magnitude head is nonlinear in $c$, so composition is
exact on the phase channel only. This arm buys fit by giving up half the exactness guarantee. A
CONFIRMED here is a result about Complex FiLM, not about the group operator, and the paper must say
so.

## Design

Identical to `pde-params` in every respect — same solver, same cached fields, same splits, same
backbone, same optimizer, same 8000 steps — except the conditioning module and the seed range.

- **Gated:** `cfilm_hyb`, seeds 10-19.
- **Reported, not gated:** `proposed_scaled_conj`, seeds 10-19. It posted the best held-out error
  of any arm in either previous run (0.01361) while fitting worst ($1.1542\times$), and whether
  that trade replicates is worth knowing.
- **Baselines:** reused from `pde-params`, seeds 0-9, unchanged.

## Pre-registered criteria

The unchanged `verdict.decide()` on TEST MSE, $N=10$ seeds, one-sided Mann-Whitney $p\le0.01$,
Cliff's $|\delta|\ge0.474$.

- **AC-1:** beats `film` by $\ge20\%$ on held-out parameter pairs.
- **AC-2:** beats the better of `hypernet` and `dynamic_linear` by $\ge20\%$.
- **AC-3:** at most 2 of 10 seeds diverge.
- **AC-5:** in-distribution MSE $\le1.10\times$ the best unstructured arm's.
- **AC-4:** FLOPs $\le1.20\times$ `film`'s, parameters $\le1.05\times$ the smallest unstructured
  arm's. Verified: $0.84\times$ and 42{,}176 against 297{,}856.

**CONFIRMED $\Leftrightarrow$ all five.**

## What the outcomes mean

**CONFIRMED** would be the first confirmed result in this program on a task with standing outside
representation learning, and it would resolve the fit penalty by routing around it rather than
removing it: the compositional advantage survives when the operator is given a magnitude channel,
and the pure isometry is the wrong tool. It would also make Complex FiLM — already confirmed in the
transformation role and non-inferior in the content role — the recommended form of the method in
three independent settings.

**KILL on AC-5** would mean the $1.030\times$ fit was seed luck, and would say the penalty follows
the family rather than the isometry. Given `proposed_conj` looked like a fix on validation and was
not, that outcome is live and I am not discounting it.

## Config

As `pde-params`. Throttled (`PDE_THROTTLE_MS=40`), fsync-per-row with resume.

Amendments after this point must be dated and recorded below.
