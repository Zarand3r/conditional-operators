# Pre-registration: does relaxing the isometry fix the fit criterion?

**Status:** pre-registered 2026-07-29, before the confirmatory run and before its test split is read.
**Name:** `pde-conj` · **Hardware:** one RTX PRO 6000.

## What this follows from

`pde-params` was a KILL on AC-5 alone, by 1.8 percentage points: in-distribution fit $1.118\times$
against a $1.10\times$ ceiling, with the compositional criteria passing by $57.5\%$ at
$\delta=-1.00$. The diagnosis written into that spec *before* the run was that the operator is a
strict isometry — it can turn features but never rescale them.

Two relaxations were then compared **on validation only**, with the test split neither read nor
stored, because a 1.8-point gap is exactly the size selection pressure invents:

| arm | in-dist | val | val vs `proposed` | fit ratio (AC-5) |
|---|---|---|---|---|
| `proposed` | 0.00352 | 0.01461 | — | $1.074\times$ |
| `proposed_scaled` | 0.00376 | 0.01487 | $-1.8\%$ | $1.148\times$ |
| `proposed_conj` | 0.00315 | 0.01468 | $-0.5\%$ | $0.962\times$ |
| `proposed_scaled_conj` | 0.00299 | 0.01447 | $+0.9\%$ | $0.912\times$ |

**The result contradicts the primary hypothesis.** Adding the scaling generator — making the
algebra $\mathbb{C}^*$ rather than $SO(2)$, which was the change I expected to matter — made the
fit *worse* ($1.148\times$). Conjugating the basis by a learned positive diagonal, the change I
listed second, removed the penalty outright: $0.962\times$, better than the best unstructured arm,
with compositional performance unchanged to within $1\%$.

So the fit penalty was not a shortage of magnitude freedom in the operator. It was the orthogonal
basis forcing the rotation to act in the wrong metric.

## Arm selection rule, stated before the run

`proposed_conj` is the **gated** arm. It is not the best validation number — `proposed_scaled_conj`
fits slightly better — and the reason for choosing it is stated rather than discovered afterwards:

> A rotation is bounded by nature, so `proposed_conj` composes exactly for any weights whatever.
> The scaled arms need a clamp on $e^s$, and a clamp is nonlinear, so composition holds only while
> $|s|\le4$ — measured error is $\sim10^{-6}$ inside the clamp and order $1$ once it engages.
> Exactness-by-construction is this project's central claim, and an arm that keeps it
> unconditionally is worth more than one point of fit.

`proposed_scaled_conj` is reported, not gated.

## Design

Identical in every respect to `pde-params`: same solver, same cached fields, same splits, same
backbone, same optimizer, same 8000 steps, same 10 seeds. Only the conditioning module changes.
The six baseline arms are **not re-run** — their 10-seed results already exist under an identical
protocol, and re-running them would only add noise.

## Pre-registered criteria

The **unchanged** `verdict.decide()`, with `proposed_conj` in the gated slot. Margins identical to
`pde-params`, which were identical to stages 4, 5 and 7. Test read exactly once.

- **AC-1:** beats `film` by $\ge20\%$ on held-out parameter pairs.
- **AC-2:** beats the better of `hypernet` and `dynamic_linear` by $\ge20\%$.
- **AC-3:** at most 2 of 10 seeds diverge.
- **AC-5:** in-distribution MSE $\le1.10\times$ the best unstructured arm's.
- **AC-4:** FLOPs $\le1.20\times$ `film`'s, parameters $\le1.05\times$ the smallest unstructured
  arm's, from the shared counter. Verified at $1.11\times$ and 44{,}288 parameters.

**CONFIRMED $\Leftrightarrow$ all five.** Any failure is a KILL.

## What the outcomes mean

**CONFIRMED** would say the fit penalty — which has shadowed every experiment in this program, at
$1.46\times$, $2.43\times$, $1.33\times$ and $1.118\times$ — was an artifact of an unnecessary
design choice, not a cost of imposing exact composition. Orthogonality was adopted for numerical
stability and never required by the group law. That would remove the standing objection to the
method and make the $57.5\%$ compositional win at $1.11\times$ FiLM cost a clean result rather than
a qualified one.

**KILL** would mean the validation improvement did not survive a held-out split, which at these
margins is entirely possible, and would be reported as such.

A caution recorded now: `proposed`'s own fit ratio was $1.118\times$ over 10 seeds and
$1.074\times$ over the 5 seeds used here, so seed noise on this quantity is comparable to the
margin being tested. The confirmatory run uses all 10.

## Config

As `pde-params`. Throttled (`PDE_THROTTLE_MS=40`). fsync-per-row with resume.

Amendments after this point must be dated and recorded below.
