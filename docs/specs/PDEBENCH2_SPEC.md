# Pre-registration: a contained operator on PDEBench

**Status:** pre-registered 2026-07-29, before the decision run and before its test split is read.
**Name:** `pdebench-split` · **Hardware:** one RTX PRO 6000.

## Disclosure

A one-seed validation screen has been run and **favours** this arm: `split_cga` is $21.4\%$ below
`concat_mlp` where `proposed` is $133\%$ worse. Validation is what selection is for; the test split
has not been read for these arms and will be read once.

The previous follow-up on the synthetic task, `pde-conj`, looked like a fix on validation
($0.962\times$) and failed on held-out data ($1.1007\times$). That is the standing reason to
distrust a validation improvement of this size until it is confirmed.

## What this follows from

Every loss in this programme is predicted by one number. In-distribution fit ratio separates the
seven completed comparisons with no overlap: $\le1.005\times$ won three of three, $\ge1.118\times$
lost four of four.

The cause is **dimensional, not geometric**. The operator's reachable family is
$d_c$-dimensional, because the algebra coordinates are a linear image of the condition. On this
benchmark $d_c=2$, so the entire space of transformations `proposed` can express is
two-dimensional, while `concat_mlp` builds arbitrary nonlinear interactions. That is why both
geometric relaxations failed: `proposed_scaled` and `proposed_conj` change the *shape* of a family
that is the wrong *size*.

So the fix is not another relaxation. It is to stop replacing the expressive path and contain it.
`split_cga` gives half the channels to a concat-MLP and half to the operator, on the bet that a
conditional map needs some directions fitted freely and others composed exactly.

## Arm selection, by the registered budget rule rather than by judgment

Two hybrids were screened. `hybrid_concat` scored better ($+30.8\%$) but costs $2.10\times$ FiLM's
FLOPs, which **violates AC-4** — the same disqualification that made stage 2's dense basis UNFAIR.
It is therefore reported, not gated, with its cost stated. `split_cga` costs $1.00\times$ FiLM's
FLOPs and $0.60\times$ `concat_mlp`'s and is the only eligible arm. The budget rule made this
choice, not the validation numbers.

## What is given up

Exact composition. A group action plus an MLP is not a group action, so `split_cga` has the
guarantee on half its channels and none on the other half. This is a deliberate trade: the
guarantee has not won a benchmark in nineteen gates, and this experiment is aimed at one. A
CONFIRMED here is a result about a hybrid conditioner, **not** about conditioning as a group action,
and the paper must say so in those words.

## Design

Identical to `pdebench-reacdiff`: same cells, same splits, same FNO backbone, same optimizer, same
8000 steps. Only the conditioning module and the seed range differ.

- **Gated:** `split_cga`, seeds 10-19 (fresh).
- **Reported, not gated:** `hybrid_concat`, seeds 10-19.
- **Baselines:** reused from `pdebench-reacdiff`, seeds 0-9, unchanged.

## Pre-registered criteria

Unchanged `verdict.decide()` on TEST MSE, $N=10$ seeds, one-sided Mann-Whitney $p\le0.01$, Cliff's
$|\delta|\ge0.474$.

- **AC-1:** beats `film` by $\ge20\%$ on held-out cells.
- **AC-2:** beats the better of `hypernet` and `dynamic_linear` by $\ge20\%$.
- **AC-3:** at most 2 of 10 seeds diverge.
- **AC-5:** in-distribution MSE $\le1.10\times$ the best unstructured arm's.
- **AC-4:** FLOPs $\le1.20\times$ `film`'s, parameters $\le1.05\times$ the smallest unstructured
  arm's. Verified: $1.00\times$ and 49{,}984 against 297{,}856.

**CONFIRMED $\Leftrightarrow$ all five.**

**Additionally reported, and the number that matters for the claim:** margin against
`concat_mlp`, the mechanism parameter-conditioned surrogates actually use. The gate does not
reference it, because the shared gate has never referenced it, but a win over `hypernet` while
losing to concatenation would be worthless and will be reported as such.

## What the outcomes mean

**CONFIRMED, with a positive margin over `concat_mlp`,** would be the first result in this
programme that improves on the standard practice of a field, on data we did not generate, with an
architecture we did not choose, at no extra cost. That is the outcome worth having.

**CONFIRMED on the gate but losing to `concat_mlp`** would be a gate artifact and will be reported
as a failure of the experiment's purpose regardless of what `decide()` returns.

**KILL** would mean the validation margin did not survive, as it did not for `pde-conj`, and that
containing the operator does not rescue it either.

## Config

As `pdebench-reacdiff`. Throttled, fsync-per-row with resume.

Amendments after this point must be dated and recorded below.
