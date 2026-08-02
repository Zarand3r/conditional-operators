# Pre-registration: does exact multiplicative aggregation beat additive, and where?

## Outcome: KILL, and the two halves separate cleanly (2026-08-02)

> Written after the run. Everything below is the pre-registration, unchanged. Numbers in
> `results/composition_summary.json`.

AC-1 required the advantage to scale in **both** `n` and `s`. It scales in `n` and not in `s`.

```
advantage of proposed over the best additive arm
          s=0.25    s=0.5     s=1.0     s=2.0
n=1       -0.8%     -1.4%     +0.6%     -9.9%
n=2       -0.8%     -1.3%    +25.5%    -12.4%
n=3       -0.7%     -1.5%    +38.7%    -14.2%
n=4       -0.8%     -1.7%    +46.6%    -14.8%
```

**The composition claim is confirmed, in exactly the predicted shape.** At the trained magnitude a
single condition buys $+0.6\%$ — nothing, as required — and the advantage grows monotonically as
conditions stack: $+25.5\%$, $+38.7\%$, $+46.6\%$ ($p=9.1\times10^{-5}$, $\delta=-1.00$). AC-2
passed independently, with a $1.6\%$ spread at `n=1, s=0.25` where the first-order expansion is
tight. A criterion we could have failed by winning, and did not.

**The magnitude claim is refuted, and the derivation was wrong.** The `½‖ΣA‖²` argument predicted
monotone growth in strength. Rotations are **periodic**: past the trained angle the operator wraps,
so the error is oscillatory rather than growing, and the advantage reverses. This is the same
non-monotonicity the E0 guidance screen measured (fidelity `0.139 → 1.979 → … → 0.800` across
`α = 1…8`), and it was not carried into the derivation. Theorem 2's bound is still correct — error
is at most linear in `‖c‖` — but it is loose, and loose in the direction that matters here.

**What the pair establishes.** The advantage comes from *composing conditions*, not from strength,
not from difficulty, and not from task selection: it is absent at one condition, absent at weak
conditions, and grows with the number of factors at fixed magnitude. That is a mechanism, not a
correlation. It also bounds the practical claim sharply — the benefit lives at the trained
conditioning scale and disappears outside it, so exact composition is not a strength knob.

**Consequence for §7 of `THEORY.md`:** the strength prediction must be withdrawn and replaced with
the periodicity caveat. The count prediction stands.


**Status:** pre-registered 2026-08-02, before the decision run.
**Name:** `composition-order` · **Hardware:** one RTX PRO 6000.

## The hypothesis, and why it is worth a run

Cross-attention aggregates condition effects through a value sum, so composing conditions is
**additive**: `I + ΣᵢAᵢ`. A group operator composes **multiplicatively**: `exp(ΣᵢAᵢ)`. These agree to
first order and diverge as

```
exp(ΣA) − (I + ΣA)  =  ½‖ΣA‖² + O(‖ΣA‖³)
```

so the discrepancy grows **quadratically in the total conditioning magnitude**, roughly `n·s` for
`n` conditions of strength `s`. The prediction is therefore *differentiated*: no benefit at one weak
condition, growing benefit as conditions are stacked or strengthened.

## What is already measured, and what is not

**Measured here.** Trained conditioners learn additive composition without being asked to.
Extracting the effective operator `J(c) = ∂/∂z[arm(c,z)]` by autograd from arms trained on dSprites
and comparing `J(c₁+c₂)` against the two candidates:

| arm | ‖J₁₂ − (J₁+J₂−I)‖ | ‖J₁₂ − J₁J₂‖ | effect size |
|---|---|---|---|
| `film` | **7.50** | 16.81 | 9.80 |
| `hypernet` | **11.97** | 37.21 | 12.03 |
| `concat_mlp` | 1.07 | 1.01 | 1.50 |

FiLM and the hypernetwork both compose additively; concatenation fits neither, consistent with an
unstructured nonlinear map. This is the hypothesis's premise, and it did not have to come out this
way.

**Also measured, retrodictively.** The structured advantage widens from two-factor to three-factor
combinations in all nine comparisons across three suites (`+41.9% → +48.0%` against FiLM on
dSprites; `+87.5% → +91.4%` against the hypernetwork on 3D Shapes). Consistent with the quadratic
term, but confounded: triples are simply harder, and this is data we had already seen.

**Not measured.** Whether the advantage scales as the argument requires — with *strength* at fixed
`n`, which difficulty alone does not explain — and whether it holds for **attention**, the mechanism
the hypothesis is actually about.

## Design

Two axes swept independently, which is what separates the prediction from "harder is harder":

- **`n`**, simultaneously changed factors: 1, 2, 3, 4.
- **`s`**, per-factor magnitude: a scalar on the condition vector, over {0.25, 0.5, 1.0, 2.0}. At
  fixed `n`, difficulty rises with `s` for every arm, but only the additive arms should suffer the
  `‖ΣA‖²` term.

Arms, matched on the shared budget counter, identical backbone, stage-4 dSprites harness:

- `film` — additive by the measurement above.
- `xattn` — **new**: conditions as separate tokens into a cross-attention block, aggregated by value
  sum. The mechanism the hypothesis is about; no arm in this repository has tested it.
- `xattn_linear` — the same without softmax, making token aggregation *exactly* additive. Isolates
  normalisation from aggregation.
- `proposed` — multiplicative by construction.

## Pre-registered criteria

`N=10` seeds, one-sided Mann-Whitney `p ≤ 0.01`, Cliff's `|δ| ≥ 0.474`, on held-out combinations.
**The strongest baseline is named in advance, as the PDEBench artifacts require: whichever of
`film`, `xattn`, `xattn_linear` has the lowest error in that `(n,s)` cell.**

- **AC-1 (the scaling claim, and the one that matters):** the relative advantage of `proposed` over
  the best additive arm **increases monotonically in `s`** at fixed `n=2`, and in `n` at fixed
  `s=1.0`. A win that does not scale refutes the mechanism even if it is large.
- **AC-2 (no benefit where the theory says none):** at `n=1, s=0.25` all arms fall within 10% of one
  another. If `proposed` wins there too, the advantage is not the composition term and the
  explanation is wrong.
- **AC-3:** at most 2 of 10 seeds diverge.
- **AC-4:** budget, from the shared counter.

**CONFIRMED ⇔ AC-1 ∧ AC-2 ∧ AC-3 ∧ AC-4.** AC-2 is a criterion we can fail *by winning*, which is
deliberate: the hypothesis predicts a shape, not a direction.

## What each outcome means

**CONFIRMED:** exact multiplicative aggregation buys something over the additive aggregation
attention performs, the benefit scales as the second-order term predicts, and it vanishes where the
expansion is tight. A claim about attention-style conditioning with a mechanism rather than a
correlation.

**KILL on AC-1:** the advantage does not scale, so whatever produces it is not the quadratic term,
and §7 of `THEORY.md` is wrong.

**KILL on AC-2:** `proposed` wins where composition cannot be the cause, pointing at task selection
or a confound rather than the mechanism.

## Scope, stated before the result

This is dSprites: four factors, `dc=4`, an enumerable condition space. By our own analysis that is
the low-dimensional regime where structure has least to offer, and a positive result here does
**not** transfer to text-conditioned models with hundreds of condition dimensions. It tests the
*mechanism*, not the application. The application claim needs a high-dimensional parametric
condition space, which we have not found.

## Config

Stage-4 harness unchanged: Adam 1e-3, batch 256, 12,000 steps, `N=10` seeds, identical backbone and
data for every arm; fsync-per-row logging with resume; throttled per the standing power rule.

Amendments after this point must be dated and recorded below.
