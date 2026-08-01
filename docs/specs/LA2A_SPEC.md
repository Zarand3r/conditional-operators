# Pre-registration: Complex FiLM against FiLM on the SignalTrain LA-2A recordings

**Status:** pre-registered 2026-08-01, before the decision run.
**Name:** `la2a-cfilm` · **Hardware:** one RTX PRO 6000.

## Disclosure

A three-seed validation screen has been run and **points against the method**: `cfilm` is $14.6\%$
worse than `film`, and the worst of four arms. The test split has not been read.

This registration exists so the negative is recorded through a gate rather than left as a screen we
happened to lose. Every margin is the shared protocol, unchanged.

## Why this experiment

This is the last application candidate the boundary map admits, and the only one where the target
was not invented by us. Black-box audio effect modelling already conditions a TCN on a device's
knobs using FiLM, and "which conditioning mechanism" is already a published question in that field
(DAFx 2024; SMC 2024; PyNeuralFx). So the data is theirs (66 pairs of 20-minute recordings from a
Teletronix LA-2A), the architecture is theirs (causal dilated TCN, 10 blocks, 32 channels, 278 ms
receptive field), the incumbent is theirs (FiLM), and the metric is theirs (error-to-signal ratio).

Our claim is narrow and pre-existing: **Complex FiLM beats FiLM in the transformation role at lower
cost.** It has held on three earlier gates. This asks whether it holds on someone else's benchmark.

## Scope, stated before the result

The LA-2A exposes one binary switch and one continuous peak-reduction knob, and SignalTrain's split
separates by **audio content**, not by knob setting. So this measures interpolation across settings,
not generalisation to unseen *combinations* — which is the axis the $36\%$ came from. A loss here
does not refute the compositional claim; it refutes the claim that Complex FiLM is a general
drop-in improvement on FiLM. Those are different, and the write-up must not blur them.

## Design

Cache of 64 fixed random windows of 32{,}768 samples from each recording, so every arm sees
byte-identical data. Identical backbone, optimizer, schedule and seeds for every arm; only the
conditioning module differs, applied inside every residual block, which is where this field puts it.

Arms and their conditioning budgets at 32 channels:

| arm | cond params | vs `film` | vs backbone |
|---|---|---|---|
| `film` (incumbent) | 32{,}640 | $1.00\times$ | $0.23\times$ |
| **`cfilm`** (candidate) | **27{,}680** | **$0.85\times$** | $0.19\times$ |
| `concat` | 42{,}880 | $1.31\times$ | $0.30\times$ |
| `hyper` | 349{,}440 | $10.71\times$ | $2.43\times$ |

`hyper` is reported, not gated: at this width it is larger than the model it conditions, so any win
by it would not be budget-fair. It is included because the published literature reports
hypernetworks beating FiLM here, and that is worth checking on identical data.

## Pre-registered criteria

$N=10$ seeds, one-sided Mann-Whitney $p\le0.01$, Cliff's $|\delta|\ge0.474$, on TEST error-to-signal
ratio. The strongest baseline is named in advance, as the PDEBench artifacts require:

- **AC-1:** `cfilm` beats `film` — the incumbent and the arm our claim names.
- **AC-2:** `cfilm` beats **the best of `film`, `concat` and `hyper`**, whichever that turns out to
  be. This is the criterion the earlier gates lacked, and its absence produced two false CONFIRMEDs
  on PDEBench.
- **AC-3:** at most 2 of 10 seeds diverge.
- **AC-4:** `cfilm` conditioning parameters $\le$ `film`'s. Verified: $0.85\times$.

**CONFIRMED $\Leftrightarrow$ AC-1 ∧ AC-2 ∧ AC-3 ∧ AC-4.** A margin below $20\%$ that is
nonetheless significant will be reported as a small effect, not upgraded.

## What the outcomes mean

**KILL** is expected. It would say the "drop-in FiLM upgrade" claim does not survive contact with a
real device on someone else's benchmark, leaving the method with no demonstrated application at
all. Given that this was the best-fitting candidate the boundary map allowed, that is close to a
terminal result for the applied line, and should be written as one.

**CONFIRMED** would be the first result in this programme to improve on a field's incumbent, on
their data, at lower cost.

## Config

Adam 3e-3, cosine schedule, gradient clip 1.0, batch 16, 4000 steps, $N=10$ seeds. ESR computed
with the first 12{,}277 samples discarded, since a causal model has not filled its receptive field.
Throttled (`NEURALFX_THROTTLE_MS=50`); unthrottled this backbone draws a sustained 585 W against a
supply that trips at 600 W.

Amendments after this point must be dated and recorded below.
