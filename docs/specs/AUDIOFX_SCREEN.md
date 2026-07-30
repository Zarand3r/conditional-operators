# Screen: parametric audio effects (not pre-registered — stopped at the screen)

**Run 2026-07-30.** Validation only, one seed, no test split read, no verdict claimed. Numbers in
`results/audiofx_screen.json`.

## Why it was tried

The boundary map made this the best remaining candidate. Audio effect controls are continuous and
parametric; gains in dB compose by addition, so exact composition is the *semantics of the control*
rather than an approximation of it; the audio signal supplies all the content, which sidesteps the
content failure that killed the diffusion suite; and the model trains from scratch, which sidesteps
the frozen-representation reversal. Every previous application violated at least one of those.

## The rigging check, run before anything was trained

A linear gain is additive in dB and filter cascades multiply, so both are exactly
`exp(linear in the condition)` — our operator's form. A task built from those alone would be won by
construction, which is what happened on the synthetic Navier-Stokes task. So the chain mixed two
aligned stages with two that cannot be aligned (saturation is pointwise nonlinear; compression
makes gain depend on the signal's own envelope).

    gain+cutoff 0.252   gain+drive 0.666   gain+threshold 0.287
    cutoff+drive 0.285  cutoff+threshold 0.120   drive+threshold 0.273

**0 of 6 pairs are exactly our form.** The nonlinear stages contaminate the linear ones downstream,
so nothing was aligned by construction. The trap was closed.

## Result: NOT WORTH A SWEEP, and not for the expected reason

| check | value | |
|---|---|---|
| conditioning share | $77.9\%$ | pass |
| **fit ratio, `proposed`** | **$1.000\times$** | **pass** |
| fit ratio, `cfilm_hyb` | $1.038\times$ | pass |
| **compositional gap** | **$1.13\times$** | **fail** (want $\ge1.50$) |

**This is the first task in the programme where the operator's inductive bias matches.**
`proposed` was the best-fitting arm of five and edged `concat_mlp` by $1.7\%$ on validation. The
bias-match check, which predicted every previous outcome, says go.

It fails the other check instead: held-out parameter pairs are barely harder than trained ones, so
there is nothing to generalise to. That is the camera-slider failure mode, and it confirms the two
checks are genuinely independent — a task can pass either while failing the other, and both are
necessary.

## Why calibration cannot rescue it

Enlarging the steps raises train-pair and test-pair interaction together, so the ratio does not move:

    steps                train pairs      test pairs
    current              0.252, 0.273     0.287, 0.285
    2x drive/threshold   0.252, 0.303     0.311, 0.300
    2x all               0.318, 0.303     0.322, 0.373

A gap would require test pairs to interact far more strongly than train pairs. The only way to
arrange that is to assign splits *after* measuring which pairs interact most, which is selecting the
task to fit the method. Not done, and recorded as the reason for stopping rather than as an
option left open.

## Cost

About ten minutes of GPU, against the month a full pre-registration and sweep would have taken. The
screen did its job.
