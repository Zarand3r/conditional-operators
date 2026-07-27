# Conditioning as Group Action

Code, data, and pre-registrations for [the paper](docs/paper/paper.pdf).

FiLM conditions a network by scaling each feature. Hypernetworks generate weights from the
condition. Both learn what a condition does by being shown it, and neither learns what two
conditions do together unless the training data happened to contain them together. When it
didn't, they fail quietly: the training loss looks fine and only the unseen combinations come
out wrong.

The idea here is to rotate the features instead, with the rotation angles a linear function of
the condition:

```
y = T(c)·x + β(c)      where      T(c) = P·exp(A(Wc))·Pᵀ
```

Rotations add, so `T(c₁+c₂) = T(c₁)T(c₂)` falls out of the parametrization. It holds for any
weights, at initialization, and far outside the training distribution — the network never has to
learn composition, because it cannot get it wrong. FiLM and RoPE turn out to be two other points
in the same family.

## What happened

Eleven experiments. The success criteria for each were written down and committed before it ran,
so the outcome couldn't be argued afterwards. Seven passed. Four didn't, and the four that
didn't are the more useful half.

When the condition is a *change* — a pose offset, an attribute delta, something you would
naturally combine with another change — rotating wins comfortably. On synthetic tasks where the
conditions form a group exactly, error on unseen combinations is five orders of magnitude below
the best baseline, and flat as combinations get longer than anything trained. On dSprites and 3D
Shapes, working through a learned latent it doesn't control, it still cuts error on unseen
combinations by 43–88%, using fewer parameters than any baseline and about 1.1× FiLM's compute.
A hypernetwork with 43× the conditioning compute does worse than plain FiLM on the same task,
which was not what I expected going in.

Then it stops working, in three places worth knowing about:

**On 3D Shapes it wins the thing it was built to win and loses anyway.** Lowest error on unseen
combinations by 87.5%, but it fits the *training* combinations 1.46× worse than the best
baseline, and the criteria I'd registered forbid buying generalization that way. Scored as a
failure.

**It cannot say what to generate.** Asked to condition a diffusion model on the image content
itself, rotation conditioning is 7× worse than FiLM. In hindsight this is obvious — a rotation
moves information around and cannot add any — but it took a failed experiment to see it. If your
condition is a class label or a caption, use FiLM.

**It doesn't fix world-model rollouts.** Exact composition should have stopped error compounding
over long rollouts. It didn't; every conditioner drifts at the same rate, which killed my own
headline prediction. The reason turned out to be more interesting than the prediction: rollout
error is dominated by the latent drifting off the decodable manifold, not by the transition
composing badly. Add a latent-consistency loss and the picture inverts — rollout error goes flat
7× past the training horizon while every baseline still drifts.

Those failures pointed at the fix. **Complex FiLM** treats each feature pair as a complex number:
an expressive magnitude carries content, a linear phase carries composition. It beats FiLM by 36%
on unseen combinations and matches it on content generation, at FiLM's cost. The same phase
channel gives guidance for free — scaling the condition is exactly raising the operator to a
power — which distorts 2–5× less than classifier-free guidance at every strength and needs no
second forward pass.

`python -m conditional_operators.suites --list` prints all of this from the committed results,
verdicts included.

## Running it

```bash
uv venv --python 3.12 .venv                                    # torch needs <3.13
uv pip install --python .venv/bin/python numpy scipy torch matplotlib h5py
.venv/bin/python -m unittest discover -s tests                 # 83 tests, seconds, no GPU
.venv/bin/python -m conditional_operators.suites --list        # every experiment and how it scored
.venv/bin/python -m conditional_operators.suites aligned --run      # ~15 min on CPU
```

The image suites want a GPU and the two datasets:

```bash
curl -L -o datasets/dsprites.npz https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
curl -L -o datasets/3dshapes.h5  https://storage.googleapis.com/3d-shapes/3dshapes.h5
.venv/bin/python -m conditional_operators.suites dsprites --run      # ~4 h on an RTX 6000
```

Sweeps survive interruption: every run is fsync'd to an append-only log, and restarting picks up
where it stopped. If your power supply objects to hours at full draw, `STAGE*_THROTTLE_MS=60`
adds a duty cycle. (Mine did. That is why the flag exists.)

Everything in the paper regenerates from the committed results:

```bash
.venv/bin/python -m conditional_operators.gen_tables
.venv/bin/python -m conditional_operators.figures
cd docs/paper && tectonic paper.tex
```

## Where things live

```
conditional_operators/
  verdict.py        the gate: acceptance criteria, Mann-Whitney U, Cliff's δ
  data.py arms.py   synthetic benchmark; the conditioning arms and the shared cost counter
  train.py sweep.py the aligned synthetic experiment
  stage2..stage11   the remaining experiments and controls
  suites.py         experiment name → code → results → verdict, and how to run each one
  gen_tables.py figures.py mechanistic.py   the paper's artifacts
docs/
  paper/            the paper; its tables and figures are generated, never hand-edited
  specs/            the pre-registrations, one per experiment, each written before its run
  RESEARCH_LOG.md   what happened and when, including the amendments and the prior-art sweep
  PROPOSAL.md RESEARCH_NOTES.md   where the idea started, kept for provenance
results/            the raw run logs and summaries every number comes from
tests/              operator invariants, gate logic, data splits, registry
```

## About the protocol

Every experiment has a spec in `docs/specs/` fixing the margin, the test, the seed count, and the
splits before the decision run, and `verdict.py` scores the result mechanically afterwards. No
criteria invented after the fact, no dropped seeds, no moved goalposts. A shared counter keeps
the conditioning path within 1.20× FiLM's FLOPs while baselines are allowed up to 48× the
parameters. Test splits get read once, after all the tuning is done on validation.

This cost me three results I wanted and one I predicted publicly, which is the point of doing it
that way. Deviations are recorded as dated amendments inside the specs. One erratum — a FLOP
undercount my own audit caught, which turned a favourable verdict into an inconclusive one — is
written up in `RESEARCH_LOG.md`.

## What this doesn't show

Everything here runs at 64×64 with known factors and width-128 models. None of it is validated at
production scale, and the obvious next experiment is the adaLN site of a real text-conditioned
diffusion model. The method suits conditions that are combinable changes; for conditions that
name content, my own results say keep FiLM, and Complex FiLM is the candidate that handles both
at the scale I could test.

## Citation

```bibtex
@misc{cga2026,
  title  = {Conditioning as Group Action: Exact Compositional Conditioning at FiLM Cost},
  author = {Bao, Richard},
  year   = {2026},
  email  = {richardbao419@gmail.com},
  url    = {https://github.com/Zarand3r/conditional-operators}
}
```

Richard Bao, <richardbao419@gmail.com>. MIT licensed. dSprites and 3D Shapes are DeepMind's,
under their own terms.
