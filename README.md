# Structured conditioning mostly doesn't help

Code, data, and pre-registrations for [the paper](docs/paper/paper.pdf).

FiLM conditions a network by scaling each feature. Hypernetworks generate weights from the
condition. Both learn what a condition does by being shown it, and neither learns what two
conditions do together unless the training data happened to contain them together. When it
didn't, they fail quietly: the training loss looks fine and only the unseen combinations come
out wrong.

The obvious fix is to rotate the features instead, with the rotation angles a linear function of
the condition:

```
y = T(c)·x + β(c)      where      T(c) = P·exp(A(Wc))·Pᵀ
```

Rotations add, so `T(c₁+c₂) = T(c₁)T(c₂)` falls out of the parametrization. It holds for any
weights, at initialization, and far outside the training distribution. The network never has to
learn composition because it cannot get it wrong. FiLM and RoPE are two other points in the same
family. I expected this to win.

## It mostly doesn't

Twenty-five experiments. The success criteria for each were written down and committed before it
ran, so the outcome couldn't be argued afterwards. **Fifteen came back negative.**

Both benchmarks where I took the data *and* the task definition from someone else were losses:
plain concatenation beat it on PDEBench, and a hypernetwork beat it on the LA-2A compressor. The
one time I made a prediction *before* running the experiment rather than after, on the
high-dimensional condition space where the theory said structure should matter most, the
advantage came out flat at −1.1%.

Anyone who read only my positive results would reach for this operator and be wrong most of the
time. That's the headline.

## When it does work, and why

The exceptions aren't random, and working out what separates them is what the project actually
produced.

Conditioning multiplies your features by a matrix. Any matrix is a rotation followed by a
stretch, and each mechanism only reaches one part: FiLM stretches and can't turn, a rotation
turns and can't stretch, a hypernetwork does both. So the question isn't how many parameters a
mechanism has, it's which of the two things your task needs.

Two consequences are flat impossibilities rather than tendencies:

**A rotation can never specify content.** Rotations preserve lengths, so they can move information
around but not add or remove any. If your condition is a class label or a caption, you need to
suppress features, and no change of coordinates fixes this because coordinates don't move
eigenvalues. Measured: 8× worse than FiLM, and the ablation fails identically. If your condition
names *what to generate*, use FiLM. This one is worth knowing before you try it.

**Content and composition can't share a channel.** Exact composition forces the
condition-to-modulation map to be linear. Content isn't additive, so it needs a nonlinear map.
You can't have both on one channel, and the smallest thing that carries both is a magnitude times
a phase. That derives **Complex FiLM** rather than proposing it: expressive magnitude for
content, linear phase for composition. It beats FiLM by 36% on unseen combinations at 0.84× its
cost, and wins by 64.6% on a Navier-Stokes surrogate.

Where the theory says structure should pay, it does. The advantage is +0.6% on one condition and
+46.6% on four, and vanishes at weak single conditions, so it's composition doing the work and
not difficulty or a generally better arm.

Where the theory is fitted rather than derived, it breaks. Two short training runs are supposed
to predict whether structure will pay. Both passed on the 16-control audio chain I built
*because* the theory liked it, and the advantage was flat. A check already sitting in this
repository had called that outcome before training: the chain's stages interact, so the combined
effect isn't the composition of the individual effects, and there's nothing for exact composition
to be exact about.

`python -m conditional_operators.suites --list` prints every experiment from the committed
results, verdicts included.

## Running it

```bash
uv venv --python 3.12 .venv                                    # torch needs <3.13
uv pip install --python .venv/bin/python numpy scipy torch matplotlib h5py
.venv/bin/python -m unittest discover -s tests                 # 160 tests
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
  discriminability.py  the screen that is supposed to predict whether structure will pay
  suites.py         experiment name → code → results → verdict, and how to run each one
  gen_tables.py figures.py mechanistic.py   the paper's artifacts
docs/
  paper/            the paper; its tables and figures are generated, never hand-edited
  specs/            the pre-registrations, one per experiment, each written before its run
  THEORY.md         what each mechanism can reach, the two impossibilities, and the proofs
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

This cost me most of the results I wanted, including one I'd predicted publicly, which is the
point of doing it that way. Two experiments *passed* their criteria and are recorded as failures
anyway, because the criteria named the wrong baseline. Deviations are dated amendments inside the
specs. Two errata are written up: a FLOP undercount my own audit caught, which turned a
favourable verdict into an inconclusive one, and a summary table in `THEORY.md` that turned out
not to reproduce from the committed results when I went to fold it into the paper.

## What this doesn't show

Everything here runs at 64×64 with known factors and width-128 models. None of it is validated at
production scale, and the obvious next experiment is the adaLN site of a real text-conditioned
diffusion model.

The wins are all on tasks I designed, which is the weakest joint in the whole thing — geometric
factor changes are exactly what rotations are good at, and I chose them. The two benchmarks I
didn't design were both losses. The maths in the theory is elementary (orthogonal Procrustes,
Cauchy's functional equation, conjugation preserving eigenvalues); its value is that nobody seems
to have written it down for conditioning, not that any step is hard.

## Citation

```bibtex
@misc{cga2026,
  title  = {Structured Conditioning Mostly Does Not Help:
            A Pre-registered Account of the Exceptions},
  author = {Bao, Richard},
  year   = {2026},
  url    = {https://github.com/Zarand3r/conditional-operators}
}
```

Richard Bao, <richardbao419@gmail.com>. MIT licensed. dSprites and 3D Shapes are DeepMind's,
under their own terms.
