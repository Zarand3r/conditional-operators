# Conditioning as Group Action

Code and data for the paper **[Conditioning as Group Action: Exact Compositional Conditioning at
FiLM Cost](docs/paper/paper.pdf)**.

Networks are conditioned by scaling features (FiLM) or by generating weights (hypernetworks).
Both learn what each condition does. Neither learns what two conditions do *together* unless
training data shows them together. We condition by **rotating** features, with rotation angles a
linear function of the condition. Rotations add, so combined conditions compose automatically:

```
y = T(c)·x + β(c)      with   T(c) = P·exp(A(Wc))·Pᵀ      ⟹      T(c₁+c₂) = T(c₁)·T(c₂)
```

The identity holds for *any* weights, so composition is a property of the parametrization rather
than a behavior learned from data. FiLM and RoPE are special cases of the same family.

## Results at a glance

Eleven pre-registered experiments; success criteria, statistical tests, and data splits were
committed to version control **before** each run, and every verdict is reported as the gate
scored it, including the negatives.

| | Question | Verdict |
|---|---|---|
| **S1–S3b** | Does structure beat unstructured conditioning on unseen condition combinations, at equal budget? | **confirmed** (up to 5 orders of magnitude; 42.9–53.8% on real images) |
| **S4** | Does it hold on a second image dataset? | **kill** — best OOD numbers, but a 1.46× in-distribution fit penalty trips our no-regression gate |
| **S5** | Can rotation conditioning specify image *content*? | **kill** — 7× worse; a rotation moves information and cannot add it |
| **S6 / S6′** | Does exact composition prevent world-model rollout drift? | **kill**, then the mechanism: with a latent-consistency loss, rollout error goes *flat* 7× beyond the training horizon |
| **S7** | **Complex FiLM**: magnitude for content, phase for composition | **confirmed** — beats FiLM 36% on unseen combinations, matches it on content (0.97×), at FiLM cost |
| **S8** | Guidance by powering the condition instead of classifier-free extrapolation | **confirmed** — 2–5× less distortion at every strength, with no second forward pass |

`python -m conditional_operators.suites --list` prints this table from the committed results.

## Quickstart

```bash
uv venv --python 3.12 .venv                                    # torch needs <3.13
uv pip install --python .venv/bin/python numpy scipy torch matplotlib h5py
.venv/bin/python -m unittest discover -s tests                 # 70 tests, seconds, no GPU
.venv/bin/python -m conditional_operators.suites --list        # suites and their verdicts
.venv/bin/python -m conditional_operators.suites S1 --run      # ~15 min, CPU
```

Image suites need the datasets and a GPU:

```bash
curl -L -o datasets/dsprites.npz https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
curl -L -o datasets/3dshapes.h5  https://storage.googleapis.com/3d-shapes/3dshapes.h5
.venv/bin/python -m conditional_operators.suites S3 --run      # ~4 h on an RTX 6000
```

Sweeps are crash-resilient: each run is fsync'd to an append-only log and re-running resumes
where it stopped. `STAGE*_THROTTLE_MS=60` inserts a duty cycle if your power supply objects to
sustained full-tilt draw.

Rebuild the paper's tables and figures from the committed results, then the PDF:

```bash
.venv/bin/python -m conditional_operators.gen_tables
.venv/bin/python -m conditional_operators.figures
cd docs/paper && tectonic paper.tex
```

## Layout

```
conditional_operators/
  verdict.py        the pre-registered gate: acceptance criteria, Mann-Whitney U, Cliff's δ
  data.py arms.py   synthetic benchmark; the six conditioning arms + shared param/FLOP counter
  train.py sweep.py suite S1
  stage2..stage11   suites S2-S8 and controls (see suites.py for the paper mapping)
  suites.py         paper label → code → results → verdict; the replication entry point
  gen_tables.py figures.py mechanistic.py   paper artifacts, generated from results/
docs/
  paper/            the paper, its figures and tables (all generated, never hand-edited)
  specs/            one pre-registration per suite, written before the run
  RESEARCH_LOG.md   program history: every verdict, every amendment, the novelty sweep
  PROPOSAL.md RESEARCH_NOTES.md   the original proposal and its prior-art review
results/            append-only run logs and summary JSONs (the source of every number)
tests/              operator invariants, gate logic, data splits, registry integrity
```

## How the protocol works

Each suite has a specification under `docs/specs/` fixing the success margin, the statistical
test, the seed count, and the data splits before any decision run. `verdict.py` then scores the
result mechanically: no post-hoc criteria, no dropped seeds, no moved goalposts. A shared counter
enforces budget parity (our conditioning path stays within 1.20× FiLM's FLOPs while baselines run
up to 48× our parameter count). Test splits are read once, after all selection on validation.

The discipline cost us three times, which is the point: S4, S5, and S6 are negatives, and S6
falsified our own headline prediction. Where we deviated, the specs carry dated amendments, and
one accounting erratum (a FLOP undercount our own audit caught, which downgraded a favorable
verdict) is documented in `RESEARCH_LOG.md`.

## Scope

Every result comes from controlled 64×64 benchmarks with known factors and width-128 models.
Nothing here is validated at production scale. The method applies to conditions that are
*combinable changes*; for conditions that specify content, our own registered negative says keep
FiLM, and Complex FiLM is the candidate that covers both roles at our evidence scale.

## Citation

```bibtex
@misc{cga2026,
  title  = {Conditioning as Group Action: Exact Compositional Conditioning at FiLM Cost},
  author = {Bao, Richard},
  year   = {2026},
  note   = {Code and pre-registrations: https://github.com/Zarand3r/conditional-operators}
}
```

MIT licensed. Datasets (dSprites, 3D Shapes) are distributed by DeepMind under their own terms.
