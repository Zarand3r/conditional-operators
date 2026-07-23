# Research Log — Conditioning as Group Action program

Append-only program history. One entry per stage: what was registered, what ran, the verdict,
the numbers that matter, and the decision it drove. Detailed per-stage reports live in
`docs/RESULTS*.md`; specs in `docs/specs/`; raw logs in `results/`.

## Completed

| Stage | Question | Verdict | Key numbers | Decision it drove |
|---|---|---|---|---|
| S1 (aligned synthetic) | Does structure beat unstructured at equal budget? | **CONFIRMED** | 61.5% vs best hypernet, 0.87× FiLM cost | Proceed |
| S2-dense (de-aligned, dense P) | Survives hidden basis? | **UNFAIR (erratum)** | true cost 1.52× FiLM | Structured bases required |
| S2 (Lie + GS basis) | Exact composition within budget? | **CONFIRMED** | 3.3e-8 OOD; triples flat; angles to 2e-4 rad | The mechanism is linearity |
| S3 (dSprites deltas) | Survives learned latents? | **CONFIRMED** | 53.8% margin; gap 1.25× vs 2.1–3.3× | Real-image evidence |
| S3b (+categorical) | Survives non-group factors? | **CONFIRMED** | 42.9%; gap 1.63× | Attenuates, holds |
| S4 (3D Shapes) | Second dataset? | **KILL (AC-5 fit)** | best OOD (87.5%) but 1.46× in-dist | Orthogonality prices fit |
| S5 (DiT content) | Drop-in adaLN replacement? | **KILL** | 7× underfit; ablation fails identically | Rotation ≠ content channel |
| S6 (rollout world model) | Composition prevents drift? | **KILL** | all arms ~0.043 @ h20; contractive arm best | Consistency, not composition, limits rollouts |

Papers: `docs/paper/paper.pdf` (long), `docs/paper/paper_short.pdf` (CGA, GRAPE-style). Both on
`main` (github.com/Zarand3r/conditional-operators), plain-language register, mechanical style
gate clean.

## Running / queued (autonomous chain; sequential GPU, throttled, fsync+resume)

| Order | Stage | Question | Spec | Status |
|---|---|---|---|---|
| now | 8a | Complex FiLM wins the transform role? | STAGE8_SPEC | running |
| next | 8b | Complex FiLM keeps FiLM's content role? | STAGE8_SPEC | queued (chained) |
| 3 | 10 | Consistency loss unlocks operator guarantees in rollouts? | STAGE10_SPEC | queued (chained) |
| 4 | 11 | Contraction rate is the rollout-stability knob? | STAGE11_SPEC | queued (chained) |
| 5 | 9 | Guidance as group power beats CFG scaling? | STAGE9_SPEC | spec'd; implement after 8b verdict (arm-selection rule pre-fixed) |

Loop protocol: each completion notification → fold verdict here + into papers → regenerate
tables/figures → recompile → commit → push. Verdicts are whatever the registered gates say.

## Standing rules

- Pre-register before running; margins never move after data exists; deviations = dated
  amendments in the spec.
- Power safety: throttle (STAGE*_THROTTLE_MS=60) or `sudo nvidia-smi -pl 300`; PSU trips at
  sustained ~600 W.
- Every number in papers regenerates from `results/*.json`; never hand-edit tables/figures.
