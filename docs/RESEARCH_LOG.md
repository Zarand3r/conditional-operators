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
| 8a (Complex FiLM, transform role) | Does magnitude x phase keep the transform wins? | **CONFIRMED** | cfilm_hyb 36.1% over film; fit parity; 0.84x film cost | Half of the improved-FiLM claim |
| 8b (Complex FiLM, content role) | Does the magnitude channel restore content ability? | **CONFIRMED** | cfilm_hyb 0.97x film (gate <=1.10x); cfilm_lin fails (0.19): expressive magnitude is necessary | **IMPROVED-FILM CONFIRMED**: one operator, both roles |
| 10 (consistency latents) | Does a consistency loss unlock rollout guarantees? | **KILL (AC-10.2)** | AC-10.1 hit huge: h20 flat at 0.0039, 90% below hypernet (growth 2.0x vs 19.8x); pairs margin (AC-10.2) washed out to 1% | Rollout recipe = isometry + consistency loss; gate reported as registered |
| 11 (contraction sweep) | Is a fixed contraction rate the rollout knob? | **KILL** | h20 flat at the 0.043 plateau for ALL eps; larger eps only hurts h10/in-dist | Correction must be adaptive (consistency loss / learned contraction), not a scalar |
| 9 (guidance as group power) | Does condition powering beat CFG extrapolation? | **CONFIRMED** | parity at strength 1 (0.96x); growth 23.7x vs CFG 50.7x (p=1.6e-4); 2-5x lower at all strengths >1; no second pass | Guidance lives in the conditioning path when conditioning is a group action |

Papers: `docs/paper/paper.pdf` (long), `docs/paper/paper_short.pdf` (CGA, GRAPE-style). Both on
`main` (github.com/Zarand3r/conditional-operators), plain-language register, mechanical style
gate clean.

## Running / queued (autonomous chain; sequential GPU, throttled, fsync+resume)

| Order | Stage | Question | Spec | Status |
|---|---|---|---|---|
| — | — | **Registered queue complete: 13 gates (7 confirmed, 5 killed as registered, 1 unfair/erratum).** | | |

Loop protocol: each completion notification → fold verdict here + into papers → regenerate
tables/figures → recompile → commit → push. Verdicts are whatever the registered gates say.

## Standing rules

- Pre-register before running; margins never move after data exists; deviations = dated
  amendments in the spec.
- Power safety: throttle (STAGE*_THROTTLE_MS=60) or `sudo nvidia-smi -pl 300`; PSU trips at
  sustained ~600 W.
- Every number in papers regenerates from `results/*.json`; never hand-edit tables/figures.

- **2026-07-23 incident:** `git checkout` between branches while a sweep held its results-log fd
  replaced the file on disk; the writer kept appending to the unlinked inode. Recovered 43/50 rows via
  `/proc/<pid>/fd`; final 7 rows preserved in `results/stage8a_stdout.log` + `stage8a_summary.json` aggregates. New rule: **no branch switching while any sweep is running**; merges to main
  wait for chain-idle windows.

## Novelty sweep (2026-07-24, three adversarial agents, ~35 works reviewed)

**No fatal collision.** Complex FiLM and guidance-as-group-power are unclaimed; the crowded
surfaces are rotation-conditioning (Worrall 2017, CLG-VAE, Attributes-as-Operators, the
2025-26 RoPE-generalization wave) and world models, where the components are prior art
(HAE/Keurti 2023: rotation transitions + latent prediction loss + long stable rollouts;
Quessard 2020; TD-MPC consistency losses; bilinear Koopman folklore). Claims narrowed
accordingly in both papers: we own the combination, the exactness-by-construction contract, the
consistency-vs-composition decomposition, the contraction null, and the pre-registered
budget-parity protocol — never the operators or the loss. CFG distortion is established fact
(Bradley & Nakkiran; APG): Stage-9 must frame it as isolated, not discovered. ~11 citations
added to both papers.
