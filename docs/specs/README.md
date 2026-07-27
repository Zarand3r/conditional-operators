# Pre-registrations

One document per experiment, written and committed before that experiment ran. Each one fixes
the success margin, the statistical test, the number of seeds, and the data splits in advance,
so there was nothing left to negotiate once the numbers came in.

Each spec now opens with an outcome section, written afterwards, sitting above the original text.
Nothing below that section was edited after the run.

Confirmed means every criterion passed; kill means at least one didn't; unfair means the budget
rule was broken, so the comparison settles nothing.

| Spec | Suite | Question | Verdict |
|---|---|---|---|
| [STAGE1_SPEC.md](STAGE1_SPEC.md) | **S1** | Does structure beat unstructured conditioning at equal budget? | confirmed |
| [STAGE3_SPEC.md](STAGE3_SPEC.md) | **S2** | Does exact composition survive a hidden basis, within budget? | confirmed |
| [STAGE4_SPEC.md](STAGE4_SPEC.md) | **S3** | Does the advantage survive a learned latent on real images? | confirmed |
| [STAGE4_SPEC.md](STAGE4_SPEC.md) | **S3b** | Does it survive a categorical (non-group) factor? | confirmed |
| [STAGE4_SPEC.md](STAGE4_SPEC.md) | **S4** | Does it hold on a second dataset (3D Shapes, RGB)? | kill |
| [STAGE6_SPEC.md](STAGE6_SPEC.md) | **S5** | Can rotation conditioning specify image content (diffusion)? | kill |
| [STAGE7_SPEC.md](STAGE7_SPEC.md) | **S6** | Does exact composition prevent world-model rollout drift? | kill |
| [STAGE10_SPEC.md](STAGE10_SPEC.md) | **S6'** | Does a latent-consistency loss unlock rollout guarantees? | kill |
| [STAGE8_SPEC.md](STAGE8_SPEC.md) | **S7a** | Complex FiLM in the transformation role | confirmed |
| [STAGE8_SPEC.md](STAGE8_SPEC.md) | **S7b** | Complex FiLM in the content role (non-inferiority) | confirmed |
| [STAGE9_SPEC.md](STAGE9_SPEC.md) | **S8** | Does condition powering beat classifier-free guidance? | confirmed |
| [STAGE2_SPEC.md](STAGE2_SPEC.md) | **aux-erratum** | Dense-basis control; failed the FLOP ceiling (erratum in the paper) | unfair |
| [STAGE11_SPEC.md](STAGE11_SPEC.md) | **aux-contraction** | Is a fixed contraction rate the rollout-stability knob? | kill |

Where a spec had to change, the change is a dated amendment inside it. One erratum — a FLOP
undercount that turned a favourable verdict into an inconclusive one — is written up in
[`../RESEARCH_LOG.md`](../RESEARCH_LOG.md).

`python -m conditional_operators.suites --list` prints the same table straight from the results.
