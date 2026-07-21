# Stage-4 Results — Real-Image Compositional Conditional Transformation (dSprites)

*Auto-generated from `results/stage4_summary.json` by `render_stage4.py`.*
Criteria pre-registered in [`specs/STAGE4_SPEC.md`](specs/STAGE4_SPEC.md) (2026-07-21, before any decision run; pre-run amendments documented there, including the BCE loss amendment and the smoke-read hygiene disclosure). Task: encode a real dSprites image, apply the conditioning operator to the **learned** latent for a factor-change condition Δ, decode, score pixel-MSE against the deterministic ground-truth image. OOD = **never-trained two-factor change types**; triples = never-trained three-factor types (diagnostic). Hardware: RTX PRO 6000 Blackwell.
Config: 10 seeds, 12000 steps, BCE training loss, pixel-MSE gated metric.

## Verdict: **CONFIRMED**

> CONFIRMED — the Lie conditioning advantage transfers to real images on a LEARNED latent, at the smallest parameter count and 1.11× FiLM cost.

- gate: AC-1..AC-5 pass; proposed beats hypernet by 53.8% (p=9.1e-05, delta=-1.00)

## Per-arm results (pixel-MSE)

| Arm | in-dist | OOD pairs (mean±sd) | OOD/in-dist | triples | params | FLOPs/FiLM |
|---|---|---|---|---|---|---|
| film | 0.001461 | 0.003044 ± 0.000170 | 2.08× | 0.004031 | 50,176 | 1.00× |
| concat_mlp | 0.001662 | 0.002886 ± 0.000108 | 1.74× | 0.003880 | 83,072 | 1.66× |
| cond_layernorm | 0.001438 | 0.003208 ± 0.000228 | 2.23× | 0.004304 | 50,176 | 1.01× |
| hypernet | 0.001431 | 0.003832 ± 0.000274 | 2.68× | 0.005379 | 2,147,200 | 43.17× |
| dynamic_linear | 0.001526 | 0.005083 ± 0.000922 | 3.33× | 0.006903 | 297,856 | 5.98× |
| **proposed** | 0.001411 | 0.001769 ± 0.000062 | 1.25× | 0.002095 | 44,160 | 1.11× |
| proposed_mlp_gs *(ablation, over budget)* | 0.001449 | 0.002746 ± 0.000204 | 1.89× | 0.003569 | 52,416 | 1.27× |

**Gate:** margin 53.8% vs `hypernet` · p=9.1e-05 · Cliff's δ=-1 · all criteria {'AC-1': True, 'AC-2': True, 'AC-3': True, 'AC-5': True}

## Reading — what the columns show

- **Every arm fits in-distribution equally** (~0.0014): the backbone is identical; all differences are compositional generalization, isolated by construction.
- **The OOD/in-dist ratio is the story**: proposed **1.25×** (near-systematic recombination) vs FiLM 2.08×, hypernet 2.68×, dynamic_linear 3.33×. More conditioning capacity made recombination *worse* — the capacity-vs-inductive-bias tradeoff, now on real images.
- **The mechanism ablation holds on real images**: same GS-P, more FLOPs, but an MLP angle head instead of linear-in-the-algebra → 1.55× worse OOD. Linearity, not the orthogonal basis, carries the win.
- **Triples degrade gracefully** for the Lie arm (1.48× its in-dist) and steeply for everything else — the length-extrapolation signature from Stage-3 survives, attenuated, on a learned latent.

## Honest scope

- Errors here are ~0.0018, not Stage-3's 1e-8: a learned conv latent does **not** support exact group action — the advantage is a robust ~2× on unseen combinations, not the orders-of-magnitude of the synthetic stages. Both facts are the finding.
- dSprites is simple, single-sprite, binary. Escalations that remain: natural images, text/class conditioning in a DiT (adaLN swap), and categorical (non-group) factors like shape changes.
