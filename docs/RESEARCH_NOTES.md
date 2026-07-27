# Research Notes — novelty review & re-scoping

> **Historical document.** This is the original prior-art review, written before any experiment ran. It is
> kept for provenance and is **not** a description of what the project concluded. Where it
> disagrees with [`paper/paper.pdf`](paper/paper.pdf), the paper is current. In particular: the
> staged plan below was revised (the proposed CIFAR-100 stage was dropped as unable to test
> compositionality), and the operator family ranked most promising here is not the one that won.
> See [`RESEARCH_LOG.md`](RESEARCH_LOG.md) for what actually happened.

*Produced with the `research-ideation` discipline (ground → Scoop-Check → Pursue/Refine/Kill).
Read this before investing: it names the nearest prior work and the defensible contribution, so
we don't reinvent OFT/LoRA under new words ("smart plagiarism").*

## Verdict: **Refine → Pursue.** Promising, but the claim must be narrowed.

High feasibility (Stage 1 is cheap and solo-doable) and high upside (FiLM is everywhere, so a
drop-in better conditioner is widely useful). The risk is **novelty crowding**: the operator
*families* are largely taken by the PEFT literature. The idea is worth pursuing **only** with the
re-scoped claim below.

## Scoop-Check — the four axes

| Axis | As written | Nearest prior work | Ruling |
|---|---|---|---|
| **Problem framing** | condition features with a richer-than-diagonal operator | FiLM family (multi-hop/GNN/temporal FiLM, EquiFiLM) | ADJACENT |
| **Core mechanism** | block-diag / orthogonal / orth+low-rank / Lie / flow operators | **OFT** (block-diag orthogonal), **BOFT** (butterfly orthogonal), **HRA** (orthogonal+low-rank Householder), **LoRA** (low-rank), RoPE/GRAPE (Lie) | **COLLISION** — each family exists as a PEFT method |
| **Key insight** | conditioning = structured transform of latent geometry with identity-init + bounded spectrum | OFT/BOFT already use identity-init + orthogonality for bounded spectrum | ADJACENT/COLLISION |
| **Application domain** | conditional *generation / compositional reasoning*, **per-sample, activation-space** | OFT/BOFT/HRA/LoRA are **weight fine-tuning** (a fixed transform); HyperLoRA/TC-LoRA generate low-rank adapters from a condition but still modify weights | **CLEAR** (this is the opening) |

### Nearest neighbors, named
- **OFT** — Orthogonal Fine-Tuning ([arXiv:2311.06243](https://arxiv.org/pdf/2311.06243)): learnable *block-diagonal orthogonal* weight transform. = Families 1+2.
- **BOFT** — Butterfly Orthogonal FT (same line): structured orthogonal interpolating identity↔full orthogonal. = Family 1/2 done efficiently.
- **HRA** — Householder Reflection Adaptation ([arXiv:2405.17484](https://arxiv.org/pdf/2405.17484)): *bridges low-rank and orthogonal adaptation*. = **Family 3** ("most promising") already exists.
- **Group-and-Shuffle** ([arXiv:2406.10019](https://arxiv.org/pdf/2406.10019)): structured orthogonal parametrization.
- **He et al., Unified View of PEFT** ([arXiv:2110.04366](https://arxiv.org/pdf/2110.04366)): already unifies adapters/prefix/LoRA. = **Q5's "unify everything" is largely done.**
- **HyperLoRA / TC-LoRA / Text-to-LoRA**: hypernetwork-generated, input-conditioned low-rank adapters.

## The re-scoped, defensible claim

The whole PEFT stack above learns a **fixed** structured transform of the **weights** for a task.
**FiLM is categorically different**: a **per-sample, activation-space** transform **generated from a
conditioning input `c`**, used for **conditional generation / compositional reasoning**. That
combination — *structured* operators in *FiLM's role* — is the gap.

> **Claim:** There exists a minimal structured operator family (candidate: input-conditioned
> **block-orthogonal + low-rank**, $T(c)=Q(c)+U(c)V(c)^\top$) that, used as a *per-sample activation
> conditioner* (a strict FiLM generalization), beats FiLM on **compositional out-of-distribution
> generalization** in conditional generation, at **<20% overhead**, with **identity-init** and
> **bounded spectrum** for stability — and whose learned rotations correspond to **interpretable
> subspace transformations of semantic concepts**.

What to explicitly change from the proposal:
1. **Drop the grand "theory of conditional computation" unification** (Q5) — scooped by He et al.
   Keep at most a one-paragraph positioning, not a claimed contribution.
2. **Position every family against its PEFT twin** ("OFT/BOFT/HRA adapt *weights*; we condition
   *activations per-sample*"). Make this the abstract's first move.
3. **Lead with two things reviewers can't dismiss:** (a) the Stage-1 synthetic *compositional*
   benchmark that isolates conditioning, and (b) the **mechanistic-interpretability** result
   (concepts ↔ subspace rotations) — the fresh angle the PEFT papers skip.

## Falsifiable experiment — cheapest kill-test first

1. **Stage 1 only, ~days:** on the synthetic analytic-transformation benchmark, at *equal
   parameters*, does input-conditioned $Q(c)+U(c)V(c)^\top$ (activation modulation) beat FiLM on
   **compositional OOD** with a statistically significant margin and <20% FLOP overhead?
   - **Falsified / KILL** if it does *not* beat FiLM even on synthetic compositional transforms —
     stop for the cost of an engineer-week. (Diagonal-is-enough = Outcome A.)
   - **Confirmed** → proceed to CIFAR-100 (Stage 2), then one DiT/adaLN swap (Stage 3).
2. Register the success margin and overhead threshold **before** running, so a soft win can't be
   laundered into a Pursue.

## Competitive-landscape watch (scoop risk is real)
OFT/BOFT/HRA + hypernetwork-LoRA groups are active. Differentiate on **conditioning + generation +
compositionality + mechanistic interp**, and move fast on Stage 1. The `auto-research` skill is a
good fit for the Stage-1 sweep (one metric — compositional OOD error — on a fixed synthetic harness).

## Fit to prior discussion
Note the natural tie-in: **Family 3 / Lie operators** are exactly the conditioning primitive a
world-model's *action-conditioned latent dynamics* would use (condition the transition on the action
via a structured operator). If Stage 1–3 land, the world-model conditioning application is a strong,
motivated follow-up — not a claimed contribution here.
