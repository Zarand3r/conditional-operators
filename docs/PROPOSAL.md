# Beyond FiLM: Structured Conditional Operators
## A Research Proposal for Generalizing Feature-wise Linear Modulation

**Status:** Research Proposal
**Area:** Neural Network Architecture, Conditional Computation, Diffusion Models, Multimodal Learning
**Risk:** Medium · **Potential Impact:** High

> **Read `RESEARCH_NOTES.md` first.** It records the prior-art collision check and the
> re-scoped, defensible claim. As written below the operator *families* overlap heavily with
> the PEFT literature (OFT/BOFT/HRA/LoRA); the contribution must be framed as *input-conditioned
> activation modulation* (FiLM's role), not weight fine-tuning.

---

## Executive summary

Feature-wise Linear Modulation (FiLM) is one of the most influential conditioning mechanisms in
deep learning — diffusion models, multimodal transformers, VLMs, RL — yet it performs an extremely
restrictive operation:

$$y = \gamma(c)\odot x + \beta(c)$$

where the conditioning variable $c$ only **scales and shifts each feature independently**.

This proposal argues **FiLM is not the fundamental object** — it is the *simplest member* of a much
larger family of conditional operators acting on feature space. The central question:

> *Can we identify the smallest mathematically richer family of conditional operators that
> significantly improves compositional reasoning and conditional generation while preserving
> FiLM's optimization stability and computational efficiency?*

Rather than replacing FiLM with unrestricted hypernetworks, we propose **structured operator
families** with desirable mathematical invariants.

---

## Motivation

Conditioning is everywhere (diffusion, VLMs, CLIP adapters, robotics, RL, conditional normalization,
world models, PEFT, hypernetworks), yet essentially all methods fall into three buckets:

1. **Concatenation** — $[x\,;\,c]$
2. **Feature-wise affine (FiLM)** — $y = \gamma(c)\odot x + \beta(c)$
3. **Hypernetworks** — $W(c)$

FiLM is efficient but restrictive; hypernetworks are expressive but expensive and unstable. **The
continuum between them is under-explored.**

---

## Core hypothesis

> Conditioning should apply **structured transformations to latent geometry** — modifying feature
> *interactions*, *subspaces*, *orientations*, and *symmetries* — rather than independently scaling
> coordinates, while preserving useful optimization properties.

---

## Mathematical view

FiLM is $T(c) = D(c) = \mathrm{diag}(\gamma(c))$ — merely a diagonal matrix. A diagonal matrix
cannot rotate, shear, exchange information, or couple features; every feature behaves independently.

**Proposed generalization:** replace $D(c)$ with a structured operator $T(c)$:

$$y = T(c)\,x + \beta(c)$$

The research question becomes: *what family should $T(c)$ belong to?*

---

## Candidate operator families

- **Family 1 — Block diagonal.** Independent small groups (e.g. $2\times2$ blocks). Cheap, local
  feature mixing, GPU-friendly.
- **Family 2 — Orthogonal.** $T(c)^\top T(c) = I$. Norm/information preserving, stable gradients;
  learned rotations.
- **Family 3 — Orthogonal + low rank.** $T(c) = Q(c) + U(c)V(c)^\top$ ($Q$ orthogonal, $UV^\top$
  low rank): a rotation plus a small task-specific deformation. *Likely the most promising.*
- **Family 4 — Lie group.** $T(c) = \exp(A(c))$ with $A(c)$ in a Lie algebra: invertible, continuous,
  compositional. Connects to Lie-group positional encodings (RoPE) and GRAPE.
- **Family 5 — Flow.** Integrate $\dfrac{dx}{dt} = f(x, c)$ for several steps — conditioning as a
  learned dynamical system.

---

## Desired properties

- **Identity initialization:** $T(c) = I$ at init, so optimization starts identically to the
  unconditioned model.
- **Bounded spectrum:** largest singular value $\approx 1$ (no exploding activations).
- **Compositionality:** ideally $T(c_2 \circ c_1) = T(c_2)\,T(c_1)$, so conditioning signals compose.
- **Low cost:** target overhead $<20\%$ vs FiLM.
- **Parallelizable:** no sequential dependence.

---

## Research questions

- **Q1.** Does cross-feature conditioning beat independent feature scaling?
- **Q2.** Which family gives the best compute–performance tradeoff?
- **Q3.** Do structured operators improve *compositional generalization*? (Train `red`, `circle`;
  test `red circle`.)
- **Q4.** Can structured operators replace cross-attention in some settings?
- **Q5.** Can one family unify FiLM / conditional LayerNorm / LoRA / adapters / hypernetworks?
  *(See `RESEARCH_NOTES.md` — a unified PEFT view already exists; scope this narrowly.)*

---

## Experimental roadmap

**Stage 1 — Synthetic transformation benchmark (the cheap kill-test).** Isolate conditioning itself.
Input $x \in \mathbb{R}^{128}$; the condition requests an analytic transformation (rotate, permute,
shear, scale, combine); ground truth generated analytically. Baselines: concatenation, FiLM,
conditional LayerNorm, hypernetwork, dynamic linear layer. Metrics: test error, OOD composition,
training stability, FLOPs, memory. **Success:** at equal parameters, the structured operator shows a
statistically significant improvement on compositional OOD tasks.

**Stage 2 — Image classification.** CIFAR-100, class-embedding condition; FiLM vs proposed operator;
measure accuracy, convergence, calibration.

**Stage 3 — Conditional diffusion.** Replace FiLM (adaLN) inside a DiT or SD U-Net; class- and
text-conditional generation; metrics: FID, CLIP score, compositional prompts, sampling stability.

**Stage 4 — Multimodal.** CLIP / LLaVA; replace feature modulation with structured operators.

**Ablations.** Diagonal → block-diagonal → orthogonal → orthogonal+low-rank → Lie group → flow, all
at identical parameter budgets.

---

## Mechanistic interpretability (the differentiator)

Measure singular values, operator spectrum/entropy, effective rank, composition error. Visualize
$T(c)$ across thousands of conditions. **Do semantic concepts correspond to rotations?** Do colors
rotate one subspace and object identities another? This mechanistic story is what most PEFT papers
do *not* do.

---

## Expected outcomes

- **A.** Diagonal FiLM is sufficient — negative result, still valuable.
- **B.** Small structured operators consistently beat FiLM — strong publication.
- **C.** Different domains need different families — a hierarchy of conditional computation.
- **D.** Orthogonal + low rank dominates every baseline — a practical FiLM replacement.

---

## Potential applications

Diffusion (replace FiLM/adaLN in DiTs) · world models (condition latent dynamics on actions) ·
robotics (task-conditioned policies) · VLMs (condition visual features without cross-attention) ·
RL (goal-conditioned policies) · PEFT (interpret LoRA as a conditional operator).

---

## Broader vision

Conditioning should be understood as **learning transformations of latent geometry** rather than
independent modulation of coordinates. FiLM is one point in a broader family of conditional
operators — just as RoPE is one point in the family of Lie-group positional encodings. The long-term
goal is a **theory of conditional computation** unifying FiLM, conditional normalization, adapters,
hypernetworks, and LoRA under a common operator-theoretic framework, while identifying the *minimal*
family that delivers real empirical gains without sacrificing efficiency or stability.
