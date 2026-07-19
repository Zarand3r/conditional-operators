# conditional-operators

**Beyond FiLM: structured conditional operators.** Research code + notes investigating whether a
minimal, richer-than-diagonal family of **input-conditioned operators** can replace Feature-wise
Linear Modulation (FiLM) — improving compositional generalization and conditional generation while
keeping FiLM's efficiency and optimization stability.

FiLM applies a diagonal transform per feature, `y = γ(c)⊙x + β(c)`. We study `y = T(c)x + β(c)` for
**structured** `T(c)` (block-diagonal, orthogonal, orthogonal+low-rank, Lie-group, flow), generated
**per-sample from a conditioning input** and applied to **activations** (FiLM's role) — *not* weight
fine-tuning.

- **Proposal:** [`docs/PROPOSAL.md`](docs/PROPOSAL.md)
- **Novelty review + re-scoped claim + first kill-test:** [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) — **read this first.**
- **Experiments (executed):** Stage-1 [`docs/RESULTS.md`](docs/RESULTS.md) — **CONFIRMED** (61.5% OOD win at 0.87× FiLM FLOPs); Stage-2 de-alignment control [`docs/RESULTS_STAGE2.md`](docs/RESULTS_STAGE2.md) — **CONFIRMED** (68.8% win when the operator must *learn* the hidden basis; no-basis ablation collapses to the floor). Specs under [`docs/specs/`](docs/specs/). Run via `.venv` (see memory) — `python -m conditional_operators.sweep` / `.stage2`. Both stages synthetic; not a novelty claim — Stage 3 (real conditional generation, GPU) is the next bar.

**The honest framing (see notes):** the operator *families* overlap with the PEFT literature
(OFT/BOFT/HRA/LoRA); the defensible contribution is studying them as **per-sample activation
conditioners** for **compositional generation**, with a **mechanistic** account of what the learned
rotations do. **Start with the Stage-1 synthetic benchmark — it's a cheap kill-test.**
