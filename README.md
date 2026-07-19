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

**The honest framing (see notes):** the operator *families* overlap with the PEFT literature
(OFT/BOFT/HRA/LoRA); the defensible contribution is studying them as **per-sample activation
conditioners** for **compositional generation**, with a **mechanistic** account of what the learned
rotations do. **Start with the Stage-1 synthetic benchmark — it's a cheap kill-test.**
