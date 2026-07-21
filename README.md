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
- **Experiments (executed):** Stage-1 [`docs/RESULTS.md`](docs/RESULTS.md) — **CONFIRMED** (61.5% OOD win at 0.87× FiLM FLOPs). Stage-2 de-alignment control [`docs/RESULTS_STAGE2.md`](docs/RESULTS_STAGE2.md) — **UNFAIR after erratum** (dense-`P` basis was FLOP-undercounted 2×; true cost 1.52× FiLM broke the pre-registered ceiling; MSEs remain valid but the comparison is inconclusive at budget). Stage-3 [`docs/RESULTS_STAGE3.md`](docs/RESULTS_STAGE3.md) — **CONFIRMED, decisively**: GRAPE-inspired Lie-algebra conditioning `T(c)=P·R(Wc)·Pᵀ` ([arXiv:2512.07805](https://arxiv.org/abs/2512.07805); condition linear in the Lie algebra ⇒ composition exact by construction; `P` = Group-and-Shuffle structured orthogonal at 1.11× FiLM) solves the de-aligned task to ~1e-8 MSE on unseen pairs **and never-trained triples** — ~1.3e5× better than the strongest unstructured baseline, ~4.3e4× better than the identical-`P` MLP-head ablation. Specs under [`docs/specs/`](docs/specs/); run via `.venv` — `python -m conditional_operators.sweep` / `.stage2` / `.stage3`. All stages synthetic (group-structured conditions); Stage 4 = real conditional generation on GPU.

**The honest framing (see notes):** the operator *families* overlap with the PEFT literature
(OFT/BOFT/HRA/LoRA); the defensible contribution is studying them as **per-sample activation
conditioners** for **compositional generation**, with a **mechanistic** account of what the learned
rotations do. **Start with the Stage-1 synthetic benchmark — it's a cheap kill-test.**
