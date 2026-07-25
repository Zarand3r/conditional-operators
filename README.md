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
- **Experiments (executed):** Stage-1 [`docs/RESULTS.md`](docs/RESULTS.md) — **CONFIRMED** (61.5% OOD win at 0.87× FiLM FLOPs). Stage-2 de-alignment control [`docs/RESULTS_STAGE2.md`](docs/RESULTS_STAGE2.md) — **UNFAIR after erratum** (dense-`P` basis was FLOP-undercounted 2×; true cost 1.52× FiLM broke the pre-registered ceiling; MSEs remain valid but the comparison is inconclusive at budget). Stage-3 [`docs/RESULTS_STAGE3.md`](docs/RESULTS_STAGE3.md) — **CONFIRMED, decisively**: GRAPE-inspired Lie-algebra conditioning `T(c)=P·R(Wc)·Pᵀ` ([arXiv:2512.07805](https://arxiv.org/abs/2512.07805); condition linear in the Lie algebra ⇒ composition exact by construction; `P` = Group-and-Shuffle structured orthogonal at 1.11× FiLM) solves the de-aligned task to ~1e-8 MSE on unseen pairs **and never-trained triples** — ~1.3e5× better than the strongest unstructured baseline, ~4.3e4× better than the identical-`P` MLP-head ablation. Stage-4 [`docs/RESULTS_STAGE4.md`](docs/RESULTS_STAGE4.md) — **CONFIRMED on real images**: dSprites conditional transformation on a **learned** conv latent (RTX PRO 6000); Lie arm beats the best unstructured conditioner by 53.8% (p=9.1e-05, δ=−1.0) on never-trained two-factor change types, with OOD/in-dist ratio 1.25× vs 2.1–3.3× for every baseline, at the fewest params and 1.11× FiLM FLOPs; the identical-`P` MLP-head ablation confirms linearity-in-the-algebra as the mechanism on real images too. Specs under [`docs/specs/`](docs/specs/); run via `.venv` — `python -m conditional_operators.sweep` / `.stage2` / `.stage3` / `.stage4`. S3b (dSprites+categorical shape) — **CONFIRMED** (42.9% margin; the advantage attenuates but survives non-group factors). S5 (3D Shapes, RGB) — **registered negative (KILL on AC-5)**: best OOD by 87.5% with gap 1.13× vs 13–26×, but a 1.46× in-dist fit penalty trips the pre-registered no-regression gate — the fit-vs-composition trade-off, quantified. **Submission paper (unified, ICLR format):** [`docs/paper/paper_iclr.pdf`](docs/paper/paper_iclr.pdf); source drafts [`paper.pdf`](docs/paper/paper.pdf)/[`paper_short.pdf`](docs/paper/paper_short.pdf) (tables/figures auto-generated from `results/`). Remaining escalation: DiT/adaLN at diffusion scale.

**The honest framing (see notes):** the operator *families* overlap with the PEFT literature
(OFT/BOFT/HRA/LoRA); the defensible contribution is studying them as **per-sample activation
conditioners** for **compositional generation**, with a **mechanistic** account of what the learned
rotations do. **Start with the Stage-1 synthetic benchmark — it's a cheap kill-test.**
