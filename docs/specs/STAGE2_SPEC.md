# Spec: Stage-2 — Structure Without the Answer Handed to It

**Status:** Draft — two Open Questions need your call before coding (they change hours of work).
**Derived from:** [`STAGE1_SPEC.md`](STAGE1_SPEC.md) (CONFIRMED), [`../PROPOSAL.md`](../PROPOSAL.md), [`../RESEARCH_NOTES.md`](../RESEARCH_NOTES.md)

> Stage-1 was **favorable by construction** — the operator's block structure matched the data's
> compositional structure. Stage-2's entire job is to remove that advantage: test whether a
> structured input-conditioned operator still beats FiLM **and** the best unstructured conditioner on
> **compositional OOD** in a setting where **nobody hand-designs the operator to fit the data**.
> Passing Stage-2 is the first result that could support a novelty claim; failing it means the
> Stage-1 win was structure-alignment, not a general advantage (a valid, publishable negative).

## Goal

Determine whether an input-conditioned structured operator generalizes to **unseen attribute
compositions** better than FiLM and the strongest unstructured conditioner, at equal-or-lower budget,
on a **real (non-hand-aligned) conditional task**.

## Why not CIFAR-100 classification as written in PROPOSAL.md

Classification conditions on the label being predicted — there is **no composition of two conditions**
to hold out, so it cannot test the core claim (compositional generalization). Stage-2 must use a task
with **factorizable conditions** and an **unseen-combination split**. See OQ-1.

## Requirements (EARS)

- **R1:** The system shall condition on **factored attributes** (e.g. shape × color × position) and
  train on a subset of attribute combinations, holding out unseen combinations for OOD-VAL/OOD-TEST.
- **R2:** The operator's structure shall **not** be coordinate-aligned to the data's generative
  factors (unlike Stage-1); any structural inductive bias must be learned, not hand-placed.
- **R3:** WHEN comparing arms, the system shall reuse the Stage-1 fairness machinery — shared encoder,
  shared param/FLOP counter, identical optimizer/schedule/seeds, OOD-VAL selection, single OOD-TEST read.
- **R4:** The verdict shall be computed by the **same** `conditional_operators/verdict.py` gate (AC-1..AC-6),
  re-using its pre-registration semantics.

## Acceptance Criteria (pre-registered — mirror Stage-1, one addition)

- **AC-1..AC-6:** identical roles to Stage-1 (floor vs FiLM; ≥20% vs best unstructured; MWU p≤0.01 ∧
  Cliff's δ≤−0.474; params ≤1.05× / FLOPs ≤1.20×FiLM; no in-dist regression; single OOD-TEST read).
- **AC-7 (new, the point of Stage-2):** the proposed operator receives **no coordinate alignment** to
  the generative factors — verified by an alignment-shuffle control: the result must hold when the
  data's factor subspaces are randomly rotated (so a coordinate-block operator has no free lunch).

## Invariants

- **INV-1..4:** carry over unchanged (identity-init, orthogonality of Q, bounded spectrum, composition
  error logged). The proposed operator likely needs a **more flexible orthogonal parametrization**
  (BOFT-style butterfly or larger learnable blocks) to represent non-aligned structure within budget —
  this is the central engineering risk and directly touches the OFT/BOFT prior-art boundary.

## Open Questions — RESOLVED (PI sign-off 2026-07-19)

- [x] **OQ-1 — benchmark.** **De-aligned synthetic generation** — reuse the Stage-1 MSE harness with
      `M(c)=B·R(c)·Bᵀ` (`B` fixed random orthonormal), so the operator is not handed the structure.
      Cheapest, most controlled attack on the Stage-1 caveat. (dSprites/CIFAR deferred to Stage 3+.)
- [x] **OQ-2 — compute.** Run the scoped CPU version now (minutes on 32 cores). Implemented in
      `conditional_operators/stage2.py`; results in [`../RESULTS_STAGE2.md`](../RESULTS_STAGE2.md).

## Original open questions (for the record)

- **OQ-1 — benchmark.** dSprites/Shapes3D-style factored-attribute *generation* (preserves the
      compositional thesis, moderate cost) vs CIFAR-100 classification as-proposed (cheap but does not
      test compositionality) vs a de-aligned synthetic *generation* task (cheapest, most controlled).
- [ ] **OQ-2 — compute.** This box is **CPU-only (no GPU)**; a real image sweep (6 arms × 10 seeds) is
      **multi-hour to overnight**. Options: run a scoped CPU version now; I build it and you run the
      full sweep on a GPU; or I run an overnight CPU sweep via the `elves`/`auto-research` skills.

## Dependencies / Assumptions

- Same `.venv` (torch CPU). A real dataset download (~tens–hundreds of MB) over the working network.
- No large data/checkpoints in git (CLAUDE.md) — only splits config + results log + summary.
