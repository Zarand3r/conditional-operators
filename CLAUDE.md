# CLAUDE.md — conditional-operators

Research repo investigating **structured conditional operators** as a generalization of **FiLM**
(`y = γ(c)⊙x + β(c)`, a diagonal transform) to `y = T(c)x + β(c)` for structured `T(c)`
(block-diagonal, orthogonal, orthogonal+low-rank, Lie-group, flow), **generated per-sample from a
conditioning input** and applied to **activations**.

**Source of truth:** [`docs/PROPOSAL.md`](docs/PROPOSAL.md). **Read [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) first** — it has the prior-art collision check and the re-scoped claim.

## Non-negotiable research facts (easy to get wrong)

- **We are still generalizing FiLM** — FiLM is the diagonal special case `T(c)=diag(γ(c))`. The
  generalization is richer *structured* `T(c)`, kept efficient and stable (identity-init, bounded
  spectrum). That thesis is unchanged.
- **The novelty is the *setting*, not the operators.** OFT/BOFT/HRA/LoRA already provide block-diag,
  orthogonal, orthogonal+low-rank, and low-rank transforms — but as **weight fine-tuning** (a fixed
  transform). Our contribution is those structures as **per-sample, activation-space, input-conditioned**
  modulation (FiLM's role) for **conditional generation + compositional generalization**. Position
  every family against its PEFT twin; do **not** claim the operator families as new.
- **Drop the grand "unify all PEFT" theory** — He et al. (2110.04366) already unified adapters/LoRA/prefix.
  Keep it to a one-paragraph positioning, not a contribution.
- **Stage 1 is a cheap kill-test.** On the synthetic analytic-transformation benchmark, if
  input-conditioned `Q(c)+U(c)V(c)^T` does not beat FiLM on **compositional OOD** at <20% overhead
  and equal params, **stop** (diagonal-is-enough is a valid negative result). Register the success
  margin *before* running; never launder a soft win into a Pursue.
- **The differentiator is mechanistic interpretability** — do learned rotations correspond to
  interpretable subspace transforms of semantic concepts? Most PEFT papers skip this; we lead with it.

## Skills — use these automatically

The `eng-skills` plugin is wired via `.claude/settings.json` (auto-installed + auto-updated; invoked
namespaced like `/eng-skills:python-style`, and auto-invoked by description). Most relevant here:

| Skill | Load it when… |
|---|---|
| **research-ideation** | Deciding/refining *what's worth doing* and whether it's novel — prior-art collision check (Scoop-Check) + Pursue/Refine/Kill. Already applied in `RESEARCH_NOTES.md`; re-run before adding a new sub-direction. |
| **karpathy-guidelines** | Always, for any writing/reviewing/refactoring. Surgical changes, surface assumptions, verifiable success criteria. |
| **principal-production-engineer** | Implementing/reviewing the operator layers, benchmark harness, training loops. Single entry point — simple design, explicit ownership, visible failure, honest verification. |
| **strategic-engineering-planner** | *Before* a nontrivial build (the benchmark suite, a DiT integration). Produces a written roadmap first. |
| **test-driven-verification** | Every operator implementation — derive tests from the math (identity-init `T(c)=I`, orthogonality `T^T T=I`, bounded spectrum, compositionality) first; red→green→refactor; capture re-runnable evidence. |
| **auto-research** | The **Stage-1 sweep** — optimizing *one number* (compositional-OOD error) on a *fixed* synthetic harness, unattended. Append-only results log, keep-on-improvement / reset-on-regression. A near-perfect fit. |
| **data-oriented-design** | Performance in the training/eval hot loops, batched operator application, vectorized synthetic-data generation. |
| **python-style** | Writing/reviewing Python — flat control flow, enums over magic strings, fail-fast, no optional imports/redundancy. |
| **elves** | Executing a multi-stage plan unattended/overnight (Stage 1 → 2 → 3 sweeps). Sprint-sized batches, tests + PR review. Requires `git` + `gh`. |

**Default flow** for a nontrivial change: `research-ideation` (is this sub-direction novel/worth it?)
→ `strategic-engineering-planner` (roadmap) → `principal-production-engineer` (implement, routing into
`python-style` / `data-oriented-design`), with `test-driven-verification` gating and
`karpathy-guidelines` throughout. For the Stage-1 metric sweep: **`auto-research`**.

## Conventions

- **Never fake a result.** State what was run and the real number; if a run failed or was skipped,
  say so. Register success criteria before the experiment.
- **Equal-parameter, equal-FLOP comparisons only** — any operator that beats FiLM by spending more
  compute is not a fair win; hold budget constant.
- Keep changes surgical and reversible; large checkpoints/datasets stay out of git.
