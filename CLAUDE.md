# CLAUDE.md — conditional-operators

Research repo for **Conditioning as Group Action (CGA)**: conditioning a network with a
per-sample structured operator `y = T(c)x + β(c)` where the condition enters linearly in an
abelian Lie algebra, so `T(c₁+c₂) = T(c₁)T(c₂)` holds for any weights. FiLM and RoPE are special
cases. The experimental program is **complete**: 11 pre-registered suites, 7 confirmed, 4
negatives, all folded into the paper.

**Start here:** [`README.md`](README.md) for the result and how to reproduce it,
[`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) for the full program history (every verdict, every
amendment, the adversarial novelty sweep), [`docs/paper/paper.tex`](docs/paper/paper.tex) for the
paper. [`docs/PROPOSAL.md`](docs/PROPOSAL.md) and
[`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) are the original proposal and prior-art review,
kept for provenance; where they disagree with the paper, **the paper is current**.

## Non-negotiable rules

- **Never fake a result.** State what was run and the real number. If a run failed or was skipped,
  say so. This repo's value is that its negatives are real.
- **Pre-register before running.** Every suite gets a spec under `docs/specs/` fixing the success
  margin, statistical test, seed count, and splits *before* the decision run. Margins never move
  after data exists; deviations become dated amendments in the spec, never silent edits.
- **Verdicts come from the gate, not from judgment.** `conditional_operators/verdict.py` scores
  results mechanically. A KILL is reported as a KILL.
- **Equal-budget comparisons only.** The shared counter in `arms.py` enforces ≤1.20× FiLM FLOPs
  and ≤1.05× the smallest unstructured baseline's parameters. An operator that wins by spending
  more has not won.
- **Test splits are read once**, after all selection on validation.
- **Never hand-edit `docs/paper/tables/` or `docs/paper/figs/`.** They regenerate from
  `results/*.json` via `gen_tables.py` and `figures.py`. Same for any number in the paper.
- **Claim the combination, not the components.** The operator families (OFT/BOFT/HRA/LoRA), the
  Lie parametrization (RoPE/GRAPE), and latent-consistency losses (TD-MPC) are prior art; see the
  novelty sweep in `RESEARCH_LOG.md` for what must be cited and how claims are scoped.

## Operational notes

- **Environment:** `.venv` (Python 3.12 via `uv`); the system Python is 3.14 and has no torch.
- **Suites:** `python -m conditional_operators.suites --list` maps paper labels → code → results.
- **Sweeps:** fsync per row + resume on restart. **Do not switch git branches while a sweep is
  running** — it replaces the log file under the writer's file descriptor (this happened once;
  recovery via `/proc/<pid>/fd` is in the log).
- **Power:** this box's PSU trips at sustained ~600 W. Throttle with `STAGE*_THROTTLE_MS=60` or
  cap the GPU (`sudo nvidia-smi -pl 300`), and run sweeps sequentially.

## Skills

The `eng-skills` plugin is wired via `.claude/settings.json`. Most relevant here:
`test-driven-verification` (operator invariants derive directly from the math),
`principal-production-engineer` / `python-style` (implementation),
`spec-driven-development` (new pre-registrations), `research-ideation` (before any new
direction), `karpathy-guidelines` (throughout). The `paper-writing` skill is installed locally
under `.claude/skills/` but is **not tracked here** — it is third-party (MIT, SNL-UCSB) and lives
in the `claude-skills` marketplace repo.
