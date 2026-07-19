# Stage-2 Results — Structure Without the Answer Handed To It

*Auto-generated from `results/stage2_summary.json` by `render_stage2.py`.*
Criteria: [`specs/STAGE2_SPEC.md`](specs/STAGE2_SPEC.md). Control: **de-aligned basis** (AC-7) — data is `M(c)=B·R(c)·Bᵀ` with `B` a fixed random orthonormal basis, so the operator's coordinate blocks are NOT aligned to the generative factors.
Config: 10 seeds, 4000 steps.

## Verdict: **CONFIRMED**

> CONFIRMED — the Stage-1 advantage SURVIVES de-alignment; structure is not just alignment.

- AC-1..AC-5 pass; proposed beats hypernet by 68.8% (p=9.1e-05, delta=-1.00)

## The decisive control: the no-basis ablation

| Operator | OOD-TEST MSE | reads as |
|---|---|---|
| **proposed** (T=P·Q(c)·Pᵀ, P **learned**) | **0.00107** | learns the hidden basis, generalizes |
| proposed_nobasis (P=I, coordinate blocks) | 0.01538 | cannot fit → pinned near FiLM |
| FiLM (diagonal floor) | 0.01553 | capacity floor |

The coordinate-block operator that *won* Stage-1 collapses to the FiLM floor here (0.01538 vs FiLM 0.01553); the proposed operator wins **only because it learns the hidden basis P**. This is the direct refutation of the Stage-1 'favorable-by-construction' caveat — structure helps *when it can be discovered*, not only when it is handed over.

## Per-arm results

| Arm | OOD-TEST MSE (mean±sd) | in-dist MSE | params | FLOPs/FiLM |
|---|---|---|---|---|
| film | 0.01553 ± 0.00043 | 0.01571 | 50,688 | 1.00× |
| concat_mlp | 0.01118 ± 0.00365 | 0.00001 | 83,584 | 1.65× |
| cond_layernorm | 0.02713 ± 0.00049 | 0.02719 | 50,688 | 1.01× |
| hypernet | 0.00344 ± 0.00087 | 0.00005 | 2,147,712 | 42.74× |
| dynamic_linear | 0.00608 ± 0.00044 | 0.00002 | 166,272 | 3.30× |
| **proposed** | 0.00107 ± 0.00019 | 0.00001 | 60,356 | 1.20× |

**AC-2 bar (best unstructured):** `hypernet`  ·  **margin:** 68.8%  ·  **p:** 9.1e-05  ·  **Cliff's δ:** -1

## What this changes for the novelty claim

- Stage-1 showed structure helps *when aligned*; Stage-2 shows the advantage **survives when the operator must discover the structure itself** — a materially stronger result.
- The learned-basis operator `P·Q(c)·Pᵀ` uses a **dense** `P` (two D×D matmuls), landing at the AC-4 FLOP ceiling. Scaling `P` to higher dimensions within budget needs a **butterfly / BOFT-style** parametrization — which is exactly the OFT→BOFT prior-art boundary this project must engage. That is the honest next collision to clear, not a solved problem.
- Still NOT a novelty claim on its own: both stages are synthetic. A real conditional-generation domain (Stage 3) and external review remain the bar.
