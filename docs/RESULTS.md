# Stage-1 Results — Structured Conditional Operators vs FiLM

*Auto-generated from `results/summary.json` by `render_results.py`. Do not hand-edit.*
Source of criteria: [`specs/STAGE1_SPEC.md`](specs/STAGE1_SPEC.md). Config: 10 seeds, 4000 steps, Adam lr=0.001, batch=256.

## Verdict: **CONFIRMED**

> CONFIRMED — proceed to Stage 2 (CIFAR-100).

- AC-1..AC-5 pass; proposed beats hypernet by 61.5% (p=9.1e-05, delta=-1.00)

## Pre-registered acceptance criteria

| Criterion | Meaning | Result |
|---|---|---|
| AC-1 | proposed OOD-MSE < FiLM (floor) | ✅ |
| AC-2 | proposed beats best-unstructured by ≥20% rel | ✅ |
| AC-3 | significant (MWU p≤0.01 ∧ Cliff's δ≤−0.474) | ✅ |
| AC-5 | no in-dist regression (≤1.10×) | ✅ |
| AC-4 | params ≤1.05× min-unstruct ∧ FLOPs ≤1.20× FiLM | see table |
| AC-6 | OOD-TEST read once | enforced in harness |

**Best unstructured competitor (AC-2 bar):** `hypernet`  ·  **margin:** 61.5%  ·  **p:** 9.1e-05  ·  **Cliff's δ:** -1

## Per-arm results

| Arm | OOD-TEST MSE (mean±sd) | in-dist MSE | params | FLOPs/FiLM | diverged |
|---|---|---|---|---|---|
| film | 0.01269 ± 0.00056 | 0.01262 | 50,688 | 1.00× | 0% |
| concat_mlp | 0.01462 ± 0.00292 | 0.00002 | 83,584 | 1.65× | 0% |
| cond_layernorm | 0.02445 ± 0.00062 | 0.02418 | 50,688 | 1.01× | 0% |
| hypernet | 0.00193 ± 0.00034 | 0.00002 | 2,147,712 | 42.74× | 0% |
| dynamic_linear | 0.00575 ± 0.00046 | 0.00002 | 166,272 | 3.30× | 0% |
| **proposed** | 0.00074 ± 0.00020 | 0.00000 | 43,972 | 0.87× | 0% |

## Reading

- FiLM is diagonal and cannot rotate, so its in-dist MSE is a capacity floor, not a convergence artifact — this is why AC-1 (beat FiLM) is a sanity floor, not the real bar.
- The real test (AC-2/AC-3) is proposed vs the strongest **unstructured** conditioner (`hypernet`) on **unseen compositions**, at equal-or-lower budget (AC-4).
- Proposed wins while being the **cheapest** arm (0.87× FiLM FLOPs, fewest params); the unstructured winner `hypernet` spends 42× FiLM FLOPs and still loses on OOD.

## Mechanistic interpretability (the differentiator)

From `mechanistic.py` (one trained proposed operator):

- **Learned rotations are interpretable.** Singleton conditions recover the true primitive angles to within **1.7e-04 rad** (true `[0.3, 0.4286, 0.5571, 0.6857, 0.8143, 0.9429, 1.0714, 1.2]` → learned `[0.3, 0.4286, 0.5572, 0.6858, 0.8142, 0.9428, 1.0713, 1.2]`).
- **Composition is explicit.** On held-out pairs, `‖T(cᵢⱼ)−T(cᵢ)T(cⱼ)‖_F ≈ 0.355` equals recovery error `≈ 0.355` — i.e. `T(cᵢ)T(cⱼ)` IS the true composition; the only residual is the head's imperfect extrapolation to two-hot inputs, and it stays on the orthogonal manifold (bounded spectrum).

## Honest scope — what this does and does NOT prove

- **Does:** at equal-or-lower budget, a structured (block-orthogonal) input-conditioned operator generalizes to unseen compositions far better than an unstructured `W(c)` — decisively, with a clean mechanistic account. Stage-1 is **CONFIRMED**.
- **Does NOT:** prove the general claim. Stage-1 is the *favorable case by construction* — the operator's 2-plane block structure matches the data's compositional structure. The value of CONFIRMED is that it clears the cheap kill-test, licensing **Stage 2 (CIFAR-100)** where the 'right' structure is unknown — the real test of whether structure helps when it is not handed the answer.
- **Budget finding:** the flagship *fully per-sample* `U(c)V(c)ᵀ` low-rank head exceeds the ≤1.20× FiLM FLOP budget; the proposed arm uses block-orthogonal `Q(c)` + a shared-basis, input-conditioned-gain low-rank term to stay within budget (see `arms.py`).
