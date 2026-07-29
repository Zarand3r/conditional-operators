# Pre-registration: Complex FiLM on 3D Shapes, the one kill worth revisiting

## Outcome: KILL (2026-07-29) — and it falsifies the diagnosis

> Written after the run. Everything below is the pre-registration, unchanged. Numbers in
> `results/shapes3d_cfilm_summary.json`.

**AC-5 failed, and worse than before.** Complex FiLM's in-distribution fit ratio is
$1.542\times$ against a $1.10\times$ ceiling — *higher* than the $1.457\times$ that killed
`proposed` on this task. AC-1, AC-2 and AC-3 passed, AC-2 by $+84.5\%$ at $\delta=-1.00$.

| arm | in-dist | fit ratio | OOD test |
|---|---|---|---|
| `hypernet` (best unstructured) | 0.000708 | — | 0.009284 |
| `proposed` (original run) | 0.001032 | $1.457\times$ | 0.001163 |
| **`cfilm_hyb`** (gated here) | 0.001092 | **$1.542\times$** | 0.001439 |
| `proposed_scaled_conj` (reported) | 0.000991 | $1.400\times$ | **0.001127** |

**The spec predicted this outcome and named the reason, before the run.** It said: *"3D Shapes
differs from the PDE task in the way that matters most: its condition includes a categorical shape
swap, so the conditioning is partly asked to supply content... If Complex FiLM fails AC-5 here, the
reading is that the fit penalty tracks content demand rather than the isometry, and the PDE result
is narrower than it looks."*

That is what happened, and the contrast is clean because only one thing differs:

| task | content in the condition? | Complex FiLM fit ratio |
|---|---|---|
| PDE parameters | no — the initial field carries it | $\mathbf{0.938\times}$ pass |
| 3D Shapes | yes — a categorical shape swap | $\mathbf{1.542\times}$ fail |

**So the diagnosis written into `PDE2_SPEC.md` was wrong.** I had attributed the fit penalty to the
isometry: a rotation cannot rescale features, so give it a magnitude channel. On the PDE task that
reading worked ($1.118\times \to 0.938\times$) and looked like confirmation. It was not. The
penalty tracks **how much content the conditioning is asked to supply**, and a magnitude channel
does not fix that — here it made things marginally worse.

The isometry does cost something: on the PDE task, where content demand is zero, the magnitude
channel still bought $1.118 \to 0.938$. But that is the smaller of the two effects, and it is
visible only once the larger one is absent.

**Consequences, applied rather than noted.** The PDE confirmation is a result about a task with no
content in its condition, not a general fix, and the paper must say so. `shapes3d` remains a KILL
and its verdict is untouched. Complex FiLM is *not* a universal successor to the pure operator: it
wins where content demand is zero and does not help where it is not.

One thing worth recording without claiming it: `proposed_scaled_conj` posted the best OOD error of
any arm ever run on this task (0.001127, below `proposed`'s 0.001163) *and* the best fit of the
structured arms ($1.400\times$). It still fails AC-5. It was reported, not gated, and it is not a
result.


**Status:** pre-registered 2026-07-29, before the run.
**Name:** `shapes3d-cfilm` · **Hardware:** one RTX PRO 6000.

## Why this one, and why not the others

An audit of all seven kills asked which could plausibly change with an arm that was not available
when they ran. Complex FiLM was introduced in stage 8; stages 5, 6, 7, 10 and 11 predate it.

Only **3D Shapes** survives the filter, and the case is specific rather than hopeful:

- It failed **AC-5 alone** — in-distribution fit at $1.46\times$ against a $1.10\times$ ceiling.
  AC-1, AC-2 and AC-3 all passed, with CGA $87.5\%$ below the hypernetwork on unseen combinations.
- That is the *same criterion and the same mechanism* just resolved on the physics task: a rotation
  is an isometry, so it cannot rescale features, and in-distribution accuracy pays. Complex FiLM's
  magnitude channel supplies the rescaling, and on the PDE task it moved the fit ratio from
  $1.118\times$ to $0.938\times$ and turned a KILL into a CONFIRMED.
- `cfilm_hyb` was never run on 3D Shapes.

The other kills are excluded for stated reasons, not left out:

| kill | failed on | why not revisited |
|---|---|---|
| content (stage 6) | AC-1/2/3 — composition, not fit | Complex FiLM was tested in the content role in stage 8b and only reached non-inferiority ($0.97\times$) |
| rollouts (stage 7) | AC-1/2 — composition | failure is latent drift off the decodable manifold, which a magnitude channel does not address |
| rollouts+loss (stage 10) | AC-10.2 | same |
| contraction (stage 11) | all criteria | tests a contraction rate, not a conditioning arm |
| latent-edit | fit and composition | `cfilm_hyb` **was** run there and scored 0.4532, no better than plain FiLM. A frozen representation defeats it too |

**No bug motivates this.** Nothing found this session invalidates any past verdict: the flaws were
in code written this session (the screen's yardstick and output activation, a sample-size-saturated
diversity metric, two screens that read the test split). This is a coverage gap — the strongest arm
did not exist yet — not a correctness one.

## Design

Identical to `shapes3d` in every respect: same `Data6`, same factor splits, same `Backbone(in_ch=3)`,
same BCE objective, same Adam 1e-3, batch 256, 12{,}000 steps, same evaluation protocol.

- **Gated:** `cfilm_hyb`, **seeds 10-19** — fresh, never trained on this task.
- **Reported, not gated:** `proposed_scaled_conj`, seeds 10-19. It posted the best held-out error of
  any arm on the PDE task while fitting worst, and whether that trade reappears on images is worth
  recording.
- **Baselines:** reused from `shapes3d`, seeds 0-9, unchanged. They were generated long before this
  and re-running them would add noise and nothing else.

Seeds 10-19 are used rather than 0-9 so the gated arm's data is new even though the task is not.

## Pre-registered criteria

The **unchanged** `verdict.decide()` on OOD-TEST MSE, $N=10$ seeds, one-sided Mann-Whitney
$p\le0.01$, Cliff's $|\delta|\ge0.474$. Identical to the criteria `shapes3d` was scored against.

- **AC-1:** beats `film` by $\ge20\%$ on unseen two-factor combinations.
- **AC-2:** beats the better of `hypernet` and `dynamic_linear` by $\ge20\%$.
- **AC-3:** at most 2 of 10 seeds diverge.
- **AC-5:** in-distribution MSE $\le1.10\times$ the best unstructured arm's.
- **AC-4:** FLOPs $\le1.20\times$ `film`'s, parameters $\le1.05\times$ the smallest unstructured arm's.

**CONFIRMED $\Leftrightarrow$ all five.** Any failure is a KILL and is reported as one.

## What the outcomes mean, and the honest risk

**CONFIRMED** would mean the 3D Shapes kill was a property of the isometry rather than of the
dataset, and that the same fix works on images and on physics. Combined with the PDE result it
would make the recommendation unambiguous: use the magnitude-and-phase operator, not the pure
rotation.

**KILL** is a live outcome and I am not discounting it. 3D Shapes differs from the PDE task in the
way that matters most: its condition includes a **categorical shape swap**, so the conditioning is
partly asked to supply content, and content is where this family has repeatedly failed. The PDE
task deliberately had no content in the condition at all. If Complex FiLM fails AC-5 here, the
reading is that the fit penalty tracks content demand rather than the isometry, and the PDE result
is narrower than it looks.

Whatever happens, this run does not touch the `shapes3d` verdict, which stands as recorded.

## Config

As `shapes3d`. Throttled per the standing power rule; fsync-per-row with resume.

Amendments after this point must be dated and recorded below.
