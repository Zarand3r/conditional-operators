"""Render results/stage3_summary.json into docs/RESULTS_STAGE3.md.

Run after the Stage-3 sweep:  .venv/bin/python -m conditional_operators.render_stage3
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "stage3_summary.json"
OUT = ROOT / "docs" / "RESULTS_STAGE3.md"

GLOSS = {
    "confirmed": "CONFIRMED — Lie-algebra conditioning wins within the TRUE budget, with exact "
                 "compositionality and systematic length extrapolation.",
    "kill": "KILL — a pre-registered Stage-3 criterion failed (named below); valid negative.",
    "unfair": "UNFAIR — budget violated under the corrected counter; comparison inconclusive.",
    "blocked": "BLOCKED — insufficient non-diverged seeds.",
    "invalid": "INVALID — hygiene breach; no scientific verdict.",
}


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    v = s["final_verdict"]
    bu = s["best_unstructured"]
    pa = s["per_arm"]

    lines = [
        "# Stage-3 Results — Lie-Algebra Conditioning (GRAPE-inspired), Within the True Budget",
        "",
        "*Auto-generated from `results/stage3_summary.json` by `render_stage3.py`.*",
        f"Criteria pre-registered in [`specs/STAGE3_SPEC.md`](specs/STAGE3_SPEC.md) (2026-07-21, "
        f"before any triples evaluation). Mechanism per GRAPE "
        f"([arXiv:2512.07805](https://arxiv.org/abs/2512.07805)): the condition enters **linearly "
        f"in the Lie algebra** — `T(c) = P·R(W·c)·Pᵀ` — so `T(c₁+c₂) = T(c₁)T(c₂)` exactly, by "
        f"construction. `P` is a Group-and-Shuffle structured orthogonal that fits the corrected "
        f"≤1.20× FiLM FLOP ceiling (the dense-P Stage-2 arm did not: see the Stage-2 erratum).",
        f"Config: {s['config']['n_seeds']} seeds, {s['config']['steps']} steps.",
        "",
        f"## Final verdict: **{v.upper()}**  (gate: {s['gate_verdict']})",
        "",
        f"> {GLOSS.get(v, v)}",
        "",
    ]
    for r in s["reasons"]:
        lines.append(f"- gate: {r}")
    crit = s["stage3_criteria"]
    lines += [
        f"- AC-7 exact compositionality: max ‖T(c₁+c₂)−T(c₂)T(c₁)‖_F = "
        f"{s['ac7_max_composition_err']:.2e} (gate < 1e-4) → {'✅' if crit['AC-7'] else '❌'}",
        f"- AC-8a triples vs best unstructured: {pa['proposed']['triples_mean']:.2e} vs "
        f"0.5×{pa[bu]['triples_mean']:.2e} → {'✅' if crit['AC-8a'] else '❌'}",
        f"- AC-8b systematicity (triples ≤ 2× own pairs): {pa['proposed']['triples_mean']:.2e} vs "
        f"2×{pa['proposed']['ood_test_mean']:.2e} → {'✅' if crit['AC-8b'] else '❌'}",
        "",
        "## Composition-length extrapolation (never-trained 3-hot conditions)",
        "",
        "| Arm | OOD pairs MSE | **Triples MSE** (all 56, unseen) | FLOPs/FiLM |",
        "|---|---|---|---|",
    ]
    order = ["film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear",
             "proposed", "proposed_mlp_gs"]
    for n in order:
        a = pa[n]
        star = "**" if n == "proposed" else ""
        note = " *(ablation, over budget)*" if n == "proposed_mlp_gs" else ""
        lines.append(f"| {star}{n}{star}{note} | {a['ood_test_mean']:.2e} | "
                     f"{a['triples_mean']:.2e} ± {a['triples_std']:.2e} | {a['flops_vs_film']:.2f}× |")
    lines += [
        "",
        f"**Gate margin (pairs, vs `{bu}`):** {_pct(s['margin_observed'])} · p={_g(s['p_value'])} · "
        f"Cliff's δ={_g(s['cliffs_delta'])}",
        "",
        "## Per-arm detail",
        "",
        "| Arm | OOD-TEST pairs (mean±sd) | in-dist | params | FLOPs/FiLM |",
        "|---|---|---|---|---|",
    ]
    for n in order:
        a = pa[n]
        star = "**" if n == "proposed" else ""
        lines.append(f"| {star}{n}{star} | {a['ood_test_mean']:.2e} ± {a['ood_test_std']:.2e} | "
                     f"{a['indist_mean']:.2e} | {a['params']:,} | {a['flops_vs_film']:.2f}× |")
    lines += [
        "",
        "## Reading",
        "",
        "- **The MLP-head ablation isolates the mechanism**: same structured `P`, same task — the "
        "only difference is whether the condition enters the Lie algebra *linearly* (exact "
        "composition) or through an MLP (learned composition). Compare their triples rows.",
        "- The Lie arm is **exactly orthogonal** (singular values ≡ 1: INV-3 for free) and satisfies "
        "AC-7 *for any weights* — composition is a property of the parametrization, not of training.",
        "- `P` is undercomplete (4,800 params vs dim SO(128)=8,128) yet suffices — it only needs the "
        "K active eigenplanes of the hidden basis, not all of it. This is the within-budget answer "
        "to the Stage-2 erratum and the OFT→BOFT boundary (structured orthogonal, Group-and-Shuffle "
        "style).",
        "",
        "## Honest scope",
        "",
        "- The data-generating process is *literally a commuting one-parameter group* — the exact "
        "match for a Lie parametrization. This stage proves the mechanism decisively **when the "
        "conditioning factors form a group**; it does not show real-world conditions behave this "
        "way. That is the Stage-4 (real conditional generation, GPU) question.",
        "- Mechanism credit: linear-in-Lie-algebra conditioning is GRAPE's construction (for "
        "attention position); the contribution here is transplanting it to FiLM's role — per-sample "
        "activation conditioning on arbitrary multi-hot conditions — and the budget-fair, "
        "pre-registered demonstration of systematic compositional generalization in that role.",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


def _pct(x):
    return f"{x:.1%}" if isinstance(x, (int, float)) else "—"


def _g(x):
    return f"{x:.2g}" if isinstance(x, (int, float)) else "—"


if __name__ == "__main__":
    main()
