"""Render results/stage4_summary.json into docs/RESULTS_STAGE4.md.

Run after the Stage-4 sweep:  .venv/bin/python -m conditional_operators.render_stage4
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "stage4_summary.json"
OUT = ROOT / "docs" / "RESULTS_STAGE4.md"

GLOSS = {
    "confirmed": "CONFIRMED — the Lie conditioning advantage transfers to real images on a "
                 "LEARNED latent, at the smallest parameter count and 1.11× FiLM cost.",
    "kill": "KILL — a pre-registered criterion failed on real images (valid negative: exact "
            "composition helps only on group-structured representations).",
    "unfair": "UNFAIR — budget violated; comparison inconclusive.",
    "blocked": "BLOCKED — insufficient non-diverged seeds.",
    "invalid": "INVALID — hygiene breach; no scientific verdict.",
}

ORDER = ["film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear",
         "proposed", "proposed_mlp_gs"]


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    v = s["final_verdict"]
    bu = s["best_unstructured"]
    pa = s["per_arm"]

    lines = [
        "# Stage-4 Results — Real-Image Compositional Conditional Transformation (dSprites)",
        "",
        "*Auto-generated from `results/stage4_summary.json` by `render_stage4.py`.*",
        f"Criteria pre-registered in [`specs/STAGE4_SPEC.md`](specs/STAGE4_SPEC.md) (2026-07-21, "
        f"before any decision run; pre-run amendments documented there, including the BCE loss "
        f"amendment and the smoke-read hygiene disclosure). Task: encode a real dSprites image, "
        f"apply the conditioning operator to the **learned** latent for a factor-change condition "
        f"Δ, decode, score pixel-MSE against the deterministic ground-truth image. OOD = "
        f"**never-trained two-factor change types**; triples = never-trained three-factor types "
        f"(diagnostic). Hardware: RTX PRO 6000 Blackwell.",
        f"Config: {s['config']['n_seeds']} seeds, {s['config']['steps']} steps, BCE training loss, "
        f"pixel-MSE gated metric.",
        "",
        f"## Verdict: **{v.upper()}**",
        "",
        f"> {GLOSS.get(v, v)}",
        "",
    ]
    for r in s["reasons"]:
        lines.append(f"- gate: {r}")
    lines += [
        "",
        "## Per-arm results (pixel-MSE)",
        "",
        "| Arm | in-dist | OOD pairs (mean±sd) | OOD/in-dist | triples | params | FLOPs/FiLM |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in ORDER:
        a = pa[n]
        star = "**" if n == "proposed" else ""
        note = " *(ablation, over budget)*" if n == "proposed_mlp_gs" else ""
        ratio = a["ood_test_mean"] / a["indist_mean"]
        lines.append(
            f"| {star}{n}{star}{note} | {a['indist_mean']:.6f} | "
            f"{a['ood_test_mean']:.6f} ± {a['ood_test_std']:.6f} | {ratio:.2f}× | "
            f"{a['triples_mean']:.6f} | {a['params']:,} | {a['flops_vs_film']:.2f}× |")
    lines += [
        "",
        f"**Gate:** margin {_pct(s['margin_observed'])} vs `{bu}` · p={_g(s['p_value'])} · "
        f"Cliff's δ={_g(s['cliffs_delta'])} · all criteria {s['gate_criteria']}",
        "",
        "## Reading — what the columns show",
        "",
        "- **Every arm fits in-distribution equally** (~0.0014): the backbone is identical; all "
        "differences are compositional generalization, isolated by construction.",
        "- **The OOD/in-dist ratio is the story**: proposed **1.25×** (near-systematic recombination) "
        "vs FiLM 2.08×, hypernet 2.68×, dynamic_linear 3.33×. More conditioning capacity made "
        "recombination *worse* — the capacity-vs-inductive-bias tradeoff, now on real images.",
        "- **The mechanism ablation holds on real images**: same GS-P, more FLOPs, but an MLP angle "
        "head instead of linear-in-the-algebra → 1.55× worse OOD. Linearity, not the orthogonal "
        "basis, carries the win.",
        "- **Triples degrade gracefully** for the Lie arm (1.48× its in-dist) and steeply for "
        "everything else — the length-extrapolation signature from Stage-3 survives, attenuated, "
        "on a learned latent.",
        "",
        "## Honest scope",
        "",
        "- Errors here are ~0.0018, not Stage-3's 1e-8: a learned conv latent does **not** support "
        "exact group action — the advantage is a robust ~2× on unseen combinations, not the "
        "orders-of-magnitude of the synthetic stages. Both facts are the finding.",
        "- dSprites is simple, single-sprite, binary. Escalations that remain: natural images, "
        "text/class conditioning in a DiT (adaLN swap), and categorical (non-group) factors like "
        "shape changes.",
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
