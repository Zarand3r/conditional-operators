"""Generate all paper figures from results/*.json into docs/paper/figs/ (PDF + PNG).

Idempotent: re-run after any sweep; stages 4b/5 are included automatically when their
summaries exist. Color scheme: neutral grays for baselines, one fixed accent for the
proposed arm, one for its mechanism ablation — identity is carried by axis labels, not hue.

Run:  .venv/bin/python -m conditional_operators.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
FIGS = ROOT / "docs" / "paper" / "figs"

ACCENT = "#0f62fe"        # proposed (fixed across all figures)
ACCENT2 = "#ff832b"       # proposed_mlp_gs ablation
GRAY = "#8d8d8d"          # baselines
INK = "#161616"

ARM_LABEL = {
    "film": "FiLM", "concat_mlp": "Concat-MLP", "cond_layernorm": "Cond-LN",
    "hypernet": "Hypernetwork", "dynamic_linear": "Dynamic low-rank",
    "proposed": "Lie (ours)", "proposed_mlp_gs": "MLP-head (abl.)",
}
ORDER = ["film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear",
         "proposed_mlp_gs", "proposed"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c6c6c6", "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 150,
})


def _load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def _color(arm):
    return ACCENT if arm == "proposed" else ACCENT2 if arm == "proposed_mlp_gs" else GRAY


def _barh(ax, summary, key, title, xlabel):
    # fixed 7-row grid so rows align across panels; missing arms leave a blank row
    present = [a for a in ORDER if a in summary["per_arm"]]
    y = [ORDER.index(a) for a in present]
    vals = [summary["per_arm"][a][key] for a in present]
    errs = [summary["per_arm"][a].get(key.replace("mean", "std"), 0) for a in present]
    ax.barh(y, vals, xerr=errs, height=0.62, color=[_color(a) for a in present],
            error_kw=dict(lw=0.8, ecolor=INK))
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_yticks(range(len(ORDER)), [ARM_LABEL[a] for a in ORDER])
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(axis="x", color="#e0e0e0", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_main():
    """F1: OOD error per arm across benchmarks (one panel per stage)."""
    panels = [
        (_load("summary.json"), "S1 synthetic (aligned)", "ood_test_mean"),
        (_load("stage3_summary.json"), "S2 synthetic (hidden basis)", "ood_test_mean"),
        (_load("stage4_summary.json"), "S3 dSprites (learned latent)", "ood_test_mean"),
        (_load("stage4b_summary.json"), "S3b dSprites (+categorical)", "ood_test_mean"),
        (_load("stage5_summary.json"), "S4 3D Shapes (RGB)", "ood_test_mean"),
    ]
    panels = [(s, t, k) for s, t, k in panels if s]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, (s, title, key) in zip(axes, panels):
        _barh(ax, s, key, title, "compositional OOD MSE (log)")
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.tight_layout()
    _save(fig, "f1_ood_all_stages")


def fig_length():
    """F2: composition-length generalization (stage-3 synthetic: pairs -> triples)."""
    s = _load("stage3_summary.json")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for a in ORDER:
        if a not in s["per_arm"]:
            continue
        pa = s["per_arm"][a]
        ax.plot([2, 3], [pa["ood_test_mean"], pa["triples_mean"]], marker="o", ms=4,
                lw=1.6 if a == "proposed" else 1.0, color=_color(a),
                label=ARM_LABEL[a], zorder=3 if a == "proposed" else 2)
    ax.set_yscale("log")
    ax.set_xticks([2, 3], ["2 factors\n(held-out pairs)", "3 factors\n(never trained)"])
    ax.set_ylabel("OOD MSE (log)")
    ax.set_title("Composition-length extrapolation", fontsize=9, loc="left")
    ax.grid(axis="y", color="#e0e0e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "f2_length_extrapolation")


def fig_ratio():
    """F3: OOD/in-dist ratio on real images — the compositional-gap metric."""
    stages = [("stage4_summary.json", "dSprites"),
              ("stage4b_summary.json", "dSprites +categorical"),
              ("stage5_summary.json", "3D Shapes")]
    stages = [(_load(f), t) for f, t in stages]
    stages = [(s, t) for s, t in stages if s]
    fig, axes = plt.subplots(1, len(stages), figsize=(2.9 * len(stages), 2.6))
    if len(stages) == 1:
        axes = [axes]
    for ax, (s, title) in zip(axes, stages):
        arms = [a for a in ORDER if a in s["per_arm"]]
        vals = [s["per_arm"][a]["ood_test_mean"] / s["per_arm"][a]["indist_mean"] for a in arms]
        y = range(len(arms))
        ax.barh(y, vals, height=0.62, color=[_color(a) for a in arms])
        ax.axvline(1.0, color=INK, lw=0.8, ls=":")
        ax.set_yticks(list(y), [ARM_LABEL[a] for a in arms])
        ax.set_xlabel("OOD / in-dist MSE ratio")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", color="#e0e0e0", lw=0.6)
        ax.set_axisbelow(True)
        if ax is not axes[0]:
            ax.set_yticklabels([])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "f3_ood_ratio_real")


def fig_mechanism():
    """F4: mechanistic recovery — learned singleton angles vs true primitive angles (S1)."""
    m = _load("mechanistic.json")
    fig, ax = plt.subplots(figsize=(2.7, 2.6))
    t, l = m["true_angles"], m["learned_singleton_angles"]
    lo, hi = min(t) - 0.08, max(t) + 0.08
    ax.plot([lo, hi], [lo, hi], color="#c6c6c6", lw=0.8, ls="--", zorder=1)
    ax.scatter(t, l, s=22, color=ACCENT, zorder=2)
    ax.set_xlabel("true primitive angle (rad)")
    ax.set_ylabel("learned angle (rad)")
    ax.set_title("Angle recovery", fontsize=9, loc="left")
    ax.grid(color="#e0e0e0", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "f4_angle_recovery")


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGS / name}.pdf/.png")


def fig_horizon():
    """F5: rollout horizon curves, without (stage-7) and with (stage-10) consistency loss."""
    panels = [(_load("stage7_summary.json"), "Without consistency loss"),
              (_load("stage10_summary.json"), "With consistency loss")]
    panels = [(s, t) for s, t in panels if s]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.6), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    xs = [3, 10, 20]
    for ax, (s, title) in zip(axes, panels):
        for a in ORDER:
            pa = s["per_arm"].get(a)
            if not pa:
                continue
            ys = [pa["indist"], pa["h10"], pa["h20"]]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6 if a == "proposed" else 1.0,
                    color=_color(a), label=ARM_LABEL[a], zorder=3 if a == "proposed" else 2)
        ax.set_yscale("log")
        ax.set_xticks(xs, ["3\n(trained)", "10", "20"])
        ax.set_xlabel("rollout length (actions)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="y", color="#e0e0e0", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("generation MSE (log)")
    axes[-1].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, "f5_rollout_horizon")


if __name__ == "__main__":
    fig_main()
    fig_length()
    fig_ratio()
    fig_mechanism()
    fig_horizon()
