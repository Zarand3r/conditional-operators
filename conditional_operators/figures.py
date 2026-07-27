"""Generate the paper's figures from results/*.json into docs/paper/figs/ (PDF + PNG).

Run:  .venv/bin/python -m conditional_operators.figures

Design notes, so the next person does not undo them:

* **No bars on a log axis.** Bar length encodes magnitude from zero, which a log scale destroys.
  Where the quantity spans orders of magnitude we use dots on a log axis; where it is a ratio we
  use dots on a linear axis with a reference line.
* **Prefer a ratio with a natural zero-point.** The compositional gap (error on unseen
  combinations / error in distribution) is 1.0 when a model generalizes perfectly, so the reader
  sees the result without reading an axis.
* **Every mark is labelled.** Nobody should have to measure a bar against a tick.
* Three colour roles, fixed: ours, the ablation, everything else. Blue and orange stay separable
  under the common colour-vision deficiencies; grey is neutral and recedes.
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

OURS = "#0f62fe"      # the proposed operator
ABL = "#ff832b"       # its mechanism ablation
BASE = "#8d8d8d"      # every baseline
INK = "#161616"
GRID = "#e3e3e3"

ARM_LABEL = {
    "film": "FiLM", "concat_mlp": "Concat-MLP", "cond_layernorm": "Cond-LN",
    "hypernet": "Hypernetwork", "dynamic_linear": "Dynamic low-rank",
    "proposed": "CGA (ours)", "proposed_mlp_gs": "MLP-head (ablation)",
}
ORDER = ["dynamic_linear", "hypernet", "cond_layernorm", "concat_mlp", "film",
         "proposed_mlp_gs", "proposed"]

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": "#c6c6c6", "axes.linewidth": 0.7,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "figure.dpi": 150, "xtick.major.size": 3,
    "ytick.major.size": 0,
})


def _load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def _color(arm):
    return OURS if arm == "proposed" else ABL if arm == "proposed_mlp_gs" else BASE


def _clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def _dots(ax, arms, values, labels, fmt, log=False, ref=None):
    """Horizontal dot plot: a dot per arm, a stem to the axis, and the value written on it."""
    y = range(len(arms))
    left = min(values) / (2.2 if log else 1)
    if ref is not None:
        ax.axvline(ref, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=1)
    for i, (a, v) in enumerate(zip(arms, values)):
        c = _color(a)
        ax.plot([left, v], [i, i], color=c, lw=1.4, alpha=0.45, zorder=2,
                solid_capstyle="butt")
        ax.plot([v], [i], "o", ms=6.5, color=c, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(fmt(v), (v, i), textcoords="offset points", xytext=(9, 0),
                    va="center", fontsize=7.6,
                    color=INK if a != "proposed" else OURS,
                    fontweight="bold" if a == "proposed" else "normal")
    if log:
        ax.set_xscale("log")
    ax.set_yticks(list(y), labels)
    ax.set_ylim(-0.7, len(arms) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    _clean(ax)



# ---------------------------------------------------------------- headline: the compositional gap

def fig_gap():
    """F1. How much worse does each conditioner get on unseen combinations than on trained ones?

    A ratio, so 1.0 means 'no degradation at all'. This is the paper's claim in one picture, and
    it needs no log axis: the reader sees CGA sitting near the reference line while the most
    expressive baselines sit far from it.
    """
    suites = [(_load("stage4_summary.json"), "dSprites"),
              (_load("stage4b_summary.json"), "dSprites, plus a categorical factor"),
              (_load("stage5_summary.json"), "3D Shapes")]
    suites = [(s, t) for s, t in suites if s]
    fig, axes = plt.subplots(1, len(suites), figsize=(9.6, 2.9))
    if len(suites) == 1:
        axes = [axes]
    for ax, (s, title) in zip(axes, suites):
        arms = [a for a in ORDER if a in s["per_arm"]]
        gaps = [s["per_arm"][a]["ood_test_mean"] / s["per_arm"][a]["indist_mean"] for a in arms]
        labels = [ARM_LABEL[a] for a in arms] if ax is axes[0] else [""] * len(arms)
        _dots(ax, arms, gaps, labels, lambda v: f"{v:.2f}×", ref=1.0)
        ax.set_xlim(0.55, max(gaps) * 1.4)
        ax.set_title(title, fontsize=8.5, loc="left", pad=6)
        if max(gaps) > 10:                       # 3D Shapes spans a much wider range
            ax.set_xticks([1, 5, 10, 15, 20, 25])
    fig.supxlabel("error on unseen combinations, as a multiple of error on trained ones "
                  "(1.0 = no degradation, dashed line)", fontsize=8.5, y=0.02)
    fig.suptitle("Every conditioner degrades on unseen condition combinations. CGA degrades least.",
                 fontsize=9.5, x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    _save(fig, "f1_compositional_gap")


def fig_synthetic():
    """F2. Absolute error on the two synthetic suites, where the win spans orders of magnitude.

    Dots on a log axis (not bars, which a log axis would make meaningless), with the size of the
    gap stated in words so it does not have to be read off the ticks.
    """
    suites = [(_load("summary.json"), "S1: conditions aligned to the features"),
              (_load("stage3_summary.json"), "S2: conditions hidden behind a random basis")]
    suites = [(s, t) for s, t in suites if s]
    fig, axes = plt.subplots(1, len(suites), figsize=(9.0, 2.7))
    if len(suites) == 1:
        axes = [axes]
    for ax, (s, title) in zip(axes, suites):
        arms = [a for a in ORDER if a in s["per_arm"]]
        vals = [s["per_arm"][a]["ood_test_mean"] for a in arms]
        labels = [ARM_LABEL[a] for a in arms] if ax is axes[0] else [""] * len(arms)
        _dots(ax, arms, vals, labels, lambda v: f"{v:.1e}".replace("e-0", "e-"), log=True)
        ours = s["per_arm"]["proposed"]["ood_test_mean"]
        best_other = min(v for a, v in zip(arms, vals) if a != "proposed")
        ax.set_xlim(ours / 6, max(vals) * 12)
        ax.set_title(f"{title}\n{best_other / ours:,.0f}× lower error than the next best arm",
                     fontsize=8.5, loc="left", pad=6, color=INK)
    fig.supxlabel("error on unseen condition combinations (log scale)", fontsize=8.5, y=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "f2_synthetic_error")


def fig_length():
    """F3. Error as compositions get longer than anything seen in training."""
    s = _load("stage3_summary.json")
    if not s:
        return
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    for a in ORDER:
        if a not in s["per_arm"]:
            continue
        pa = s["per_arm"][a]
        ys = [pa["ood_test_mean"], pa["triples_mean"]]
        ax.plot([2, 3], ys, marker="o", ms=5, lw=2.0 if a == "proposed" else 1.0,
                color=_color(a), label=ARM_LABEL[a],
                zorder=3 if a == "proposed" else 2,
                alpha=1.0 if a in ("proposed", "proposed_mlp_gs") else 0.75)
        if a in ("proposed", "film"):
            ax.annotate(ARM_LABEL[a], (3, ys[1]), textcoords="offset points",
                        xytext=(8, 0), va="center", fontsize=7.6, color=_color(a),
                        fontweight="bold" if a == "proposed" else "normal")
    ax.set_yscale("log")
    ax.set_xticks([2, 3], ["two factors\n(held out)", "three factors\n(never trained)"])
    ax.set_xlim(1.85, 3.6)
    ax.set_ylabel("error (log scale)")
    ax.set_title("Longer combinations than training contained", fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    _clean(ax)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", length=3)
    ax.legend(fontsize=6.8, frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    _save(fig, "f3_length_extrapolation")


def fig_mechanism():
    """F4. The learned rotation angles against the true generators of the data."""
    m = _load("mechanistic.json")
    if not m:
        return
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    t, l = m["true_angles"], m["learned_singleton_angles"]
    lo, hi = min(t) - 0.1, max(t) + 0.1
    ax.plot([lo, hi], [lo, hi], color="#bdbdbd", lw=1.0, ls="--", zorder=1)
    ax.scatter(t, l, s=42, color=OURS, zorder=2, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("true generator angle (rad)")
    ax.set_ylabel("angle the model learned (rad)")
    ax.set_title(f"Recovered to {m['max_singleton_angle_abs_err']:.0e} rad",
                 fontsize=9, loc="left")
    ax.grid(color=GRID, lw=0.6)
    _clean(ax)
    for s in ("left",):
        ax.spines[s].set_visible(True)
    ax.tick_params(axis="y", length=3)
    fig.tight_layout()
    _save(fig, "f4_angle_recovery")


def fig_horizon():
    """F5. Rollout error against horizon, without and with the latent-consistency loss."""
    panels = [(_load("stage7_summary.json"), "Without a consistency loss:\neveryone drifts"),
              (_load("stage10_summary.json"), "With one:\nthe isometric operator stays flat")]
    panels = [(s, t) for s, t in panels if s]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, 2.9), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    xs = [3, 10, 20]
    for ax, (s, title) in zip(axes, panels):
        for a in ORDER:
            pa = s["per_arm"].get(a)
            if not pa:
                continue
            ys = [pa["indist"], pa["h10"], pa["h20"]]
            ax.plot(xs, ys, marker="o", ms=4.5, lw=2.0 if a == "proposed" else 1.0,
                    color=_color(a), label=ARM_LABEL[a] if ax is axes[0] else None,
                    zorder=3 if a == "proposed" else 2,
                    alpha=1.0 if a in ("proposed", "proposed_mlp_gs") else 0.7)
        p = s["per_arm"].get("proposed")
        if p:
            ax.annotate(f"{p['h20']:.4f}", (20, p["h20"]), textcoords="offset points",
                        xytext=(6, -2), fontsize=7.4, color=OURS, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xticks(xs, ["3\n(trained)", "10", "20"])
        ax.set_xlim(2.4, 23)
        ax.set_xlabel("rollout length (actions)")
        ax.set_title(title, fontsize=8.5, loc="left", pad=6)
        ax.grid(axis="y", color=GRID, lw=0.6)
        _clean(ax)
    axes[0].set_ylabel("error (log scale)")
    axes[0].spines["left"].set_visible(True)
    axes[0].tick_params(axis="y", length=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    _save(fig, "f5_rollout_horizon")


def fig_guidance():
    """F6. What happens to generated images as guidance strength is turned up."""
    s = _load("stage9_summary.json")
    if not s:
        return
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    ws = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    series = [("film_cfg", BASE, "FiLM + classifier-free guidance"),
              ("cfilm_gp", OURS, "Complex FiLM, powering the condition")]
    for arm, col, lab in series:
        ys = [s["per_arm"][arm][f"ood_w{w}"] for w in ws]
        ax.plot(ws, ys, marker="o", ms=5, lw=2.0 if arm == "cfilm_gp" else 1.3,
                color=col, label=lab, zorder=3 if arm == "cfilm_gp" else 2)
        ax.annotate(f"{ys[-1] / ys[0]:.0f}× worse", (ws[-1], ys[-1]),
                    textcoords="offset points", xytext=(-4, 9), fontsize=7.4,
                    color=col, ha="right",
                    fontweight="bold" if arm == "cfilm_gp" else "normal")
    ax.set_yscale("log")
    ax.set_xlabel("guidance strength")
    ax.set_ylabel("error against the target image (log)")
    ax.set_title("Both distort as guidance rises; powering distorts less", fontsize=9, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    _clean(ax)
    ax.spines["left"].set_visible(True)
    ax.tick_params(axis="y", length=3)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "f6_guidance_distortion")


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


if __name__ == "__main__":
    fig_gap()
    fig_synthetic()
    fig_length()
    fig_mechanism()
    fig_horizon()
    fig_guidance()
