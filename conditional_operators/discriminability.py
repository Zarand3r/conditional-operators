"""Pre-flight screen: can this task tell conditioning mechanisms apart at all?

Run this before committing GPU-hours to a sweep. It exists because the camera-slider experiment
was fully built, pre-registered and calibrated before we noticed that its error was dominated by
scene reconstruction rather than by conditioning, which no conditioning mechanism can improve. A
sweep on such a task returns a null that describes the harness, not the method.

Three numbers decide it:

* **conditioning share** -- how much of the trivial "copy the input" error a trained model
  actually removes. If a model cannot fit its own training distribution, differences between
  conditioning modules are noise on top of a larger failure. dSprites reached 0.0014 against a
  much larger identity baseline; the camera task reached 0.020 and could not separate arms.
* **compositional gap** -- error on held-out condition combinations over error on trained ones.
  If it is near 1.0 the task has no compositional failure for a method to fix, so even a perfect
  operator has nothing to win.
* **observed separation** -- how far apart two deliberately different conditioners land after a
  short run. Near zero here, after the first two look healthy, means something subtler is wrong.

Usage: implement the three callables of `Task` for a new benchmark and call `screen(task)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

# A task is discriminative enough to be worth a sweep when a trained model removes most of the
# identity-baseline error AND held-out combinations are meaningfully harder than trained ones.
MIN_CONDITIONING_SHARE = 0.60
MIN_COMPOSITIONAL_GAP = 1.50


@dataclass
class Task:
    """What a benchmark must provide to be screened."""
    name: str
    sample_train: Callable          # (n, generator) -> (x_in, condition, x_target)
    sample_heldout: Callable        # (n, generator) -> same, with held-out condition combinations
    build: Callable                 # (arm_name) -> nn.Module mapping (x_in, condition) -> logits
    arms: tuple = ("film", "proposed")


@dataclass
class Screen:
    task: str
    identity_mse: float             # predict the input unchanged, ignoring the condition
    mean_mse: float                 # predict the dataset mean, ignoring everything
    fitted_mse: float               # what a trained model actually reaches in distribution
    heldout_mse: float
    conditioning_share: float       # fraction of the identity baseline the model removes
    compositional_gap: float        # heldout / in-distribution
    separation: float               # relative gap between the two arms tried

    @property
    def discriminative(self) -> bool:
        return (self.conditioning_share >= MIN_CONDITIONING_SHARE
                and self.compositional_gap >= MIN_COMPOSITIONAL_GAP)

    def report(self) -> str:
        verdict = "WORTH A SWEEP" if self.discriminative else "NOT DISCRIMINATIVE"
        why = []
        if self.conditioning_share < MIN_CONDITIONING_SHARE:
            why.append(f"the model removes only {self.conditioning_share:.0%} of the identity"
                       f" baseline, so most error is not the conditioning's to fix")
        if self.compositional_gap < MIN_COMPOSITIONAL_GAP:
            why.append(f"held-out combinations are only {self.compositional_gap:.2f}x harder than"
                       f" trained ones, so there is little compositional failure to repair")
        lines = [
            f"{self.task}: {verdict}",
            f"  identity baseline (copy input) : {self.identity_mse:.5f}",
            f"  mean-image baseline            : {self.mean_mse:.5f}",
            f"  fitted, in distribution        : {self.fitted_mse:.5f}",
            f"  fitted, held-out combinations  : {self.heldout_mse:.5f}",
            f"  conditioning share             : {self.conditioning_share:.0%} "
            f"(want >= {MIN_CONDITIONING_SHARE:.0%})",
            f"  compositional gap              : {self.compositional_gap:.2f}x "
            f"(want >= {MIN_COMPOSITIONAL_GAP:.2f}x)",
            f"  separation between two arms    : {self.separation:+.1%}",
        ]
        if why:
            lines.append("  why not: " + "; ".join(why) + ".")
        return "\n".join(lines)


def _baselines(task: Task, n: int, gen) -> tuple[float, float]:
    x1, _, x2 = task.sample_train(n, gen)
    identity = torch.mean((x1 - x2) ** 2).item()
    mean_img = torch.mean((x2.mean(dim=(0, 2, 3), keepdim=True) - x2) ** 2).item()
    return identity, mean_img


def _fit(task: Task, arm: str, steps: int, batch: int, lr: float, seed: int):
    torch.manual_seed(seed)
    m = task.build(arm)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x1, c, x2 = task.sample_train(batch, g)
        loss = torch.mean((torch.sigmoid(m(x1, c)) - x2) ** 2)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"{task.name}/{arm} diverged during screening")
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def ev(sampler):
        eg = torch.Generator().manual_seed(9_999)
        x1, c, x2 = sampler(512, eg)
        return torch.mean((torch.sigmoid(m(x1, c)) - x2) ** 2).item()

    return ev(task.sample_train), ev(task.sample_heldout)


def screen(task: Task, steps: int = 2000, batch: int = 128, lr: float = 1e-3,
           seed: int = 0) -> Screen:
    """Short training run plus trivial baselines. Cheap relative to a full sweep."""
    g = torch.Generator().manual_seed(seed)
    identity, mean_img = _baselines(task, 512, g)
    fits = {a: _fit(task, a, steps, batch, lr, seed) for a in task.arms}
    ref, alt = task.arms[0], task.arms[-1]
    tr, ho = fits[ref]
    return Screen(
        task=task.name, identity_mse=identity, mean_mse=mean_img,
        fitted_mse=tr, heldout_mse=ho,
        conditioning_share=max(0.0, 1.0 - tr / identity) if identity > 0 else 0.0,
        compositional_gap=ho / tr if tr > 0 else float("inf"),
        separation=(fits[ref][1] - fits[alt][1]) / fits[ref][1] if fits[ref][1] > 0 else 0.0,
    )
