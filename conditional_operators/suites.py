"""Registry mapping the paper's suite labels to the code and results that produced them.

The paper (docs/paper/paper.tex) labels experiments S1-S8; the modules are named by the order
they were run. This module is the bridge, and the single entry point for replication:

    python -m conditional_operators.suites --list        # what exists, and its verdict
    python -m conditional_operators.suites S3            # print the command for suite S3
    python -m conditional_operators.suites S3 --run      # run it (GPU; hours)

Every suite writes an append-only log and a summary JSON under results/; the paper's tables and
figures are generated from those files by gen_tables.py and figures.py, never by hand.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Suite:
    label: str            # as printed in the paper
    question: str         # what it tests
    module: str           # python -m conditional_operators.<module>
    args: str             # arguments for the decision run
    summary: str          # results/<summary>.json holding the registered verdict
    spec: str             # docs/specs/<spec>.md: the pre-registration
    gpu: bool             # needs a GPU (dSprites/3D-Shapes/diffusion suites)


SUITES: tuple[Suite, ...] = (
    Suite("S1", "Does structure beat unstructured conditioning at equal budget?",
          "sweep", "10 4000", "summary", "STAGE1_SPEC", False),
    Suite("S2", "Does exact composition survive a hidden basis, within budget?",
          "stage3", "10 4000", "stage3_summary", "STAGE3_SPEC", False),
    Suite("S3", "Does the advantage survive a learned latent on real images?",
          "stage4", "10 12000", "stage4_summary", "STAGE4_SPEC", True),
    Suite("S3b", "Does it survive a categorical (non-group) factor?",
          "stage4b", "10 12000", "stage4b_summary", "STAGE4_SPEC", True),
    Suite("S4", "Does it hold on a second dataset (3D Shapes, RGB)?",
          "stage5", "10 12000", "stage5_summary", "STAGE4_SPEC", True),
    Suite("S5", "Can rotation conditioning specify image content (diffusion)?",
          "stage6", "10 40000", "stage6_summary", "STAGE6_SPEC", True),
    Suite("S6", "Does exact composition prevent world-model rollout drift?",
          "stage7", "10 6000", "stage7_summary", "STAGE7_SPEC", True),
    Suite("S6'", "Does a latent-consistency loss unlock rollout guarantees?",
          "stage10", "10 6000", "stage10_summary", "STAGE10_SPEC", True),
    Suite("S7a", "Complex FiLM in the transformation role",
          "stage8", "a 10 12000", "stage8a_summary", "STAGE8_SPEC", True),
    Suite("S7b", "Complex FiLM in the content role (non-inferiority)",
          "stage8", "b 10 40000", "stage8b_summary", "STAGE8_SPEC", True),
    Suite("S8", "Does condition powering beat classifier-free guidance?",
          "stage9", "10 40000", "stage9_summary", "STAGE9_SPEC", True),
)

# Suites that inform the paper without being one of its numbered rows.
SUPPORTING: tuple[Suite, ...] = (
    Suite("aux-erratum", "Dense-basis control; failed the FLOP ceiling (erratum in the paper)",
          "stage2", "10 4000", "stage2_summary", "STAGE2_SPEC", False),
    Suite("aux-contraction", "Is a fixed contraction rate the rollout-stability knob?",
          "stage11", "10 6000", "stage11_summary", "STAGE11_SPEC", True),
)

BY_LABEL = {s.label: s for s in SUITES + SUPPORTING}


def verdict_of(suite: Suite) -> str:
    """Read the registered verdict from the committed summary, or report it as not yet run."""
    path = RESULTS / f"{suite.summary}.json"
    if not path.exists():
        return "not run"
    data = json.loads(path.read_text())
    return data.get("final_verdict") or data.get("verdict") or "unknown"


def command(suite: Suite) -> str:
    return f"python -m conditional_operators.{suite.module} {suite.args}".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("label", nargs="?", help="suite label, e.g. S3 (see --list)")
    ap.add_argument("--list", action="store_true", help="list all suites with their verdicts")
    ap.add_argument("--run", action="store_true", help="execute the suite instead of printing it")
    args = ap.parse_args()

    if args.list or not args.label:
        print(f"{'suite':<17}{'verdict':<12}{'gpu':<5}{'module':<10}question")
        for suite in SUITES + SUPPORTING:
            print(f"{suite.label:<17}{verdict_of(suite):<12}{'yes' if suite.gpu else 'no':<5}"
                  f"{suite.module:<10}{suite.question}")
        print("\nRun one with:  python -m conditional_operators.suites <label> --run")
        return

    if args.label not in BY_LABEL:
        sys.exit(f"unknown suite {args.label!r}; use --list")
    suite = BY_LABEL[args.label]
    cmd = command(suite)
    if not args.run:
        print(f"suite   {suite.label}: {suite.question}")
        print(f"spec    docs/specs/{suite.spec}.md  (pre-registered success criteria)")
        print(f"verdict {verdict_of(suite)}  (results/{suite.summary}.json)")
        print(f"command {cmd}")
        return
    print(f"running {suite.label}: {cmd}", flush=True)
    subprocess.run([sys.executable, "-m", f"conditional_operators.{suite.module}",
                    *suite.args.split()], check=True)


if __name__ == "__main__":
    main()
