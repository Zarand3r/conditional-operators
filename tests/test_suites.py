"""The suite registry is the paper-to-code bridge; these tests keep it honest.

A published repo must not claim a result whose spec, code, or log is missing.
"""

import unittest
from pathlib import Path

from conditional_operators.suites import ROOT, SUITES, SUPPORTING, command, verdict_of

ALL = SUITES + SUPPORTING


class TestSuiteRegistry(unittest.TestCase):
    def test_every_suite_has_its_preregistration(self):
        for s in ALL:
            self.assertTrue((ROOT / "docs" / "specs" / f"{s.spec}.md").exists(),
                            f"{s.label}: missing spec {s.spec}.md")

    def test_every_suite_has_its_module(self):
        for s in ALL:
            self.assertTrue((ROOT / "conditional_operators" / f"{s.module}.py").exists(),
                            f"{s.label}: missing module {s.module}.py")

    def test_every_verdict_is_readable_or_the_experiment_is_pending(self):
        """A registered experiment that has not run yet reports 'not run'; anything else must be
        a real verdict. This catches a summary that exists but is corrupt, and a claim made for
        an experiment whose results were never committed."""
        for s in ALL:
            v = verdict_of(s)
            self.assertIn(v, {"confirmed", "kill", "unfair", "blocked", "invalid", "not run"},
                          f"{s.label}: unreadable verdict {v!r}")
            if v != "not run":
                self.assertTrue((ROOT / "results" / f"{s.summary}.json").exists())

    def test_completed_experiments_outnumber_pending_ones(self):
        """A sanity check on the registry as a whole: if most entries are pending, something has
        been registered speculatively rather than run."""
        pending = [s.label for s in ALL if verdict_of(s) == "not run"]
        self.assertLess(len(pending), len(ALL) / 2, f"too many pending: {pending}")

    def test_labels_are_unique(self):
        labels = [s.label for s in ALL]
        self.assertEqual(len(labels), len(set(labels)))

    def test_command_is_runnable_form(self):
        for s in ALL:
            self.assertTrue(command(s).startswith("python -m conditional_operators."))


if __name__ == "__main__":
    unittest.main()
