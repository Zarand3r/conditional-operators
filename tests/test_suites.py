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

    def test_every_suite_has_a_committed_verdict(self):
        for s in ALL:
            self.assertTrue((ROOT / "results" / f"{s.summary}.json").exists(),
                            f"{s.label}: missing results/{s.summary}.json")
            self.assertIn(verdict_of(s),
                          {"confirmed", "kill", "unfair", "blocked", "invalid"},
                          f"{s.label}: unreadable verdict")

    def test_labels_are_unique(self):
        labels = [s.label for s in ALL]
        self.assertEqual(len(labels), len(set(labels)))

    def test_command_is_runnable_form(self):
        for s in ALL:
            self.assertTrue(command(s).startswith("python -m conditional_operators."))


if __name__ == "__main__":
    unittest.main()
