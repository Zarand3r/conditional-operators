"""The pre-flight screen must correctly separate a task worth running from one that is not.

Tested against two synthetic tasks with known answers rather than against a real benchmark, so
the screen's own logic is what is under test.
"""

import unittest

try:
    import torch
    from torch import nn

    from conditional_operators.discriminability import Screen, Task, _fit, screen
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


if HAVE_TORCH:
    D = 16

    class _Toy(nn.Module):
        """Maps (x, c) -> logits. `use_c` controls whether the condition is used at all."""

        def __init__(self, use_c: bool):
            super().__init__()
            self.use_c = use_c
            self.f = nn.Linear(D + (D if use_c else 0), D)

        def forward(self, x, c):
            flat = x.flatten(1)
            inp = torch.cat([flat, c], 1) if self.use_c else flat
            return self.f(inp).view_as(x)

    def _make_task(name, gap_factor, arms=("film", "proposed")):
        """A task where the target is the input plus a condition-dependent shift.

        gap_factor scales how much harder the held-out conditions are, letting a test dial the
        compositional gap to a known value.
        """
        def sample(n, g, held=False):
            x = torch.rand(n, 1, 4, 4, generator=g)
            c = torch.randn(n, D, generator=g) * (gap_factor if held else 1.0)
            y = (x + c.view(n, 1, 4, 4).mean(dim=(2, 3), keepdim=True) * 0.3).clamp(0, 1)
            return x, c, y
        return Task(name=name,
                    sample_train=lambda n, g: sample(n, g, held=False),
                    sample_heldout=lambda n, g: sample(n, g, held=True),
                    build=lambda arm: _Toy(use_c=(arm != "blind")),
                    arms=arms)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestScreen(unittest.TestCase):
    def test_reports_the_three_deciding_numbers(self):
        s = screen(_make_task("toy", 1.0), steps=50)
        for v in (s.identity_mse, s.fitted_mse, s.conditioning_share, s.compositional_gap):
            self.assertTrue(v == v and v != float("inf"), "screen produced a non-finite number")
        self.assertIn("toy", s.report())

    def test_a_task_with_no_compositional_gap_is_rejected(self):
        """Held-out conditions drawn from the training distribution: nothing to generalise to."""
        s = screen(_make_task("no-gap", 1.0), steps=200)
        self.assertLess(s.compositional_gap, 1.5)
        self.assertFalse(s.discriminative)
        self.assertIn("little compositional failure", s.report())

    def test_a_model_that_cannot_fit_is_rejected(self):
        """If the conditioning share is low the task cannot discriminate, whatever the gap."""
        s = Screen(task="unfittable", identity_mse=0.10, mean_mse=0.11, fitted_mse=0.09,
                   heldout_mse=0.30, conditioning_share=0.10, compositional_gap=3.3,
                   separation=0.0)
        self.assertFalse(s.discriminative)
        self.assertIn("not the conditioning's to fix", s.report())

    def test_a_healthy_task_is_accepted(self):
        s = Screen(task="healthy", identity_mse=0.10, mean_mse=0.12, fitted_mse=0.01,
                   heldout_mse=0.025, conditioning_share=0.90, compositional_gap=2.5,
                   separation=0.3)
        self.assertTrue(s.discriminative)
        self.assertIn("WORTH A SWEEP", s.report())

    def test_camera_and_dsprites_thresholds_match_what_we_observed(self):
        """The thresholds must classify the two real cases the way the sweeps actually went."""
        camera = Screen("camera", 0.0878, 0.1004, 0.0400, 0.0503, 1 - 0.0400 / 0.0878,
                        0.0503 / 0.0400, 0.017)
        self.assertFalse(camera.discriminative)          # 1.26x gap: no room, as observed
        dsprites = Screen("dSprites", 0.0500, 0.0600, 0.0014, 0.0030, 1 - 0.0014 / 0.0500,
                          0.0030 / 0.0014, 0.538)
        self.assertTrue(dsprites.discriminative)         # fits well, 2.1x gap, arms separate

    def test_the_yardstick_is_the_best_arm_not_the_first(self):
        """The bug this guards against threw away a good task because FiLM could not fit it.

        Frozen-latent editing: FiLM removes 15% of the identity baseline and would be scored as
        unfittable, while a hypernet removes 88% and shows a 2.1x compositional gap. The task is
        fine; the reference arm was the wrong instrument. Arms are ordered worst-first here so a
        regression to "use arms[0]" fails this test.
        """
        task = _make_task("frozen-latent", 3.0, arms=("blind", "proposed"))
        blind_fit, _ = _fit(task, "blind", 300, 128, 1e-3, 0)
        strong_fit, _ = _fit(task, "proposed", 300, 128, 1e-3, 0)
        self.assertGreater(blind_fit, strong_fit, "the arm that sees c must fit better")
        s = screen(task, steps=300)
        self.assertAlmostEqual(s.fitted_mse, strong_fit, places=6)

    def test_separation_spans_all_arms_tried(self):
        s = Screen("spread", 0.10, 0.12, 0.01, 0.025, 0.90, 2.5, 0.0)
        self.assertTrue(s.discriminative)                # separation is reported, never gates



@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestBiasMatch(unittest.TestCase):
    """The check with the best track record: does the structured arm's in-distribution fit
    predict the outcome? Across seven completed comparisons it separated every win from every
    loss with no overlap, so these observed values are pinned as a regression."""

    OBSERVED = [
        ("dSprites", 0.986, True), ("dSprites+shape", 1.005, True),
        ("synthetic PDE (cfilm)", 0.938, True),
        ("synthetic PDE (proposed)", 1.118, False), ("3D Shapes", 1.457, False),
        ("PDEBench", 2.381, False), ("latent-edit", 2.433, False),
    ]

    def test_the_threshold_reproduces_every_observed_outcome(self):
        from conditional_operators.discriminability import MAX_BIAS_MISMATCH
        for name, ratio, won in self.OBSERVED:
            predicted = ratio <= MAX_BIAS_MISMATCH
            self.assertEqual(predicted, won,
                             f"{name}: fit ratio {ratio} predicts "
                             f"{'win' if predicted else 'loss'}, but it was "
                             f"{'won' if won else 'lost'}")

    def test_the_wins_and_losses_do_not_overlap(self):
        wins = [r for _, r, w in self.OBSERVED if w]
        losses = [r for _, r, w in self.OBSERVED if not w]
        self.assertLess(max(wins), min(losses),
                        "the separation this check rests on has gone; revisit the threshold")

    def test_report_flags_a_mismatch(self):
        s = Screen("mismatched", 0.10, 0.12, 0.01, 0.03, 0.90, 3.0, 0.4, bias_match=2.4)
        self.assertIn("MISMATCH", s.report())
        self.assertIn("every task in this project with a ratio this high was lost", s.report())

    def test_report_omits_the_line_when_unmeasured(self):
        s = Screen("nostructured", 0.10, 0.12, 0.01, 0.03, 0.90, 3.0, 0.4)
        self.assertNotIn("inductive bias", s.report())

if __name__ == "__main__":
    unittest.main()
