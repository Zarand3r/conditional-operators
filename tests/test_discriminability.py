"""The pre-flight screen must correctly separate a task worth running from one that is not.

Tested against two synthetic tasks with known answers rather than against a real benchmark, so
the screen's own logic is what is under test.
"""

import unittest

try:
    import torch
    from torch import nn

    from conditional_operators.discriminability import Screen, Task, screen
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


if __name__ == "__main__":
    unittest.main()
