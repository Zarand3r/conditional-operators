"""The audio-effects conditioners, rewritten for this field's channel widths.

Two properties are load-bearing. Causality, because an effect model that peeks at future samples
is not modelling an effect. And exact phase composition, because that is the entire reason Complex
FiLM is the candidate rather than just another head.
"""

import unittest

try:
    import torch

    from conditional_operators.neuralfx import CONDS, TCN
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestConditioners(unittest.TestCase):
    def test_every_arm_starts_as_the_identity(self):
        """Zero-init means training starts from the unconditioned model, for all arms equally."""
        x = torch.randn(4, 32, 64)
        c = torch.randn(4, 2)
        for name, cls in CONDS.items():
            torch.manual_seed(0)
            out = cls(2, 32)(x, c)
            self.assertLess((out - x).abs().max().item(), 1e-6, f"{name} is not identity at init")

    def test_complex_film_phase_composes_exactly(self):
        """phase(c1+c2) = phase(c1) + phase(c2), for any weights: the property being claimed."""
        torch.manual_seed(0)
        arm = CONDS["cfilm"](2, 32)
        with torch.no_grad():
            arm.phase.weight.normal_(0, 0.5)
            c1, c2 = torch.randn(16, 2) * 0.4, torch.randn(16, 2) * 0.4
            both = arm.phase(c1 + c2)
            summed = arm.phase(c1) + arm.phase(c2)
            self.assertLess((both - summed).abs().max().item(), 1e-5)

    def test_complex_film_is_cheaper_than_the_incumbent(self):
        """The claim is better AND cheaper; if this flips, the claim changes."""
        film = TCN("film").cond_params()
        cfilm = TCN("cfilm").cond_params()
        self.assertLess(cfilm, film)

    def test_the_hypernetwork_outweighs_the_backbone_it_conditions(self):
        """Recorded because the published literature reports hypernetworks winning here, and at
        this width one is 2.4x the model it modulates -- so any such win is not budget-fair."""
        m = TCN("hyper")
        self.assertGreater(m.cond_params(), m.backbone_params())

    def test_magnitude_is_expressive_and_phase_is_linear_in_the_condition(self):
        """The content/composition split: the magnitude head sees the encoder, the phase head
        sees the raw condition. Swapping these would undo the mechanism."""
        arm = CONDS["cfilm"](2, 32)
        self.assertEqual(arm.phase.in_features, 2)                  # raw condition
        self.assertEqual(arm.mag.in_features, arm.enc[-1].out_features)   # encoded


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestBackbone(unittest.TestCase):
    def test_the_model_is_causal(self):
        """Changing a sample must never alter any earlier output. An effect model that sees the
        future is not modelling an effect, and the error would be invisible in the loss."""
        for name in CONDS:
            torch.manual_seed(0)
            m = TCN(name, ch=8, blocks=3, kernel=5).eval()
            with torch.no_grad():
                for p in m.parameters():
                    p.normal_(0, 0.1)
                x = torch.randn(1, 1, 256)
                c = torch.randn(1, 2)
                a = m(x, c)
                x2 = x.clone(); x2[0, 0, 200] += 5.0
                b = m(x2, c)
            self.assertLess((a[..., :200] - b[..., :200]).abs().max().item(), 1e-5,
                            f"{name}: output before sample 200 changed when 200 changed")
            self.assertGreater((a[..., 200:] - b[..., 200:]).abs().max().item(), 1e-4,
                               f"{name}: the change had no effect at all, so the test is vacuous")

    def test_shapes_round_trip(self):
        m = TCN("cfilm", ch=8, blocks=3)
        y = m(torch.randn(2, 1, 512), torch.randn(2, 2))
        self.assertEqual(y.shape, (2, 1, 512))

    def test_receptive_field_covers_compressor_timescales(self):
        """An LA-2A's attack and release run tens to hundreds of ms; the model must see that far
        back or no conditioner can help."""
        m = TCN("cfilm")
        self.assertGreater(m.receptive / 44100 * 1000, 100.0)


if __name__ == "__main__":
    unittest.main()
