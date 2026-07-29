"""The relaxed operators must buy expressiveness without spending the guarantee.

The whole claim of this project is that composition is exact *for any weights*, not approximately
or after training. If a relaxation breaks that, it is a different method and the paper's central
argument does not cover it. These tests are run with deliberately large random weights, because
exactness that only holds near initialization is not exactness.
"""

import unittest

try:
    import torch

    from conditional_operators import improved as I
    from conditional_operators.stage4 import ARM_CLASSES, DC, DZ
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _armed(name, scale=0.25, seed=0):
    """An arm with real weights, and beta zeroed since it is a shift rather than part of T."""
    torch.manual_seed(seed)
    arm = ARM_CLASSES[name](dc=DC)
    with torch.no_grad():
        for p in arm.parameters():
            p.normal_(0, scale)
        arm.beta.weight.zero_(); arm.beta.bias.zero_()
    return arm


def _apply(arm, c, z):
    return arm.op(arm.enc(c), c, z)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestGuaranteesSurvive(unittest.TestCase):
    def test_composition_is_still_exact(self):
        """T(c1+c2) = T(c1)T(c2), for arbitrary weights, which is the entire claim."""
        for name in I.NEW_ARMS:
            arm = _armed(name)
            z = torch.randn(32, DZ)
            c1, c2 = torch.randn(32, DC) * 0.3, torch.randn(32, DC) * 0.3
            both = _apply(arm, c1 + c2, z)
            seq = _apply(arm, c1, _apply(arm, c2, z))
            self.assertLess((both - seq).abs().max().item(), 1e-4, name)

    def test_composition_holds_at_large_weights_while_the_clamp_is_inactive(self):
        """Exactness that only held near initialization would not be exactness."""
        for name in I.NEW_ARMS:
            arm = _armed(name, scale=0.5, seed=3)
            z = torch.randn(16, DZ)
            c1, c2 = torch.randn(16, DC) * 0.5, torch.randn(16, DC) * 0.5
            if hasattr(arm, "S"):
                self.assertLess(arm.S(c1 + c2).abs().max().item(), I.SCLAMP,
                                "this case is meant to keep the clamp inactive")
            both = _apply(arm, c1 + c2, z)
            seq = _apply(arm, c1, _apply(arm, c2, z))
            self.assertLess((both - seq).abs().max().item() / z.abs().max().item(), 1e-3, name)

    def test_the_pure_rotation_relaxation_is_exact_unconditionally(self):
        """A rotation is bounded by nature, so nothing nonlinear ever touches the algebra."""
        for scale in (0.25, 0.5, 0.8, 1.2):
            arm = _armed("proposed_conj", scale=scale, seed=3)
            z = torch.randn(16, DZ)
            c1, c2 = torch.randn(16, DC) * 0.5, torch.randn(16, DC) * 0.5
            both = _apply(arm, c1 + c2, z)
            seq = _apply(arm, c1, _apply(arm, c2, z))
            self.assertLess((both - seq).abs().max().item() / z.abs().max().item(), 1e-3,
                            f"conjugated rotation should be exact at weight scale {scale}")

    def test_the_magnitude_clamp_is_what_costs_exactness(self):
        """Documented cost of touching the magnitude channel, not a bug: a clamp is nonlinear,
        so clamp(s1+s2) != clamp(s1)+clamp(s2) and composition fails once it engages."""
        arm = _armed("proposed_scaled", scale=0.8, seed=3)
        z = torch.randn(16, DZ)
        c1, c2 = torch.randn(16, DC) * 0.5, torch.randn(16, DC) * 0.5
        self.assertGreater(arm.S(c1 + c2).abs().max().item(), I.SCLAMP,
                           "this case is meant to drive the clamp active")
        both = _apply(arm, c1 + c2, z)
        seq = _apply(arm, c1, _apply(arm, c2, z))
        self.assertGreater((both - seq).abs().max().item() / z.abs().max().item(), 1.0,
                           "if this no longer breaks, the documented caveat is stale")

    def test_identity_at_zero_condition(self):
        for name in I.NEW_ARMS:
            arm = _armed(name)
            z = torch.randn(32, DZ)
            self.assertLess((_apply(arm, torch.zeros(32, DC), z) - z).abs().max().item(), 1e-5,
                            name)

    def test_powering_is_exact(self):
        """T(3c) must equal applying T(c) three times: the group-power property."""
        for name in I.NEW_ARMS:
            arm = _armed(name)
            z = torch.randn(32, DZ)
            c = torch.randn(32, DC) * 0.2
            direct = _apply(arm, 3 * c, z)
            step = z
            for _ in range(3):
                step = _apply(arm, c, step)
            self.assertLess((direct - step).abs().max().item() / z.abs().max().item(), 1e-4, name)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestTheRelaxationIsReal(unittest.TestCase):
    def test_the_original_operator_preserves_norms(self):
        """The baseline this is trying to fix: a strict isometry, which cannot rescale features."""
        arm = _armed("proposed")
        z = torch.randn(64, DZ)
        ratio = _apply(arm, torch.randn(64, DC) * 0.3, z).norm(dim=1) / z.norm(dim=1)
        self.assertLess((ratio - 1.0).abs().max().item(), 1e-4,
                        "the unrelaxed operator should be an isometry")

    def test_the_relaxed_operators_do_not(self):
        for name in I.NEW_ARMS:
            arm = _armed(name)
            z = torch.randn(64, DZ)
            ratio = _apply(arm, torch.randn(64, DC) * 0.3, z).norm(dim=1) / z.norm(dim=1)
            self.assertGreater((ratio - 1.0).abs().max().item(), 0.01,
                               f"{name} is still an isometry, so it buys nothing")


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestBudget(unittest.TestCase):
    def test_all_relaxations_stay_under_the_film_ceiling(self):
        """A relaxation that wins by spending more has not won. Stage 2's erratum was exactly this."""
        film = ARM_CLASSES["film"](dc=DC)
        for name in I.NEW_ARMS:
            arm = ARM_CLASSES[name](dc=DC)
            self.assertLessEqual(arm.flops() / film.flops(), 1.20, name)

    def test_parameter_count_stays_within_the_smallest_unstructured_arm(self):
        smallest = min(ARM_CLASSES[a](dc=DC).n_params() for a in ("hypernet", "dynamic_linear"))
        for name in I.NEW_ARMS:
            self.assertLessEqual(ARM_CLASSES[name](dc=DC).n_params(), 1.05 * smallest, name)


if __name__ == "__main__":
    unittest.main()
