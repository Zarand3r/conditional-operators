"""Stage-4 invariants: arm budgets, Lie identity, split disjointness. Dataset-dependent tests skip
if datasets/dsprites.npz is absent; all others run on CPU."""

import unittest
from pathlib import Path

try:
    import torch

    from conditional_operators import stage4 as s4
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

HAVE_DATA = HAVE_TORCH and (Path(__file__).resolve().parent.parent / "datasets" / "dsprites.npz").exists()


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestArms4(unittest.TestCase):
    def test_ac4_budget(self):
        film = s4.FiLM4().flops()
        prop = s4.Lie4()
        self.assertLessEqual(prop.flops(), 1.20 * film)
        min_unstruct = min(s4.Hypernet4().n_params(), s4.DynLin4().n_params())
        self.assertLessEqual(prop.n_params(), 1.05 * min_unstruct)

    def test_ablation_over_budget_and_not_gated(self):
        self.assertGreater(s4.MLPGS4().flops(), 1.20 * s4.FiLM4().flops())
        self.assertNotIn("proposed_mlp_gs", s4.GATE_ARMS)

    def test_identity_init_all_arms(self):
        # cond_layernorm excluded: it normalizes by design, so it is never identity at init
        # (same exception as Stage-1's test_all_arms_identity_init_except_layernorm).
        z = torch.randn(4, s4.DZ)
        d = torch.randn(4, s4.DC)
        for name, cls in s4.ARM_CLASSES.items():
            if name == "cond_layernorm":
                continue
            torch.manual_seed(0)
            a = cls()
            out = a(d, z)
            self.assertLess((out - z).abs().max().item(), 1e-5, f"{name} not identity at init")

    def test_lie_structural_identity_at_zero_delta(self):
        # T(0)=I even with trained weights — the Lie arm's structural property.
        lie = s4.Lie4()
        with torch.no_grad():
            lie.P.skew.normal_(0, 0.4)
            lie.W.weight.normal_(0, 0.5)
        z = torch.randn(4, s4.DZ)
        self.assertLess((lie.op(None, torch.zeros(4, s4.DC), z) - z).abs().max().item(), 1e-5)

    def test_split_types_disjoint(self):
        train = set(s4.TRAIN_TYPES)
        self.assertEqual(train & set(s4.VAL_PAIRS), set())
        self.assertEqual(train & set(s4.TEST_PAIRS), set())
        self.assertEqual(set(s4.VAL_PAIRS) & set(s4.TEST_PAIRS), set())
        # all 6 two-factor types are covered across the three splits
        two = {t for t in train if len(t) == 2} | set(s4.VAL_PAIRS) | set(s4.TEST_PAIRS)
        self.assertEqual(len(two), 6)


@unittest.skipUnless(HAVE_DATA, "datasets/dsprites.npz not present")
class TestData4(unittest.TestCase):
    def test_zero_delta_is_identity_pair(self):
        data = s4.Data()
        g = torch.Generator().manual_seed(0)
        x1, d, x2 = data.sample(((),), 16, g)
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(bool((d == 0).all()))

    def test_delta_moves_to_ground_truth(self):
        # A pure +posX delta must yield a different image with identical pixel mass (same sprite).
        data = s4.Data()
        g = torch.Generator().manual_seed(1)
        x1, d, x2 = data.sample(((2,),), 32, g)
        moved = (x1 != x2).any(dim=(1, 2, 3))
        self.assertGreater(moved.float().mean().item(), 0.9)  # rare clamp-to-same allowed
        self.assertTrue(torch.allclose(x1.sum(dim=(1, 2, 3)), x2.sum(dim=(1, 2, 3)), rtol=0.2))


if __name__ == "__main__":
    unittest.main()
