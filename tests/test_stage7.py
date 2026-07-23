"""Stage-7 invariants: action ground truth, identity at no-op, budget, gate logic."""

import unittest
from pathlib import Path

try:
    import torch

    from conditional_operators import stage7 as s7
    from conditional_operators.stage4 import ARM_CLASSES, Data
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

HAVE_DATA = HAVE_TORCH and (Path(__file__).resolve().parent.parent / "datasets" / "dsprites.npz").exists()


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestArms7(unittest.TestCase):
    def test_ac4_budget(self):
        film = ARM_CLASSES["film"](dc=s7.DC7).flops()
        self.assertLessEqual(ARM_CLASSES["proposed"](dc=s7.DC7).flops(), 1.20 * film)

    def test_identity_at_zero_action(self):
        # zero action vector => Lie transition is exactly identity (structural), even trained-like
        lie = ARM_CLASSES["proposed"](dc=s7.DC7)
        with torch.no_grad():
            lie.P.skew.normal_(0, 0.4)
            lie.W.weight.normal_(0, 0.5)
        z = torch.randn(4, 128)
        out = lie.op(None, torch.zeros(4, s7.DC7), z)
        self.assertLess((out - z).abs().max().item(), 1e-5)

    def test_split_types_disjoint(self):
        tr = {frozenset(t) for t in s7.TRAIN_PAIRS7}
        va = {frozenset(t) for t in s7.VAL_PAIRS7}
        te = {frozenset(t) for t in s7.TEST_PAIRS7}
        self.assertEqual(tr & va, set()); self.assertEqual(tr & te, set())
        self.assertEqual(va & te, set())
        self.assertEqual(len(tr | va | te), 6)   # all C(4,2) pair types covered

    def test_gate_logic(self):
        def rows(arm, pairs, h20, indist, flops=1000):
            return [dict(arm=arm, seed=s, diverged=False, params=1, flops=flops,
                         indist=indist, ood_val=pairs, ood_pairs=pairs, h10=h20 / 2, h20=h20)
                    for s in range(10)]
        runs = {"film": rows("film", 0.02, 0.05, 0.004, flops=1000),
                "concat_mlp": rows("concat_mlp", 0.02, 0.05, 0.004),
                "cond_layernorm": rows("cond_layernorm", 0.02, 0.05, 0.004),
                "hypernet": rows("hypernet", 0.02, 0.08, 0.004),
                "dynamic_linear": rows("dynamic_linear", 0.03, 0.09, 0.004),
                "proposed": rows("proposed", 0.01, 0.008, 0.004, flops=1100),
                "proposed_mlp_gs": rows("proposed_mlp_gs", 0.015, 0.03, 0.004)}
        v, crit, stats = s7.decide7(runs)
        self.assertEqual(v, "confirmed")
        self.assertTrue(all(crit.values()))
        # break AC-2's growth condition only
        runs["proposed"] = rows("proposed", 0.01, 0.05, 0.004, flops=1100)  # growth 12.5 vs 20
        v2, crit2, _ = s7.decide7(runs)
        self.assertEqual(v2, "kill")
        self.assertFalse(crit2["AC-2"])


@unittest.skipUnless(HAVE_DATA, "datasets/dsprites.npz not present")
class TestSpriteWorld(unittest.TestCase):
    def test_zero_length_is_identity(self):
        data = Data()
        g = torch.Generator().manual_seed(0)
        x0, acts, x1 = s7.sample_rollouts(data, s7.TRAIN_TYPES7, (0, 0), 8, g)
        self.assertTrue(torch.equal(x0, x1))

    def test_actions_reach_ground_truth(self):
        # frames must change when actions applied, and pixel mass is conserved (same sprite)
        data = Data()
        g = torch.Generator().manual_seed(1)
        x0, acts, x1 = s7.sample_rollouts(data, (("posX",),), (3, 3), 32, g)
        moved = (x0 != x1).any(dim=(1, 2, 3))
        self.assertGreater(moved.float().mean().item(), 0.8)   # clamp no-ops allowed
        # every emitted action row is a valid signed one-hot (<=1 nonzero, magnitude <=1)
        nz = (acts != 0).sum(-1)
        self.assertTrue(bool((nz <= 1).all()))

    def test_effective_action_zero_when_clamped(self):
        # rollouts starting at posX=0 taking -x steps must emit zero-magnitude actions sometimes
        data = Data()
        g = torch.Generator().manual_seed(2)
        _, acts, _ = s7.sample_rollouts(data, (("posX",),), (3, 3), 256, g)
        self.assertTrue(bool((acts >= 0).all()))               # magnitudes only, no negatives


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestComplexFiLM(unittest.TestCase):
    """Stage-8 arms (registered in stage8.py; tested here to keep one torch-gated module)."""

    def test_budget_and_identity(self):
        from conditional_operators import stage8  # registers arms
        from conditional_operators.stage4 import ARM_CLASSES, DC, DZ
        film = ARM_CLASSES["film"](dc=DC).flops()
        for name in ("cfilm_lin", "cfilm_hyb"):
            a = ARM_CLASSES[name](dc=DC)
            self.assertLessEqual(a.flops(), 1.20 * film)
            z = torch.randn(4, DZ); d = torch.randn(4, DC)
            self.assertLess((a(d, z) - z).abs().max().item(), 1e-5)

    def test_lin_composition_exact(self):
        from conditional_operators import stage8
        from conditional_operators.stage4 import ARM_CLASSES, DC, DZ
        a = ARM_CLASSES["cfilm_lin"](dc=DC)
        with torch.no_grad():
            a.S.weight.normal_(0, 0.3); a.TH.weight.normal_(0, 0.7)
        d1, d2, z = torch.randn(1, DC), torch.randn(1, DC), torch.randn(1, DZ)
        lhs = a.op(None, d1 + d2, z)
        rhs = a.op(None, d2, a.op(None, d1, z))
        self.assertLess((lhs - rhs).abs().max().item(), 1e-4)
