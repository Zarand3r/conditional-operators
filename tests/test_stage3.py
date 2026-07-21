"""Stage-3 invariants: GS orthogonality, exact Lie compositionality (AC-7), budget, verdict logic."""

import unittest

try:
    import torch

    from conditional_operators import stage3
    from conditional_operators.arms import FiLM
    from conditional_operators.data import D, K
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestGSOrthogonal(unittest.TestCase):
    def _randomized(self):
        m = stage3.GSOrthogonal()
        with torch.no_grad():
            m.skew.normal_(0, 0.5)
        return m

    def test_orthogonal_permutation_at_init(self):
        # At init the Cayley blocks are I but the fixed shuffles still permute: P is a permutation
        # matrix (orthogonal), NOT I. The OPERATOR is still exactly identity because
        # T = P R(0) P^T = P P^T = I — asserted by TestProposedLie.test_identity_init.
        P = stage3.GSOrthogonal().dense()
        self.assertTrue(torch.allclose(P.T @ P, torch.eye(D), atol=1e-6))
        self.assertTrue(torch.all((P == 0) | (P == 1)).item())  # a permutation matrix

    def test_orthogonal_when_trained(self):
        P = self._randomized().dense()
        self.assertLess((P.T @ P - torch.eye(D)).norm().item(), 1e-4)

    def test_apply_t_is_inverse(self):
        m = self._randomized()
        x = torch.randn(8, D)
        self.assertLess((m.apply_t(m.apply(x)) - x).abs().max().item(), 1e-5)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestProposedLie(unittest.TestCase):
    def _randomized(self):
        m = stage3.build3("proposed", 0)
        with torch.no_grad():
            m.W.weight.normal_(0, 0.7)
            m.P.skew.normal_(0, 0.4)
        return m

    def test_identity_init(self):
        m = stage3.build3("proposed", 0)
        x = torch.randn(6, D); c = torch.zeros(6, K); c[:, 3] = 1
        self.assertLess((m(c, x) - x).abs().max().item(), 1e-5)

    def test_ac7_exact_compositionality_any_weights(self):
        # Structural: T(c1+c2) = T(c2) T(c1) for ANY W, P — this is the GRAPE property.
        self.assertLess(stage3.composition_error(self._randomized(), n_pairs=16), stage3.AC7_TOL)

    def test_bounded_spectrum(self):
        # T = P R P^T is exactly orthogonal -> all singular values 1 (INV-3 for free).
        m = self._randomized()
        c = torch.zeros(1, K); c[0, 0] = 1; c[0, 5] = 1
        sv = torch.linalg.svdvals(m.dense_operator(c))
        self.assertLess((sv - 1).abs().max().item(), 1e-4)

    def test_ac4_true_budget(self):
        self.assertLessEqual(stage3.build3("proposed", 0).flops(), 1.20 * FiLM().flops())

    def test_ablation_is_over_budget_and_flagged(self):
        # The MLP-head ablation deliberately exceeds the ceiling; it must never sit in GATE_ARMS.
        self.assertGreater(stage3.build3("proposed_mlp_gs", 0).flops(), 1.20 * FiLM().flops())
        self.assertNotIn("proposed_mlp_gs", stage3.GATE_ARMS)

    def test_triples_are_unseen(self):
        splits = stage3.make_splits()
        tri = stage3.triples()
        self.assertEqual(len(tri), 56)
        self.assertTrue(all(len(t) == 3 for t in tri))
        self.assertEqual(set(tri) & set(splits.all_conditions()), set())


class TestStage3Verdict(unittest.TestCase):
    """Pure-logic verdict composition — runs without torch."""

    def test_confirmed_needs_all(self):
        v, crit = stage3.stage3_verdict("confirmed", 1e-6, 0.001, 0.010, 0.001) \
            if HAVE_TORCH else (None, None)
        if not HAVE_TORCH:
            self.skipTest("stage3 module needs torch to import")
        self.assertEqual(v, "confirmed")
        self.assertTrue(all(crit.values()))

    def test_ac7_failure_kills(self):
        if not HAVE_TORCH: self.skipTest("needs torch import")
        v, crit = stage3.stage3_verdict("confirmed", 1e-2, 0.001, 0.010, 0.001)
        self.assertEqual(v, "kill"); self.assertFalse(crit["AC-7"])

    def test_ac8a_failure_kills(self):
        if not HAVE_TORCH: self.skipTest("needs torch import")
        v, crit = stage3.stage3_verdict("confirmed", 1e-6, 0.009, 0.010, 0.009)  # not <0.5x
        self.assertEqual(v, "kill"); self.assertFalse(crit["AC-8a"])

    def test_ac8b_systematicity_failure_kills(self):
        if not HAVE_TORCH: self.skipTest("needs torch import")
        v, crit = stage3.stage3_verdict("confirmed", 1e-6, 0.004, 0.010, 0.001)  # 4x own pairs
        self.assertEqual(v, "kill"); self.assertFalse(crit["AC-8b"])

    def test_gate_failure_passes_through(self):
        if not HAVE_TORCH: self.skipTest("needs torch import")
        v, _ = stage3.stage3_verdict("unfair", 1e-6, 0.001, 0.010, 0.001)
        self.assertEqual(v, "unfair")


if __name__ == "__main__":
    unittest.main()
