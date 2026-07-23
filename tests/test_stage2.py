"""Stage-2 invariants: de-aligned data properties + learned-basis operator (STAGE2_SPEC AC-7)."""

import unittest

try:
    import torch

    from conditional_operators import stage2
    from conditional_operators.arms import FiLM
    from conditional_operators.data import D, K
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestDealignedData(unittest.TestCase):
    def test_basis_is_orthonormal(self):
        self.assertLess((stage2.B.T @ stage2.B - torch.eye(D)).norm().item(), 1e-4)

    def test_transform_is_orthogonal(self):
        m = stage2.dealigned_matrix(frozenset({1, 4}))
        self.assertLess((m.T @ m - torch.eye(D)).norm().item(), 1e-4)

    def test_not_coordinate_aligned(self):
        # A single primitive under B is NOT a 2x2 coordinate block -> off-block mass is large.
        m = stage2.dealigned_matrix(frozenset({0}))
        off = m.clone()
        off[0:2, 0:2] = 0.0  # zero the coordinate block primitive 0 would occupy if aligned
        self.assertGreater(off.abs().sum().item(), 1.0)

    def test_composition_still_commutes(self):
        # B R_i B^T and B R_j B^T commute because R_i, R_j do (disjoint planes).
        a = stage2.dealigned_matrix(frozenset({0})) @ stage2.dealigned_matrix(frozenset({5}))
        b = stage2.dealigned_matrix(frozenset({0, 5}))
        self.assertLess((a - b).norm().item(), 1e-4)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestLearnedBasisOperator(unittest.TestCase):
    def test_identity_init(self):
        m = stage2.build2("proposed", 0)
        x = torch.randn(8, D); c = torch.zeros(8, K); c[:, 2] = 1
        self.assertLess((m(c, x) - x).abs().max().item(), 1e-5)

    def test_P_orthogonal_when_trained(self):
        m = stage2.build2("proposed", 0)
        with torch.no_grad():
            m.skew.normal_(0, 0.5)
        P = m._P()
        # float32 matrix_exp on 128x128 -> orthogonal to ~1e-4.
        self.assertLess((P.T @ P - torch.eye(D)).norm().item(), 5e-4)

    def test_bounded_spectrum(self):
        m = stage2.build2("proposed", 0)
        with torch.no_grad():
            m.skew.normal_(0, 0.5)
            m.angles.weight.normal_(0, 1.0); m.angles.bias.normal_(0, 1.0)
        worst = 0.0
        for _ in range(32):
            c = torch.bernoulli(torch.full((1, K), 0.5))
            worst = max(worst, torch.linalg.svdvals(m.dense_operator(c))[0].item())
        self.assertLessEqual(worst, 1.02)  # orthogonal P,Q + bounded low-rank

    def test_ac4_budget_violated_erratum(self):
        # ERRATUM 2026-07-21: the dense-P basis costs 4*D*D per sample (two [1,D]@[D,D] matmuls);
        # the corrected counter puts this arm at ~1.52x FiLM — OVER the 1.20x ceiling. This test
        # pins the corrected accounting so the undercount can never silently return. The
        # within-budget replacement is Stage-3's GSOrthogonal (see tests/test_stage3.py).
        m = stage2.build2("proposed", 0)
        self.assertGreater(m.flops(), 1.20 * FiLM().flops())
        self.assertAlmostEqual(m.flops() / FiLM().flops(), 1.522, places=3)

    def test_nobasis_ablation_is_identity_operator(self):
        m = stage2.build2("proposed_nobasis", 0)
        self.assertIsNone(getattr(m, "skew", None))
        self.assertTrue(torch.allclose(m._P(), torch.eye(D)))


if __name__ == "__main__":
    unittest.main()
