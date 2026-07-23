"""Data-split tests (R2): TRAIN / OOD-VAL / OOD-TEST disjoint; OOD are unseen compositions."""

import unittest

try:
    import torch

    from conditional_operators import data
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestSplits(unittest.TestCase):
    def setUp(self):
        self.s = data.make_splits()

    def test_splits_disjoint(self):
        tr, va, te = set(self.s.train), set(self.s.ood_val), set(self.s.ood_test)
        self.assertEqual(tr & va, set())
        self.assertEqual(tr & te, set())
        self.assertEqual(va & te, set())

    def test_ood_are_unseen_pairs(self):
        # Every OOD condition is a length-2 composition never present in TRAIN.
        for c in self.s.ood_val + self.s.ood_test:
            self.assertEqual(len(c), 2)
            self.assertNotIn(c, self.s.train)

    def test_train_has_all_singletons(self):
        singles = {frozenset({i}) for i in range(data.K)}
        self.assertTrue(singles.issubset(set(self.s.train)))

    def test_partition_covers_all_pairs(self):
        import itertools
        all_pairs = {frozenset(p) for p in itertools.combinations(range(data.K), 2)}
        got = {c for c in self.s.all_conditions() if len(c) == 2}
        self.assertEqual(got, all_pairs)

    def test_splits_are_deterministic(self):
        self.assertEqual(self.s.ood_test, data.make_splits().ood_test)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestGroundTruth(unittest.TestCase):
    def test_rotation_preserves_norm(self):
        # Block rotations are orthogonal -> ||M x|| == ||x||.
        g = torch.Generator().manual_seed(0)
        _, x, y = data.sample_batch((frozenset({0, 3}),), 64, g)
        self.assertTrue(torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=1e-5))

    def test_commuting_composition(self):
        # Disjoint-plane primitives commute: M({0,1}) == M({0})M({1}) == M({1})M({0}).
        m01 = data.transform_matrix(frozenset({0, 1}))
        a = data.transform_matrix(frozenset({0})) @ data.transform_matrix(frozenset({1}))
        b = data.transform_matrix(frozenset({1})) @ data.transform_matrix(frozenset({0}))
        self.assertTrue(torch.allclose(m01, a, atol=1e-6))
        self.assertTrue(torch.allclose(m01, b, atol=1e-6))

    def test_identity_condition(self):
        self.assertTrue(torch.allclose(data.transform_matrix(frozenset()), torch.eye(data.D)))


if __name__ == "__main__":
    unittest.main()
