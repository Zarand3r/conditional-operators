"""The partial-conditioning harness, the exactness of the power, and the metrics.

The load-bearing test is that p(x|c) really does have 64 distinct members. The whole reason this
harness exists is that stages 6 and 9 had a point-mass conditional and so could not measure
diversity at all.
"""

import math
import unittest

try:
    import torch

    from conditional_operators import guidance2 as G
    from conditional_operators import stage6
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestConditional(unittest.TestCase):
    def test_the_condition_does_not_determine_the_image(self):
        """The point of the harness: one condition, 64 genuinely different images."""
        data = G.Data()
        bank = G.free_bank(data, G.CONDITIONS[0])
        self.assertEqual(bank.shape[0], G.N_FREE)
        flat = bank.flatten(1)
        pair = torch.cdist(flat, flat)
        off = pair + torch.eye(G.N_FREE, device=pair.device) * 1e9
        self.assertGreater(off.min().item(), 0.0,
                           "two 'distinct' positions produced identical images")

    def test_position_is_absent_from_the_condition(self):
        """If position leaked into the condition vector the conditional would collapse."""
        self.assertEqual(G.PCDIM, 5)
        a = G.cond_vec([(1, 2, 3)])
        b = G.cond_vec([(1, 2, 3)])
        self.assertTrue(torch.equal(a, b))
        # same condition, different positions -> different images, identical condition vectors
        data = G.Data()
        i1 = G.images_for(data, [(1, 2, 3)], [0], [0])
        i2 = G.images_for(data, [(1, 2, 3)], [7], [7])
        self.assertGreater((i1 - i2).abs().max().item(), 0.5)

    def test_training_pairs_spread_over_the_free_positions(self):
        data = G.Data()
        g = torch.Generator().manual_seed(0)
        c, x = G.sample_batch(data, 256, g)
        self.assertEqual(c.shape, (256, G.PCDIM))
        self.assertGreater(len(torch.unique(x.flatten(1), dim=0)), 32,
                           "training targets are not spread over free positions")


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestExactPower(unittest.TestCase):
    def test_powering_is_exact_on_both_channels(self):
        """psi-linear heads mean alpha scales the algebra coordinates, not an approximation."""
        m = G.PsiLinearDiT()
        with torch.no_grad():
            m.Smag.weight.normal_(0, 0.05)
            m.TH.weight.normal_(0, 0.3)
            c = torch.randn(4, G.PCDIM)
            psi = m.c_mlp(c)
            for alpha in (2.0, 3.5):
                s1, t1 = m.Smag(psi * alpha), m.TH(psi * alpha)
                s2, t2 = m.Smag(psi) * alpha, m.TH(psi) * alpha
                self.assertLess((s1 - s2).abs().max().item(), 1e-5)
                self.assertLess((t1 - t2).abs().max().item(), 1e-5)

    def test_alpha_one_is_the_plain_conditional(self):
        m = G.PsiLinearDiT().eval()
        x, t, c = torch.randn(2, 1, 64, 64), torch.zeros(2, dtype=torch.long), torch.randn(2, G.PCDIM)
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(x, t, c), m(x, t, c, alpha=1.0), atol=1e-6))

    def test_zero_condition_is_the_identity_operator_at_init(self):
        m = G.PsiLinearDiT().eval()
        with torch.no_grad():
            psi = m.c_mlp(torch.zeros(2, G.PCDIM))
            self.assertLess(m.Smag(psi).abs().max().item(), 1e-6)
            self.assertLess(m.TH(psi).abs().max().item(), 1e-6)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.bank = G.free_bank(G.Data(), G.CONDITIONS[0])

    def test_perfect_uniform_samples_score_perfectly(self):
        s = G.score(self.bank.clone(), self.bank)
        self.assertAlmostEqual(s["fidelity"], 0.0, places=6)
        self.assertAlmostEqual(s["coverage"], 1.0, places=6)
        self.assertAlmostEqual(s["entropy"], 1.0, places=5)
        self.assertAlmostEqual(s["tv_from_uniform"], 0.0, places=5)

    def test_total_collapse_scores_as_no_diversity(self):
        """Every sample identical: fidelity can still be perfect, diversity must not be."""
        collapsed = self.bank[0:1].repeat(G.N_FREE, 1, 1, 1)
        s = G.score(collapsed, self.bank)
        self.assertAlmostEqual(s["fidelity"], 0.0, places=6)
        self.assertAlmostEqual(s["coverage"], 1.0 / G.N_FREE, places=5)
        self.assertAlmostEqual(s["entropy"], 0.0, places=5)

    def test_a_perfect_sampler_scores_as_perfect_relative_to_ideal(self):
        """The bug this guards against: with K=N=64 a flawless uniform sampler covers only 63.5%
        of the cells, so measuring collapse against 1.0 leaves almost no headroom and a saturated
        number gets compared against a threshold. Metrics are reported against `ideal_stats`."""
        for k in (64, 512):
            ideal = G.ideal_stats(k, G.N_FREE)
            self.assertAlmostEqual(ideal["coverage"], 1 - (1 - 1 / G.N_FREE) ** k, places=2)
            g = torch.Generator().manual_seed(11)
            which = torch.randint(G.N_FREE, (k,), generator=g)
            samples = self.bank[which]
            s = G.score(samples, self.bank)
            self.assertAlmostEqual(s["coverage"] / ideal["coverage"], 1.0, delta=0.1)

    def test_a_collapsed_sampler_scores_far_below_ideal(self):
        for k in (64, 512):
            ideal = G.ideal_stats(k, G.N_FREE)
            s = G.score(self.bank[0:1].repeat(k, 1, 1, 1), self.bank)
            self.assertLess(s["coverage"] / ideal["coverage"], 0.1)

    def test_the_two_axes_are_independent(self):
        """Fidelity measures distance to the support, diversity measures spread across it."""
        noisy = self.bank + torch.randn_like(self.bank) * 0.5
        s = G.score(noisy, self.bank)
        self.assertGreater(s["fidelity"], 0.0, "noise must cost fidelity")
        self.assertGreater(s["coverage"], 0.5, "noise must not, by itself, cost coverage")


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestNFE(unittest.TestCase):
    def test_cfg_costs_two_evaluations_per_step_and_group_power_one(self):
        """The speed claim, verified by counting rather than asserted."""
        film = G.FilmDiT().to(G.DEVICE).eval()
        psi = G.PsiLinearDiT().to(G.DEVICE).eval()
        c = G.cond_vec([G.CONDITIONS[0]] * 2)
        noise = torch.randn(2, 1, 64, 64, device=G.DEVICE)
        with torch.no_grad():
            _, n_plain = G.ddim(film, c, noise, w=1.0, steps=5)
            _, n_cfg = G.ddim(film, c, noise, w=3.0, steps=5)
            _, n_gp = G.ddim(psi, c, noise, alpha=3.0, steps=5)
        self.assertEqual(n_plain, 5)
        self.assertEqual(n_cfg, 10)
        self.assertEqual(n_gp, 5)


if __name__ == "__main__":
    unittest.main()
