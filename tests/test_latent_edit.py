"""Latent-space attribute editing: the ground truth, the splits, the budget, and the freezing.

The freezing test is the load-bearing one. The whole point of this harness is that every arm
shares one representation it cannot change, so if gradients leaked into the encoder the
comparison would silently become a comparison of encoders.
"""

import unittest

try:
    import torch

    from conditional_operators import latent_edit as L
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestEdits(unittest.TestCase):
    def test_edit_vector_describes_the_factor_change(self):
        """Every nonzero entry must equal the direction the attribute actually moved."""
        g = torch.Generator().manual_seed(0)
        lat = torch.stack([torch.randint(int(s), (400,), generator=g) for s in L.SIZES6], 1)
        which = torch.randint(len(L.TRAIN_TYPES), (400,), generator=g)
        edit, tgt = L.apply_edits(lat, L.TRAIN_TYPES, which, g)
        for ax, col in enumerate(L.AX_COL):
            size, st = L.AX_MAX[ax], L.STEP[ax]
            moved = edit[:, ax] != 0
            expected = (lat[moved, col] + edit[moved, ax].long() * st) % size
            self.assertTrue(torch.equal(tgt[moved, col], expected),
                            f"axis {L.EDIT_AXES[ax]} moved somewhere other than its edit says")

    def test_untouched_attributes_do_not_move(self):
        g = torch.Generator().manual_seed(1)
        lat = torch.stack([torch.randint(int(s), (400,), generator=g) for s in L.SIZES6], 1)
        which = torch.randint(len(L.TRAIN_TYPES), (400,), generator=g)
        edit, tgt = L.apply_edits(lat, L.TRAIN_TYPES, which, g)
        for ax, col in enumerate(L.AX_COL):
            still = edit[:, ax] == 0
            self.assertTrue(torch.equal(tgt[still, col], lat[still, col]))
        self.assertTrue(torch.equal(tgt[:, 4], lat[:, 4]), "shape is not an edit axis")

    def test_scale_never_leaves_its_range(self):
        """Scale is not circular, so a step is turned around rather than wrapped."""
        g = torch.Generator().manual_seed(2)
        lat = torch.stack([torch.randint(int(s), (600,), generator=g) for s in L.SIZES6], 1)
        which = torch.randint(len(L.TRAIN_TYPES), (600,), generator=g)
        _, tgt = L.apply_edits(lat, L.TRAIN_TYPES, which, g)
        col = L.AX_COL[3]
        self.assertGreaterEqual(int(tgt[:, col].min()), 0)
        self.assertLess(int(tgt[:, col].max()), L.SIZES6[col])

    def test_every_edit_type_changes_exactly_its_own_axes(self):
        g = torch.Generator().manual_seed(3)
        for ti, t in enumerate(L.TRAIN_TYPES):
            lat = torch.stack([torch.randint(int(s), (64,), generator=g) for s in L.SIZES6], 1)
            which = torch.full((64,), ti)
            edit, _ = L.apply_edits(lat, L.TRAIN_TYPES, which, g)
            self.assertEqual(set(torch.nonzero(edit[0]).flatten().tolist()), set(t))


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestSplits(unittest.TestCase):
    def test_train_val_and_test_pairs_are_disjoint(self):
        tr, va, te = set(L.TRAIN_PAIRS), set(L.VAL_PAIRS), set(L.TEST_PAIRS)
        self.assertEqual(tr & va, set())
        self.assertEqual(tr & te, set())
        self.assertEqual(va & te, set())
        self.assertEqual(len(tr | va | te), 10, "all ten attribute pairs must be accounted for")

    def test_triples_are_never_trained(self):
        for t in L.TRIPLES:
            self.assertNotIn(t, L.TRAIN_TYPES)
            self.assertEqual(len(t), 3)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestModel(unittest.TestCase):
    def test_the_encoder_is_frozen_and_shared(self):
        ae = L.AE()
        ae.eval().requires_grad_(False)
        m = L.Editor("proposed", ae)
        trained = {id(p) for p in m.cond.parameters()}
        for p in m.ae.parameters():
            self.assertFalse(p.requires_grad, "a gradient can reach the shared encoder")
            self.assertNotIn(id(p), trained)

        x = torch.rand(4, 3, 64, 64)
        m(x, torch.zeros(4, L.DC)).sum().backward()
        for p in m.ae.parameters():
            self.assertIsNone(p.grad, "the shared encoder accumulated a gradient")

    def test_every_arm_starts_at_the_identity(self):
        """Zero edit must leave the latent alone, before any training.

        cond_layernorm is excluded because it normalizes by design and so is never identity at
        init, the same exception the stage-1 and stage-4 suites carry.
        """
        ae = L.AE().eval().requires_grad_(False)
        x = torch.rand(8, 3, 64, 64)
        with torch.no_grad():
            z = ae.encode(x)
            for arm in L.ALL_ARMS:
                if arm == "cond_layernorm":
                    continue
                out = L.Editor(arm, ae)(x, torch.zeros(8, L.DC))
                self.assertLess(torch.abs(out - z).max().item(), 1e-5, f"{arm} is not identity")

    def test_the_proposed_arm_is_identity_at_zero_even_once_trained(self):
        """T(0)=I is structural, not an initialization: it must survive arbitrary weights."""
        ae = L.AE().eval().requires_grad_(False)
        m = L.Editor("proposed", ae)
        with torch.no_grad():
            for p in m.cond.parameters():
                p.normal_(0, 0.3)
            m.cond.beta.weight.zero_(); m.cond.beta.bias.zero_()   # beta is a shift, not part of T
            x = torch.rand(8, 3, 64, 64)
            out = m(x, torch.zeros(8, L.DC))
            self.assertLess(torch.abs(out - ae.encode(x)).max().item(), 1e-4)

    def test_the_proposed_arm_stays_inside_its_registered_budget(self):
        arms = {a: L.ARM_CLASSES[a](dc=L.DC) for a in L.GATE_ARMS}
        prop, film = arms["proposed"], arms["film"]
        self.assertLessEqual(prop.flops() / film.flops(), 1.20)
        smallest_unstructured = min(arms[a].n_params() for a in ("hypernet", "dynamic_linear"))
        self.assertLessEqual(prop.n_params(), 1.05 * smallest_unstructured)

    def test_standardizing_the_latent_is_invertible(self):
        ae = L.AE().eval().requires_grad_(False)
        with torch.no_grad():
            ae.mu.copy_(torch.randn(L.DZ))
            ae.sd.copy_(torch.rand(L.DZ) + 0.5)
            x = torch.rand(4, 3, 64, 64)
            raw = ae.enc(x)
            self.assertTrue(torch.allclose(ae.encode(x) * ae.sd + ae.mu, raw, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
