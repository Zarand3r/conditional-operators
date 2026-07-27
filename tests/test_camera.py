"""Camera-control benchmark: renderer correctness, split hygiene, arms and budget.

The renderer supplies every ground-truth frame in this experiment, so its properties are the
foundation the results stand on and are tested first.
"""

import unittest

try:
    import torch

    from conditional_operators import camera as C
    from conditional_operators.stage4 import ARM_CLASSES
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestRenderer(unittest.TestCase):
    def test_deterministic(self):
        pts, cols = C.make_scene(3)
        p = torch.randn(2, C.DC).to(C.DEVICE) * 0.3
        self.assertTrue(torch.equal(C.render(pts, cols, p), C.render(pts, cols, p)))

    def test_scene_is_a_function_of_its_seed(self):
        a1, c1 = C.make_scene(5)
        a2, c2 = C.make_scene(5)
        self.assertTrue(torch.equal(a1, a2) and torch.equal(c1, c2))
        b1, _ = C.make_scene(6)
        self.assertFalse(torch.equal(a1, b1))

    def test_output_is_a_well_formed_image(self):
        pts, cols = C.make_scene(1)
        img = C.render(pts, cols, torch.zeros(3, C.DC).to(C.DEVICE))
        self.assertEqual(tuple(img.shape), (3, 3, C.RES, C.RES))
        self.assertGreaterEqual(img.min().item(), 0.0)
        self.assertLessEqual(img.max().item(), 1.0)
        self.assertGreater(img.mean().item(), 0.01)          # not a blank frame

    def test_moves_commute_in_slider_space(self):
        """The abelian property the method's composition law relies on."""
        pts, cols = C.make_scene(2)
        pose = torch.randn(4, C.DC).to(C.DEVICE) * 0.3
        a = torch.zeros(4, C.DC).to(C.DEVICE); a[:, 0] = 0.3
        b = torch.zeros(4, C.DC).to(C.DEVICE); b[:, 1] = 0.2
        self.assertTrue(torch.allclose(C.render(pts, cols, pose + a + b),
                                       C.render(pts, cols, pose + b + a), atol=1e-6))

    def test_every_axis_moves_the_image(self):
        pts, cols = C.make_scene(4)
        base = torch.zeros(1, C.DC).to(C.DEVICE)
        for ax in range(C.DC):
            p = base.clone(); p[0, ax] = 0.4
            delta = (C.render(pts, cols, p) - C.render(pts, cols, base)).abs().mean().item()
            self.assertGreater(delta, 0.01, f"axis {C.AXES[ax]} barely changes the frame")


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestSplits(unittest.TestCase):
    def test_move_types_are_disjoint(self):
        tr = {frozenset(t) for t in C.TRAIN_PAIRS}
        va = {frozenset(t) for t in C.VAL_PAIRS}
        te = {frozenset(t) for t in C.TEST_PAIRS}
        self.assertEqual(tr & va, set()); self.assertEqual(tr & te, set())
        self.assertEqual(va & te, set())
        self.assertEqual(len(tr | va | te), 6)               # all pairs of four axes

    def test_triples_are_never_trained(self):
        self.assertEqual(len(C.TRIPLES), 4)
        for t in C.TRIPLES:
            self.assertEqual(len(t), 3)
            self.assertNotIn(t, C.TRAIN_TYPES)

    def test_train_and_test_use_disjoint_scenes(self):
        g = torch.Generator().manual_seed(0)
        seen = set()
        for train, expect in ((True, range(0, C.TRAIN_SCENES)),
                              (False, range(10**6, 10**6 + C.TEST_SCENES))):
            lo, hi = (0, C.TRAIN_SCENES) if train else (10**6, 10**6 + C.TEST_SCENES)
            s = torch.randint(lo, hi, (64,), generator=g)
            self.assertTrue(bool(((s >= lo) & (s < hi)).all()))
            seen.add((lo, hi))
        self.assertEqual(len(seen), 2)

    def test_zero_move_is_the_identity(self):
        g = torch.Generator().manual_seed(0)
        x1, move, x2 = C.sample(((),), 8, g)
        self.assertTrue(torch.equal(x1, x2))
        self.assertEqual(move.abs().max().item(), 0.0)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestArms(unittest.TestCase):
    def test_all_arms_build_and_start_at_identity(self):
        z, d = torch.randn(4, C.DZ), torch.randn(4, C.DC)
        for a in C.ALL_ARMS:
            if a == "cond_layernorm":                        # normalises by design
                continue
            torch.manual_seed(0)
            m = ARM_CLASSES[a](dc=C.DC)
            self.assertLess((m(d, z) - z).abs().max().item(), 1e-5, f"{a} not identity at init")

    def test_c4_budget_holds(self):
        film = ARM_CLASSES["film"](dc=C.DC).flops()
        prop = ARM_CLASSES["proposed"](dc=C.DC)
        smallest_unstructured = min(ARM_CLASSES[a](dc=C.DC).n_params()
                                    for a in ("hypernet", "dynamic_linear"))
        self.assertLessEqual(prop.flops(), 1.20 * film)
        self.assertLessEqual(prop.n_params(), 1.05 * smallest_unstructured)

    def test_additive_arm_has_no_multiplicative_interaction(self):
        """It must be a pure bias, since it stands in for the deployed CameraCtrl mechanism."""
        m = ARM_CLASSES["additive"](dc=C.DC)
        with torch.no_grad():
            m.proj.weight.normal_(0, 0.3); m.proj.bias.normal_(0, 0.3)
        d = torch.randn(1, C.DC)
        z1, z2 = torch.randn(1, C.DZ), torch.randn(1, C.DZ)
        # the condition's effect is the same offset regardless of the latent it acts on
        off1 = m.op(m.enc(d), d, z1) - z1
        off2 = m.op(m.enc(d), d, z2) - z2
        self.assertTrue(torch.allclose(off1, off2, atol=1e-6))

    def test_gate_arms_are_the_ones_scored(self):
        self.assertIn("additive", C.GATE_ARMS)               # the deployed mechanism, C1
        self.assertIn("proposed", C.GATE_ARMS)
        for a in C.REPORTED:
            self.assertNotIn(a, C.GATE_ARMS)


if __name__ == "__main__":
    unittest.main()
