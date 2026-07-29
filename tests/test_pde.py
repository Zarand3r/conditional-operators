"""The PDE harness: solver physics, split hygiene, and budget.

The solver tests are the load-bearing ones. Everything downstream is a claim about conditioning,
and a claim about conditioning on top of wrong physics is worth nothing.
"""

import unittest

try:
    import torch

    from conditional_operators import pde as P
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestSolver(unittest.TestCase):
    def setUp(self):
        self.s = P.Solver()
        self.g = torch.Generator().manual_seed(0)

    def test_diffusion_matches_the_analytic_solution(self):
        """At small amplitude advection is negligible and each mode must decay as exp(-nu k^2 t)."""
        w = 1e-3 * P.initial_field(1, self.g)
        p = {"nu": 5e-3, "alpha": 0.0, "force": 0.0, "vx": 0.0}
        got = self.s.run(w, 0.5 / 200, 200, p)
        _, _, k2, _, _ = P._wavenumbers(P.DEVICE)
        want = torch.fft.ifft2(torch.fft.fft2(w) * torch.exp(-p["nu"] * k2 * 0.5)).real
        self.assertLess(((got - want).norm() / want.norm()).item(), 1e-3)

    def test_inviscid_flow_conserves_enstrophy(self):
        w = P.initial_field(1, self.g)
        p = {"nu": 0.0, "alpha": 0.0, "force": 0.0, "vx": 0.0}
        e0 = (w ** 2).sum().item()
        eT = (self.s.run(w, 1e-3, 300, p) ** 2).sum().item()
        self.assertLess(abs(eT - e0) / e0, 0.05)

    def test_advection_is_a_rigid_translation(self):
        """Background flow must move the field without deforming it."""
        w = 1e-4 * P.initial_field(1, self.g)
        p = {"nu": 0.0, "alpha": 0.0, "force": 0.0, "vx": 1.0}
        shifted = self.s.run(w, P.LBOX / P.N / 8, 8, p)
        rolled = torch.roll(w, shifts=1, dims=1)
        self.assertLess(((shifted - rolled).norm() / rolled.norm()).item(), 0.05)

    def test_decay_is_monotone_in_viscosity(self):
        w = P.initial_field(1, self.g)
        e = [(self.s.run(w, 2e-3, 200,
                         {"nu": nu, "alpha": 0.0, "force": 0.0, "vx": 0.0}) ** 2).sum().item()
             for nu in (1e-3, 4e-3, 1.6e-2)]
        self.assertGreater(e[0], e[1])
        self.assertGreater(e[1], e[2])

    def test_solver_is_deterministic(self):
        w = P.initial_field(1, self.g)
        self.assertTrue(torch.equal(self.s.run(w, 1e-3, 20, P.BASE),
                                    self.s.run(w, 1e-3, 20, P.BASE)))

    def test_each_axis_actually_changes_the_outcome(self):
        """A parameter that does not move the field cannot be conditioned on."""
        w = P.initial_field(4, self.g)
        base = self.s.run(w, P.DT, 200, P.params_from([0, 0, 0, 0]))
        for ax in range(P.DC):
            d = [0] * P.DC; d[ax] = 1
            moved = self.s.run(w, P.DT, 200, P.params_from(d))
            rel = ((moved - base).norm() / base.norm()).item()
            self.assertGreater(rel, 0.01, f"axis {P.AXES[ax]} barely moves the outcome")


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestSplits(unittest.TestCase):
    def test_pairs_are_disjoint_and_complete(self):
        tr, va, te = set(P.TRAIN_PAIRS), set(P.VAL_PAIRS), set(P.TEST_PAIRS)
        self.assertEqual(tr & va, set())
        self.assertEqual(tr & te, set())
        self.assertEqual(va & te, set())
        self.assertEqual(tr | va | te, set(P.ALL_PAIRS))

    def test_training_never_sees_a_held_out_pair(self):
        train = set(P._settings("train"))
        for kind in ("val", "test"):
            self.assertEqual(train & set(P._settings(kind)), set())

    def test_every_training_delta_moves_at_most_two_axes(self):
        for d in P._settings("train"):
            self.assertLessEqual(sum(1 for v in d if v != 0), 2)

    def test_held_out_deltas_move_exactly_two_axes(self):
        for kind in ("val", "test"):
            for d in P._settings(kind):
                self.assertEqual(sum(1 for v in d if v != 0), 2)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestBudget(unittest.TestCase):
    def test_proposed_stays_inside_the_registered_budget(self):
        from conditional_operators.stage4 import ARM_CLASSES
        arms = {a: ARM_CLASSES[a](dc=P.DC) for a in P.GATE}
        self.assertLessEqual(arms["proposed"].flops() / arms["film"].flops(), 1.20)
        smallest = min(arms[a].n_params() for a in ("hypernet", "dynamic_linear"))
        self.assertLessEqual(arms["proposed"].n_params(), 1.05 * smallest)


if __name__ == "__main__":
    unittest.main()
