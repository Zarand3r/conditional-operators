"""Operator invariant property tests (INV-1..4) + AC-4 structural budget checks.

Requires the torch stack (see .venv). These are the property layer of STAGE1_TEST_PLAN.md.
"""

import unittest

try:
    import torch

    from conditional_operators import arms
    from conditional_operators.data import D, K
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def _random_conditions(n: int) -> "torch.Tensor":
    torch.manual_seed(123)
    return torch.bernoulli(torch.full((n, K), 0.5))


def _train_like(m: arms.Proposed) -> None:
    """Randomize heads to emulate a trained (non-identity) operator."""
    with torch.no_grad():
        for layer in (m.angles, m.gains):
            layer.weight.normal_(0, 1.0)
            layer.bias.normal_(0, 1.0)


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestInvariants(unittest.TestCase):
    def test_inv1_identity_init(self):
        # ||T(c) - I||_F < 1e-5 for every sampled c on the untrained proposed operator.
        m = arms.build("proposed", 0)
        worst = 0.0
        for c in _random_conditions(256):
            T = m.dense_operator(c.unsqueeze(0))
            worst = max(worst, (T - torch.eye(D)).norm().item())
        self.assertLess(worst, 1e-5, f"identity-init violated: {worst}")

    def test_inv2_orthogonality(self):
        # ||Q(c)^T Q(c) - I||_F < 1e-4 for every c, including trained-like weights.
        m = arms.build("proposed", 0)
        _train_like(m)
        worst = 0.0
        for c in _random_conditions(256):
            Q = m.rotation_only(c.unsqueeze(0))
            worst = max(worst, (Q.T @ Q - torch.eye(D)).norm().item())
        self.assertLess(worst, 1e-4, f"orthogonality violated: {worst}")

    def test_inv3_bounded_spectrum(self):
        # sigma_max(T(c)) <= 1 + 1e-2 for every c, including trained-like weights.
        m = arms.build("proposed", 0)
        _train_like(m)
        worst = 0.0
        for c in _random_conditions(256):
            T = m.dense_operator(c.unsqueeze(0))
            worst = max(worst, torch.linalg.svdvals(T)[0].item())
        self.assertLessEqual(worst, 1.01, f"spectrum unbounded: {worst}")

    def test_inv4_composition_error_finite(self):
        # Diagnostic (not a gate): ||T(c2 o c1) - T(c2)T(c1)||_F is finite and >= 0.
        m = arms.build("proposed", 0)
        _train_like(m)
        c1 = torch.zeros(1, K); c1[0, 0] = 1
        c2 = torch.zeros(1, K); c2[0, 3] = 1
        c12 = torch.zeros(1, K); c12[0, 0] = 1; c12[0, 3] = 1
        err = (m.dense_operator(c12) - m.dense_operator(c2) @ m.dense_operator(c1)).norm().item()
        self.assertTrue(err >= 0 and err == err and err != float("inf"))


@unittest.skipUnless(HAVE_TORCH, "torch not installed (run under .venv)")
class TestBudgetAC4(unittest.TestCase):
    def test_proposed_flops_within_120pct_film(self):
        film = arms.FiLM().flops()
        proposed = arms.Proposed().flops()
        self.assertLessEqual(proposed, 1.20 * film)

    def test_proposed_params_within_105pct_smallest_unstructured(self):
        proposed = arms.Proposed().n_params()
        smallest_unstruct = min(arms.Hypernet().n_params(), arms.DynamicLinear().n_params())
        self.assertLessEqual(proposed, 1.05 * smallest_unstruct)

    def test_all_arms_identity_init_except_layernorm(self):
        x = torch.randn(4, D)
        c = torch.zeros(4, K); c[:, 0] = 1
        for name in ("film", "concat_mlp", "hypernet", "dynamic_linear", "proposed"):
            m = arms.build(name, 0)
            self.assertLess((m(c, x) - x).abs().max().item(), 1e-5, f"{name} not identity-init")


if __name__ == "__main__":
    unittest.main()
