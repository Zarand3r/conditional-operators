"""Cross-check the stdlib Mann-Whitney U p-value against scipy (AC-3 trustworthiness).

The verdict gate must not hinge on a subtly-wrong significance test (CLAUDE.md non-negotiable).
scipy is available only in the venv; this test is skipped if scipy is absent.
"""

import unittest

from conditional_operators.verdict import mann_whitney_u

try:
    from scipy.stats import mannwhitneyu as _scipy_mwu
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class TestMWUvsScipy(unittest.TestCase):
    def _check(self, a, b):
        _, p_ours = mann_whitney_u(a, b, alternative="less")
        p_scipy = _scipy_mwu(a, b, alternative="less", method="asymptotic").pvalue
        # Same normal approximation + tie/continuity correction -> agree to ~1e-9.
        self.assertAlmostEqual(p_ours, float(p_scipy), places=6)

    def test_separated(self):
        self._check(tuple(0.10 + 0.005 * i for i in range(10)),
                    tuple(0.30 + 0.005 * i for i in range(10)))

    def test_overlapping(self):
        self._check((0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6),
                    (0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.9))

    def test_with_ties(self):
        self._check((0.2, 0.2, 0.3, 0.3, 0.4), (0.3, 0.3, 0.5, 0.5, 0.6))


if __name__ == "__main__":
    unittest.main()
