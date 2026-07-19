"""Statistics tests (AC-3 primitives): Mann-Whitney U + Cliff's delta."""

import unittest

from conditional_operators.verdict import cliffs_delta, mann_whitney_u


class TestMannWhitneyU(unittest.TestCase):
    def test_fully_separated_is_significant(self):
        # proposed clearly below unstructured -> tiny left-tail p.
        a = tuple(0.10 + 0.01 * i for i in range(10))   # 0.10..0.19
        b = tuple(0.30 + 0.01 * i for i in range(10))   # 0.30..0.39
        u, p = mann_whitney_u(a, b, alternative="less")
        self.assertEqual(u, 0.0)            # every a-rank below every b-rank
        self.assertLess(p, 0.01)

    def test_identical_samples_not_significant(self):
        a = b = tuple([0.2] * 10)
        _, p = mann_whitney_u(a, b, alternative="less")
        self.assertGreaterEqual(p, 0.5)     # no evidence a < b

    def test_reversed_direction_not_significant(self):
        # a ABOVE b -> H1(a<b) should not fire.
        a = tuple(0.30 + 0.01 * i for i in range(10))
        b = tuple(0.10 + 0.01 * i for i in range(10))
        _, p = mann_whitney_u(a, b, alternative="less")
        self.assertGreater(p, 0.99)

    def test_only_less_supported(self):
        with self.assertRaises(ValueError):
            mann_whitney_u((1.0,), (2.0,), alternative="greater")

    def test_empty_sample_raises(self):
        with self.assertRaises(ValueError):
            mann_whitney_u((), (1.0,))


class TestCliffsDelta(unittest.TestCase):
    def test_all_smaller_is_minus_one(self):
        a = tuple(range(10))
        b = tuple(range(100, 110))
        self.assertEqual(cliffs_delta(a, b), -1.0)

    def test_all_larger_is_plus_one(self):
        a = tuple(range(100, 110))
        b = tuple(range(10))
        self.assertEqual(cliffs_delta(a, b), 1.0)

    def test_identical_is_zero(self):
        a = b = tuple([5.0] * 6)
        self.assertEqual(cliffs_delta(a, b), 0.0)

    def test_large_effect_threshold(self):
        # 8 of 10 below, 2 above -> delta favoring a, magnitude beyond 0.474.
        a = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9)
        b = tuple([0.5] * 10)
        self.assertLessEqual(cliffs_delta(a, b), -0.474)


if __name__ == "__main__":
    unittest.main()
