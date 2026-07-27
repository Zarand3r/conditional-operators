"""Verdict decision-logic tests — one per acceptance criterion in docs/specs/STAGE1_SPEC.md."""

import unittest

from conditional_operators.verdict import Arm, ArmResult, Verdict, decide


def _arm(arm, ood, indist=None, *, n_diverged=0, params=1000, flops=1000, reads=1):
    """Build an ArmResult; ood/indist may be a single float (broadcast to 10 seeds)."""
    if isinstance(ood, (int, float)):
        ood = tuple([float(ood)] * 10)
    if indist is None:
        indist = tuple([0.05] * len(ood))
    elif isinstance(indist, (int, float)):
        indist = tuple([float(indist)] * len(ood))
    return ArmResult(arm, tuple(ood), tuple(indist), n_diverged, params, flops, reads)


def _base_confirmed():
    """A full six-arm result set engineered to satisfy AC-1..AC-5 fairly."""
    proposed_ood = tuple(0.10 + 0.005 * i for i in range(10))   # mean ~0.1225
    best_unstruct_ood = tuple(0.30 + 0.005 * i for i in range(10))  # mean ~0.3225, separated
    return {
        Arm.FILM: _arm(Arm.FILM, 0.80, flops=1000),               # floor: worst OOD
        Arm.CONCAT_MLP: _arm(Arm.CONCAT_MLP, 0.50),
        Arm.COND_LAYERNORM: _arm(Arm.COND_LAYERNORM, 0.45),
        Arm.HYPERNET: _arm(Arm.HYPERNET, best_unstruct_ood, indist=0.05, params=1000),
        Arm.DYNAMIC_LINEAR: _arm(Arm.DYNAMIC_LINEAR, 0.40, indist=0.05, params=1000),
        # proposed: fair budget (params <=1.05x1000, flops <=1.20x1000).
        Arm.PROPOSED: _arm(Arm.PROPOSED, proposed_ood, indist=0.05, params=1050, flops=1200),
    }


class TestVerdict(unittest.TestCase):
    def test_confirmed_happy_path(self):
        r = decide(_base_confirmed())
        self.assertEqual(r.verdict, Verdict.CONFIRMED)
        # best unstructured = lowest-OOD competitor = HYPERNET (~0.32 < DYNAMIC_LINEAR 0.40).
        self.assertEqual(r.best_unstructured, Arm.HYPERNET)
        self.assertTrue(all(r.criteria.values()))

    def test_ac1_floor_required(self):
        # Make proposed worse than FiLM on OOD -> AC-1 fails -> KILL.
        res = _base_confirmed()
        res[Arm.FILM] = _arm(Arm.FILM, 0.05)  # FiLM now better than proposed
        r = decide(res)
        self.assertEqual(r.verdict, Verdict.KILL)
        self.assertFalse(r.criteria["AC-1"])

    def test_ac2_margin_fail_is_kill(self):
        # proposed only ~5% better than best unstructured -> AC-2 (needs 20%) fails.
        res = _base_confirmed()
        res[Arm.HYPERNET] = _arm(Arm.HYPERNET, 0.128, params=1000)   # proposed mean ~0.1225
        res[Arm.DYNAMIC_LINEAR] = _arm(Arm.DYNAMIC_LINEAR, 0.130, params=1000)
        r = decide(res)
        self.assertEqual(r.verdict, Verdict.KILL)
        self.assertFalse(r.criteria["AC-2"])
        self.assertLess(r.margin_observed, 0.20)

    def test_ac3_significance_fail_is_kill(self):
        # Big margin but overlapping distributions -> p not significant -> AC-3 fails.
        res = _base_confirmed()
        # proposed and unstruct interleave heavily; means far apart via one outlier only.
        res[Arm.PROPOSED] = _arm(Arm.PROPOSED, (0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6),
                                 params=1050, flops=1200)
        res[Arm.HYPERNET] = _arm(Arm.HYPERNET, (0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.2, 0.5, 0.9),
                                 params=1000)
        res[Arm.DYNAMIC_LINEAR] = _arm(Arm.DYNAMIC_LINEAR, 0.9, params=1000)
        r = decide(res)
        self.assertEqual(r.verdict, Verdict.KILL)
        self.assertFalse(r.criteria["AC-3"])

    def test_ac4_unfair_blocks_confirm(self):
        # Proposed spends 2x FiLM FLOPs -> UNFAIR, not CONFIRMED even if AC-1..AC-5 pass.
        res = _base_confirmed()
        res[Arm.PROPOSED] = _arm(Arm.PROPOSED, tuple(0.10 + 0.005 * i for i in range(10)),
                                 indist=0.05, params=1050, flops=2000)
        r = decide(res)
        self.assertEqual(r.verdict, Verdict.UNFAIR)

    def test_ac4_param_bloat_is_unfair(self):
        res = _base_confirmed()
        res[Arm.PROPOSED] = _arm(Arm.PROPOSED, tuple(0.10 + 0.005 * i for i in range(10)),
                                 indist=0.05, params=5000, flops=1200)
        r = decide(res)
        self.assertEqual(r.verdict, Verdict.UNFAIR)

    def test_ac5_indist_regression_is_kill(self):
        # Proposed wins OOD but its in-dist MSE is 2x best unstructured -> AC-5 fails.
        res = _base_confirmed()
        res[Arm.PROPOSED] = _arm(Arm.PROPOSED, tuple(0.10 + 0.005 * i for i in range(10)),
                                 indist=0.20, params=1050, flops=1200)  # unstruct indist 0.05
        r = decide(res)
        self.assertEqual(r.verdict, Verdict.KILL)
        self.assertFalse(r.criteria["AC-5"])

    def test_ac6_double_read_is_invalid(self):
        res = _base_confirmed()
        res[Arm.PROPOSED] = _arm(Arm.PROPOSED, tuple(0.10 + 0.005 * i for i in range(10)),
                                 indist=0.05, params=1050, flops=1200, reads=2)
        r = decide(res)
        self.assertEqual(r.verdict, Verdict.INVALID)

    def test_leakage_is_invalid(self):
        r = decide(_base_confirmed(), leakage=True)
        self.assertEqual(r.verdict, Verdict.INVALID)

    def test_insufficient_seeds_blocks(self):
        res = _base_confirmed()
        short = tuple([0.12] * 7)  # only 7 non-diverged seeds
        res[Arm.PROPOSED] = ArmResult(Arm.PROPOSED, short, tuple([0.05] * 7),
                                      n_diverged=3, params=1050, flops=1200, ood_test_reads=1)
        r = decide(res)
        self.assertEqual(r.verdict, Verdict.BLOCKED)

    def test_divergence_rate_reported(self):
        res = _base_confirmed()
        # 10 non-diverged + 5 diverged -> rate 1/3, reported even on a clean verdict.
        base = res[Arm.FILM]
        res[Arm.FILM] = ArmResult(Arm.FILM, base.ood_test_mse, base.indist_test_mse,
                                  n_diverged=5, params=base.params, flops=base.flops,
                                  ood_test_reads=1)
        r = decide(res)
        self.assertAlmostEqual(r.divergence_rates["film"], 5 / 15)

    def test_missing_arm_raises(self):
        res = _base_confirmed()
        del res[Arm.CONCAT_MLP]
        with self.assertRaises(ValueError):
            decide(res)


if __name__ == "__main__":
    unittest.main()
