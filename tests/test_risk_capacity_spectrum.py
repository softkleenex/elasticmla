import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "risk_v5", ROOT / "experiments" / "analyze_risk_capacity_spectrum.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RiskCapacitySpectrumTests(unittest.TestCase):
    def test_tail_counts_include_mean_and_max_and_are_nested(self):
        self.assertEqual(MODULE.normalize_tail_counts([8, 4], 16), [16, 8, 4, 1])

    def test_upper_tail_mean_matches_discrete_expected_shortfall(self):
        values = [-2.0, 1.0, 3.0, 4.0]
        self.assertEqual(MODULE.upper_tail_mean(values, 4), 1.5)
        self.assertEqual(MODULE.upper_tail_mean(values, 2), 3.5)
        self.assertEqual(MODULE.upper_tail_mean(values, 1), 4.0)

    def test_risk_safe_rates_are_ordered_without_rank_curve_monotonicity(self):
        ranks = [1, 2, 3]
        # Each row is one retained rate and columns are future offsets. Rank-loss
        # curves are deliberately nonmonotone at some offsets.
        deltas = np.array([
            [0.00, 0.05, 0.70, -0.20],
            [0.00, 0.12, 0.25, -0.10],
            [0.00, 0.00, 0.00, 0.00],
        ])
        safe = []
        for tail_count in (4, 2, 1):
            curve = [MODULE.upper_tail_mean(row, tail_count) for row in deltas]
            safe.append(MODULE.suffix_all_satisfy_r_star(curve, ranks, 0.10))
        self.assertTrue(all(a <= b for a, b in zip(safe, safe[1:])))

    def test_float32_mean_and_max_endpoints_match_v4_aggregators(self):
        values = np.asarray([0.10000001, -0.3, 0.7, 0.2], dtype=np.float32)
        self.assertEqual(MODULE.upper_tail_mean(values, len(values)), float(values.mean()))
        self.assertEqual(MODULE.upper_tail_mean(values, 1), float(values.max()))

    def test_no_suffix_safe_rank_raises_instead_of_falsely_returning_full(self):
        with self.assertRaises(ValueError):
            MODULE.suffix_all_satisfy_r_star([0.4, 0.3, 0.2], [1, 2, 3], 0.1)

    def test_invalid_tail_count_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.upper_tail_mean([1.0, 2.0], 0)


if __name__ == "__main__":
    unittest.main()
