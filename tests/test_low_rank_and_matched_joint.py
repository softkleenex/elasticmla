import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "low_rank_joint", ROOT / "experiments" / "analyze_low_rank_and_matched_joint.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ExtendRStarTest(unittest.TestCase):
    def test_stored_above_floor_is_unchanged(self):
        self.assertEqual(mod.extend_r_star(64, np.zeros(6), mod.LOW_RANKS, 0.1), 64)

    def test_floor_extends_to_smallest_suffix_safe_rank(self):
        # ranks 0, 1, 2, 4, 8, 16: rank 1 fails, so the suffix starts at 2.
        curve = np.array([0.0, 0.5, 0.05, 0.05, 0.05, 0.05])
        self.assertEqual(mod.extend_r_star(16, curve, mod.LOW_RANKS, 0.1), 2)

    def test_floor_all_safe_gives_zero(self):
        self.assertEqual(mod.extend_r_star(16, np.zeros(6), mod.LOW_RANKS, 0.1), 0)


class AllocationTest(unittest.TestCase):
    def test_uniform_matches_mean_rate(self):
        out = mod.uniform_ranks({3: 16, 9: 48, 20: 32})
        self.assertEqual(set(out.values()), {32})
        self.assertEqual(sorted(out), [3, 9, 20])

    def test_shuffled_is_a_permutation_and_deterministic(self):
        ranks = {p: r for p, r in zip(range(10), range(0, 100, 10))}
        a, b = mod.shuffled_ranks(ranks, 5), mod.shuffled_ranks(ranks, 5)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a.values()), sorted(ranks.values()))


if __name__ == "__main__":
    unittest.main()
