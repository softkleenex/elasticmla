import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rank_v4", ROOT / "experiments" / "analyze_rank_variance_v4.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RankVarianceWindowTests(unittest.TestCase):
    def test_valid_positions_include_last_exact_horizon(self):
        np.testing.assert_array_equal(
            MODULE.valid_probe_positions(8, 3), np.arange(6)
        )

    def test_loss_window_starts_at_source_logit(self):
        window = MODULE.future_loss_slice(2, 3, 8)
        self.assertEqual((window.start, window.stop), (2, 5))

    def test_loss_window_rejects_short_tail(self):
        with self.assertRaises(ValueError):
            MODULE.future_loss_slice(6, 3, 8)

    def test_sampled_sequence_spans_do_not_overlap(self):
        first = MODULE.sample_starts(
            np.random.default_rng(1), 0, 10000, 12, min_separation=257
        )
        second = MODULE.sample_starts(
            np.random.default_rng(2), 0, 10000, 12,
            min_separation=257, excluded=first,
        )
        combined = np.sort(np.concatenate((first, second)))
        self.assertTrue(np.all(np.diff(combined) >= 257))

    def test_default_grid_covers_high_384_channels(self):
        self.assertEqual(
            MODULE.normalize_rank_grid(MODULE.DEFAULT_RANK_GRID, 384)[-4:],
            [288, 320, 352, 384],
        )
        self.assertEqual(
            MODULE.normalize_rank_grid(MODULE.DEFAULT_RANK_GRID, 256)[-1], 256
        )


if __name__ == "__main__":
    unittest.main()
