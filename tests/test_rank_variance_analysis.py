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

    def test_masked_layers_can_leave_layer_zero_full(self):
        import torch
        torch.manual_seed(4)
        model = MODULE.MLAGPT(
            vocab_size=31, d_model=24, n_layers=2, n_heads=2,
            d_head=8, d_rope=4, d_c=12, max_len=8,
        ).eval()
        idx = torch.randint(0, 31, (1, 6))
        positions = torch.tensor([2])
        masks = torch.zeros(2, 1, 12)
        baseline = MODULE.forward_with_layer_masks(model, idx)
        no_active = MODULE.forward_with_layer_masks(
            model, idx, channel_masks=masks, probe_positions=positions,
            masked_layers=(),
        )
        torch.testing.assert_close(no_active, baseline)
        downstream_only = MODULE.forward_with_layer_masks(
            model, idx, channel_masks=masks, probe_positions=positions,
            masked_layers=(1,),
        )
        self.assertFalse(torch.allclose(downstream_only, baseline))


if __name__ == "__main__":
    unittest.main()
