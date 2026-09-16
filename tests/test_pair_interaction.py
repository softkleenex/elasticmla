import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT

SPEC = importlib.util.spec_from_file_location(
    "pair_interaction", ROOT / "experiments" / "analyze_pair_interaction.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SamplePairsTests(unittest.TestCase):
    def setUp(self):
        self.valid = MODULE._RISK.valid_probe_positions(64, 8)  # 0..56

    def test_pairs_have_the_requested_separation_and_stay_probeable(self):
        rng = np.random.default_rng(0)
        pairs = MODULE.sample_pairs(rng, self.valid, 8, 5)
        self.assertEqual(len(pairs), 5)
        for first, second in pairs:
            self.assertEqual(second - first, 8)
            self.assertIn(first, self.valid)
            self.assertIn(second, self.valid)

    def test_a_separation_too_wide_for_the_window_yields_no_pairs(self):
        rng = np.random.default_rng(0)
        self.assertEqual(len(MODULE.sample_pairs(rng, self.valid, 999, 5)), 0)

    def test_pair_count_is_capped_by_the_available_first_positions(self):
        rng = np.random.default_rng(0)
        pairs = MODULE.sample_pairs(rng, self.valid, 55, 50)
        self.assertGreater(len(pairs), 0)
        self.assertLessEqual(len(pairs), 2)  # only first positions 0 and 1 fit


class ResidualInvariantTests(unittest.TestCase):
    """The mixed difference must vanish where additivity cannot be violated."""

    def setUp(self):
        torch.manual_seed(19)
        self.d_c = 20
        self.model = MLAGPT(
            vocab_size=71, d_model=48, n_layers=3, n_heads=3,
            d_head=12, d_rope=8, d_c=self.d_c, max_len=40, dropout=0.0,
        ).eval()
        self.device = torch.device("cpu")
        self.tokens = np.random.default_rng(4).integers(0, 71, size=(1, 33)).astype(np.int64)
        self.orders = np.stack([
            np.random.default_rng(i).permutation(self.d_c) for i in range(3)
        ])

    def residual(self, rank, t1, t2):
        masks = MODULE._RISK.make_channel_masks(self.orders, rank, 1, self.device)
        base = MODULE.loss_vectors(self.model, self.tokens, [], masks, self.device)
        d1 = MODULE.loss_vectors(self.model, self.tokens, [t1], masks, self.device) - base
        d2 = MODULE.loss_vectors(self.model, self.tokens, [t2], masks, self.device) - base
        d12 = MODULE.loss_vectors(self.model, self.tokens, [t1, t2], masks, self.device) - base
        return d12 - d1 - d2

    def test_full_rank_truncation_is_a_no_op_so_the_residual_is_zero(self):
        residual = self.residual(self.d_c, 5, 9)
        self.assertLess(float(np.abs(residual).max()), 1e-5)

    def test_the_residual_is_symmetric_in_the_two_positions(self):
        forward = self.residual(8, 5, 11)
        backward = self.residual(8, 11, 5)
        np.testing.assert_allclose(forward, backward, atol=1e-6)

    def test_a_real_truncation_produces_a_nonzero_residual(self):
        """Sanity bound on the two zero-tests: the measurement can detect an effect."""
        residual = self.residual(4, 5, 6)
        self.assertGreater(float(np.abs(residual).sum()), 1e-4)


if __name__ == "__main__":
    unittest.main()
