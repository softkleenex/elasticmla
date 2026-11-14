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
    "perturbation_null", ROOT / "experiments" / "analyze_perturbation_null.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EqualNormNullTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.d_c = 16
        self.model = MLAGPT(
            vocab_size=50, d_model=32, n_layers=2, n_heads=2,
            d_head=8, d_rope=4, d_c=self.d_c, max_len=16, dropout=0.0,
        ).eval()
        self.idx = torch.randint(0, 50, (3, 10))
        self.positions = torch.tensor([1, 4, 7])
        self.orders = np.stack([np.random.default_rng(i).permutation(self.d_c) for i in range(2)])
        dirs = np.stack([MODULE.null_directions(i, 2, self.d_c) for i in range(3)])
        self.dirs = torch.as_tensor(dirs.transpose(1, 0, 2).copy(), dtype=torch.float32)

    def test_full_rank_null_is_the_unperturbed_model(self):
        masks = MODULE.risk.make_channel_masks(self.orders, self.d_c, 3, torch.device("cpu"))
        with torch.no_grad():
            logits, norms = MODULE.forward_with_equal_norm_noise(
                self.model, self.idx, channel_masks=masks, directions=self.dirs,
                probe_positions=self.positions)
            reference = MODULE.risk.forward_with_layer_masks(self.model, self.idx)
        torch.testing.assert_close(logits, reference)
        self.assertTrue(torch.all(norms == 0))

    def test_noise_norm_equals_the_truncated_residual_norm(self):
        rank = 5
        masks = MODULE.risk.make_channel_masks(self.orders, rank, 3, torch.device("cpu"))
        with torch.no_grad():
            _, latents = MODULE.risk.forward_with_all_latents(self.model, self.idx)
            _, norms = MODULE.forward_with_equal_norm_noise(
                self.model, self.idx, channel_masks=masks, directions=self.dirs,
                probe_positions=self.positions)
        # Layer 0's latent is upstream of any intervention, so its residual is exact.
        rows = latents[0][torch.arange(3), self.positions]
        expected = (rows * (1 - masks[0])).norm(dim=-1)
        torch.testing.assert_close(norms[0], expected)
        self.assertTrue(torch.all(norms[1] > 0))

    def test_directions_are_unit_and_seeded_per_probe(self):
        a = MODULE.null_directions(3, 4, 32)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-12)
        np.testing.assert_array_equal(a, MODULE.null_directions(3, 4, 32))
        self.assertFalse(np.allclose(a, MODULE.null_directions(4, 4, 32)))


if __name__ == "__main__":
    unittest.main()
