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
    "latent_quantization", ROOT / "experiments" / "analyze_latent_quantization.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LatentQuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.d_c = 16
        self.model = MLAGPT(
            vocab_size=50, d_model=32, n_layers=2, n_heads=2,
            d_head=8, d_rope=4, d_c=self.d_c, max_len=16, dropout=0.0,
        ).eval()
        self.idx = torch.randint(0, 50, (1, 12))
        self.orders = np.stack([np.random.default_rng(i).permutation(self.d_c) for i in range(2)])

    def test_quantization_error_is_within_half_a_step(self):
        x = torch.randn(4, 9, 32)
        for qmax in (127, 7):
            q = MODULE.quantize_per_token(x, qmax)
            step = x.abs().amax(dim=-1, keepdim=True) / qmax
            self.assertTrue(torch.all((q - x).abs() <= step / 2 + 1e-3 * step))
            levels = torch.unique((q / step).round())
            self.assertLessEqual(len(levels), 2 * qmax + 1)

    def test_joint_mask_with_one_position_equals_the_probe_forward(self):
        pos, rank = 5, 6
        masks = MODULE.joint_rank_masks(self.orders, {pos: rank}, 12, torch.device("cpu"))
        channel_masks = MODULE.risk.make_channel_masks(self.orders, rank, 1, torch.device("cpu"))
        with torch.no_grad():
            joint = MODULE.forward_with_full_masks(self.model, self.idx, masks)
            probe = MODULE.risk.forward_with_layer_masks(
                self.model, self.idx, channel_masks=channel_masks,
                probe_positions=torch.tensor([pos]))
        torch.testing.assert_close(joint, probe)

    def test_probe_truncation_equals_cached_decoding_with_that_entry_masked(self):
        """The full-attention probe is what incremental cached decoding computes."""
        pos, rank = 4, 5
        channel_masks = MODULE.risk.make_channel_masks(self.orders, rank, 1, torch.device("cpu"))
        with torch.no_grad():
            probe = MODULE.risk.forward_with_layer_masks(
                self.model, self.idx, channel_masks=channel_masks,
                probe_positions=torch.tensor([pos]))
            caches, steps = None, []
            for t in range(self.idx.shape[1]):
                masks = [channel_masks[layer, :, None, :] if t == pos else None
                         for layer in range(2)]
                logits, caches = self.model.forward_cached(
                    self.idx[:, t:t + 1], caches=caches, rank_masks=masks)
                steps.append(logits)
        torch.testing.assert_close(torch.cat(steps, dim=1), probe, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
