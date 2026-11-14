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

    def run_both(self, rank):
        masks = MODULE.risk.make_channel_masks(self.orders, rank, 3, torch.device("cpu"))
        with torch.no_grad():
            trunc, norms = MODULE.forward_truncation_with_norms(
                self.model, self.idx, channel_masks=masks, probe_positions=self.positions)
            plain = MODULE.risk.forward_with_layer_masks(
                self.model, self.idx, channel_masks=masks, probe_positions=self.positions)
            null = MODULE.forward_with_added_noise(
                self.model, self.idx, norms=norms, directions=self.dirs,
                probe_positions=self.positions)
        return masks, trunc, plain, norms, null

    def test_full_rank_null_is_the_unperturbed_model(self):
        _, _, _, norms, null = self.run_both(self.d_c)
        with torch.no_grad():
            reference = MODULE.risk.forward_with_layer_masks(self.model, self.idx)
        torch.testing.assert_close(null, reference)
        self.assertTrue(torch.all(norms == 0))

    def test_recording_norms_does_not_change_truncation(self):
        _, trunc, plain, _, _ = self.run_both(5)
        torch.testing.assert_close(trunc, plain)

    def test_norms_are_the_truncation_streams_residuals(self):
        masks, _, _, norms, _ = self.run_both(5)
        captured = []
        hooks = [b.attn.W_DKV.register_forward_hook(lambda m, i, o: captured.append(o.clone()))
                 for b in self.model.blocks]
        with torch.no_grad():
            MODULE.risk.forward_with_layer_masks(
                self.model, self.idx, channel_masks=masks, probe_positions=self.positions)
        for h in hooks:
            h.remove()
        for layer in range(2):
            rows = captured[layer][torch.arange(3), self.positions]
            torch.testing.assert_close(norms[layer], (rows * (1 - masks[layer])).norm(dim=-1))

    def test_covariance_directions_are_unit(self):
        roots = np.stack([np.diag(np.arange(1, 33, dtype=float))] * 4)
        a = MODULE.null_directions(3, 4, 32, roots)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-12)
        self.assertFalse(np.allclose(a, MODULE.null_directions(3, 4, 32)))

    def test_directions_are_unit_and_seeded_per_probe(self):
        a = MODULE.null_directions(3, 4, 32)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-12)
        np.testing.assert_array_equal(a, MODULE.null_directions(3, 4, 32))
        self.assertFalse(np.allclose(a, MODULE.null_directions(4, 4, 32)))


if __name__ == "__main__":
    unittest.main()
