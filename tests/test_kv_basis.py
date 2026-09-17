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
    "kv_basis", ROOT / "experiments" / "analyze_kv_basis.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HeadBasisTests(unittest.TestCase):
    def test_both_bases_are_orthonormal_and_ordered_by_their_criterion(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(500, 8)) @ rng.normal(size=(8, 8))
        moment = x.T @ x / 500
        bases, eigenvalues, energy = MODULE.head_bases(moment)
        for name, basis in bases.items():
            np.testing.assert_allclose(basis.T @ basis, np.eye(8), atol=1e-9, err_msg=name)
        self.assertTrue(np.all(np.diff(eigenvalues) <= 1e-12))
        self.assertTrue(np.all(np.diff(energy) <= 1e-12))

    def test_the_coordinate_basis_is_a_permutation_of_the_axes(self):
        moment = np.diag([3.0, 1.0, 5.0, 2.0])
        bases, _, _ = MODULE.head_bases(moment)
        coord = bases["coord"]
        # every column is a unit axis, ordered by descending diagonal energy
        self.assertEqual([int(np.argmax(coord[:, j])) for j in range(4)], [2, 0, 3, 1])
        self.assertTrue(np.all((coord == 0) | (coord == 1)))

    def test_pca_captures_at_least_as_much_energy_as_the_axes_at_every_rank(self):
        """Ky Fan: no axis selection can beat the eigenbasis at capturing energy."""
        rng = np.random.default_rng(11)
        x = rng.normal(size=(2000, 10)) @ rng.normal(size=(10, 10))
        moment = x.T @ x / 2000
        bases, _, _ = MODULE.head_bases(moment)
        for rank in range(1, 10):
            captured = {
                name: np.trace(b[:, :rank].T @ moment @ b[:, :rank])
                for name, b in bases.items()
            }
            self.assertGreaterEqual(captured["pca"] + 1e-9, captured["coord"])


class TruncationEquivalenceTests(unittest.TestCase):
    """Folding the projection into W_UK/W_UV must equal truncating the cached vectors."""

    def setUp(self):
        torch.manual_seed(13)
        self.model = MLAGPT(
            vocab_size=61, d_model=48, n_layers=2, n_heads=3,
            d_head=8, d_rope=8, d_c=20, max_len=24, dropout=0.0,
        ).eval()
        self.device = torch.device("cpu")
        self.idx = torch.randint(0, 61, (2, 12))
        gen = torch.Generator().manual_seed(2)
        self.bases = {
            name: [[np.linalg.qr(torch.randn(8, 8, generator=gen).numpy())[0]
                    for _ in range(3)] for _ in range(2)]
            for name in ("key", "value")
        }

    def test_full_rank_truncation_leaves_the_model_unchanged(self):
        truncated = MODULE.truncated_model(self.model, self.bases, 8, self.device)
        with torch.no_grad():
            torch.testing.assert_close(
                self.model(self.idx)[0], truncated(self.idx)[0], rtol=1e-4, atol=1e-4)

    def test_folding_the_projection_equals_projecting_the_cached_vectors(self):
        rank = 3
        truncated = MODULE.truncated_model(self.model, self.bases, rank, self.device)
        hidden = torch.randn(2, 5, 48)
        for layer in range(2):
            original, changed = self.model.blocks[layer].attn, truncated.blocks[layer].attn
            with torch.no_grad():
                latent = original.W_DKV(hidden)
                for name, weight in (("key", "W_UK"), ("value", "W_UV")):
                    direct = getattr(original, weight)(latent).reshape(2, 5, 3, 8)
                    folded = getattr(changed, weight)(latent).reshape(2, 5, 3, 8)
                    for head in range(3):
                        basis = torch.as_tensor(
                            self.bases[name][layer][head][:, :rank], dtype=torch.float32)
                        expected = direct[:, :, head, :] @ (basis @ basis.T)
                        torch.testing.assert_close(
                            folded[:, :, head, :], expected, rtol=1e-4, atol=1e-4)

    def test_truncation_is_per_head_not_across_the_concatenated_vector(self):
        """A per-head compressor cannot mix heads; rank 1 must leave one direction each."""
        truncated = MODULE.truncated_model(self.model, self.bases, 1, self.device)
        rows = truncated.blocks[0].attn.W_UK.weight.reshape(3, 8, -1)
        for head in range(3):
            self.assertLessEqual(int(np.linalg.matrix_rank(rows[head].detach().numpy(), tol=1e-5)), 1)

    def test_a_rank_below_full_actually_changes_the_output(self):
        truncated = MODULE.truncated_model(self.model, self.bases, 2, self.device)
        with torch.no_grad():
            self.assertFalse(torch.allclose(
                self.model(self.idx)[0], truncated(self.idx)[0], rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
