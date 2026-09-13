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
    "basis_dependence", ROOT / "experiments" / "analyze_basis_dependence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RotationEquivalenceTests(unittest.TestCase):
    """The whole method rests on a rotation of the model being an exact identity."""

    def setUp(self):
        torch.manual_seed(31)
        self.d_c = 24
        self.model = MLAGPT(
            vocab_size=61, d_model=48, n_layers=3, n_heads=3,
            d_head=12, d_rope=8, d_c=self.d_c, max_len=32, dropout=0.0,
        ).eval()
        self.idx = torch.randint(0, 61, (2, 12))
        generator = torch.Generator().manual_seed(7)
        self.bases = np.stack([
            np.linalg.qr(torch.randn(self.d_c, self.d_c, generator=generator).numpy())[0]
            for _ in range(3)
        ])

    def test_rotating_the_model_does_not_change_the_full_rank_function(self):
        rotated = MODULE.rotate_model(self.model, self.bases, torch.device("cpu"))
        deviation, exact = MODULE.check_rotation_is_exact(
            self.model, rotated, self.idx, tolerance=1e-4
        )
        self.assertTrue(exact, f"full-rank deviation {deviation}")

    def test_identity_rotation_leaves_the_weights_untouched(self):
        identity = np.stack([np.eye(self.d_c) for _ in range(3)])
        rotated = MODULE.rotate_model(self.model, identity, torch.device("cpu"))
        for original, copy_ in zip(self.model.blocks, rotated.blocks):
            torch.testing.assert_close(original.attn.W_DKV.weight, copy_.attn.W_DKV.weight)
            torch.testing.assert_close(original.attn.W_UK.weight, copy_.attn.W_UK.weight)

    def test_truncating_in_the_rotated_basis_equals_projecting_in_the_original(self):
        """Masking the first k rotated coordinates == projecting onto span(U[:, :k]).

        Checked on the quantities the mask actually reaches -- the per-head key and
        value produced from the (possibly truncated) latent -- because the model
        forward masks one probe position only, while an explicit projection baked
        into the weights would apply at every position.
        """
        keep = 9
        rotated = MODULE.rotate_model(self.model, self.bases, torch.device("cpu"))
        hidden = torch.randn(2, 5, 48)
        mask = torch.zeros(self.d_c)
        mask[:keep] = 1.0
        for layer, u in enumerate(self.bases):
            original, rotated_attn = self.model.blocks[layer].attn, rotated.blocks[layer].attn
            projection = torch.as_tensor(u[:, :keep] @ u[:, :keep].T, dtype=torch.float32)
            with torch.no_grad():
                truncated_latent = rotated_attn.W_DKV(hidden) * mask
                via_rotation_k = rotated_attn.W_UK(truncated_latent)
                via_rotation_v = rotated_attn.W_UV(truncated_latent)
                projected_latent = original.W_DKV(hidden) @ projection.T
                via_projection_k = original.W_UK(projected_latent)
                via_projection_v = original.W_UV(projected_latent)
            torch.testing.assert_close(via_rotation_k, via_projection_k, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(via_rotation_v, via_projection_v, rtol=1e-4, atol=1e-4)

    def test_a_full_rank_mask_in_the_rotated_basis_changes_nothing(self):
        """Sanity bound on the previous test: at keep = d_c the projection is identity."""
        rotated = MODULE.rotate_model(self.model, self.bases, torch.device("cpu"))
        identity_order = np.tile(np.arange(self.d_c), (3, 1))
        masks = MODULE._RISK.make_channel_masks(identity_order, self.d_c, 2, torch.device("cpu"))
        positions = torch.tensor([4, 7])
        with torch.no_grad():
            masked = MODULE._RISK.forward_with_layer_masks(
                rotated, self.idx, channel_masks=masks, probe_positions=positions
            )
            reference = self.model(self.idx)[0]
        torch.testing.assert_close(masked, reference, rtol=1e-4, atol=1e-4)

    def test_the_rotation_check_rejects_a_non_orthogonal_matrix(self):
        broken = self.bases.copy()
        broken[0] *= 1.5
        rotated = MODULE.rotate_model(self.model, broken, torch.device("cpu"))
        _, exact = MODULE.check_rotation_is_exact(
            self.model, rotated, self.idx, tolerance=1e-4
        )
        self.assertFalse(exact)


class BasisConstructionTests(unittest.TestCase):
    def test_each_basis_is_orthogonal_and_ordered_by_its_own_criterion(self):
        rng = np.random.default_rng(11)
        d_c, layers = 6, 2
        c = rng.normal(size=(layers, 400, d_c))
        g = rng.normal(size=(layers, 400, d_c))
        cc = np.stack([x.T @ x / 400 for x in c])
        gc = np.stack([gg.T @ x / 400 for gg, x in zip(g, c)])
        saliency = np.abs(g * c).sum(axis=1)
        bases, diagnostics = MODULE.build_bases(cc, gc, saliency)
        for name in MODULE.BASES:
            for layer in range(layers):
                u = bases[name][layer]
                np.testing.assert_allclose(u.T @ u, np.eye(d_c), atol=1e-9)
        for layer in range(layers):
            values = diagnostics["pca"][layer]["criterion_values"]
            self.assertEqual(values, sorted(values, reverse=True))
            grad_values = [abs(v) for v in diagnostics["grad"][layer]["criterion_values"]]
            self.assertEqual(grad_values, sorted(grad_values, reverse=True))

    def test_coord_basis_is_the_published_saliency_permutation(self):
        rng = np.random.default_rng(3)
        d_c = 5
        cc = np.stack([np.eye(d_c)])
        gc = np.stack([rng.normal(size=(d_c, d_c))])
        saliency = np.array([[0.1, 0.9, 0.4, 0.7, 0.2]])
        bases, _ = MODULE.build_bases(cc, gc, saliency)
        u = bases["coord"][0]
        # Column j must be the unit vector of the j-th most salient channel.
        expected_order = np.argsort(-saliency[0])
        for j, channel in enumerate(expected_order):
            self.assertEqual(u[channel, j], 1.0)
            self.assertEqual(u[:, j].sum(), 1.0)


class SamplingParityTests(unittest.TestCase):
    """The ablation must probe the same population the published spectrum probed.

    The first version of this script drew starts with a plain ``rng.integers``
    instead of the rejection sampler the published analysis uses, which silently
    selected different calibration and evaluation windows and made the ``coord``
    baseline miss the published numbers by ~0.9pp. These pin the protocol.
    """

    def setUp(self):
        self.risk = MODULE._RISK
        self.block_size = 256
        self.sequence_span = self.block_size + 1
        self.calibration_high = 50_000
        self.evaluation_low = self.calibration_high + self.block_size
        self.n_possible_starts = 120_000

    def published_starts(self, repeats, n_calib, n_eval):
        used = []
        calibration = []
        for repeat_idx in range(repeats):
            rng = np.random.default_rng(self.risk.CALIBRATION_SEED_BASE + repeat_idx)
            starts = self.risk.sample_starts(
                rng, 0, self.calibration_high, n_calib,
                min_separation=self.sequence_span, excluded=used,
            )
            calibration.append(starts)
            used.extend(map(int, starts))
        eval_rng = np.random.default_rng(self.risk.EVALUATION_SEED)
        evaluation = self.risk.sample_starts(
            eval_rng, self.evaluation_low, self.n_possible_starts, n_eval,
            min_separation=self.sequence_span,
        )
        return calibration, evaluation, eval_rng

    def test_calibration_windows_are_disjoint_within_and_across_repeats(self):
        calibration, _, _ = self.published_starts(3, 16, 24)
        flat = [int(v) for starts in calibration for v in starts]
        self.assertEqual(len(flat), len(set(flat)))
        for i, a in enumerate(flat):
            for b in flat[i + 1:]:
                self.assertGreaterEqual(abs(a - b), self.sequence_span)

    def test_plain_integers_would_not_reproduce_the_published_starts(self):
        """Guard against the exact mistake: rng.integers is not the sampler."""
        _, evaluation, _ = self.published_starts(3, 16, 24)
        naive = np.random.default_rng(self.risk.EVALUATION_SEED).integers(
            self.evaluation_low, self.n_possible_starts, size=24
        )
        self.assertFalse(np.array_equal(np.sort(naive), evaluation))

    def test_probe_positions_come_from_the_generator_after_the_eval_starts(self):
        _, _, eval_rng = self.published_starts(3, 16, 24)
        valid = self.risk.valid_probe_positions(self.block_size, 32)
        first = np.sort(eval_rng.choice(valid, size=32, replace=False))
        # A generator freshly seeded (i.e. positions drawn before the starts)
        # gives a different draw, so the ordering of rng use is part of the protocol.
        fresh = np.sort(
            np.random.default_rng(self.risk.EVALUATION_SEED).choice(valid, size=32, replace=False)
        )
        self.assertFalse(np.array_equal(first, fresh))


if __name__ == "__main__":
    unittest.main()
