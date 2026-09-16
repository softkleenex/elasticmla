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
        rotated = MODULE.rotate_model(
            self.model, self.bases.transpose(0, 2, 1), self.bases, torch.device("cpu"))
        deviation, exact = MODULE.check_rotation_is_exact(
            self.model, rotated, self.idx, tolerance=1e-4
        )
        self.assertTrue(exact, f"full-rank deviation {deviation}")

    def test_identity_rotation_leaves_the_weights_untouched(self):
        identity = np.stack([np.eye(self.d_c) for _ in range(3)])
        rotated = MODULE.rotate_model(self.model, identity, identity, torch.device("cpu"))
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
        rotated = MODULE.rotate_model(
            self.model, self.bases.transpose(0, 2, 1), self.bases, torch.device("cpu"))
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
        rotated = MODULE.rotate_model(
            self.model, self.bases.transpose(0, 2, 1), self.bases, torch.device("cpu"))
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
        rotated = MODULE.rotate_model(
            self.model, broken.transpose(0, 2, 1), self.bases, torch.device("cpu"))
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
        forward, inverse, diagnostics = MODULE.build_bases(cc, gc, saliency)
        for name in ("coord", "pca", "grad"):
            for layer in range(layers):
                u = inverse[name][layer]
                np.testing.assert_allclose(u.T @ u, np.eye(d_c), atol=1e-9)
                np.testing.assert_allclose(forward[name][layer], u.T, atol=1e-12)
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
        _, inverse, _ = MODULE.build_bases(cc, gc, saliency)
        u = inverse["coord"][0]
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


class PairedDifferenceTests(unittest.TestCase):
    """Bases are measured on identical probes, so comparisons must use the pairing."""

    def entry(self, values):
        return {"r_star_by_sequence": {"0": values[:4], "1": values[4:]}}

    def test_a_uniform_shift_is_detected_and_excludes_zero(self):
        first = self.entry([16, 32, 48, 64, 16, 32, 48, 64])
        second = self.entry([48, 64, 80, 96, 48, 64, 80, 96])
        out = MODULE.paired_difference(first, second, 256, 2000, 7)
        self.assertAlmostEqual(out["mean_difference"], -32.0)
        self.assertTrue(out["excludes_zero"])
        self.assertEqual(out["probe_fraction_first_lower"], 1.0)

    def test_identical_bases_difference_to_exactly_zero(self):
        same = self.entry([16, 32, 48, 64, 96, 128, 160, 192])
        out = MODULE.paired_difference(same, same, 256, 2000, 7)
        self.assertEqual(out["mean_difference"], 0.0)
        self.assertEqual(out["probe_fraction_tied"], 1.0)
        self.assertFalse(out["excludes_zero"])

    def test_pairing_can_resolve_a_difference_that_marginal_spreads_hide(self):
        """The point of pairing: large common variation, small consistent offset."""
        base = [16, 96, 192, 256, 32, 128, 224, 256]
        first = self.entry(base)
        second = self.entry([v + 16 for v in base])
        out = MODULE.paired_difference(first, second, 256, 2000, 7)
        self.assertAlmostEqual(out["mean_difference"], -16.0)
        self.assertTrue(out["excludes_zero"])
        # the marginal spread of either arm dwarfs the 16-rank offset
        self.assertGreater(float(np.std(base)), 3 * 16)


class ComparisonBookkeepingTests(unittest.TestCase):
    def test_pairs_follow_the_declared_basis_order_not_the_measured_order(self):
        shuffled = {"grad": None, "model_metric": None, "coord": None, "pca": None}
        self.assertEqual(
            MODULE.comparison_pairs(shuffled),
            [("pca", "coord"), ("grad", "coord"), ("model_metric", "coord"),
             ("model_metric", "pca"), ("pca", "grad")],
        )

    def test_a_subset_drops_only_the_comparisons_it_cannot_make(self):
        self.assertEqual(
            MODULE.comparison_pairs({"pca": None, "grad": None}), [("pca", "grad")])
        self.assertEqual(MODULE.comparison_pairs({"pca": None}), [])

    def test_the_seed_of_a_comparison_does_not_depend_on_what_else_was_measured(self):
        """A position-derived seed would silently change with the --bases subset."""
        seed = MODULE.comparison_seed("pca", "coord", "maximum")
        self.assertEqual(seed, MODULE.comparison_seed("pca", "coord", "maximum"))
        self.assertNotEqual(seed, MODULE.comparison_seed("pca", "coord", "signed_mean"))
        self.assertNotEqual(seed, MODULE.comparison_seed("grad", "coord", "maximum"))
        self.assertNotEqual(seed, MODULE.comparison_seed("coord", "pca", "maximum"))


class ModelMetricBasisTests(unittest.TestCase):
    """The frame that minimises reconstruction error as the model actually sees it."""

    def setUp(self):
        rng = np.random.default_rng(21)
        self.d_c, self.d_out, self.n = 12, 20, 2000
        self.w1 = rng.normal(size=(self.d_out, self.d_c))
        self.w2 = rng.normal(size=(self.d_out, self.d_c))
        latents = rng.normal(size=(self.n, self.d_c)) @ rng.normal(size=(self.d_c, self.d_c))
        self.latents = latents
        self.cc = latents.T @ latents / self.n

    def transform(self):
        return MODULE.model_metric_transform(self.cc, (self.w1, self.w2))

    def test_forward_and_inverse_compose_to_the_identity(self):
        forward, inverse, _ = self.transform()
        np.testing.assert_allclose(inverse @ forward, np.eye(self.d_c), atol=1e-8)
        np.testing.assert_allclose(forward @ inverse, np.eye(self.d_c), atol=1e-8)

    def output_error(self, forward, inverse, keep):
        coords = self.latents @ forward.T
        coords[:, keep:] = 0.0
        residual = self.latents - coords @ inverse.T
        return (np.linalg.norm(residual @ self.w1.T) ** 2
                + np.linalg.norm(residual @ self.w2.T) ** 2) / self.n

    def test_truncation_cost_equals_the_predicted_eigenvalue_tail(self):
        forward, inverse, spectrum = self.transform()
        for keep in (2, 5, 9):
            self.assertAlmostEqual(
                self.output_error(forward, inverse, keep), spectrum[keep:].sum(), places=6)

    def test_it_beats_plain_pca_on_output_space_error(self):
        forward, inverse, _ = self.transform()
        values, vectors = np.linalg.eigh(self.cc)
        pca = vectors[:, np.argsort(-values)]
        for keep in (2, 5, 9):
            self.assertLess(
                self.output_error(forward, inverse, keep),
                self.output_error(pca.T, pca, keep),
            )

    def test_it_is_not_orthogonal_so_it_is_a_frame_change_not_a_rotation(self):
        forward, _, _ = self.transform()
        self.assertFalse(np.allclose(forward @ forward.T, np.eye(self.d_c), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
