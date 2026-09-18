import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_kv", ROOT / "experiments" / "analyze_public_kv_basis.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SliceTests(unittest.TestCase):
    def test_qkv_slices_partition_the_concatenated_projection(self):
        s = MODULE.qkv_slices(768)
        self.assertEqual((s["query"].start, s["query"].stop), (0, 768))
        self.assertEqual((s["key"].start, s["key"].stop), (768, 1536))
        self.assertEqual((s["value"].start, s["value"].stop), (1536, 2304))


class BatchTests(unittest.TestCase):
    def test_batches_are_whole_blocks_and_respect_the_requested_count(self):
        ids = np.arange(1000)
        batches = MODULE.build_batches(ids, block_size=100, batch_size=2, count=3)
        self.assertEqual(len(batches), 3)
        self.assertEqual(tuple(batches[0].shape), (2, 100))
        self.assertTrue(torch.equal(batches[0][0], torch.arange(100)))

    def test_a_short_stream_yields_fewer_batches_rather_than_a_ragged_one(self):
        ids = np.arange(250)
        batches = MODULE.build_batches(ids, block_size=100, batch_size=2, count=5)
        self.assertEqual(len(batches), 1)
        self.assertEqual(tuple(batches[0].shape), (2, 100))


class HeadBasisTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        n = 40000
        self.keys = rng.normal(size=(n, 12)) @ rng.normal(size=(12, 12))
        self.queries = rng.normal(size=(n, 12)) @ rng.normal(size=(12, 12))
        self.moment = self.keys.T @ self.keys / n
        self.metric = self.queries.T @ self.queries / n

    def test_pca_captures_at_least_as_much_energy_as_the_axes_at_every_rank(self):
        bases, _, _ = MODULE.head_bases(self.moment)
        for rank in range(1, 12):
            captured = {}
            for n, (_, inverse) in bases.items():
                sub = inverse[:, :rank]
                captured[n] = np.trace(sub.T @ self.moment @ sub)
            self.assertGreaterEqual(captured["pca"] + 1e-9, captured["coord"])

    def test_every_row_projection_is_idempotent_and_full_rank_is_the_identity(self):
        bases, _, _ = MODULE.head_bases(self.moment, self.metric)
        self.assertIn("task_aware", bases)
        for name, (forward, inverse) in bases.items():
            full = MODULE.row_projection(forward, inverse, 12)
            np.testing.assert_allclose(full, np.eye(12), atol=1e-8, err_msg=name)
            partial = MODULE.row_projection(forward, inverse, 5)
            np.testing.assert_allclose(partial @ partial, partial, atol=1e-7, err_msg=name)
            self.assertAlmostEqual(np.linalg.matrix_rank(partial, tol=1e-6), 5, msg=name)

    def test_the_task_aware_frame_is_oblique_not_a_rotation(self):
        bases, _, _ = MODULE.head_bases(self.moment, self.metric)
        forward, _ = bases["task_aware"]
        self.assertFalse(np.allclose(forward @ forward.T, np.eye(12), atol=1e-6))

    def test_the_task_aware_frame_minimises_the_score_error_energy_ignores(self):
        """Truncation cost is the query-weighted score error, not the key's own energy."""
        bases, _, _ = MODULE.head_bases(self.moment, self.metric)

        def score_error(name, rank):
            forward, inverse = bases[name]
            projection = MODULE.row_projection(forward, inverse, rank)
            residual = self.keys - self.keys @ projection
            return float(np.mean(np.einsum("ij,ij->i", self.queries, residual) ** 2))

        for rank in (3, 5, 8):
            self.assertLess(score_error("task_aware", rank), score_error("pca", rank))
            self.assertLess(score_error("task_aware", rank), score_error("coord", rank))

    def test_its_truncation_cost_equals_the_predicted_eigenvalue_tail(self):
        forward, inverse, spectrum = MODULE.metric_frame(self.metric, self.moment)
        for rank in (3, 6, 9):
            projection = MODULE.row_projection(forward, inverse, rank)
            residual = self.keys - self.keys @ projection
            measured = float(np.mean(np.einsum("ij,ij->i", self.queries, residual) ** 2))
            # identity holds in expectation for independent queries and keys
            self.assertAlmostEqual(measured / spectrum[rank:].sum(), 1.0, delta=0.05)


@unittest.skipUnless(
    (Path.home() / ".cache/huggingface/hub").exists(), "no local HF cache")
class GPT2FoldTests(unittest.TestCase):
    """Folding the projection into c_attn must equal truncating the cached vectors."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForCausalLM
        cls.model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        cls.device = torch.device("cpu")
        cfg = cls.model.config
        cls.n_heads, cls.d_head = cfg.n_head, cfg.n_embd // cfg.n_head
        gen = torch.Generator().manual_seed(4)
        def random_frame():
            basis = np.linalg.qr(torch.randn(cls.d_head, cls.d_head, generator=gen).numpy())[0]
            return (basis.T, basis)
        cls.bases = {
            name: [[random_frame() for _ in range(cls.n_heads)] for _ in range(cfg.n_layer)]
            for name in ("key", "value")
        }

    def test_full_rank_fold_leaves_the_model_unchanged(self):
        truncated = MODULE.truncated_model(self.model, self.bases, self.d_head, self.device)
        ids = torch.randint(0, 50000, (1, 16))
        with torch.no_grad():
            torch.testing.assert_close(
                self.model(ids).logits, truncated(ids).logits, rtol=2e-4, atol=2e-4)

    def test_the_fold_projects_the_cached_key_and_value_but_not_the_query(self):
        rank = 5
        truncated = MODULE.truncated_model(self.model, self.bases, rank, self.device)
        hidden = torch.randn(1, 4, self.model.config.n_embd)
        slices = MODULE.qkv_slices(self.model.config.n_embd)
        with torch.no_grad():
            before = self.model.transformer.h[0].attn.c_attn(hidden)
            after = truncated.transformer.h[0].attn.c_attn(hidden)
        torch.testing.assert_close(
            before[..., slices["query"]], after[..., slices["query"]], rtol=1e-5, atol=1e-5)
        for name in ("key", "value"):
            start = slices[name].start
            for head in range(self.n_heads):
                lo, hi = start + head * self.d_head, start + (head + 1) * self.d_head
                forward, inverse = self.bases[name][0][head]
                projection = torch.as_tensor(
                    MODULE.row_projection(forward, inverse, rank), dtype=torch.float32)
                expected = before[..., lo:hi] @ projection
                torch.testing.assert_close(
                    after[..., lo:hi], expected, rtol=2e-4, atol=2e-4)

    def test_a_low_rank_fold_actually_changes_the_logits(self):
        truncated = MODULE.truncated_model(self.model, self.bases, 2, self.device)
        ids = torch.randint(0, 50000, (1, 16))
        with torch.no_grad():
            self.assertFalse(torch.allclose(
                self.model(ids).logits, truncated(ids).logits, rtol=1e-2, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
