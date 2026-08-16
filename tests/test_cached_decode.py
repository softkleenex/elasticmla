import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT


class CachedDecodeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = MLAGPT(
            vocab_size=101,
            d_model=48,
            n_layers=3,
            n_heads=3,
            d_head=12,
            d_rope=8,
            d_c=20,
            max_len=64,
            dropout=0.0,
        ).eval()
        self.idx = torch.randint(0, 101, (2, 11))

    @torch.no_grad()
    def test_token_by_token_matches_full_forward(self):
        full_logits, _ = self.model(self.idx)
        caches = None
        pieces = []
        for pos in range(self.idx.shape[1]):
            logits, caches = self.model.forward_cached(
                self.idx[:, pos : pos + 1], caches=caches
            )
            pieces.append(logits)
        cached_logits = torch.cat(pieces, dim=1)
        torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=2e-6)

    @torch.no_grad()
    def test_chunked_prefill_matches_full_forward(self):
        full_logits, _ = self.model(self.idx)
        first, caches = self.model.forward_cached(self.idx[:, :6])
        second, caches = self.model.forward_cached(self.idx[:, 6:], caches=caches)
        chunked = torch.cat((first, second), dim=1)
        torch.testing.assert_close(chunked, full_logits, rtol=1e-5, atol=2e-6)

    @torch.no_grad()
    def test_cache_is_compressed_and_bytes_are_exact(self):
        _, caches = self.model.forward_cached(self.idx)
        batch, length = self.idx.shape
        for cache in caches:
            self.assertEqual(cache["c_kv"].shape, (batch, length, 20))
            self.assertEqual(cache["k_rope"].shape, (batch, 1, length, 8))
            self.assertFalse(cache["c_kv"].requires_grad)
            self.assertFalse(cache["k_rope"].requires_grad)

        expected = batch * length * self.model.n_layers * (20 + 8) * 4
        self.assertEqual(self.model.cache_num_bytes(caches), expected)
        mha_bytes = self.model.theoretical_mha_cache_num_bytes(
            batch, length, dtype=torch.float32
        )
        self.assertLess(expected, mha_bytes)
        # Primary baseline is conservative standard MHA: K and V each d_head.
        expected_ratio = (20 + 8) / (2 * 3 * 12)
        self.assertAlmostEqual(expected / mha_bytes, expected_ratio)
        shape_matched = self.model.theoretical_shape_matched_mha_cache_num_bytes(
            batch, length, dtype=torch.float32
        )
        self.assertGreater(shape_matched, mha_bytes)

    @torch.no_grad()
    def test_rank_mask_only_changes_new_cache_entries(self):
        _, caches = self.model.forward_cached(self.idx[:, :5])
        old = [cache["c_kv"].clone() for cache in caches]
        mask = torch.zeros(20)
        mask[:7] = 1
        _, caches2 = self.model.forward_cached(
            self.idx[:, 5:7], caches=caches, rank_masks=mask
        )
        for before, after in zip(old, caches2):
            torch.testing.assert_close(after["c_kv"][:, :5], before)
            self.assertTrue(torch.count_nonzero(after["c_kv"][:, 5:, 7:]) == 0)

    @torch.no_grad()
    def test_layer_specific_masks(self):
        masks = []
        for kept in (3, 7, 11):
            mask = torch.zeros(20)
            mask[:kept] = 1
            masks.append(mask)
        _, caches = self.model.forward_cached(self.idx[:, :2], rank_masks=masks)
        for cache, kept in zip(caches, (3, 7, 11)):
            self.assertTrue(torch.count_nonzero(cache["c_kv"][:, :, kept:]) == 0)


    @torch.no_grad()
    def test_rejects_mixed_layer_cache_lengths(self):
        _, short = self.model.forward_cached(self.idx[:, :2])
        _, long = self.model.forward_cached(self.idx[:, :4])
        mixed = [short[0], long[1], short[2]]
        with self.assertRaisesRegex(ValueError, "same past length"):
            self.model.forward_cached(self.idx[:, 4:5], caches=mixed)


    @torch.no_grad()
    def test_cpu_autocast_cache_round_trip(self):
        # RoPE arithmetic may use a different dtype from linear projection outputs;
        # both cache components must be validated against their own new counterpart.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, caches = self.model.forward_cached(self.idx[:, :3])
            logits, caches = self.model.forward_cached(self.idx[:, 3:4], caches=caches)
        self.assertEqual(logits.shape, (2, 1, 101))

    @torch.no_grad()
    def test_rejects_cache_dtype_mismatch(self):
        _, caches = self.model.forward_cached(self.idx[:, :2])
        bad = [{key: value.clone() for key, value in cache.items()} for cache in caches]
        bad[0]["c_kv"] = bad[0]["c_kv"].double()
        bad[0]["k_rope"] = bad[0]["k_rope"].double()
        with self.assertRaisesRegex(ValueError, "incompatible"):
            self.model.forward_cached(self.idx[:, 2:3], caches=bad)

    @torch.no_grad()
    def test_one_token_generation_has_no_unused_decode_forward(self):
        with mock.patch.object(
            self.model, "forward_cached", wraps=self.model.forward_cached
        ) as wrapped:
            out = self.model.generate_cached(self.idx[:1, :4], max_new_tokens=1, top_k=1)
        self.assertEqual(out.shape, (1, 5))
        self.assertEqual(wrapped.call_count, 1)  # prefill only

    @torch.no_grad()
    def test_generate_cached_extends_sequence(self):
        torch.manual_seed(11)
        out = self.model.generate_cached(self.idx[:1, :4], max_new_tokens=3, top_k=1)
        self.assertEqual(out.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
