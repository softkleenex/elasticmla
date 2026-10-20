import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT
from elastic_mla.elastic_cache import append_packed_latents, pack_latents, unpack_latents


class PackedElasticCacheTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = MLAGPT(
            vocab_size=97, d_model=48, n_layers=3, n_heads=3,
            d_head=12, d_rope=8, d_c=20, max_len=64, dropout=0.0,
        ).eval()
        self.idx = torch.randint(0, 97, (2, 9))
        generator = torch.Generator().manual_seed(9)
        self.orders = [torch.randperm(20, generator=generator) for _ in range(3)]
        tier_values = torch.tensor([4, 8, 12, 20])
        choices = torch.randint(0, 4, (3, 2, 9), generator=generator)
        self.layer_ranks = [tier_values[choices[layer]] for layer in range(3)]

    def test_pack_unpack_preserves_selected_channels(self):
        c = torch.randn(2, 5, 20)
        ranks = torch.tensor([[4, 8, 12, 20, 4], [20, 12, 8, 4, 8]])
        order = self.orders[0]
        packed = pack_latents(c, ranks, order)
        dense = unpack_latents(packed, 20, order)
        for b in range(2):
            for t in range(5):
                r = int(ranks[b, t])
                torch.testing.assert_close(dense[b, t, order[:r]], c[b, t, order[:r]])
                self.assertEqual(torch.count_nonzero(dense[b, t, order[r:]]), 0)

    @torch.no_grad()
    def test_truncated_packed_decode_matches_the_analysis_rank_mask_intervention(self):
        """The deployed packed cache implements exactly what the analyses measure.

        The rank-variance/risk-capacity scripts intervene through the dense
        ``Block.forward(rank_mask=...)`` path with mask[b, t, order[:r]] = 1. Here
        per-token, per-layer ranks are drawn freely (not only tiers), the packed
        path runs a prefill plus single-token decode steps, and every step's
        logits must equal the dense masked forward over the same prefix.
        """
        model = self.model.double()
        generator = torch.Generator().manual_seed(23)
        B, T, prefill = 2, 14, 5
        idx = torch.randint(0, 97, (B, T), generator=generator)
        ranks = torch.randint(1, 21, (3, B, T), generator=generator)
        masks = torch.zeros(3, B, T, 20, dtype=torch.float64)
        for layer in range(3):
            for b in range(B):
                for t in range(T):
                    masks[layer, b, t, self.orders[layer][: int(ranks[layer, b, t])]] = 1.0
        x = model.drop(model.tok_emb(idx))
        for layer, block in enumerate(model.blocks):
            x = block(x, rank_mask=masks[layer])
        dense = model.head(model.ln_f(x))
        unmasked, _ = model(idx)
        self.assertGreater(float((dense - unmasked).abs().max()), 1e-2)  # truncation matters

        logits, caches = model.forward_cached_packed(
            idx[:, :prefill], ranks=[ranks[l, :, :prefill] for l in range(3)],
            channel_orders=self.orders,
        )
        torch.testing.assert_close(logits, dense[:, :prefill], atol=1e-10, rtol=1e-10)
        for t in range(prefill, T):
            logits, caches = model.forward_cached_packed(
                idx[:, t : t + 1], ranks=[ranks[l, :, t : t + 1] for l in range(3)],
                channel_orders=self.orders, caches=caches,
            )
            torch.testing.assert_close(logits[:, 0], dense[:, t], atol=1e-10, rtol=1e-10)
        model.float()

    @torch.no_grad()
    def test_full_rank_packed_matches_full_forward(self):
        full, _ = self.model(self.idx)
        ranks = torch.full(self.idx.shape, 20, dtype=torch.long)
        packed_logits, _ = self.model.forward_cached_packed(
            self.idx, ranks=ranks, channel_orders=self.orders
        )
        torch.testing.assert_close(packed_logits, full, rtol=1e-5, atol=2e-6)

    @torch.no_grad()
    def test_mixed_rank_packed_matches_dense_mask_cache(self):
        packed_caches = None
        dense_caches = None
        packed_parts, dense_parts = [], []
        for pos in range(self.idx.shape[1]):
            token = self.idx[:, pos:pos+1]
            ranks_now = [r[:, pos:pos+1] for r in self.layer_ranks]
            packed_logits, packed_caches = self.model.forward_cached_packed(
                token,
                ranks=ranks_now,
                channel_orders=self.orders,
                caches=packed_caches,
            )
            masks = []
            for order, layer_rank in zip(self.orders, ranks_now):
                mask = torch.zeros(2, 1, 20)
                for b in range(2):
                    mask[b, 0, order[:int(layer_rank[b, 0])]] = 1
                masks.append(mask)
            dense_logits, dense_caches = self.model.forward_cached(
                token, caches=dense_caches, rank_masks=masks
            )
            packed_parts.append(packed_logits)
            dense_parts.append(dense_logits)
        torch.testing.assert_close(
            torch.cat(packed_parts, 1), torch.cat(dense_parts, 1),
            rtol=1e-5, atol=2e-6,
        )


    @torch.no_grad()
    def test_rejects_channel_order_change(self):
        ranks = torch.full((2, 2), 8, dtype=torch.long)
        _, caches = self.model.forward_cached_packed(
            self.idx[:, :2], ranks=ranks, channel_orders=self.orders
        )
        changed = list(self.orders)
        changed[0] = torch.flip(changed[0], dims=(0,))
        with self.assertRaisesRegex(ValueError, "channel_order does not match"):
            self.model.forward_cached_packed(
                self.idx[:, 2:3],
                ranks=torch.full((2, 1), 8, dtype=torch.long),
                channel_orders=changed,
                caches=caches,
            )

    @torch.no_grad()
    def test_cpu_autocast_packed_values_stay_compact(self):
        ranks = torch.full((2, 3), 8, dtype=torch.long)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, caches = self.model.forward_cached_packed(
                self.idx[:, :3], ranks=ranks, channel_orders=self.orders
            )
            self.assertEqual(caches[0]["values"].dtype, torch.bfloat16)
            self.model.forward_cached_packed(
                self.idx[:, 3:4],
                ranks=torch.full((2, 1), 8, dtype=torch.long),
                channel_orders=self.orders,
                caches=caches,
            )

    @torch.no_grad()
    def test_persistent_bytes_match_packed_payload(self):
        _, caches = self.model.forward_cached_packed(
            self.idx, ranks=self.layer_ranks, channel_orders=self.orders
        )
        measured = self.model.packed_cache_num_bytes(caches)
        expected = 0
        B, S = self.idx.shape
        for ranks in self.layer_ranks:
            expected += int(ranks.sum()) * 4              # packed float values
            expected += (B * S + 1) * 4                  # int32 offsets
            expected += B * S * 2                        # int16 ranks
            expected += 20 * 2                           # int16 channel order metadata
            expected += B * S * 8 * 4                    # shared k_rope
        self.assertEqual(measured, expected)
        dense_bytes = B * S * 3 * (20 + 8) * 4
        self.assertLess(measured, dense_bytes)

    @torch.no_grad()
    def test_persistent_bytes_stay_exact_after_decode_appends(self):
        """Proposition 1's byte count also holds after the in-place append path."""
        prefill = 4
        B, S = self.idx.shape
        _, caches = self.model.forward_cached_packed(
            self.idx[:, :prefill], ranks=[r[:, :prefill] for r in self.layer_ranks],
            channel_orders=self.orders,
        )
        for t in range(prefill, S):
            _, caches = self.model.forward_cached_packed(
                self.idx[:, t : t + 1], ranks=[r[:, t : t + 1] for r in self.layer_ranks],
                channel_orders=self.orders, caches=caches,
            )
        expected = sum(
            int(ranks.sum()) * 4 + (B * S + 1) * 4 + B * S * 2 + 20 * 2 + B * S * 8 * 4
            for ranks in self.layer_ranks
        )
        self.assertEqual(self.model.packed_cache_num_bytes(caches), expected)


class IncrementalAppendTest(unittest.TestCase):
    """The decode path appends into the packed buffers instead of rebuilding them."""

    def setUp(self):
        self.d_c = 20
        generator = torch.Generator().manual_seed(5150)
        self.order = torch.randperm(self.d_c, generator=generator)
        self.c = torch.randn(3, 11, self.d_c, generator=generator)
        self.ranks = torch.randint(1, self.d_c + 1, (3, 11), generator=generator)

    def test_incremental_append_matches_one_shot_pack(self):
        one_shot = pack_latents(self.c, self.ranks, self.order)
        cache, pos = None, 0
        for step in (1, 1, 3, 1, 2, 3):
            cache = append_packed_latents(
                cache,
                self.c[:, pos:pos + step],
                self.ranks[:, pos:pos + step],
                self.order,
            )
            pos += step
        self.assertEqual(pos, self.c.shape[1])
        for key in ("values", "offsets", "ranks", "channel_order"):
            self.assertTrue(torch.equal(cache[key], one_shot[key]), key)
        torch.testing.assert_close(
            unpack_latents(cache, self.d_c, self.order),
            unpack_latents(one_shot, self.d_c, self.order),
        )

    def test_append_does_not_rewrite_existing_packed_values(self):
        """Appending must leave the already-stored prefix of the buffer untouched."""
        cache = pack_latents(self.c[:, :6], self.ranks[:, :6], self.order)
        before = cache["values"].clone()
        grown = append_packed_latents(
            cache, self.c[:, 6:7], self.ranks[:, 6:7], self.order
        )
        self.assertTrue(torch.equal(grown["values"][: before.numel()], before))
        self.assertTrue(torch.equal(grown["offsets"][: cache["offsets"].numel()], cache["offsets"]))

    def test_append_rejects_int32_offset_overflow(self):
        cache = pack_latents(self.c[:, :1], self.ranks[:, :1], self.order)
        cache["offsets"] = torch.full_like(cache["offsets"], 2**31 - 8)
        cache["offsets"][0] = 0
        with self.assertRaises(OverflowError):
            append_packed_latents(
                cache, self.c[:, 1:2], self.ranks[:, 1:2], self.order
            )

    def test_append_rejects_channel_order_change(self):
        cache = pack_latents(self.c[:, :4], self.ranks[:, :4], self.order)
        with self.assertRaisesRegex(ValueError, "channel_order does not match"):
            append_packed_latents(
                cache, self.c[:, 4:5], self.ranks[:, 4:5], torch.flip(self.order, dims=(0,))
            )

    def test_append_rejects_batch_size_change(self):
        cache = pack_latents(self.c[:, :4], self.ranks[:, :4], self.order)
        with self.assertRaisesRegex(ValueError, "batch size does not match"):
            append_packed_latents(
                cache, self.c[:1, 4:5], self.ranks[:1, 4:5], self.order
            )


if __name__ == "__main__":
    unittest.main()
