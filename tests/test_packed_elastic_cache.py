import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT
from elastic_mla.elastic_cache import pack_latents, unpack_latents


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


if __name__ == "__main__":
    unittest.main()
