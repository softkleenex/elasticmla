import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import ElasticMLAGPT, MLAGPT, TieredRankRouter


class TieredRouterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.base = MLAGPT(
            vocab_size=89, d_model=48, n_layers=2, n_heads=3,
            d_head=12, d_rope=8, d_c=20, dropout=0.0,
        ).eval()
        self.orders = [torch.randperm(20), torch.randperm(20)]
        self.elastic = ElasticMLAGPT(
            self.base, self.orders, tiers=(4, 8, 12, 20)
        ).eval()
        self.idx = torch.randint(0, 89, (2, 6))

    def test_router_shapes_and_expected_rank(self):
        _, logits = self.elastic.routing_logits_full(self.idx)
        self.assertEqual(len(logits), 2)
        self.assertEqual(logits[0].shape, (2, 6, 4))
        expected = self.elastic.routers[0].expected_rank(logits[0])
        self.assertEqual(expected.shape, (2, 6))
        self.assertTrue(torch.all(expected >= 4))
        self.assertTrue(torch.all(expected <= 20))

    @torch.no_grad()
    def test_forced_router_path_matches_base_packed_path(self):
        ranks = [
            torch.tensor([[4, 8, 12, 20, 4, 8], [20, 12, 8, 4, 8, 12]]),
            torch.tensor([[8, 8, 12, 12, 20, 4], [4, 20, 12, 8, 4, 20]]),
        ]
        base_logits, base_cache = self.base.forward_cached_packed(
            self.idx, ranks=ranks, channel_orders=self.orders
        )
        elastic_logits, elastic_cache, chosen, _ = self.elastic.forward_cached_packed(
            self.idx, forced_ranks=ranks
        )
        torch.testing.assert_close(elastic_logits, base_logits)
        self.assertEqual(
            self.elastic.packed_cache_num_bytes(elastic_cache),
            self.base.packed_cache_num_bytes(base_cache),
        )
        for got, expected in zip(chosen, ranks):
            torch.testing.assert_close(got, expected)

    @torch.no_grad()
    def test_predicted_ranks_are_valid_tiers(self):
        _, _, ranks, _ = self.elastic.forward_cached_packed(self.idx)
        tiers = {4, 8, 12, 20}
        for layer_ranks in ranks:
            self.assertTrue(set(layer_ranks.flatten().tolist()).issubset(tiers))


    @torch.no_grad()
    def test_rejects_non_tier_forced_rank(self):
        bad = [torch.full((2, 6), 5), torch.full((2, 6), 8)]
        with self.assertRaisesRegex(ValueError, "must belong"):
            self.elastic.forward_cached_packed(self.idx, forced_ranks=bad)

    @torch.no_grad()
    def test_rejects_mixed_wrapper_cache_lengths(self):
        ranks = [torch.full((2, 2), 8), torch.full((2, 2), 8)]
        _, short, _, _ = self.elastic.forward_cached_packed(
            self.idx[:, :2], forced_ranks=ranks
        )
        ranks_long = [torch.full((2, 4), 8), torch.full((2, 4), 8)]
        _, long, _, _ = self.elastic.forward_cached_packed(
            self.idx[:, :4], forced_ranks=ranks_long
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            self.elastic.forward_cached_packed(
                self.idx[:, 4:5],
                caches=[short[0], long[1]],
                forced_ranks=[torch.full((2, 1), 8), torch.full((2, 1), 8)],
            )

    def test_supervised_router_loss_backpropagates(self):
        router = TieredRankRouter(48, tiers=(4, 8, 12, 20))
        hidden = torch.randn(2, 5, 48)
        logits = router(hidden)
        targets = torch.tensor([[4, 8, 12, 20, 4], [20, 12, 8, 4, 8]])
        loss = router.supervised_loss(logits, targets)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in router.parameters()))

    def test_freeze_base(self):
        self.elastic.freeze_base()
        self.assertTrue(all(not p.requires_grad for p in self.base.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.elastic.routers.parameters()))


if __name__ == "__main__":
    unittest.main()
