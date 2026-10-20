import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from elastic_mla import ContextualElasticMLAGPT, MLAGPT
from train_joint_rollout_router import build_keep, forward_joint
from confirm_fresh_contextual_router import sample_fresh_starts


class JointRolloutRouterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.base = MLAGPT(
            vocab_size=67, d_model=32, n_layers=3, n_heads=2,
            d_head=8, d_rope=4, d_c=16, max_len=12,
        ).eval()
        self.orders = [torch.randperm(16) for _ in range(3)]
        self.model = ContextualElasticMLAGPT(
            self.base, self.orders, tiers=(4, 8, 12, 16)
        ).eval().freeze_base()
        self.idx = torch.randint(0, 67, (2, 7))

    def test_straight_through_hard_forward_matches_packed_inference(self):
        keep = build_keep(
            [order.tolist() for order in self.orders],
            [4, 8, 12, 16], torch.device("cpu"), torch.float32,
        )
        logits, _, chosen = forward_joint(
            self.base, self.model.router, self.idx,
            [4, 8, 12, 16], keep,
        )
        with torch.no_grad():
            packed, _, packed_ranks, _ = self.model.forward_cached_packed(self.idx)
        torch.testing.assert_close(chosen, packed_ranks)
        torch.testing.assert_close(logits, packed, atol=2e-5, rtol=2e-5)

    def test_straight_through_updates_router_but_not_frozen_base(self):
        keep = build_keep(
            [order.tolist() for order in self.orders],
            [4, 8, 12, 16], torch.device("cpu"), torch.float32,
        )
        logits, expected, _ = forward_joint(
            self.base, self.model.router, self.idx,
            [4, 8, 12, 16], keep,
        )
        # Backpropagate the LM term only: the rank penalty (`expected`) is a smooth
        # function of the soft probabilities and would reach the router even if the
        # straight-through path were broken (gate = hard one-hot, no gradient).
        logits.square().mean().backward()
        grads = [p.grad for p in self.model.router.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        self.assertTrue(expected.requires_grad)

    def test_hard_gate_without_straight_through_gives_router_no_lm_gradient(self):
        """Control for the test above: it must fail if the estimator is removed."""
        keep = build_keep(
            [order.tolist() for order in self.orders],
            [4, 8, 12, 16], torch.device("cpu"), torch.float32,
        )
        x = self.base.blocks[0](self.base.drop(self.base.tok_emb(self.idx)))
        probs = torch.softmax(self.model.router(self.base.blocks[1].ln1(x)), dim=-1)
        gate = torch.nn.functional.one_hot(probs.argmax(-1), 4).to(probs.dtype)  # no ST term
        for i, block in enumerate(self.base.blocks[1:], start=1):
            x = block(x, rank_mask=torch.einsum("btk,kc->btc", gate, keep[i]))
        out = self.base.head(self.base.ln_f(x)).square().mean()
        self.assertFalse(out.requires_grad and any(
            g is not None and float(g.abs().sum()) > 0
            for g in torch.autograd.grad(out, list(self.model.router.parameters()), allow_unused=True)
        ))

    def test_fresh_start_sampler_excludes_all_prior_spans(self):
        old = [6000, 9000, 12000]
        starts = sample_fresh_starts(30000, 255, 12, 91827, old)
        self.assertEqual(len(starts), len(set(starts)))
        combined = sorted(starts)
        self.assertTrue(all(b - a >= 256 for a, b in zip(combined, combined[1:])))
        self.assertTrue(all(abs(a - b) >= 256 for a in starts for b in old))


if __name__ == "__main__":
    unittest.main()
