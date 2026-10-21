"""The DeepSeek-V2 latent probe: exactness, causality and the calibration gradient.

Uses a tiny randomly initialised ``DeepseekV2ForCausalLM`` on CPU, so it checks the
hook mechanics, not anything about the pretrained model.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "public_mla", ROOT / "experiments" / "analyze_public_mla_spectrum.py")
PUBLIC = importlib.util.module_from_spec(_SPEC)
sys.modules["public_mla"] = PUBLIC
_SPEC.loader.exec_module(PUBLIC)


def tiny_model():
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

    torch.manual_seed(0)
    config = DeepseekV2Config(
        vocab_size=97, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=8, qk_nope_head_dim=8,
        v_head_dim=8, n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        first_k_dense_replace=1, max_position_embeddings=64,
    )
    config._experts_implementation = "eager"  # grouped_mm has no float64 kernel
    model = DeepseekV2ForCausalLM(config).double().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class PublicMlaProbeTest(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.probe = PUBLIC.LatentProbe(
            [layer.self_attn.kv_a_layernorm for layer in self.model.model.layers])
        self.x = torch.randint(0, 97, (2, 20))
        self.d_c = self.model.config.kv_lora_rank
        self.n_layers = self.model.config.num_hidden_layers

    def tearDown(self):
        self.probe.remove()

    def logits(self, positions=None, rank=None, forward=None, inverse=None):
        if positions is None:
            self.probe.mode = "off"
        else:
            self.probe.set_truncation(forward, inverse, rank)
            self.probe.positions, self.probe.mode = torch.tensor(positions), "probe"
        with torch.no_grad():
            out = self.model.lm_head(PUBLIC.hidden_states(self.model, self.x))
        self.probe.mode = "off"
        return out

    def orthogonal_basis(self):
        rng = np.random.default_rng(1)
        bases = [np.linalg.qr(rng.normal(size=(self.d_c, self.d_c)))[0] for _ in range(self.n_layers)]
        return [b.T for b in bases], bases

    def test_full_rank_is_identity_and_truncation_is_causal(self):
        reference = self.logits()
        forward, inverse = self.orthogonal_basis()
        full = self.logits([5, 9], self.d_c, forward, inverse)
        self.assertLess(float((full - reference).abs().max()), 1e-5)
        cut = self.logits([5, 9], 3, forward, inverse)
        # Row b is truncated at positions[b]; nothing before it can change.
        self.assertTrue(torch.equal(cut[0, :5], reference[0, :5]))
        self.assertTrue(torch.equal(cut[1, :9], reference[1, :9]))
        self.assertGreater(float((cut[0, 5:] - reference[0, 5:]).abs().max()), 1e-6)

    def test_coordinate_truncation_zeroes_exactly_the_dropped_channels(self):
        order = np.random.default_rng(2).permutation(self.d_c)
        permutation = np.zeros((self.d_c, self.d_c))
        permutation[order, np.arange(self.d_c)] = 1.0
        forward, inverse = [permutation.T] * self.n_layers, [permutation] * self.n_layers
        self.assertTrue(torch.equal(self.logits([4, 4], self.d_c, forward, inverse), self.logits()))
        captured = []
        # Layer 0: a later layer's latent at the probed position already differs because
        # the truncation at earlier layers changed its input.
        handle = self.model.model.layers[0].self_attn.kv_a_layernorm.register_forward_hook(
            lambda m, i, o: captured.append(o.detach().clone()))
        self.logits([4, 7], 5, forward, inverse)
        self.logits()
        handle.remove()
        truncated, original = captured
        # The probe hook is registered first, so the capture hook sees its output.
        kept = np.zeros(self.d_c, dtype=bool)
        kept[order[:5]] = True
        self.assertTrue(torch.equal(truncated[0, 4, kept], original[0, 4, kept]))
        self.assertTrue(torch.all(truncated[0, 4, ~kept] == 0))
        self.assertTrue(torch.equal(truncated[0, :4], original[0, :4]))

    def test_calibration_gradient_is_the_total_derivative(self):
        row = np.concatenate([[1], np.random.default_rng(3).integers(0, 97, 16)]).astype(np.int64)
        cc, gc, saliency = PUBLIC.latent_moments(self.model, self.probe, [row], self.d_c, "cpu")
        self.assertTrue(np.allclose(cc, cc.transpose(0, 2, 1)))
        x = torch.from_numpy(row[None, :-1])
        y = torch.from_numpy(row[None, 1:])
        layer, direction = 0, torch.randn(self.d_c, dtype=torch.float64)

        def loss_at(eps):
            def shift(module, inputs, output):
                return output + eps * direction
            handle = self.model.model.layers[layer].self_attn.kv_a_layernorm.register_forward_hook(shift)
            with torch.no_grad():
                value = PUBLIC.full_losses(self.model, PUBLIC.hidden_states(self.model, x), y).sum()
            handle.remove()
            return float(value)

        # Shifting every token's latent by eps*v changes the summed loss by
        # eps * sum_t g_t.v; gc's row sums cannot recover that, so recompute g directly.
        # Plain autograd, no checkpointing: the reference latent_moments must match.
        self.probe.latents, self.probe.grads, self.probe.mode = {}, {}, "record"
        inputs = self.model.get_input_embeddings()(x).detach().requires_grad_(True)
        hidden = self.model.model(inputs_embeds=inputs, use_cache=False).last_hidden_state
        self.probe.mode = "off"
        loss = PUBLIC.full_losses(self.model, hidden, y).sum()
        grads = torch.autograd.grad(loss, [self.probe.latents[i] for i in range(self.n_layers)])
        grad = grads[layer]
        analytic = float((grad.sum(dim=(0, 1)) * direction).sum())
        # The MoE router runs in float32 even in a float64 model, so a tiny step is
        # dominated by its rounding; a larger central step keeps O(eps^2) error small.
        numeric = (loss_at(1e-2) - loss_at(-1e-2)) / 2e-2
        self.assertAlmostEqual(analytic, numeric, delta=2e-3 * max(1.0, abs(numeric)))
        # latent_moments (checkpointed) gives the same statistics at every layer.
        for i in range(self.n_layers):
            latent = self.probe.latents[i].detach().reshape(-1, self.d_c)
            g = grads[i].reshape(-1, self.d_c)
            self.assertTrue(np.allclose(saliency[i], (g * latent).abs().sum(dim=0).numpy()))
            self.assertTrue(np.allclose(gc[i], (g.T @ latent).numpy() / latent.shape[0]))
        # and leaves the model in eval mode without checkpointing.
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.is_gradient_checkpointing)

    def test_window_losses_match_full_losses(self):
        y = torch.randint(0, 97, (2, 20))
        hidden = PUBLIC.hidden_states(self.model, self.x)
        full = PUBLIC.full_losses(self.model, hidden, y, chunk=7)
        starts = torch.tensor([3, 11])
        window = PUBLIC.window_losses(self.model, hidden, y, starts, 6)
        self.assertTrue(torch.allclose(window[0], full[0, 3:9]))
        self.assertTrue(torch.allclose(window[1], full[1, 11:17]))


if __name__ == "__main__":
    unittest.main()
