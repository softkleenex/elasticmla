"""R8 on the DeepSeek-V2 latent: equal-norm null and joint masks (hook mechanics only).

Uses the tiny randomly initialised ``DeepseekV2ForCausalLM`` of
`test_public_mla_probe.py` on CPU in float64, so it checks the hooks and loops of
`experiments/analyze_public_mla_null.py`, not anything about the pretrained model.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R8 = _load("public_mla_null", ROOT / "experiments" / "analyze_public_mla_null.py")
PROBE_TESTS = _load("public_mla_probe_tests", ROOT / "tests" / "test_public_mla_probe.py")


class PublicMlaNullTest(unittest.TestCase):
    def setUp(self):
        self.model = PROBE_TESTS.tiny_model()
        self.layers = [layer.self_attn.kv_a_layernorm for layer in self.model.model.layers]
        self.probe = R8.NullProbe(self.layers)
        self.d_c = self.model.config.kv_lora_rank
        self.n_layers = self.model.config.num_hidden_layers
        self.x = torch.randint(0, 97, (2, 20), generator=torch.Generator().manual_seed(5))
        order = np.random.default_rng(2).permutation(self.d_c)
        permutation = np.zeros((self.d_c, self.d_c))
        permutation[order, np.arange(self.d_c)] = 1.0
        self.order = order
        self.forward, self.inverse = [permutation.T] * self.n_layers, [permutation] * self.n_layers

    def tearDown(self):
        self.probe.remove()

    def logits(self, mode="off"):
        self.probe.mode = mode
        with torch.no_grad():
            out = self.model.lm_head(R8.PUBLIC.hidden_states(self.model, self.x))
        self.probe.mode = "off"
        return out

    def truncate(self, positions, rank):
        self.probe.set_truncation(self.forward, self.inverse, rank)
        self.probe.positions, self.probe.residual_norms = torch.tensor(positions), {}
        return self.logits("probe"), self.probe.stacked_norms()

    def test_null_at_full_rank_is_the_unperturbed_forward(self):
        reference = self.logits()
        _, norms = self.truncate([4, 9], self.d_c)
        self.assertTrue(torch.all(norms == 0))
        unit = torch.as_tensor(R8.NULL.null_directions(0, self.n_layers, self.d_c))[:, None, :]
        self.probe.noise = norms[:, :, None] * unit.expand(-1, 2, -1)
        self.assertTrue(torch.equal(self.logits("noise"), reference))

    def test_full_rank_null_deltas_are_zero_in_the_probe_loop(self):
        tokens = np.concatenate([np.ones((2, 1), dtype=np.int64),
                                 np.random.default_rng(4).integers(0, 97, (2, 24))], axis=1)
        positions = np.array([[2, 5, 7, 11], [1, 3, 8, 12]])
        dirs = {"null": np.stack([R8.NULL.null_directions(i, self.n_layers, self.d_c)
                                  for i in range(8)])}
        deltas, norms, _, spread = R8.run_r2(self.model, self.probe, self.forward, self.inverse,
                                             tokens, positions, [4, 8, self.d_c], dirs, 2, 6, "cpu")
        self.assertEqual(spread, 0.0)
        self.assertTrue(np.all(deltas["truncation"][:, -1] == 0))
        self.assertTrue(np.all(deltas["null"][:, -1] == 0))
        self.assertTrue(np.all(norms[:, -1] == 0))
        self.assertTrue(np.all(norms[:, 0] > 0))
        self.assertGreater(np.abs(deltas["null"][:, 0]).max(), 0)

    def test_recorded_norms_are_the_truncation_streams_residuals(self):
        positions, rank = [4, 7], 5
        captured, clean = {}, {}

        def capture(store, layer):
            def hook(module, inputs, output):
                store[layer] = output.detach().clone()
            return hook

        # prepend=True: the capture sees each layer's latent before the probe truncates it,
        # on the truncation run's own (already perturbed upstream) stream.
        handles = [norm.register_forward_hook(capture(captured, i), prepend=True)
                   for i, norm in enumerate(self.layers)]
        _, norms = self.truncate(positions, rank)
        for handle in handles:
            handle.remove()
        handles = [norm.register_forward_hook(capture(clean, i), prepend=True)
                   for i, norm in enumerate(self.layers)]
        self.logits()
        for handle in handles:
            handle.remove()

        dropped = np.ones(self.d_c, dtype=bool)
        dropped[self.order[:rank]] = False
        rows = torch.arange(2)
        for layer in range(self.n_layers):
            expected = captured[layer][rows, positions][:, dropped].norm(dim=-1)
            self.assertTrue(torch.allclose(norms[layer].double(), expected, rtol=1e-6, atol=0))
            on_clean = clean[layer][rows, positions][:, dropped].norm(dim=-1)
            if layer == 0:
                self.assertTrue(torch.allclose(expected, on_clean))
            else:  # later layers' norms come from the truncation stream, not the clean one
                self.assertFalse(torch.allclose(expected, on_clean))

        # The null adds a perturbation of exactly that norm (checked at layer 0, whose
        # input is unperturbed in both runs).
        unit = torch.as_tensor(R8.NULL.null_directions(3, self.n_layers, self.d_c))[:, None, :]
        self.probe.noise = norms[:, :, None] * unit.expand(-1, 2, -1)
        self.probe.positions = torch.tensor(positions)
        after = {}
        handle = self.layers[0].register_forward_hook(capture(after, 0))
        self.logits("noise")
        handle.remove()
        added = (after[0] - clean[0])[rows, positions].norm(dim=-1)
        self.assertTrue(torch.allclose(added, norms[0].double(), rtol=1e-6))
        others = torch.ones(20, dtype=torch.bool)
        others[positions[0]] = False
        self.assertTrue(torch.equal(after[0][0, others], clean[0][0, others]))

    def test_joint_mask_with_one_position_equals_single_probe_truncation(self):
        positions, rank = [4, 7], 5
        single, _ = self.truncate(positions, rank)
        masks = R8.coordinate_masks(self.forward, self.inverse, rank)  # (n_layers, d_c)
        self.assertEqual(int(masks[0].sum()), rank)
        self.probe.joint_positions = torch.tensor(positions)[:, None]
        self.probe.joint_masks = [torch.as_tensor(masks[layer]).expand(2, 1, -1)
                                  for layer in range(self.n_layers)]
        self.assertTrue(torch.equal(self.logits("joint"), single))

    def test_joint_loop_matches_separate_truncations_for_one_position(self):
        tokens = np.concatenate([np.ones((1, 1), dtype=np.int64),
                                 np.random.default_rng(6).integers(0, 97, (1, 19))], axis=1)
        positions = np.array([[6]])
        masks = {r: R8.coordinate_masks(self.forward, self.inverse, r) for r in (4, 8, self.d_c)}
        r_stars = np.array([[4, self.d_c]])
        joint = R8.run_r5(self.model, self.probe, masks, tokens, positions, r_stars, "cpu")
        self.assertTrue(np.all(joint[0, 1] == 0))  # full rank changes nothing
        self.assertTrue(np.all(joint[0, 0, :6] == 0))  # causal
        x = torch.from_numpy(np.repeat(tokens[:, :-1], 2, axis=0))
        y = torch.from_numpy(np.repeat(tokens[:, 1:], 2, axis=0))
        self.probe.set_truncation(self.forward, self.inverse, 4)
        self.probe.positions, self.probe.mode = torch.tensor([6, 6]), "probe"
        with torch.no_grad():
            cut = R8.PUBLIC.full_losses(self.model, R8.PUBLIC.hidden_states(self.model, x), y)
            self.probe.mode = "off"
            base = R8.PUBLIC.full_losses(self.model, R8.PUBLIC.hidden_states(self.model, x), y)
        self.assertTrue(np.array_equal(joint[0, 0], (cut - base)[0].numpy().astype(np.float32)))
        summary = R8.joint_summary(np.repeat(joint, 2, axis=0), np.repeat(positions, 2, axis=0),
                                   np.repeat(r_stars, 2, axis=0), self.d_c, 1000)
        self.assertEqual(summary["max_safe"]["max_position_loss_change_nats"]["mean"], 0.0)
        self.assertEqual(summary["mean_safe"]["mean_rank_sum_per_sequence"], 4.0)

    def test_calibration_mean_and_covariance_root(self):
        row = np.concatenate([[1], np.random.default_rng(3).integers(0, 97, 16)]).astype(np.int64)
        cc, _, _, mean = R8.calibrate(self.model, self.probe, [row], self.d_c, "cpu")
        captured = {}
        handle = self.layers[1].register_forward_hook(
            lambda m, i, o: captured.__setitem__(1, o.detach()))
        with torch.no_grad():
            R8.PUBLIC.hidden_states(self.model, torch.from_numpy(row[None, :-1]))
        handle.remove()
        latent = captured[1].reshape(-1, self.d_c)
        self.assertTrue(np.allclose(mean[1], latent.mean(dim=0).numpy()))
        root = R8.sqrt_covariances(cc, mean)[1]
        centered = (latent - latent.mean(dim=0)).numpy()
        self.assertTrue(np.allclose(root @ root, centered.T @ centered / len(centered), atol=1e-10))


if __name__ == "__main__":
    unittest.main()
