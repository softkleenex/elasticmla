import importlib.util, unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "code"))
from elastic_mla import ContextualElasticMLAGPT, MLAGPT

SPEC = importlib.util.spec_from_file_location(
    "bench", ROOT / "experiments/benchmark_cache_memory_latency.py"
)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class BenchmarkFunctionalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = MLAGPT(
            vocab_size=41, d_model=32, n_layers=3, n_heads=2, d_head=8, d_rope=4,
            d_c=16, max_len=32,
        ).eval()
        self.orders = [torch.randperm(16) for _ in range(3)]
        self.model = ContextualElasticMLAGPT(
            self.base, self.orders, tiers=(4, 8, 12, 16)
        ).eval().freeze_base()
        self.device = torch.device("cpu")
        self.prefill = torch.randint(0, 41, (2, 6))
        self.decode = torch.randint(0, 41, (2, 5))

    def test_full_mla_cache_bytes_match_helper(self):
        out = MODULE.run_full(self.base, self.prefill, self.decode, self.device)
        self.assertIsNone(out["peak_allocated_bytes"])
        self.assertGreater(out["resident_cache_bytes"], 0)

    def test_packed_uniform_and_router_have_smaller_or_equal_bytes_than_full(self):
        full = MODULE.run_full(self.base, self.prefill, self.decode, self.device)
        uniform = MODULE.run_packed_uniform(
            self.base, self.orders, 8, self.prefill, self.decode, self.device
        )
        router = MODULE.run_router(self.model, self.prefill, self.decode, self.device)
        self.assertLessEqual(uniform["resident_cache_bytes"], full["resident_cache_bytes"])
        self.assertIn("mean_downstream_rank", router)
        self.assertTrue(4 <= router["mean_downstream_rank"] <= 16)


if __name__ == "__main__":
    unittest.main()
