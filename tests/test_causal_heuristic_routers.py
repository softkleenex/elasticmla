import importlib.util, unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];S=importlib.util.spec_from_file_location("causal",ROOT/"experiments/evaluate_causal_heuristic_routers.py");M=importlib.util.module_from_spec(S);S.loader.exec_module(M)
class CausalHeuristicTests(unittest.TestCase):
 def test_upward_tier_quantization_and_clipping(self):
  got=M.quantize([-5,16,17,63,64,300],0,[16,64,160,256]);np.testing.assert_array_equal(got,[16,16,64,64,64,256])
 def test_bias_changes_rate_monotonically(self):
  scores=np.asarray([20,70,150]);tiers=[16,64,160,256]
  low=M.quantize(scores,-32,tiers);high=M.quantize(scores,32,tiers);self.assertTrue(np.all(low<=high))
 def test_token_type_uses_only_current_id(self):
  import tiktoken
  enc=tiktoken.get_encoding("gpt2");self.assertEqual(M.kind(13,enc),"punct");self.assertEqual(M.kind(220,enc),"space")
if __name__=="__main__":unittest.main()
