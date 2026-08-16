import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT

device = "mps" if torch.backends.mps.is_available() else "cpu"
ckpt = torch.load("experiments/exp0_rank_variance/ckpt/latest.pt", map_location=device, weights_only=False)
model = MLAGPT(**ckpt["config"]).to(device).eval()
model.load_state_dict(ckpt["model"], strict=True)
torch.manual_seed(5)
idx = torch.randint(0, ckpt["config"]["vocab_size"], (1, 16), device=device)
with torch.no_grad():
    full, _ = model(idx)
    caches = None
    parts = []
    for i in range(idx.shape[1]):
        out, caches = model.forward_cached(idx[:, i:i+1], caches=caches)
        parts.append(out)
    cached = torch.cat(parts, dim=1)
max_diff = (full-cached).abs().max().item()
actual = model.cache_num_bytes(caches)
mha = model.theoretical_mha_cache_num_bytes(1, 16, full.dtype)
print("device", device)
print("max_abs_logit_diff", max_diff)
print("mla_cache_bytes", actual)
print("mha_cache_bytes", mha)
print("mla_over_mha_ratio", actual/mha)
assert torch.allclose(full, cached, rtol=1e-4, atol=2e-5)
