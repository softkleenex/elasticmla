
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import MLAGPT
ckpt = torch.load("experiments/exp0_rank_variance/ckpt/latest.pt", map_location="cpu", weights_only=False)
print("saved step:", ckpt["step"])
print("config:", ckpt["config"])
model = MLAGPT(**ckpt["config"])
missing, unexpected = model.load_state_dict(ckpt["model"], strict=True), None
print("state_dict load: OK (strict=True, no missing/unexpected keys -> no error raised)")
print("num params:", model.num_params())
# NaN/Inf 체크 - 강종/손상됐다면 weight가 깨져있을 수 있음
import math
bad = 0
for n,p in model.named_parameters():
    if torch.isnan(p).any() or torch.isinf(p).any():
        bad += 1
        print("BAD TENSOR:", n)
print("NaN/Inf 있는 파라미터 텐서 개수:", bad)
