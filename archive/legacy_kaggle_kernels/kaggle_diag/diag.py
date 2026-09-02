
import subprocess, sys
print("=== reinstall torch 2.2.0+cu118 WITH deps (need bundled cu11 runtime libs) ===", flush=True)
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--force-reinstall",
     "torch==2.2.0", "--index-url", "https://download.pytorch.org/whl/cu118"],
    capture_output=True, text=True
)
print("returncode:", r.returncode, flush=True)
print("STDOUT tail:", r.stdout[-1500:], flush=True)
print("STDERR tail:", r.stderr[-1500:], flush=True)

import torch
print("torch version:", torch.__version__, flush=True)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), flush=True)
    print("capability:", torch.cuda.get_device_capability(0), flush=True)
    try:
        t = torch.randn(64, 64, device="cuda")
        _ = (t @ t).sum().item()
        print("CUDA matmul sanity: OK", flush=True)
    except Exception as e:
        print("CUDA matmul sanity FAILED:", repr(e), flush=True)
else:
    print("cuda not available", flush=True)
