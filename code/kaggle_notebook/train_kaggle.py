"""
ElasticMLA Experiment 1 - scale-up training on Kaggle GPU (P100 / T4x2).
Self-contained script (Kaggle "script" kernel can't easily import local packages,
so the elastic_mla module is inlined below).

Usage: push with
  kaggle kernels push -p code/kaggle_notebook
then monitor with
  kaggle kernels status softkleenex/elastic-mla-exp1-scaleup
and pull output with
  kaggle kernels output softkleenex/elastic-mla-exp1-scaleup -p ./kaggle_output
"""
import subprocess, sys

# Kaggle's preinstalled torch can lack compiled kernels for the GPU actually
# assigned to the session (observed: "CUDA error: no kernel image is available
# for execution on the device" on a P100 with a too-new torch build). Reinstall
# a broadly-compatible torch (cu118 wheel covers Pascal/Turing/Ampere) BEFORE
# importing torch anywhere else in this process.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "torch==2.1.2", "--index-url", "https://download.pytorch.org/whl/cu118"],
    check=False,
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=False)

import os, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch version:", torch.__version__, flush=True)
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0), flush=True)
    print("cuda capability:", torch.cuda.get_device_capability(0), flush=True)

# ============ elastic_mla/mla.py (inlined) ============
"""
Minimal DeepSeek-V2 style Multi-head Latent Attention (MLA) implementation.

Reference: DeepSeek-V2 (arXiv:2405.04434).
- KV is compressed into a shared low-rank latent c_KV (dim d_c).
- Q is also low-rank compressed (dim d_c' ) for training memory efficiency (optional here).
- RoPE is "decoupled": a small extra per-head dimension carries positional info
  and is NOT compressed, so latent c_KV stays purely content-based.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D_rope)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t, device, dtype):
        freqs = torch.einsum("i,j->ij", t.float(), self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class MultiHeadLatentAttention(nn.Module):
    """
    d_model:   model hidden size
    n_heads:   number of attention heads
    d_head:    per-head dim for the "content" part (non-RoPE)
    d_rope:    per-head dim carrying RoPE (decoupled, shared K rope across heads)
    d_c:       KV latent compression dim   (this is the thing we want to study/vary)
    d_c_q:     Query latent compression dim (optional, can equal d_c)
    """

    def __init__(self, d_model, n_heads, d_head=64, d_rope=32, d_c=256, d_c_q=None, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_rope = d_rope
        self.d_c = d_c
        d_c_q = d_c_q or d_c

        # --- KV latent compression / decompression ---
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)              # down-proj to latent
        self.W_UK = nn.Linear(d_c, n_heads * d_head, bias=False)      # up-proj -> K content
        self.W_UV = nn.Linear(d_c, n_heads * d_head, bias=False)      # up-proj -> V
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)            # decoupled rope key (shared across heads)

        # --- Query latent compression / decompression ---
        self.W_DQ = nn.Linear(d_model, d_c_q, bias=False)
        self.W_UQ = nn.Linear(d_c_q, n_heads * d_head, bias=False)
        self.W_QR = nn.Linear(d_c_q, n_heads * d_rope, bias=False)    # per-head rope query

        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.rope = RotaryEmbedding(d_rope)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_head + d_rope)

    def forward(self, x, attn_mask=None, return_latent=False, rank_mask=None):
        """
        x: (B, T, d_model)
        rank_mask: optional (d_c,) boolean/float mask to zero out latent dims
                   -> lets us simulate "using only rank r" at inference time.
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        c_kv = self.W_DKV(x)  # (B, T, d_c)  <-- THE latent we care about
        if rank_mask is not None:
            c_kv = c_kv * rank_mask

        k_content = self.W_UK(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        v = self.W_UV(c_kv).view(B, T, self.n_heads, self.d_head).transpose(1, 2)          # (B,H,T,Dh)
        k_rope = self.W_KR(x).view(B, T, 1, self.d_rope).transpose(1, 2)                   # (B,1,T,Dr)

        c_q = self.W_DQ(x)
        q_content = self.W_UQ(c_q).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q_rope = self.W_QR(c_q).view(B, T, self.n_heads, self.d_rope).transpose(1, 2)

        pos = torch.arange(T, device=device)
        cos, sin = self.rope(pos, device, dtype)  # (T, Dr)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)
        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)

        q = torch.cat([q_content, q_rope], dim=-1)  # (B,H,T,Dh+Dr)
        k = torch.cat([k_content, k_rope], dim=-1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,T,Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        out = self.W_O(out)

        if return_latent:
            return out, c_kv
        return out


# ============ elastic_mla/model.py (inlined) ============
"""
Small GPT-style LM using MultiHeadLatentAttention blocks.
Sized to run comfortably on Apple M4 Pro (MPS) or a single 4090.
"""
import torch
import torch.nn as nn
# (MultiHeadLatentAttention defined above, inlined)


class MLP(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_rope, d_c, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, n_heads, d_head, d_rope, d_c, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x, return_latent=False, rank_mask=None):
        h = self.ln1(x)
        if return_latent:
            a, c_kv = self.attn(h, return_latent=True, rank_mask=rank_mask)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x, c_kv
        else:
            x = x + self.attn(h, rank_mask=rank_mask)
            x = x + self.mlp(self.ln2(x))
            return x


class MLAGPT(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_layers=8, n_heads=6,
                 d_head=64, d_rope=32, d_c=256, max_len=1024, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_head, d_rope, d_c, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.max_len = max_len
        self.d_c = d_c
        self.n_layers = n_layers
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, rank_mask=None, layer_idx_for_latent=None):
        x = self.drop(self.tok_emb(idx))
        latents = {}
        for i, blk in enumerate(self.blocks):
            want_latent = (layer_idx_for_latent is not None and i == layer_idx_for_latent)
            if want_latent:
                x, c_kv = blk(x, return_latent=True, rank_mask=rank_mask)
                latents[i] = c_kv
            else:
                x = blk(x, rank_mask=rank_mask if layer_idx_for_latent is None else None)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        if layer_idx_for_latent is not None:
            return logits, loss, latents.get(layer_idx_for_latent)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============ data prep (TinyStories, larger subset than local run) ============
import tiktoken
from datasets import load_dataset

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

# Fail fast with a clear message if the GPU still can't run kernels after the
# torch reinstall, instead of burning the whole session on data download first.
if device == "cuda":
    try:
        _t = torch.randn(64, 64, device=device)
        _ = (_t @ _t).sum().item()
        print("CUDA sanity matmul OK", flush=True)
    except Exception as e:
        print("CUDA sanity check FAILED:", repr(e), flush=True)
        print("Falling back to CPU for this run.", flush=True)
        device = "cpu"

WORK_DIR = "/kaggle/working"
DATA_DIR = os.path.join(WORK_DIR, "data")
CKPT_DIR = os.path.join(WORK_DIR, "ckpt")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token

TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")

if not (os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN)):
    print("tokenizing TinyStories (larger subset for scale-up run)...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    N_DOCS = 400000   # 10x the local pilot
    tokens = []
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i >= N_DOCS:
            break
        ids = enc.encode_ordinary(ex["text"])
        tokens.extend(ids)
        tokens.append(eot)
        if i % 50000 == 0:
            print(i, "docs,", len(tokens), "tokens,", round(time.time() - t0, 1), "s", flush=True)
    tokens = np.array(tokens, dtype=np.uint16)
    n = len(tokens)
    split = int(n * 0.98)
    tokens[:split].tofile(TRAIN_BIN)
    tokens[split:].tofile(VAL_BIN)
    print("train tokens:", split, "val tokens:", n - split, flush=True)

train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

# ============ model / training config (114M-ish) ============
VOCAB_SIZE = 50257
BLOCK_SIZE = 512
BATCH_SIZE = 48
D_MODEL = 768
N_LAYERS = 12
N_HEADS = 12
D_HEAD = 64
D_ROPE = 32
D_C = 384            # KV latent dim -- primary axis of interest for ElasticMLA
MAX_STEPS = 8000
EVAL_INTERVAL = 250
EVAL_ITERS = 40
LR = 3e-4
WARMUP = 200

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - BLOCK_SIZE - 1, size=BATCH_SIZE)
    x = np.stack([data[i:i + BLOCK_SIZE].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + BLOCK_SIZE].astype(np.int64) for i in ix])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

model = MLAGPT(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
               d_head=D_HEAD, d_rope=D_ROPE, d_c=D_C, max_len=BLOCK_SIZE, dropout=0.0).to(device)
print("params:", model.num_params() / 1e6, "M", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

def lr_at(step):
    if step < WARMUP:
        return LR * step / WARMUP
    progress = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return 0.1 * LR + 0.9 * LR * 0.5 * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch("val")
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

log_f = open(os.path.join(CKPT_DIR, "train_log.jsonl"), "w")
t0 = time.time()
model.train()
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
for step in range(1, MAX_STEPS + 1):
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    if step % 20 == 0:
        log_f.write(json.dumps({"step": step, "train_loss": loss.item(), "lr": lr,
                                  "elapsed_s": round(time.time() - t0, 1)}) + "\n")
        log_f.flush()

    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
        val_loss = evaluate()
        log_f.write(json.dumps({"step": step, "val_loss": val_loss,
                                  "elapsed_s": round(time.time() - t0, 1)}) + "\n")
        log_f.flush()
        print(f"step {step}/{MAX_STEPS}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
        torch.save({"model": model.state_dict(),
                     "config": dict(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
                                     n_heads=N_HEADS, d_head=D_HEAD, d_rope=D_ROPE, d_c=D_C,
                                     max_len=BLOCK_SIZE),
                     "step": step},
                   os.path.join(CKPT_DIR, "latest.pt"))

log_f.close()
print("DONE", flush=True)
