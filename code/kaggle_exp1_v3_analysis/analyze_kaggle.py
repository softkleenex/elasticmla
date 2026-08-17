"""Kaggle self-contained Exp0-v3 replication on the completed 122M MLA checkpoint."""
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
import subprocess, sys, time

SCRIPT_START_TIME = time.time()

# Kaggle's preinstalled torch can lack compiled kernels for the GPU actually
# assigned to the session (observed: "CUDA error: no kernel image is available
# for execution on the device" on a P100 with a too-new torch build). Reinstall
# a broadly-compatible torch (cu118 wheel covers Pascal/Turing/Ampere) BEFORE
# importing torch anywhere else in this process.
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--force-reinstall",
     "torch==2.2.0", "--index-url", "https://download.pytorch.org/whl/cu118"],
    check=True,
)
print("torch reinstall returncode:", r.returncode, flush=True)

# torch==2.2.0 was built against the numpy<2 C ABI. Kaggle's preinstalled numpy is
# newer (2.x), which breaks torch<->numpy interop ("_ARRAY_API not found" /
# "RuntimeError: Numpy is not available" at torch.from_numpy time). Pin a compatible
# numpy alongside the torch downgrade.
r2 = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.4"],
    check=True,
)
print("numpy reinstall returncode:", r2.returncode, flush=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "tiktoken"],
    check=True,
)

import os, json, math
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
                # Returning one layer's latent must not disable rank truncation in
                # the other layers. Keep this in sync with elastic_mla/model.py.
                x = blk(x, rank_mask=rank_mask)
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



import argparse, gc
from pathlib import Path
from typing import Iterable, Sequence

INPUT_CANDIDATES = sorted(Path("/kaggle/input").glob("*/ckpt/latest.pt"))
if not INPUT_CANDIDATES:
    raise FileNotFoundError("kernel-source checkpoint not mounted under /kaggle/input")
INPUT_ROOT = INPUT_CANDIDATES[0].parents[1]
DATA_DIR = INPUT_ROOT / "data"
CKPT_DIR = INPUT_ROOT / "ckpt"
OUT_DIR = Path("/kaggle/working/results")
print(f"input root: {INPUT_ROOT}", flush=True)

DEFAULT_RANK_GRID = (16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384)
CALIBRATION_SEED_BASE = 12_341
EVALUATION_SEED = 23_452
BOOTSTRAP_SEED = 34_563


def choose_device(requested: str = "auto") -> torch.device:
    """Select cuda > mps > cpu, or validate an explicit selection."""
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def empty_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def normalize_rank_grid(ranks: Iterable[int], d_c: int) -> list[int]:
    """Return a sorted valid grid that always includes exact full rank."""
    grid = sorted({int(rank) for rank in ranks if 0 < int(rank) <= d_c})
    if d_c not in grid:
        grid.append(d_c)
    if not grid:
        raise ValueError("rank grid contains no positive rank at or below d_c")
    return grid


def valid_probe_positions(sequence_length: int, horizon: int) -> np.ndarray:
    """Positions with exactly ``horizon`` available subsequent loss terms."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    # The window for pos is [pos + 1, pos + 1 + horizon).
    count = sequence_length - horizon
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(count, dtype=np.int64)


def suffix_all_satisfy_r_star(
    deltas: Sequence[float], ranks: Sequence[int], epsilon: float
) -> int:
    """Smallest rank whose value and every higher-rank value satisfy epsilon."""
    values = np.asarray(deltas, dtype=np.float64)
    if values.shape != (len(ranks),):
        raise ValueError("deltas and ranks must have the same non-empty length")
    satisfies = np.isfinite(values) & (values <= epsilon)
    suffix_satisfies = np.logical_and.accumulate(satisfies[::-1])[::-1]
    indices = np.flatnonzero(suffix_satisfies)
    return int(ranks[int(indices[0])]) if len(indices) else int(ranks[-1])


def is_nonmonotonic(deltas: Sequence[float], tolerance: float = 1e-8) -> bool:
    """Whether loss effect rises at any adjacent increase in retained rank."""
    values = np.asarray(deltas, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return bool(len(finite) > 1 and np.any(np.diff(finite) > tolerance))


def sequence_cluster_bootstrap_mean_ci(
    values_by_sequence: dict[int, Sequence[float]],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI after resampling whole sequences with replacement."""
    if n_bootstrap < 1000:
        raise ValueError("headline sequence-cluster bootstrap requires >= 1000 draws")
    if not values_by_sequence:
        raise ValueError("cannot bootstrap an empty collection")
    clusters = [np.asarray(v, dtype=np.float64) for _, v in sorted(values_by_sequence.items())]
    if any(cluster.size == 0 for cluster in clusters):
        raise ValueError("every sequence cluster must contain at least one value")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n_clusters = len(clusters)
    for draw in range(n_bootstrap):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        total = sum(float(clusters[i].sum()) for i in sampled)
        count = sum(int(clusters[i].size) for i in sampled)
        means[draw] = total / count
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def per_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    batch, length, vocab = logits.shape
    losses = F.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    )
    return losses.view(batch, length).detach().cpu().numpy()


def forward_with_layer_masks(
    model: MLAGPT,
    idx: torch.Tensor,
    *,
    channel_masks: torch.Tensor | None = None,
    probe_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Model forward with a different position-isolated channel mask per layer.

    ``channel_masks`` has shape (n_layers, batch, d_c).  For item b, its mask is
    applied only at ``probe_positions[b]``; all other positions remain full rank.
    Keeping this helper local avoids changing the training model's public API.
    """
    if (channel_masks is None) != (probe_positions is None):
        raise ValueError("channel_masks and probe_positions must be supplied together")
    if channel_masks is not None:
        expected = (model.n_layers, idx.shape[0], model.d_c)
        if tuple(channel_masks.shape) != expected:
            raise ValueError(f"channel_masks shape must be {expected}")
        if tuple(probe_positions.shape) != (idx.shape[0],):
            raise ValueError("probe_positions must have shape (batch,)")

    hidden = model.drop(model.tok_emb(idx))
    batch_indices = torch.arange(idx.shape[0], device=idx.device)
    for layer_idx, block in enumerate(model.blocks):
        rank_mask = None
        if channel_masks is not None:
            # Only one (source) position per batch item is intervened on.
            rank_mask = torch.ones(
                idx.shape[0], idx.shape[1], model.d_c,
                device=hidden.device, dtype=hidden.dtype,
            )
            rank_mask[batch_indices, probe_positions] = channel_masks[layer_idx].to(hidden.dtype)
        hidden = block(hidden, rank_mask=rank_mask)
    return model.head(model.ln_f(hidden))


def forward_with_all_latents(
    model: MLAGPT, idx: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return logits and every layer's latent from one calibration forward."""
    hidden = model.drop(model.tok_emb(idx))
    latents = []
    for block in model.blocks:
        hidden, latent = block(hidden, return_latent=True)
        latents.append(latent)
    return model.head(model.ln_f(hidden)), latents


def sample_starts(
    rng: np.random.Generator, low: int, high_exclusive: int, count: int
) -> np.ndarray:
    population = high_exclusive - low
    if population < count:
        raise ValueError(f"cannot sample {count} unique starts from {population}")
    return np.sort(rng.choice(population, size=count, replace=False) + low)


def sequence_batch(data: np.memmap, starts: Sequence[int], block_size: int) -> np.ndarray:
    return np.stack(
        [np.asarray(data[int(s): int(s) + block_size + 1], dtype=np.int64) for s in starts]
    )


def calibrate_layer_orders(
    model: MLAGPT,
    data: np.memmap,
    starts_by_repeat: Sequence[Sequence[int]],
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream calibration and return aggregate orders plus repeat saliencies."""
    repeats = len(starts_by_repeat)
    saliency_by_repeat = np.zeros((repeats, model.n_layers, model.d_c), dtype=np.float64)
    for repeat_idx, starts in enumerate(starts_by_repeat):
        for offset in range(0, len(starts), batch_size):
            tokens = sequence_batch(data, starts[offset: offset + batch_size], block_size)
            x = torch.from_numpy(tokens[:, :-1]).to(device)
            y = torch.from_numpy(tokens[:, 1:]).to(device)
            model.zero_grad(set_to_none=True)
            logits, latents = forward_with_all_latents(model, x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            grads = torch.autograd.grad(loss, latents, only_inputs=True)
            for layer_idx, (latent, grad) in enumerate(zip(latents, grads)):
                # loss is a token mean; restore batch-size weighting so a short
                # final microbatch does not count like a full microbatch.
                saliency_by_repeat[repeat_idx, layer_idx] += (
                    (grad * latent.detach()).abs().sum(dim=(0, 1)).cpu().double().numpy()
                    * latent.shape[0]
                )
            del tokens, x, y, logits, loss, latents, grads
            empty_device_cache(device)
        print(f"calibration repeat={repeat_idx + 1}/{repeats}", flush=True)
    aggregate = saliency_by_repeat.mean(axis=0)
    orders = np.argsort(-aggregate, axis=1)
    return orders, saliency_by_repeat


def make_channel_masks(
    layer_orders: np.ndarray, rank: int, batch_size: int, device: torch.device
) -> torch.Tensor:
    masks = torch.zeros(
        layer_orders.shape[0], batch_size, layer_orders.shape[1],
        dtype=torch.float32, device=device,
    )
    for layer_idx, order in enumerate(layer_orders):
        keep = torch.as_tensor(order[:rank].copy(), dtype=torch.long, device=device)
        masks[layer_idx, :, keep] = 1.0
    return masks


def token_type(token_id: int, encoder) -> str:
    text = encoder.decode([int(token_id)])
    stripped = text.strip()
    if not stripped:
        return "space"
    if all(char in ".,!?;:'\"-()" for char in stripped):
        return "punct"
    if stripped[0].isupper():
        return "capitalized"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-horizon", type=int, default=32)
    parser.add_argument("--n-calib-sequences", type=int, default=16,
                        help="sequences per calibration-seed repeat")
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--n-eval-sequences", type=int, default=24)
    parser.add_argument("--positions-per-sequence", type=int, default=32)
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--probe-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--nonmonotonic-tolerance", type=float, default=0.0)
    parser.add_argument("--rank-grid", type=int, nargs="+", default=list(DEFAULT_RANK_GRID))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=CKPT_DIR / "latest.pt")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "val.bin")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_repeats < 2:
        raise ValueError("calibration_repeats must be >= 2")
    if args.calibration_batch_size <= 0 or args.probe_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.positions_per_sequence <= 0:
        raise ValueError("positions_per_sequence must be positive")
    if args.bootstrap_draws < 1000:
        raise ValueError("bootstrap_draws must be >= 1000")

    device = choose_device(args.device)
    torch.manual_seed(CALIBRATION_SEED_BASE)
    np.random.seed(CALIBRATION_SEED_BASE)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(CALIBRATION_SEED_BASE)
    print(f"device: {device}", flush=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MLAGPT(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    block_size = int(config["max_len"])
    d_c = int(config["d_c"])
    rank_grid = normalize_rank_grid(args.rank_grid, d_c)
    valid_positions = valid_probe_positions(block_size, args.future_horizon)
    if len(valid_positions) == 0:
        raise ValueError("future horizon leaves no valid probe positions")

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n_possible_starts = len(data) - block_size
    # A sequence-sized gap makes the calibration and evaluation token spans exact
    # and disjoint, not merely the sampled start indices.
    calibration_high = int(n_possible_starts * 0.55)
    evaluation_low = calibration_high + block_size
    if evaluation_low >= n_possible_starts:
        raise ValueError("validation stream is too short for disjoint regions")

    calibration_starts = []
    for repeat_idx in range(args.calibration_repeats):
        rng = np.random.default_rng(CALIBRATION_SEED_BASE + repeat_idx)
        calibration_starts.append(
            sample_starts(rng, 0, calibration_high, args.n_calib_sequences)
        )
    eval_rng = np.random.default_rng(EVALUATION_SEED)
    eval_starts = sample_starts(
        eval_rng, evaluation_low, n_possible_starts, args.n_eval_sequences
    )

    layer_orders, saliency_repeats = calibrate_layer_orders(
        model, data, calibration_starts,
        block_size=block_size,
        batch_size=args.calibration_batch_size,
        device=device,
    )
    expected_channels = np.arange(d_c)
    if any(not np.array_equal(np.sort(order), expected_channels) for order in layer_orders):
        raise AssertionError("each layer channel order must be a permutation of 0..d_c-1")

    eval_tokens = sequence_batch(data, eval_starts, block_size)
    baseline_loss = np.empty((args.n_eval_sequences, block_size), dtype=np.float32)
    with torch.no_grad():
        for offset in range(0, args.n_eval_sequences, args.probe_batch_size):
            tokens = eval_tokens[offset: offset + args.probe_batch_size]
            x = torch.from_numpy(tokens[:, :-1]).to(device)
            y = torch.from_numpy(tokens[:, 1:]).to(device)
            logits = forward_with_layer_masks(model, x)
            baseline_loss[offset: offset + len(tokens)] = per_token_loss(logits, y)
            del x, y, logits
    empty_device_cache(device)

    positions_by_sequence: list[np.ndarray] = []
    for _ in range(args.n_eval_sequences):
        count = min(args.positions_per_sequence, len(valid_positions))
        positions_by_sequence.append(
            np.sort(eval_rng.choice(valid_positions, size=count, replace=False))
        )

    # Curves are indexed [sequence][position][rank].
    mean_curves = [
        np.empty((len(positions), len(rank_grid)), dtype=np.float32)
        for positions in positions_by_sequence
    ]
    max_curves = [np.empty_like(curves) for curves in mean_curves]

    for rank_idx, rank in enumerate(rank_grid):
        for seq_idx, positions in enumerate(positions_by_sequence):
            for offset in range(0, len(positions), args.probe_batch_size):
                pos_batch = positions[offset: offset + args.probe_batch_size]
                batch_count = len(pos_batch)
                tokens = eval_tokens[seq_idx: seq_idx + 1]
                x = torch.from_numpy(np.repeat(tokens[:, :-1], batch_count, axis=0)).to(device)
                y = torch.from_numpy(np.repeat(tokens[:, 1:], batch_count, axis=0)).to(device)
                pos_tensor = torch.as_tensor(pos_batch.copy(), dtype=torch.long, device=device)
                channel_masks = make_channel_masks(layer_orders, rank, batch_count, device)
                with torch.no_grad():
                    logits = forward_with_layer_masks(
                        model, x, channel_masks=channel_masks, probe_positions=pos_tensor
                    )
                    losses = per_token_loss(logits, y)
                for local_idx, pos in enumerate(pos_batch):
                    start = int(pos) + 1
                    stop = start + args.future_horizon
                    delta = losses[local_idx, start:stop] - baseline_loss[seq_idx, start:stop]
                    if len(delta) != args.future_horizon:
                        raise AssertionError("probe did not have the exact requested horizon")
                    mean_curves[seq_idx][offset + local_idx, rank_idx] = float(delta.mean())
                    max_curves[seq_idx][offset + local_idx, rank_idx] = float(delta.max())
                del x, y, pos_tensor, channel_masks, logits, losses
        empty_device_cache(device)
        print(f"rank={rank} complete", flush=True)

    import tiktoken

    encoder = tiktoken.get_encoding("gpt2")
    records = []
    mean_by_sequence: dict[int, list[int]] = {}
    max_by_sequence: dict[int, list[int]] = {}
    for seq_idx, positions in enumerate(positions_by_sequence):
        mean_by_sequence[seq_idx] = []
        max_by_sequence[seq_idx] = []
        for pos_idx, pos_value in enumerate(positions):
            pos = int(pos_value)
            mean_values = mean_curves[seq_idx][pos_idx].astype(float).tolist()
            max_values = max_curves[seq_idx][pos_idx].astype(float).tolist()
            r_star_mean = suffix_all_satisfy_r_star(mean_values, rank_grid, args.epsilon)
            r_star_max = suffix_all_satisfy_r_star(max_values, rank_grid, args.epsilon)
            input_token_id = int(eval_tokens[seq_idx, pos])  # x_eval[b, pos]
            mean_nonmonotonic = is_nonmonotonic(mean_values, args.nonmonotonic_tolerance)
            max_nonmonotonic = is_nonmonotonic(max_values, args.nonmonotonic_tolerance)
            mean_by_sequence[seq_idx].append(r_star_mean)
            max_by_sequence[seq_idx].append(r_star_max)
            records.append({
                "seq": seq_idx,
                "sequence_start": int(eval_starts[seq_idx]),
                "pos": pos,
                "input_token_id": input_token_id,
                "input_token_type": token_type(input_token_id, encoder),
                "future_horizon": args.future_horizon,
                "r_star_future_mean": r_star_mean,
                "r_star_future_max": r_star_max,
                "future_mean_delta_by_rank": dict(zip(map(str, rank_grid), mean_values)),
                "future_max_delta_by_rank": dict(zip(map(str, rank_grid), max_values)),
                "future_mean_nonmonotonic": mean_nonmonotonic,
                "future_max_nonmonotonic": max_nonmonotonic,
            })

    mean_rstars = np.asarray([r["r_star_future_mean"] for r in records])
    max_rstars = np.asarray([r["r_star_future_max"] for r in records])
    mean_ci = sequence_cluster_bootstrap_mean_ci(
        mean_by_sequence, n_bootstrap=args.bootstrap_draws, seed=BOOTSTRAP_SEED
    )
    max_ci = sequence_cluster_bootstrap_mean_ci(
        max_by_sequence, n_bootstrap=args.bootstrap_draws, seed=BOOTSTRAP_SEED + 1
    )

    repeat_top32_overlap = []
    top_k = min(32, d_c)
    for layer_idx in range(model.n_layers):
        repeat_orders = np.argsort(-saliency_repeats[:, layer_idx], axis=1)
        overlaps = []
        for left in range(args.calibration_repeats):
            for right in range(left + 1, args.calibration_repeats):
                overlaps.append(len(set(repeat_orders[left, :top_k]) &
                                    set(repeat_orders[right, :top_k])) / top_k)
        repeat_top32_overlap.append(float(np.mean(overlaps)))

    summary = {
        "method": "per-layer gradient*activation ordering; one-position truncation; fixed future horizon",
        "scope_limitation": "full-attention training/truncation simulation, not cache-aware decoding",
        "device": str(device),
        "checkpoint_step": int(checkpoint["step"]),
        "rank_grid": rank_grid,
        "epsilon_nats": args.epsilon,
        "future_horizon_exact": args.future_horizon,
        "n_layers": model.n_layers,
        "d_c": d_c,
        "n_calibration_sequences_per_repeat": args.n_calib_sequences,
        "calibration_repeats": args.calibration_repeats,
        "n_evaluation_sequences": args.n_eval_sequences,
        "positions_per_sequence": args.positions_per_sequence,
        "n_positions_total": len(records),
        "seeds": {
            "calibration": [CALIBRATION_SEED_BASE + i for i in range(args.calibration_repeats)],
            "evaluation": EVALUATION_SEED,
            "bootstrap_mean": BOOTSTRAP_SEED,
            "bootstrap_max": BOOTSTRAP_SEED + 1,
        },
        "calibration_repeat_mean_pairwise_top32_overlap_by_layer": repeat_top32_overlap,
        "layer_channel_orders": layer_orders.tolist(),
        "future_mean": {
            "mean_r_star": float(mean_rstars.mean()),
            "sequence_cluster_bootstrap_95pct_ci": list(mean_ci),
            "bootstrap_draws": args.bootstrap_draws,
            "histogram": {str(rank): int((mean_rstars == rank).sum()) for rank in rank_grid},
            "nonmonotonic_frequency": float(np.mean(
                [record["future_mean_nonmonotonic"] for record in records]
            )),
            "nonmonotonic_count": int(sum(
                record["future_mean_nonmonotonic"] for record in records
            )),
        },
        "future_max": {
            "mean_r_star": float(max_rstars.mean()),
            "sequence_cluster_bootstrap_95pct_ci": list(max_ci),
            "bootstrap_draws": args.bootstrap_draws,
            "histogram": {str(rank): int((max_rstars == rank).sum()) for rank in rank_grid},
            "nonmonotonic_frequency": float(np.mean(
                [record["future_max_nonmonotonic"] for record in records]
            )),
            "nonmonotonic_count": int(sum(
                record["future_max_nonmonotonic"] for record in records
            )),
        },
        "token_attribution": "x_eval[b, pos] (the intervened source token)",
        "r_star_rule": "smallest grid rank for which this and every higher rank are <= epsilon",
        "nonmonotonic_rule": (
            "any adjacent raw delta increase greater than "
            f"{args.nonmonotonic_tolerance} as retained rank increases"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "exp0_v3_summary.json"
    records_path = args.output_dir / "exp0_v3_records.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with records_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {summary_path} and {records_path}", flush=True)


if __name__ == "__main__":
    main()
