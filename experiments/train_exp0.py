
import os, sys, time, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np
import torch
from elastic_mla import MLAGPT

torch.manual_seed(1337)

DATA_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)
LOG_PATH = os.path.join(CKPT_DIR, "train_log.jsonl")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device, flush=True)

VOCAB_SIZE = 50257
BLOCK_SIZE = 256
BATCH_SIZE = 32
D_MODEL = 384
N_LAYERS = 6
N_HEADS = 6
D_HEAD = 64
D_ROPE = 32
D_C = 256          # KV latent dim -- the thing Experiment 1 will make adaptive
MAX_STEPS = 3000
EVAL_INTERVAL = 200
EVAL_ITERS = 40
LR = 3e-4
WARMUP = 100

train_data = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - BLOCK_SIZE - 1, size=BATCH_SIZE)
    x = np.stack([data[i:i+BLOCK_SIZE].astype(np.int64) for i in ix])
    y = np.stack([data[i+1:i+1+BLOCK_SIZE].astype(np.int64) for i in ix])
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y

model = MLAGPT(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
               d_head=D_HEAD, d_rope=D_ROPE, d_c=D_C, max_len=BLOCK_SIZE, dropout=0.0).to(device)
n_params = model.num_params()
print("params:", n_params/1e6, "M", flush=True)

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

log_f = open(LOG_PATH, "w")
t0 = time.time()
model.train()
for step in range(1, MAX_STEPS + 1):
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr

    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 20 == 0:
        rec = {"step": step, "train_loss": loss.item(), "lr": lr, "elapsed_s": round(time.time()-t0, 1)}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
        val_loss = evaluate()
        rec = {"step": step, "val_loss": val_loss, "elapsed_s": round(time.time()-t0, 1)}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"step {step}/{MAX_STEPS}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  elapsed={time.time()-t0:.1f}s", flush=True)
        torch.save({"model": model.state_dict(),
                    "config": dict(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
                                    n_heads=N_HEADS, d_head=D_HEAD, d_rope=D_ROPE, d_c=D_C,
                                    max_len=BLOCK_SIZE),
                    "step": step},
                   os.path.join(CKPT_DIR, "latest.pt"))

log_f.close()
print("DONE", flush=True)
