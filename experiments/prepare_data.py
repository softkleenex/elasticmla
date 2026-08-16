
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np
import tiktoken
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "exp0_rank_variance", "data")
os.makedirs(OUT_DIR, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token  # 50256, use as separator

print("loading TinyStories (streaming subset)...")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

N_DOCS = 40000     # small subset, enough for a few epochs of a small model
tokens = []
t0 = time.time()
for i, ex in enumerate(ds):
    if i >= N_DOCS:
        break
    ids = enc.encode_ordinary(ex["text"])
    tokens.extend(ids)
    tokens.append(eot)
    if i % 5000 == 0:
        print(i, "docs,", len(tokens), "tokens,", round(time.time()-t0,1), "s")

tokens = np.array(tokens, dtype=np.uint16)
n = len(tokens)
split = int(n * 0.98)
train_ids = tokens[:split]
val_ids = tokens[split:]

train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))
print("train tokens:", len(train_ids), "val tokens:", len(val_ids))
print("vocab size:", enc.n_vocab)
