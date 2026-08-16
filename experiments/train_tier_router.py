"""Train a global token-tier router to imitate Exp0-v3 oracle ranks.

The v3 oracle intervenes on every MLA layer at one shared rank.  Accordingly this
script trains one global token-difficulty router from layer-0 pre-attention features
and applies its chosen tier to every layer.  Overlapping 256-token windows are kept
in the same split component to prevent token-level leakage.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from elastic_mla import GlobalElasticMLAGPT, MLAGPT

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"experiments/exp0_rank_variance/data/val.bin"
CKPT=ROOT/"experiments/exp0_rank_variance/ckpt/latest.pt"
SUMMARY=ROOT/"experiments/exp0_rank_variance/results/exp0_v3_summary.json"
RECORDS=ROOT/"experiments/exp0_rank_variance/results/exp0_v3_records.json"

def device_auto():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def quantize_up(values, tiers):
    values=torch.as_tensor(values,dtype=torch.long); tiers=torch.as_tensor(tiers,dtype=torch.long)
    if values.max()>tiers.max(): raise ValueError("largest tier must cover every oracle rank")
    return tiers[torch.searchsorted(tiers,values)]

def macro_f1(pred,target,tiers):
    scores=[]
    for tier in tiers:
        p,t=pred==tier,target==tier; tp=(p&t).sum().item(); fp=(p&~t).sum().item(); fn=(~p&t).sum().item()
        scores.append(0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    return float(np.mean(scores))

def nonoverlap_split(starts,window,seed):
    ordered=sorted(starts,key=starts.get); groups=[]; current=[]; end=-1
    for seq in ordered:
        st=starts[seq]
        if current and st>=end: groups.append(current); current=[]; end=-1
        current.append(seq); end=max(end,st+window)
    if current: groups.append(current)
    rng=np.random.RandomState(seed); rng.shuffle(groups)
    capacities={"train":16,"val":4,"test":4}; result={k:[] for k in capacities}
    for group in groups:
        candidates=[k for k in capacities if capacities[k]>=len(group)]
        if not candidates: raise RuntimeError("cannot fit overlap group into split capacities")
        dest=max(candidates,key=lambda k:capacities[k])
        result[dest].extend(group); capacities[dest]-=len(group)
    if any(capacities.values()): raise RuntimeError(f"split capacities not filled: {capacities}")
    return {k:sorted(v) for k,v in result.items()}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--objective",choices=("mean","max"),default="max")
    p.add_argument("--epochs",type=int,default=250); p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--lr",type=float,default=2e-3); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--tiers",type=int,nargs="+",default=[16,64,160,256]); args=p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); device=device_auto()
    ckpt=torch.load(CKPT,map_location=device,weights_only=False); base=MLAGPT(**ckpt["config"]).to(device).eval()
    base.load_state_dict(ckpt["model"],strict=True); summary=json.load(open(SUMMARY)); records=json.load(open(RECORDS))
    orders=[torch.tensor(o) for o in summary["layer_channel_orders"]]
    elastic=GlobalElasticMLAGPT(base,orders,tiers=args.tiers).to(device).freeze_base()
    starts={r["seq"]:r["sequence_start"] for r in records}; splits=nonoverlap_split(starts,ckpt["config"]["max_len"],args.seed)
    split_of={seq:name for name,seqs in splits.items() for seq in seqs}
    data=np.memmap(DATA,dtype=np.uint16,mode="r"); by_seq={seq:[] for seq in starts}
    for r in records: by_seq[r["seq"]].append(r)
    feature_rows=[]; labels=[]; split_names=[]
    with torch.no_grad():
        for seq in sorted(starts):
            st=starts[seq]; arr=np.array(data[st:st+ckpt["config"]["max_len"]],dtype=np.int64)
            x=torch.from_numpy(arr)[None].to(device); features=elastic.routing_features(x)
            for r in by_seq[seq]:
                feature_rows.append(features[0,r["pos"]].cpu())
                key="r_star_future_mean" if args.objective=="mean" else "r_star_future_max"
                labels.append(r[key]); split_names.append(split_of[seq])
    features=torch.stack(feature_rows); labels=quantize_up(labels,args.tiers)
    masks={name:torch.tensor([s==name for s in split_names]) for name in splits}
    train_labels=labels[masks["train"]]; counts=torch.tensor([(train_labels==t).sum() for t in args.tiers],dtype=torch.float32)
    weights=((counts.sum()/counts.clamp_min(1)).sqrt()); weights=(weights/weights.mean()).to(device)
    opt=torch.optim.AdamW(elastic.router.parameters(),lr=args.lr,weight_decay=1e-3); train_idx=torch.where(masks["train"])[0]
    gen=torch.Generator().manual_seed(args.seed); best=-1.; state=None
    for epoch in range(args.epochs):
        elastic.router.train(); perm=train_idx[torch.randperm(len(train_idx),generator=gen)]
        for st in range(0,len(perm),args.batch_size):
            ix=perm[st:st+args.batch_size]; logits=elastic.router(features[ix].to(device)); target=labels[ix].to(device)
            loss=elastic.router.supervised_loss(logits,target,weights); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        elastic.router.eval()
        with torch.no_grad():
            ix=torch.where(masks["val"])[0]; target=labels[ix].to(device)
            pred=elastic.router.select_ranks(elastic.router(features[ix].to(device)))
            score=macro_f1(pred.cpu(),target.cpu(),args.tiers)
        if score>best:
            best=score; state={k:v.detach().cpu().clone() for k,v in elastic.router.state_dict().items()}
    elastic.router.load_state_dict(state); elastic.router.eval(); ix=torch.where(masks["test"])[0]; target=labels[ix].to(device)
    with torch.no_grad(): pred=elastic.router.select_ranks(elastic.router(features[ix].to(device)))
    metrics={"objective":args.objective,"tiers":args.tiers,"split_sequences":splits,
      "split_intervals_nonoverlap":True,"train_class_counts":{str(t):int(c) for t,c in zip(args.tiers,counts)},
      "best_val_macro_f1":best,"test_accuracy":float((pred==target).float().mean()),
      "test_macro_f1":macro_f1(pred.cpu(),target.cpu(),args.tiers),"predicted_mean_rank":float(pred.float().mean()),
      "target_mean_rank":float(target.float().mean()),
      "interpretation":"global token-difficulty router; one tier is shared by every MLA layer"}
    out=ROOT/"experiments/exp_router"; out.mkdir(exist_ok=True)
    torch.save({"router":state,"tiers":args.tiers,"objective":args.objective,"channel_orders":summary["layer_channel_orders"],
      "base_checkpoint_step":ckpt["step"],"split_sequences":splits},out/f"global_router_{args.objective}.pt")
    json.dump(metrics,open(out/f"global_metrics_{args.objective}.json","w"),indent=2); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
