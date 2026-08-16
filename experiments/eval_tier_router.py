"""Evaluate global tier routers with simultaneous packed-cache compression."""
import json, math, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","code"))
from elastic_mla import GlobalElasticMLAGPT, MLAGPT
ROOT=Path(__file__).resolve().parents[1]
def device_auto():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
def ce(logits,y): return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1)).item()
def bootstrap_ci(values,seed=77,draws=2000):
    values=np.asarray(values); rng=np.random.RandomState(seed); means=[]
    for _ in range(draws): means.append(values[rng.randint(0,len(values),len(values))].mean())
    return [float(x) for x in np.percentile(means,[2.5,97.5])]
def main():
    device=device_auto(); ckpt=torch.load(ROOT/"experiments/exp0_rank_variance/ckpt/latest.pt",map_location=device,weights_only=False)
    base=MLAGPT(**ckpt["config"]).to(device).eval(); base.load_state_dict(ckpt["model"],strict=True)
    summary=json.load(open(ROOT/"experiments/exp0_rank_variance/results/exp0_v3_summary.json")); records=json.load(open(ROOT/"experiments/exp0_rank_variance/results/exp0_v3_records.json"))
    starts={r["seq"]:r["sequence_start"] for r in records}; data=np.memmap(ROOT/"experiments/exp0_rank_variance/data/val.bin",dtype=np.uint16,mode="r")
    tiers=[16,64,160,256]; orders=[torch.tensor(o) for o in summary["layer_channel_orders"]]; policies={}
    splits=[]
    for objective in ("mean","max"):
        rck=torch.load(ROOT/f"experiments/exp_router/global_router_{objective}.pt",map_location="cpu",weights_only=False)
        elastic=GlobalElasticMLAGPT(base,orders,tiers).to(device).eval(); elastic.router.load_state_dict(rck["router"])
        policies[objective]=elastic; splits.append(rck["split_sequences"])
    if splits[0]!=splits[1]: raise ValueError("mean/max policies must use identical grouped splits")
    test_sequences=splits[0]["test"]; N_SHUFFLES=20
    result={"device":str(device),"test_sequences":test_sequences,"n_shuffles_per_sequence":N_SHUFFLES,"policies":{}}
    full_losses=[]; fixed={t:{"loss":[],"bytes":[]} for t in tiers}; acc={n:{"loss":[],"bytes":[],"rank_sum":0,"rank_n":0,"hist":{str(t):0 for t in tiers},"shuffle_loss":[]} for n in policies}
    with torch.no_grad():
      for seq in test_sequences:
        st=starts[seq]; arr=np.array(data[st:st+ckpt["config"]["max_len"]+1],dtype=np.int64); x=torch.from_numpy(arr[:-1])[None].to(device); y=torch.from_numpy(arr[1:])[None].to(device)
        logits,_=base(x); full_losses.append(ce(logits,y))
        for tier in tiers:
          ranks=torch.full(x.shape,tier,device=device,dtype=torch.long); logits,caches=base.forward_cached_packed(x,ranks,orders); fixed[tier]["loss"].append(ce(logits,y)); fixed[tier]["bytes"].append(base.packed_cache_num_bytes(caches))
        for name,elastic in policies.items():
          logits,caches,ranks,_=elastic.forward_cached_packed(x); a=acc[name]; a["loss"].append(ce(logits,y)); a["bytes"].append(elastic.packed_cache_num_bytes(caches)); a["rank_sum"]+=int(ranks.sum()); a["rank_n"]+=ranks.numel()
          for tier in tiers:a["hist"][str(tier)]+=int((ranks==tier).sum())*base.n_layers
          shuffled=[]; flat=ranks.cpu().flatten()
          for rep in range(N_SHUFFLES):
            gen=torch.Generator().manual_seed(90000+1000*seq+100*rep+(0 if name=="mean" else 1)); perm=torch.randperm(flat.numel(),generator=gen)
            sr=flat[perm].view_as(ranks).to(device); sl,_=base.forward_cached_packed(x,sr,orders); shuffled.append(ce(sl,y))
          a["shuffle_loss"].append(float(np.mean(shuffled)))
    full=float(np.mean(full_losses)); result["full_mla"]={"loss":full,"ppl":math.exp(full)}; B,T=1,ckpt["config"]["max_len"]
    result["fixed_width_mla_bytes"]=B*T*base.n_layers*(base.d_c+base.blocks[0].attn.d_rope)*4; result["standard_mha_theoretical_bytes"]=base.theoretical_mha_cache_num_bytes(B,T,torch.float32)
    result["fixed_tiers"]={}
    for tier,a in fixed.items():
      loss=float(np.mean(a["loss"])); byte=float(np.mean(a["bytes"])); result["fixed_tiers"][str(tier)]={"loss":loss,"ppl":math.exp(loss),"delta_loss":loss-full,"bytes":byte,"over_fixed_mla":byte/result["fixed_width_mla_bytes"]}
    for name,a in acc.items():
      loss=float(np.mean(a["loss"])); sh=float(np.mean(a["shuffle_loss"])); diffs=np.asarray(a["loss"])-np.asarray(a["shuffle_loss"]); byte=float(np.mean(a["bytes"])); result["policies"][name]={"loss":loss,"ppl":math.exp(loss),"delta_loss":loss-full,"bytes":byte,"over_fixed_mla":byte/result["fixed_width_mla_bytes"],"average_rank":a["rank_sum"]/a["rank_n"],"rank_hist_layer_expanded":a["hist"],"matched_budget_shuffled_loss_mean":sh,"router_minus_shuffled_loss":loss-sh,"router_minus_shuffled_sequence_bootstrap_95pct_ci":bootstrap_ci(diffs,seed=80+(0 if name=="mean" else 1))}
    out=ROOT/"experiments/exp_router/global_pareto_eval.json"; json.dump(result,open(out,"w"),indent=2); print(json.dumps(result,indent=2))
if __name__=="__main__":main()
