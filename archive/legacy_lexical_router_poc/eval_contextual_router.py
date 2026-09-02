"""Evaluate contextual layer0-full routers and matched-budget controls."""
import argparse,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from elastic_mla import ContextualElasticMLAGPT,MLAGPT
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"experiments/contextual_router_30m";CKPT=ROOT/"experiments/exp0_rank_variance/ckpt/latest.pt";DATA=ROOT/"experiments/exp0_rank_variance/data/val.bin"
def dev():
 if torch.cuda.is_available():return torch.device("cuda")
 if torch.backends.mps.is_available():return torch.device("mps")
 return torch.device("cpu")
def ce(l,y):return F.cross_entropy(l.reshape(-1,l.shape[-1]),y.reshape(-1)).item()
def file_sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def bootstrap_ci(values,seed,draws=2000):
 v=np.asarray(values);g=np.random.default_rng(seed);means=[v[g.integers(0,len(v),len(v))].mean() for _ in range(draws)];return [float(x) for x in np.percentile(means,[2.5,97.5])]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--data",type=Path,default=DATA);ap.add_argument("--base-dir",type=Path,default=BASE);args=ap.parse_args();base_dir=args.base_dir
 d=dev();summary_path=base_dir/"contextual_oracle_v1/contextual_oracle_v1_summary.json";records_path=base_dir/"contextual_oracle_v1/contextual_oracle_v1_records.json";oracle=json.load(open(summary_path));records=json.load(open(records_path));orders=[torch.tensor(o) for o in oracle["layer_channel_orders"]];tiers=oracle["deployment_tiers"]
 if file_sha(args.checkpoint)!=oracle["checkpoint_sha256"] or file_sha(args.data)!=oracle["data_sha256"]:raise ValueError("evaluation inputs do not match oracle")
 ck=torch.load(args.checkpoint,map_location=d,weights_only=False);base=MLAGPT(**ck["config"]).to(d).eval();base.load_state_dict(ck["model"]);policies={};splits=[]
 for name in ("mean","max"):
  c=torch.load(base_dir/f"router_{name}.pt",map_location="cpu",weights_only=False)
  if c["objective"]!=name or c["tiers"]!=tiers or c["channel_orders"]!=oracle["layer_channel_orders"] or c["checkpoint_sha256"]!=oracle["checkpoint_sha256"] or c["data_sha256"]!=oracle["data_sha256"] or c["oracle_summary_sha256"]!=file_sha(summary_path) or c["oracle_records_sha256"]!=file_sha(records_path):raise ValueError("router checkpoint provenance does not match oracle")
  m=ContextualElasticMLAGPT(base,orders,tiers).to(d).eval();m.router.load_state_dict(c["router"]);policies[name]=m;splits.append(c["split_sequences"])
 if splits[0]!=splits[1]:raise ValueError("policy splits differ")
 test=splits[0]["test"];starts={r["seq"]:r["sequence_start"] for r in records};data=np.memmap(args.data,dtype=np.uint16,mode="r");full=[];fixed={t:{"loss":[],"bytes":[]} for t in tiers};acc={n:{"loss":[],"shuffle":[],"bytes":[],"sum":0,"n":0,"hist":{str(t):0 for t in tiers}} for n in policies};N=20
 with torch.no_grad():
  for s in test:
   st=starts[s];a=np.asarray(data[st:st+ck["config"]["max_len"]+1],dtype=np.int64);x=torch.from_numpy(a[:-1].copy())[None].to(d);y=torch.from_numpy(a[1:].copy())[None].to(d);l,_=base(x);full.append(ce(l,y))
   template=next(iter(policies.values()))
   for t in tiers:
    ranks=torch.full(x.shape,t,device=d,dtype=torch.long);l,c,_,_=template.forward_cached_packed(x,forced_ranks=ranks);fixed[t]["loss"].append(ce(l,y));fixed[t]["bytes"].append(template.packed_cache_num_bytes(c))
   for name,m in policies.items():
    l,c,ranks,_=m.forward_cached_packed(x);z=acc[name];z["loss"].append(ce(l,y));z["bytes"].append(m.packed_cache_num_bytes(c));z["sum"]+=int(ranks.sum());z["n"]+=ranks.numel()
    for t in tiers:z["hist"][str(t)]+=int((ranks==t).sum())
    flat=ranks.cpu().flatten();sh=[]
    for rep in range(N):
     g=torch.Generator().manual_seed(80000+10000*s+rep+(name=="max"));perm=torch.randperm(flat.numel(),generator=g);sr=flat[perm].view_as(ranks).to(d);sl,_,_,_=m.forward_cached_packed(x,forced_ranks=sr);sh.append(ce(sl,y))
    z["shuffle"].append(float(np.mean(sh)))
 # Static constant-rank baselines at each router's rounded mean downstream rank.
 static_matched={}
 for name,z in acc.items():
  rank=int(round(z["sum"]/z["n"]));losses=[];byte_values=[]
  for s in test:
   st=starts[s];a=np.asarray(data[st:st+ck["config"]["max_len"]+1],dtype=np.int64);x=torch.from_numpy(a[:-1].copy())[None].to(d);y=torch.from_numpy(a[1:].copy())[None].to(d);layer_ranks=[torch.full(x.shape,base.d_c,device=d,dtype=torch.long)]+[torch.full(x.shape,rank,device=d,dtype=torch.long) for _ in range(base.n_layers-1)];l,c=base.forward_cached_packed(x,ranks=layer_ranks,channel_orders=orders);losses.append(ce(l,y));byte_values.append(base.packed_cache_num_bytes(c))
  static_matched[name]={"downstream_rank":rank,"loss":float(np.mean(losses)),"bytes":float(np.mean(byte_values))}
 full_loss=float(np.mean(full));B,T=1,ck["config"]["max_len"];fixed_bytes=B*T*base.n_layers*(base.d_c+base.blocks[0].attn.d_rope)*4;out={"device":str(d),"test_sequences":test,"n_shuffles":N,"full_mla":{"loss":full_loss,"ppl":math.exp(full_loss)},"fixed_width_mla_bytes":fixed_bytes,"standard_mha_theoretical_bytes":base.theoretical_mha_cache_num_bytes(B,T,torch.float32),"fixed_downstream_tiers":{},"policies":{},"method":"layer0 full rank; tier applies to layers1+"}
 for t,z in fixed.items():
  loss=float(np.mean(z["loss"]));byte=float(np.mean(z["bytes"]));out["fixed_downstream_tiers"][str(t)]={"loss":loss,"delta_loss":loss-full_loss,"bytes":byte,"over_fixed_mla":byte/fixed_bytes}
 for n,z in acc.items():
  loss=float(np.mean(z["loss"]));sh=float(np.mean(z["shuffle"]));out["policies"][n]={"loss":loss,"ppl":math.exp(loss),"delta_loss":loss-full_loss,"bytes":float(np.mean(z["bytes"])),"over_fixed_mla":float(np.mean(z["bytes"]))/fixed_bytes,"average_downstream_rank":z["sum"]/z["n"],"rank_hist":z["hist"],"matched_budget_shuffled_loss":sh,"router_minus_shuffled_loss":loss-sh,"sequence_differences":[float(a-b) for a,b in zip(z["loss"],z["shuffle"])],"exploratory_sequence_bootstrap_95pct_ci":bootstrap_ci([a-b for a,b in zip(z["loss"],z["shuffle"])],900+(n=="max")),"matched_budget_control":"exact per-sequence tier histogram shuffled across token positions","static_rounded_mean_rank_control":static_matched[n]}
 json.dump(out,open(base_dir/"pareto_eval.json","w"),indent=2);print(json.dumps(out,indent=2))
if __name__=="__main__":main()
