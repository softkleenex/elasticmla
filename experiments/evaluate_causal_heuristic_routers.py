"""Train/select simple causal routing baselines without using fresh-window outcomes.

Position, lexical identity, rarity, and token-type scores are estimated only on the original 16
router-training sequences. An additive rate bias is selected on the original four validation
sequences under a loss budget. The selected causal rules are then evaluated once on the already
opened fresh windows and are explicitly post-confirmation exploratory baselines.
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"code"))
from elastic_mla import ContextualElasticMLAGPT,MLAGPT

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def ce(l,y):return F.cross_entropy(l.reshape(-1,l.shape[-1]),y.reshape(-1)).item()
def kind(tok,enc):
 s=enc.decode([int(tok)]).strip()
 if not s:return "space"
 if all(c in ".,!?;:'\"-()" for c in s):return "punct"
 if s[0].isupper():return "capitalized"
 return "other"
def quantize(scores,bias,tiers):
 values=np.asarray(scores,dtype=np.float64)+float(bias);grid=np.asarray(tiers,dtype=np.int64)
 return grid[np.clip(np.searchsorted(grid,values,side="left"),0,len(grid)-1)]
def bootstrap(v,seed,draws=10000):
 v=np.asarray(v,dtype=float);rng=np.random.default_rng(seed);m=np.empty(draws)
 for i in range(draws):m[i]=v[rng.integers(0,len(v),len(v))].mean()
 return [float(x) for x in np.percentile(m,[2.5,97.5])]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--base-dir",type=Path,required=True);p.add_argument("--policy",type=Path,required=True);p.add_argument("--fresh-result",type=Path,required=True);p.add_argument("--fresh-audit",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",choices=("cuda","mps","cpu"),required=True);p.add_argument("--biases",type=int,nargs="+",default=[-192,-128,-96,-64,-32,0,32,64,96,128,192]);p.add_argument("--max-delta-loss",type=float,default=.15);p.add_argument("--seed",type=int,default=72119);args=p.parse_args()
 if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 if args.device=="mps" and not torch.backends.mps.is_available():raise RuntimeError("MPS unavailable")
 d=torch.device(args.device);fresh=json.load(open(args.fresh_result));audit=json.load(open(args.fresh_audit));policy=torch.load(args.policy,map_location="cpu",weights_only=False)
 op=args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_summary.json";rp=args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_records.json";oracle=json.load(open(op));records=json.load(open(rp));source=torch.load(args.base_dir/"router_max.pt",map_location="cpu",weights_only=False)
 if audit.get("status")!="passed" or audit.get("result_sha256")!=sha(args.fresh_result):raise ValueError("fresh artifact lacks matching passed audit")
 if sha(args.checkpoint)!=fresh["checkpoint_sha256"] or sha(args.data)!=fresh["data_sha256"] or sha(args.policy)!=fresh["policy_sha256"]:raise ValueError("checkpoint/data/policy provenance mismatch")
 if source.get("oracle_summary_sha256")!=sha(op) or source.get("oracle_records_sha256")!=sha(rp):raise ValueError("split source does not authenticate contextual oracle")
 if source.get("objective")!="max" or source.get("checkpoint_sha256")!=fresh["checkpoint_sha256"] or source.get("data_sha256")!=fresh["data_sha256"]:raise ValueError("split source objective/model provenance mismatch")
 if source.get("split_sequences")!=policy.get("split_sequences") or policy["channel_orders"]!=oracle["layer_channel_orders"]:raise ValueError("policy split/order provenance mismatch")
 splits=policy["split_sequences"];flat=[int(x) for n in ("train","val","test") for x in splits[n]];seq_ids={int(r["seq"]) for r in records}
 if len(flat)!=len(set(flat)) or set(flat)!=seq_ids or len(splits["train"])!=16 or len(splits["val"])!=4:raise ValueError("unexpected split coverage")
 starts={}
 for row in records:
  q=int(row["seq"]);st=int(row["sequence_start"])
  if q in starts and starts[q]!=st:raise ValueError("inconsistent start within oracle sequence")
  starts[q]=st
 if len(fresh["fresh_starts"])!=24 or len(set(fresh["fresh_starts"]))!=24:raise ValueError("expected 24 unique fresh starts")
 T_from_audit=int(audit["authenticated_checkpoint_config"]["max_len"])
 if any(abs(int(a)-int(b))<T_from_audit+1 for a in fresh["fresh_starts"] for b in starts.values()):raise ValueError("fresh window overlaps a causal-baseline train/validation/test window")

 ck=torch.load(args.checkpoint,map_location=d,weights_only=False);base=MLAGPT(**ck["config"]).to(d).eval();base.load_state_dict(ck["model"]);orders=[torch.tensor(o) for o in policy["channel_orders"]];model=ContextualElasticMLAGPT(base,orders,policy["tiers"]).to(d).eval();model.router.load_state_dict(policy["router"]);data=np.memmap(args.data,dtype=np.uint16,mode="r");T=int(ck["config"]["max_len"]);enc=tiktoken.get_encoding("gpt2")
 def load_start(st):
  a=np.asarray(data[st:st+T+1],dtype=np.int64);return a,torch.from_numpy(a[:-1].copy())[None].to(d),torch.from_numpy(a[1:].copy())[None].to(d)
 train_r=[];train_x=[]
 with torch.no_grad():
  for seq in splits["train"]:
   a,x,_=load_start(starts[int(seq)]);_,_,r,_=model.forward_cached_packed(x);train_r.append(r.cpu().numpy().reshape(-1));train_x.append(a[:-1])
 train_r=np.stack(train_r);train_x=np.stack(train_x);global_mean=float(train_r.mean());position=train_r.mean(0)
 sums={};counts={}
 for tok,r in zip(train_x.ravel(),train_r.ravel()):tok=int(tok);sums[tok]=sums.get(tok,0.)+float(r);counts[tok]=counts.get(tok,0)+1
 lexical={t:(s+4*global_mean)/(counts[t]+4) for t,s in sums.items()};ts={};tc={}
 for tok,r in zip(train_x.ravel(),train_r.ravel()):k=kind(tok,enc);ts[k]=ts.get(k,0.)+float(r);tc[k]=tc.get(k,0)+1
 types={k:ts[k]/tc[k] for k in ts};rar_x=np.asarray([-math.log1p(counts[int(t)]) for t in train_x.ravel()]);rar_y=train_r.ravel();rar_a,rar_b=np.polyfit(rar_x,rar_y,1)
 def scores(name,tokens):
  if name=="position":return position
  if name=="lexical_identity":return np.asarray([lexical.get(int(t),global_mean) for t in tokens])
  if name=="token_rarity":return rar_a*np.asarray([-math.log1p(counts.get(int(t),0)) for t in tokens])+rar_b
  if name=="token_type":return np.asarray([types.get(kind(t,enc),global_mean) for t in tokens])
  raise KeyError(name)
 names=("position","lexical_identity","token_rarity","token_type")
 def evaluate_rule(name,bias,sequence_starts):
  rows=[]
  with torch.no_grad():
   for st in sequence_starts:
    a,x,y=load_start(st);full,_=base(x);rank_np=quantize(scores(name,a[:-1]),bias,policy["tiers"])[None];r=torch.from_numpy(rank_np).to(device=d,dtype=torch.long);layer=[torch.full_like(r,base.d_c)]+[r for _ in range(base.n_layers-1)];l,c=base.forward_cached_packed(x,ranks=layer,channel_orders=orders);rows.append({"start":int(st),"loss":ce(l,y),"full_loss":ce(full,y),"average_rank":float(r.float().mean()),"bytes":base.packed_cache_num_bytes(c)})
  return rows
 val_starts=[starts[int(s)] for s in splits["val"]];validation={};chosen={}
 for name in names:
  cand=[]
  for bias in sorted(set(args.biases)):
   rows=evaluate_rule(name,bias,val_starts);loss=float(np.mean([x["loss"] for x in rows]));full=float(np.mean([x["full_loss"] for x in rows]));cand.append({"bias":bias,"loss":loss,"full_loss":full,"delta_loss":loss-full,"average_rank":float(np.mean([x["average_rank"] for x in rows])),"bytes":float(np.mean([x["bytes"] for x in rows]))})
  feasible=[x for x in cand if x["delta_loss"]<=args.max_delta_loss];pick=min(feasible,key=lambda x:(x["bytes"],x["loss"])) if feasible else min(cand,key=lambda x:x["loss"]);validation[name]=cand;chosen[name]={**pick,"selection":"minimum validation bytes under loss budget" if feasible else "fallback minimum validation loss"}

 fresh_out={};fresh_starts=[int(x) for x in fresh["fresh_starts"]]
 for idx,name in enumerate(names):
  rows=evaluate_rule(name,chosen[name]["bias"],fresh_starts);router_rows={int(r["start"]):r for r in fresh["rows"]};diff=[router_rows[r["start"]]["router_loss"]-r["loss"] for r in rows];fresh_out[name]={"selected_bias":chosen[name]["bias"],"mean_loss":float(np.mean([r["loss"] for r in rows])),"mean_full_loss":float(np.mean([r["full_loss"] for r in rows])),"delta_loss":float(np.mean([r["loss"]-r["full_loss"] for r in rows])),"average_rank":float(np.mean([r["average_rank"] for r in rows])),"mean_bytes":float(np.mean([r["bytes"] for r in rows])),"mean_router_minus_control":float(np.mean(diff)),"router_minus_control_bootstrap_95pct_ci":bootstrap(diff,args.seed+idx),"rows":rows}
 output={"status":"complete","interpretation":"post-confirmation exploratory causal baselines; selection used only original train/validation splits","checkpoint_sha256":fresh["checkpoint_sha256"],"data_sha256":fresh["data_sha256"],"policy_sha256":fresh["policy_sha256"],"fresh_result_sha256":sha(args.fresh_result),"fresh_audit_sha256":sha(args.fresh_audit),"contextual_oracle_summary_sha256":sha(op),"contextual_oracle_records_sha256":sha(rp),"split_source_sha256":sha(args.base_dir/"router_max.pt"),"max_delta_loss":args.max_delta_loss,"biases":sorted(set(args.biases)),"training_sequences":splits["train"],"validation_sequences":splits["val"],"fresh_starts":fresh_starts,"score_causality":"rank at t uses only absolute t and/or current token x[t], never future tokens or sequence-global budgets","validation_candidates":validation,"chosen_validation":chosen,"fresh":fresh_out}
 args.output.parent.mkdir(parents=True,exist_ok=True);json.dump(output,open(args.output,"w"),indent=2);print(json.dumps({"chosen_validation":chosen,"fresh":{k:{q:v for q,v in x.items() if q!="rows"} for k,x in fresh_out.items()}},indent=2))
if __name__=="__main__":main()
