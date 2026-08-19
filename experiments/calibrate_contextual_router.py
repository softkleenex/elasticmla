"""Validation-select rollout calibration, then evaluate once on held-out test."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from elastic_mla import ContextualElasticMLAGPT,MLAGPT
def device():
 if torch.cuda.is_available():return torch.device("cuda")
 if torch.backends.mps.is_available():return torch.device("mps")
 return torch.device("cpu")
def ce(l,y):return F.cross_entropy(l.reshape(-1,l.shape[-1]),y.reshape(-1)).item()
def load_seq(data,start,T,d):
 a=np.asarray(data[start:start+T+1],dtype=np.int64);return torch.from_numpy(a[:-1].copy())[None].to(d),torch.from_numpy(a[1:].copy())[None].to(d)
def choose_ranks(model,x,bias,floor):
 feat=model.routing_features(x);logits=model.router(feat);ramp=torch.linspace(0,1,logits.shape[-1],device=logits.device);idx=(logits+bias*ramp).argmax(-1).clamp_min(floor);return model.router.tiers[idx]
def evaluate(model,base,orders,data,starts,seqs,T,bias,floor,d,shuffles=0,seedbase=0):
 losses=[];bytes_=[];ranks_all=[];shuffle_losses=[];full=[]
 with torch.no_grad():
  for s in seqs:
   x,y=load_seq(data,starts[s],T,d);fl,_=base(x);full.append(ce(fl,y));r=choose_ranks(model,x,bias,floor);l,c,_,_=model.forward_cached_packed(x,forced_ranks=r);losses.append(ce(l,y));bytes_.append(model.packed_cache_num_bytes(c));ranks_all.append(r.cpu())
   if shuffles:
    flat=r.cpu().flatten();sl=[]
    for rep in range(shuffles):
     g=torch.Generator().manual_seed(seedbase+10000*s+rep);sr=flat[torch.randperm(flat.numel(),generator=g)].view_as(r).to(d);z,_,_,_=model.forward_cached_packed(x,forced_ranks=sr);sl.append(ce(z,y))
    shuffle_losses.append(float(np.mean(sl)))
 avg=float(torch.cat([r.flatten() for r in ranks_all]).float().mean());static_rank=int(round(avg));static_loss=[];static_bytes=[]
 with torch.no_grad():
  for s in seqs:
   x,y=load_seq(data,starts[s],T,d);lr=[torch.full(x.shape,base.d_c,device=d,dtype=torch.long)]+[torch.full(x.shape,static_rank,device=d,dtype=torch.long) for _ in range(base.n_layers-1)];l,c=base.forward_cached_packed(x,ranks=lr,channel_orders=orders);static_loss.append(ce(l,y));static_bytes.append(base.packed_cache_num_bytes(c))
 out={"bias":bias,"floor_index":floor,"floor_tier":int(model.router.tiers[floor]),"loss":float(np.mean(losses)),"full_loss":float(np.mean(full)),"delta_loss":float(np.mean(losses)-np.mean(full)),"bytes":float(np.mean(bytes_)),"average_rank":avg,"static_rank":static_rank,"static_loss":float(np.mean(static_loss)),"static_bytes":float(np.mean(static_bytes)),"router_minus_static_loss":float(np.mean(losses)-np.mean(static_loss)),"rank_hist":{str(int(t)):int(sum((r==int(t)).sum() for r in ranks_all)) for t in model.router.tiers}}
 if shuffles:out.update({"matched_shuffle_loss":float(np.mean(shuffle_losses)),"router_minus_shuffle_loss":float(np.mean(losses)-np.mean(shuffle_losses)),"sequence_router_minus_shuffle":[float(a-b) for a,b in zip(losses,shuffle_losses)]})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--base-dir",type=Path,required=True);p.add_argument("--max-delta-loss",type=float,default=.15);args=p.parse_args();d=device();summary=json.load(open(args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_summary.json"));records=json.load(open(args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_records.json"));router_ck=torch.load(args.base_dir/"router_max.pt",map_location="cpu",weights_only=False);ck=torch.load(args.checkpoint,map_location=d,weights_only=False);base=MLAGPT(**ck["config"]).to(d).eval();base.load_state_dict(ck["model"]);orders=[torch.tensor(o) for o in summary["layer_channel_orders"]];model=ContextualElasticMLAGPT(base,orders,summary["deployment_tiers"]).to(d).eval();model.router.load_state_dict(router_ck["router"]);splits=router_ck["split_sequences"];starts={r["seq"]:r["sequence_start"] for r in records};data=np.memmap(args.data,dtype=np.uint16,mode="r");T=ck["config"]["max_len"]
 candidates=[]
 for bias in [0,.25,.5,1,1.5,2,3,4]:
  for floor in range(len(summary["deployment_tiers"])):
   candidates.append(evaluate(model,base,orders,data,starts,splits["val"],T,bias,floor,d))
 feasible=[x for x in candidates if x["delta_loss"]<=args.max_delta_loss and x["router_minus_static_loss"]<0];pool=feasible or [x for x in candidates if x["delta_loss"]<=args.max_delta_loss] or candidates
 chosen=min(pool,key=lambda x:(x["bytes"],x["loss"]));test=evaluate(model,base,orders,data,starts,splits["test"],T,chosen["bias"],chosen["floor_index"],d,shuffles=20,seedbase=99000)
 out={"selection":"validation only: minimum bytes subject to delta-loss threshold and beating same-rank static when feasible","max_delta_loss":args.max_delta_loss,"validation_sequences":splits["val"],"test_sequences":splits["test"],"n_candidates":len(candidates),"n_feasible":len(feasible),"chosen_validation":chosen,"test":test,"validation_candidates":candidates};json.dump(out,open(args.base_dir/"rollout_calibration.json","w"),indent=2);print(json.dumps({k:v for k,v in out.items() if k!="validation_candidates"},indent=2))
if __name__=="__main__":main()
