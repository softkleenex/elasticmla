"""Select joint-rollout router checkpoints on validation and test once."""
import argparse,glob,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from elastic_mla import ContextualElasticMLAGPT,MLAGPT
from calibrate_contextual_router import evaluate,device

def file_sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--base-dir",type=Path,required=True);p.add_argument("--max-delta-loss",type=float,default=.15);p.add_argument("--pattern",default="joint_lambda_*.pt");p.add_argument("--output-name",default="joint_rollout_selection.json");args=p.parse_args();d=device();oracle=json.load(open(args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_summary.json"));records=json.load(open(args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_records.json"));ck=torch.load(args.checkpoint,map_location=d,weights_only=False);base=MLAGPT(**ck["config"]).to(d).eval();base.load_state_dict(ck["model"]);orders=[torch.tensor(o) for o in oracle["layer_channel_orders"]];starts={r["seq"]:r["sequence_start"] for r in records};data=np.memmap(args.data,dtype=np.uint16,mode="r");T=ck["config"]["max_len"];rows=[];checkpoints=sorted(args.base_dir.glob(args.pattern))
 if not checkpoints:raise ValueError("no checkpoints matched")
 first=torch.load(checkpoints[0],map_location="cpu",weights_only=False)
 if first["checkpoint_sha256"]!=file_sha(args.checkpoint) or first["data_sha256"]!=file_sha(args.data):raise ValueError("current inputs do not match candidates")
 model=ContextualElasticMLAGPT(base,orders,first["tiers"]).to(d).eval()
 for path in checkpoints:
  c=torch.load(path,map_location="cpu",weights_only=False)
  if c["tiers"]!=first["tiers"] or c["checkpoint_sha256"]!=first["checkpoint_sha256"] or c["data_sha256"]!=first["data_sha256"] or c["channel_orders"]!=first["channel_orders"] or c["split_sequences"]!=first["split_sequences"]:raise ValueError("candidate provenance/splits differ")
  model.router.load_state_dict(c["router"]);row=evaluate(model,base,orders,data,starts,c["split_sequences"]["val"],T,0,0,d);row.update({"checkpoint":path.name,"rank_lambda":c["rank_lambda"]});rows.append(row)
 feasible=[x for x in rows if x["delta_loss"]<=args.max_delta_loss and x["router_minus_static_loss"]<0];pool=feasible or [x for x in rows if x["delta_loss"]<=args.max_delta_loss] or rows;chosen=min(pool,key=lambda x:(x["bytes"],x["loss"]));c=torch.load(args.base_dir/chosen["checkpoint"],map_location="cpu",weights_only=False);model.router.load_state_dict(c["router"]);test=evaluate(model,base,orders,data,starts,c["split_sequences"]["test"],T,0,0,d,shuffles=20,seedbase=123000);selection=("validation minimum bytes under loss constraint and exact-byte static dominance" if feasible else "fallback: validation minimum bytes under loss constraint; no static-dominating candidate")
 out={"selection":selection,"max_delta_loss":args.max_delta_loss,"n_feasible":len(feasible),"validation_candidates":rows,"chosen_validation":chosen,"exploratory_reused_windows":test};json.dump(out,open(args.base_dir/args.output_name,"w"),indent=2);print(json.dumps(out,indent=2))
if __name__=="__main__":main()
