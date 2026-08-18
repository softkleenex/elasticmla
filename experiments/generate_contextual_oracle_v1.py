"""Generate oracle labels aligned to contextual ElasticMLA routing.

Layer 0 remains full rank.  At one source position, layers 1+ share a tested
retained rank.  Effects cover logits[pos:pos+horizon], i.e. predictions for the
immediate next token through the exact fixed horizon.  Channel orders and probe
locations come from a validated Exp0-v4 run.
"""
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Sequence
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"));sys.path.insert(0,str(ROOT/"experiments"))
from elastic_mla import MLAGPT
from analyze_rank_variance_v4 import (
 choose_device, empty_device_cache, forward_with_layer_masks, future_loss_slice,
 is_nonmonotonic, make_channel_masks, per_token_loss, sample_starts,
 sequence_cluster_bootstrap_mean_ci, suffix_all_satisfy_r_star, valid_probe_positions,
)

def file_sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
 return h.hexdigest()

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--source-summary",type=Path,required=True);p.add_argument("--source-records",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",choices=("auto","cuda","mps","cpu"),default="auto");p.add_argument("--probe-batch-size",type=int,default=8);p.add_argument("--bootstrap-draws",type=int,default=2000);p.add_argument("--nonmonotonic-tolerance",type=float,default=0.0);p.add_argument("--tiers",type=int,nargs="+",required=True,help="deployment tiers; must be increasing and include full d_c");return p.parse_args()

def main():
 args=parse_args();device=choose_device(args.device);print(f"device: {device}",flush=True)
 source=json.load(open(args.source_summary));source_records=json.load(open(args.source_records))
 if source.get("status")!="valid" or not source.get("method_version","").startswith("v4"):
  raise ValueError("source must be a validated Exp0-v4 result")
 checkpoint_hash=file_sha256(args.checkpoint);data_hash=file_sha256(args.data)
 if checkpoint_hash!=source.get("checkpoint_sha256") or data_hash!=source.get("data_sha256"):
  raise ValueError("checkpoint/data hashes do not match source v4 provenance")
 checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False);model=MLAGPT(**checkpoint["config"]).to(device);model.load_state_dict(checkpoint["model"],strict=True);model.eval()
 if checkpoint["step"]!=source["checkpoint_step"] or model.d_c!=source["d_c"] or model.n_layers!=source["n_layers"]:
  raise ValueError("checkpoint does not match source v4 metadata")
 tiers=sorted(set(map(int,args.tiers)))
 if tiers!=list(map(int,args.tiers)) or tiers[0]<=0 or tiers[-1]!=model.d_c:
  raise ValueError("tiers must be strictly increasing, positive, and end at d_c")
 if model.n_layers<2:raise ValueError("contextual oracle requires at least two layers")
 rank_grid=source["rank_grid"];horizon=source["future_horizon_exact"];epsilon=source["epsilon_nats"];orders=np.asarray(source["layer_channel_orders"],dtype=np.int64)
 seq_ids=sorted({int(r["seq"]) for r in source_records});by_seq={s:[] for s in seq_ids}
 for r in source_records:by_seq[int(r["seq"])].append(r)
 for rows in by_seq.values():rows.sort(key=lambda r:int(r["pos"]))
 if len(source_records)!=source["n_positions_total"] or len(seq_ids)!=source["n_evaluation_sequences"]:
  raise ValueError("source record/sequence count does not match summary")
 expected_per_seq=source["positions_per_sequence"]
 if any(len(rows)!=expected_per_seq for rows in by_seq.values()):
  raise ValueError("source records do not have the expected per-sequence count")
 if any(len({int(r["sequence_start"]) for r in rows})!=1 for rows in by_seq.values()):
  raise ValueError("a source sequence has inconsistent start offsets")
 starts={s:int(by_seq[s][0]["sequence_start"]) for s in seq_ids};positions={s:np.asarray([int(r["pos"]) for r in by_seq[s]],dtype=np.int64) for s in seq_ids}
 block=int(checkpoint["config"]["max_len"]);data=np.memmap(args.data,dtype=np.uint16,mode="r")
 if any(start<0 or start+block>=len(data) for start in starts.values()):
  raise ValueError("source sequence span is outside the data file")
 tokens={s:np.asarray(data[starts[s]:starts[s]+block+1],dtype=np.int64) for s in seq_ids}
 for s,rows in by_seq.items():
  for record in rows:
   pos=int(record["pos"])
   if pos<0 or pos>=block or int(tokens[s][pos])!=int(record["input_token_id"]):
    raise ValueError("source record token does not match the supplied data")
 # Reproduce v4's seeded, nonoverlapping evaluation sample exactly; this rejects
 # internally consistent but substituted starts or probe positions.
 eval_rng=np.random.default_rng(int(source["seeds"]["evaluation"]));n_possible=len(data)-block;calibration_high=int(n_possible*0.55);evaluation_low=calibration_high+block
 expected_starts=sample_starts(eval_rng,evaluation_low,n_possible,source["n_evaluation_sequences"],min_separation=block+1)
 if seq_ids!=list(range(source["n_evaluation_sequences"])) or any(starts[s]!=int(expected_starts[s]) for s in seq_ids):
  raise ValueError("source evaluation starts do not reproduce from the recorded seed")
 valid_positions=valid_probe_positions(block,horizon);count=min(source["positions_per_sequence"],len(valid_positions))
 for s in seq_ids:
  expected_positions=np.sort(eval_rng.choice(valid_positions,size=count,replace=False))
  if not np.array_equal(positions[s],expected_positions):
   raise ValueError("source probe positions do not reproduce from the recorded seed")
 baseline={}
 with torch.no_grad():
  for off in range(0,len(seq_ids),args.probe_batch_size):
   ids=seq_ids[off:off+args.probe_batch_size];batch=np.stack([tokens[s] for s in ids]);x=torch.from_numpy(batch[:,:-1]).to(device);y=torch.from_numpy(batch[:,1:]).to(device);loss=per_token_loss(forward_with_layer_masks(model,x),y)
   for i,s in enumerate(ids):baseline[s]=loss[i]
 curves={s:{int(pos):{"mean":[],"max":[]} for pos in positions[s]} for s in seq_ids};masked_layers=tuple(range(1,model.n_layers))
 for rank in rank_grid:
  for s in seq_ids:
   pos_arr=positions[s]
   for off in range(0,len(pos_arr),args.probe_batch_size):
    pb=pos_arr[off:off+args.probe_batch_size];n=len(pb);tok=tokens[s];x=torch.from_numpy(np.repeat(tok[None,:-1],n,axis=0)).to(device);y=torch.from_numpy(np.repeat(tok[None,1:],n,axis=0)).to(device);pt=torch.as_tensor(pb.copy(),dtype=torch.long,device=device);masks=make_channel_masks(orders,rank,n,device)
    with torch.no_grad():losses=per_token_loss(forward_with_layer_masks(model,x,channel_masks=masks,probe_positions=pt,masked_layers=masked_layers),y)
    for j,pos in enumerate(pb):
     w=future_loss_slice(int(pos),horizon,losses.shape[1]);delta=losses[j,w]-baseline[s][w];curves[s][int(pos)]["mean"].append(float(delta.mean()));curves[s][int(pos)]["max"].append(float(delta.max()))
  empty_device_cache(device);print(f"rank={rank} complete",flush=True)
 def tier_up(raw_rank):
  return next(tier for tier in tiers if tier>=raw_rank)
 out_records=[];mean_by_seq={s:[] for s in seq_ids};max_by_seq={s:[] for s in seq_ids}
 for s in seq_ids:
  source_by_pos={int(r["pos"]):r for r in by_seq[s]}
  for pos in positions[s]:
   pos=int(pos);mc=curves[s][pos]["mean"];xc=curves[s][pos]["max"];mr=suffix_all_satisfy_r_star(mc,rank_grid,epsilon);xr=suffix_all_satisfy_r_star(xc,rank_grid,epsilon);mean_by_seq[s].append(mr);max_by_seq[s].append(xr);src=source_by_pos[pos]
   out_records.append({"seq":s,"sequence_start":starts[s],"pos":pos,"input_token_id":src["input_token_id"],"input_token_type":src["input_token_type"],"future_horizon":horizon,"r_star_context_future_mean":mr,"r_star_context_future_max":xr,"tier_context_future_mean":tier_up(mr),"tier_context_future_max":tier_up(xr),"context_future_mean_delta_by_rank":dict(zip(map(str,rank_grid),mc)),"context_future_max_delta_by_rank":dict(zip(map(str,rank_grid),xc)),"context_future_mean_nonmonotonic":is_nonmonotonic(mc,args.nonmonotonic_tolerance),"context_future_max_nonmonotonic":is_nonmonotonic(xc,args.nonmonotonic_tolerance)})
 mean=np.asarray([r["r_star_context_future_mean"] for r in out_records]);maximum=np.asarray([r["r_star_context_future_max"] for r in out_records]);mean_tiers=np.asarray([r["tier_context_future_mean"] for r in out_records]);max_tiers=np.asarray([r["tier_context_future_max"] for r in out_records])
 summary={"status":"valid","method_version":"contextual-oracle-v1: layer0 full, shared rank intervention in layers1+","scope_limitation":"isolated-position full-attention intervention, not joint rollout-optimal allocation","device":str(device),"checkpoint_step":int(checkpoint["step"]),"checkpoint_sha256":file_sha256(args.checkpoint),"data_sha256":file_sha256(args.data),"source_summary_sha256":file_sha256(args.source_summary),"source_records_sha256":file_sha256(args.source_records),"rank_grid":rank_grid,"deployment_tiers":tiers,"epsilon_nats":epsilon,"future_horizon_exact":horizon,"n_layers":model.n_layers,"full_rank_layers":[0],"routed_layers":list(masked_layers),"d_c":model.d_c,"n_evaluation_sequences":len(seq_ids),"positions_per_sequence":len(out_records)//len(seq_ids),"n_positions_total":len(out_records),"layer_channel_orders":orders.tolist(),"future_mean":{"mean_r_star":float(mean.mean()),"sequence_cluster_bootstrap_95pct_ci":list(sequence_cluster_bootstrap_mean_ci(mean_by_seq,n_bootstrap=args.bootstrap_draws,seed=45679)),"histogram":{str(r):int((mean==r).sum()) for r in rank_grid},"tier_mean":float(mean_tiers.mean()),"tier_histogram":{str(t):int((mean_tiers==t).sum()) for t in tiers}},"future_max":{"mean_r_star":float(maximum.mean()),"sequence_cluster_bootstrap_95pct_ci":list(sequence_cluster_bootstrap_mean_ci(max_by_seq,n_bootstrap=args.bootstrap_draws,seed=45680)),"histogram":{str(r):int((maximum==r).sum()) for r in rank_grid},"tier_mean":float(max_tiers.mean()),"tier_histogram":{str(t):int((max_tiers==t).sum()) for t in tiers}},"r_star_rule":"smallest grid rank for which this and every higher rank are <= epsilon"}
 args.output_dir.mkdir(parents=True,exist_ok=True);json.dump(summary,open(args.output_dir/"contextual_oracle_v1_summary.json","w"),indent=2);json.dump(out_records,open(args.output_dir/"contextual_oracle_v1_records.json","w"),indent=2);print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
