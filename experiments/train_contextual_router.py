"""Train a contextual layer0-full router on aligned oracle tier labels."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from elastic_mla import ContextualElasticMLAGPT,MLAGPT
ROOT=Path(__file__).resolve().parents[1]
CKPT=ROOT/"experiments/exp0_rank_variance/ckpt/latest.pt";DATA=ROOT/"experiments/exp0_rank_variance/data/val.bin";ORACLE_DIR=ROOT/"experiments/contextual_router_30m/contextual_oracle_v1"
def device_auto():
 if torch.cuda.is_available():return torch.device("cuda")
 if torch.backends.mps.is_available():return torch.device("mps")
 return torch.device("cpu")
def file_sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def macro_f1(p,t,tiers,observed_only=False):
 if observed_only:tiers=[x for x in tiers if bool((t==x).any())]
 out=[]
 for x in tiers:
  a,b=p==x,t==x;tp=(a&b).sum().item();fp=(a&~b).sum().item();fn=(~a&b).sum().item();out.append(0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
 return float(np.mean(out))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--objective",choices=("mean","max"),required=True);ap.add_argument("--epochs",type=int,default=250);ap.add_argument("--seed",type=int,default=2027);ap.add_argument("--lr",type=float,default=2e-3);ap.add_argument("--batch-size",type=int,default=64);ap.add_argument("--checkpoint",type=Path,default=CKPT);ap.add_argument("--data",type=Path,default=DATA);ap.add_argument("--oracle-dir",type=Path,default=ORACLE_DIR);ap.add_argument("--output-dir",type=Path,default=ROOT/"experiments/contextual_router_30m");args=ap.parse_args();torch.manual_seed(args.seed);np.random.seed(args.seed);dev=device_auto()
 summary_path=args.oracle_dir/"contextual_oracle_v1_summary.json";records_path=args.oracle_dir/"contextual_oracle_v1_records.json";summary=json.load(open(summary_path));records=json.load(open(records_path));tiers=summary["deployment_tiers"]
 if file_sha(args.checkpoint)!=summary["checkpoint_sha256"] or file_sha(args.data)!=summary["data_sha256"]:raise ValueError("training checkpoint/data do not match oracle provenance")
 ck=torch.load(args.checkpoint,map_location=dev,weights_only=False);base=MLAGPT(**ck["config"]).to(dev).eval();base.load_state_dict(ck["model"],strict=True)
 model=ContextualElasticMLAGPT(base,[torch.tensor(o) for o in summary["layer_channel_orders"]],tiers).to(dev).freeze_base();seqs=sorted({r["seq"] for r in records});splits={"val":[0,7,13,22],"test":[9,12,19,21]};splits["train"]=sorted(set(seqs)-set(splits["val"])-set(splits["test"]));split_of={s:k for k,v in splits.items() for s in v};by={s:[] for s in seqs}
 for r in records:by[r["seq"]].append(r)
 data=np.memmap(args.data,dtype=np.uint16,mode="r");features=[];labels=[];split_names=[]
 with torch.no_grad():
  for s in seqs:
   rows=sorted(by[s],key=lambda r:r["pos"]);start=rows[0]["sequence_start"];arr=np.asarray(data[start:start+ck["config"]["max_len"]],dtype=np.int64);f=model.routing_features(torch.from_numpy(arr.copy())[None].to(dev))[0].cpu()
   for r in rows:features.append(f[r["pos"]]);labels.append(r[f"tier_context_future_{args.objective}"]);split_names.append(split_of[s])
 features=torch.stack(features);labels=torch.tensor(labels,dtype=torch.long);masks={k:torch.tensor([x==k for x in split_names]) for k in splits};train_labels=labels[masks["train"]];counts=torch.tensor([(train_labels==t).sum() for t in tiers],dtype=torch.float32);weights=(counts.sum()/counts.clamp_min(1)).sqrt();weights=(weights/weights.mean()).to(dev);opt=torch.optim.AdamW(model.router.parameters(),lr=args.lr,weight_decay=1e-3);train_idx=torch.where(masks["train"])[0];gen=torch.Generator().manual_seed(args.seed);best=-1;state=None
 for _ in range(args.epochs):
  model.router.train();perm=train_idx[torch.randperm(len(train_idx),generator=gen)]
  for st in range(0,len(perm),args.batch_size):
   ix=perm[st:st+args.batch_size];logits=model.router(features[ix].to(dev));loss=model.router.supervised_loss(logits,labels[ix].to(dev),weights);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  model.router.eval()
  with torch.no_grad():
   ix=torch.where(masks["val"])[0];target=labels[ix].to(dev);pred=model.router.select_ranks(model.router(features[ix].to(dev)));score=macro_f1(pred.cpu(),target.cpu(),tiers,observed_only=True)
  if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in model.router.state_dict().items()}
 model.router.load_state_dict(state);ix=torch.where(masks["test"])[0];target=labels[ix].to(dev)
 with torch.no_grad():pred=model.router.select_ranks(model.router(features[ix].to(dev)))
 metrics={"objective":args.objective,"tiers":tiers,"split_sequences":splits,"best_val_macro_f1":best,"test_accuracy":float((pred==target).float().mean()),"test_macro_f1":macro_f1(pred.cpu(),target.cpu(),tiers),"test_observed_tier_macro_f1":macro_f1(pred.cpu(),target.cpu(),tiers,observed_only=True),"predicted_mean_rank":float(pred.float().mean()),"target_mean_rank":float(target.float().mean()),"train_class_counts":{str(t):int(c) for t,c in zip(tiers,counts)},"feature":"layer1 pre-attention LN after full-rank contextual layer0","split_strategy":"fixed sequence-group stratification ensuring rare mean-tier coverage in val/test"}
 out=args.output_dir;out.mkdir(parents=True,exist_ok=True);torch.save({"router":state,"tiers":tiers,"objective":args.objective,"split_sequences":splits,"channel_orders":summary["layer_channel_orders"],"checkpoint_sha256":summary["checkpoint_sha256"],"data_sha256":summary["data_sha256"],"oracle_summary_sha256":file_sha(summary_path),"oracle_records_sha256":file_sha(records_path)},out/f"router_{args.objective}.pt");json.dump(metrics,open(out/f"metrics_{args.objective}.json","w"),indent=2);print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
