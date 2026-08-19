"""Straight-through joint-rollout fine-tuning of a frozen contextual router."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from elastic_mla import ContextualElasticMLAGPT,MLAGPT

def file_sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def state_sha(state):
 h=hashlib.sha256()
 for key in sorted(state):
  tensor=state[key].detach().cpu().contiguous();h.update(key.encode());h.update(str(tensor.dtype).encode());h.update(bytes(tensor.numpy()))
 return h.hexdigest()
def dev():
 if torch.cuda.is_available():return torch.device("cuda")
 if torch.backends.mps.is_available():return torch.device("mps")
 return torch.device("cpu")
def build_keep(orders,tiers,device,dtype):
 mats=[]
 for order in orders:
  m=torch.zeros(len(tiers),len(order),device=device,dtype=dtype)
  for k,t in enumerate(tiers):m[k,torch.as_tensor(order[:t],device=device)]=1
  mats.append(m)
 return mats
def forward_joint(base,router,idx,tiers,keep,temperature=1.0,straight_through=True):
 x=base.drop(base.tok_emb(idx));x=base.blocks[0](x);feature=base.blocks[1].ln1(x);route_logits=router(feature);probs=F.softmax(route_logits/temperature,dim=-1);hard=F.one_hot(probs.argmax(-1),num_classes=len(tiers)).to(probs.dtype);gate=hard+probs-probs.detach() if straight_through else probs
 for i,block in enumerate(base.blocks[1:],start=1):
  mask=torch.einsum("btk,kc->btc",gate,keep[i]).to(x.dtype);x=block(x,rank_mask=mask)
 logits=base.head(base.ln_f(x));expected=(probs*torch.as_tensor(tiers,device=probs.device,dtype=probs.dtype)).sum(-1);chosen=torch.as_tensor(tiers,device=probs.device)[hard.argmax(-1)];return logits,expected,chosen
def evaluate(base,router,seq_tokens,tiers,keep,temperature):
 losses=[];ranks=[]
 with torch.no_grad():
  for a in seq_tokens:
   x=a[:-1][None];y=a[1:][None];l,_,r=forward_joint(base,router,x,tiers,keep,temperature);losses.append(F.cross_entropy(l.reshape(-1,l.shape[-1]),y.reshape(-1)).item());ranks.append(float(r.float().mean()))
 return float(np.mean(losses)),float(np.mean(ranks))
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--base-dir",type=Path,required=True);p.add_argument("--rank-lambda",type=float,required=True);p.add_argument("--epochs",type=int,default=30);p.add_argument("--lr",type=float,default=5e-4);p.add_argument("--temperature",type=float,default=1.0);p.add_argument("--tiers",type=int,nargs="+");p.add_argument("--random-init",action="store_true");p.add_argument("--tag",default="joint");p.add_argument("--output-dir",type=Path);p.add_argument("--seed",type=int,default=3031);args=p.parse_args();torch.manual_seed(args.seed);np.random.seed(args.seed);d=dev();oracle=json.load(open(args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_summary.json"));records=json.load(open(args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_records.json"))
 if file_sha(args.checkpoint)!=oracle["checkpoint_sha256"] or file_sha(args.data)!=oracle["data_sha256"]:raise ValueError("training inputs do not match oracle provenance")
 init=torch.load(args.base_dir/"router_max.pt",map_location="cpu",weights_only=False)
 oracle_summary_path=args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_summary.json";oracle_records_path=args.base_dir/"contextual_oracle_v1/contextual_oracle_v1_records.json"
 if init.get("checkpoint_sha256")!=oracle["checkpoint_sha256"] or init.get("data_sha256")!=oracle["data_sha256"]:raise ValueError("initial router provenance differs from oracle")
 if init.get("channel_orders")!=oracle["layer_channel_orders"]:raise ValueError("initial router channel orders differ from oracle")
 if init.get("objective")!="max" or init.get("tiers")!=oracle["deployment_tiers"]:raise ValueError("initializer must be the max-objective oracle router")
 if init.get("oracle_summary_sha256")!=file_sha(oracle_summary_path) or init.get("oracle_records_sha256")!=file_sha(oracle_records_path):raise ValueError("initializer oracle hashes do not match current oracle files")
 splits=init.get("split_sequences",{});expected={int(r["seq"]) for r in records};observed=[int(i) for name in ("train","val","test") for i in splits.get(name,[])]
 if len(observed)!=len(set(observed)) or set(observed)!=expected:raise ValueError("initial router splits must uniquely cover every oracle sequence")
 ck=torch.load(args.checkpoint,map_location=d,weights_only=False);base=MLAGPT(**ck["config"]).to(d).eval();base.load_state_dict(ck["model"]);tiers=args.tiers or oracle["deployment_tiers"]
 if sorted(set(tiers))!=tiers or tiers[-1]!=base.d_c:raise ValueError("tiers must be increasing and end at d_c")
 model=ContextualElasticMLAGPT(base,[torch.tensor(o) for o in oracle["layer_channel_orders"]],tiers).to(d).freeze_base()
 if not args.random_init:model.router.load_state_dict(init["router"])
 initial_router_state_sha256=state_sha(model.router.state_dict());initialization_mode="seeded_random" if args.random_init else "router_max_weights"
 keep=build_keep(oracle["layer_channel_orders"],tiers,d,next(base.parameters()).dtype);starts={r["seq"]:r["sequence_start"] for r in records};data=np.memmap(args.data,dtype=np.uint16,mode="r");T=ck["config"]["max_len"]
 def toks(seqs):return [torch.from_numpy(np.asarray(data[starts[s]:starts[s]+T+1],dtype=np.int64).copy()).to(d) for s in seqs]
 train=toks(init["split_sequences"]["train"]);val=toks(init["split_sequences"]["val"]);opt=torch.optim.AdamW(model.router.parameters(),lr=args.lr,weight_decay=1e-3);best=None;best_score=float("inf");history=[];gen=torch.Generator().manual_seed(args.seed);started=time.time()
 for epoch in range(args.epochs):
  model.router.train();order=torch.randperm(len(train),generator=gen).tolist();train_loss=[];train_rank=[]
  for j in order:
   a=train[j];x=a[:-1][None];y=a[1:][None];logits,expected,_=forward_joint(base,model.router,x,tiers,keep,args.temperature);lm=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1));rank=expected.mean()/base.d_c;loss=lm+args.rank_lambda*rank;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.router.parameters(),1.0);opt.step();train_loss.append(float(lm.detach()));train_rank.append(float(expected.mean().detach()))
  model.router.eval();vl,vr=evaluate(base,model.router,val,tiers,keep,args.temperature);score=vl+args.rank_lambda*vr/base.d_c;row={"epoch":epoch+1,"train_lm_loss":float(np.mean(train_loss)),"train_expected_rank":float(np.mean(train_rank)),"val_loss":vl,"val_hard_rank":vr,"selection_score":score};history.append(row);print(json.dumps(row),flush=True)
  if score<best_score:best_score=score;best={k:v.detach().cpu().clone() for k,v in model.router.state_dict().items()}
 lam_tag=str(args.rank_lambda).replace(".","p");output_dir=args.output_dir or args.base_dir;output_dir.mkdir(parents=True,exist_ok=True);out=output_dir/f"{args.tag}_lambda_{lam_tag}.pt";torch.save({"router":best,"rank_lambda":args.rank_lambda,"temperature":args.temperature,"tiers":tiers,"split_sequences":init["split_sequences"],"channel_orders":oracle["layer_channel_orders"],"checkpoint_sha256":oracle["checkpoint_sha256"],"data_sha256":oracle["data_sha256"],"history":history,"best_score":best_score,"random_init":args.random_init,"seed":args.seed,"epochs":args.epochs,"learning_rate":args.lr,"tag":args.tag,"initialization_mode":initialization_mode,"initial_router_state_sha256":initial_router_state_sha256,"weight_initializer_sha256":None if args.random_init else file_sha(args.base_dir/"router_max.pt"),"split_source_sha256":file_sha(args.base_dir/"router_max.pt"),"split_source_objective":init["objective"],"oracle_summary_sha256":file_sha(oracle_summary_path),"oracle_records_sha256":file_sha(oracle_records_path)},out);json.dump({"rank_lambda":args.rank_lambda,"temperature":args.temperature,"best_score":best_score,"history":history,"elapsed_s":time.time()-started,"random_init":args.random_init,"seed":args.seed,"epochs":args.epochs,"learning_rate":args.lr,"tag":args.tag,"checkpoint_sha256":oracle["checkpoint_sha256"],"data_sha256":oracle["data_sha256"]},open(output_dir/f"{args.tag}_lambda_{lam_tag}.json","w"),indent=2)
if __name__=="__main__":main()
