"""Compare normalized Exp0-v4 rank requirements across 30M and 122M models."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUTS={
 "30M":ROOT/"experiments/exp0_rank_variance/results_v4/exp0_v4_records.json",
 "122M":ROOT/"experiments/exp1_rank_variance_122m/results_v4/exp0_v4_records.json",
}
DCS={"30M":256,"122M":384}
SEED=45678; DRAWS=10000

def load(path): return json.load(open(path))
def sequence_clusters(records,key,dc):
 return [np.asarray([r[key]/dc for r in records if r["seq"]==seq],dtype=np.float64)
         for seq in sorted({r["seq"] for r in records})]
def main():
 records={name:load(path) for name,path in INPUTS.items()};rng=np.random.default_rng(SEED)
 out={"method":"independent sequence-cluster bootstrap of normalized r* difference (122M - 30M)","seed":SEED,"draws":DRAWS,"models":{},"differences":{}}
 for name,rows in records.items():
  out["models"][name]={"d_c":DCS[name],"n_records":len(rows)}
  for label,key in [("future_mean","r_star_future_mean"),("future_max","r_star_future_max")]:
   vals=np.asarray([r[key] for r in rows]);out["models"][name][label]={"mean_rank":float(vals.mean()),"normalized_mean":float(vals.mean()/DCS[name])}
 for label,key in [("future_mean","r_star_future_mean"),("future_max","r_star_future_max")]:
  a=sequence_clusters(records["30M"],key,DCS["30M"]);b=sequence_clusters(records["122M"],key,DCS["122M"]);draws=np.empty(DRAWS)
  for d in range(DRAWS):
   ma=np.concatenate([a[i] for i in rng.integers(0,len(a),len(a))]).mean();mb=np.concatenate([b[i] for i in rng.integers(0,len(b),len(b))]).mean();draws[d]=mb-ma
  observed=out["models"]["122M"][label]["normalized_mean"]-out["models"]["30M"][label]["normalized_mean"]
  out["differences"][label]={"observed":observed,"bootstrap_95pct_ci":[float(x) for x in np.percentile(draws,[2.5,97.5])]}
 path=ROOT/"experiments/exp0_v4_scale_comparison.json";json.dump(out,open(path,"w"),indent=2);print(json.dumps(out,indent=2))
if __name__=="__main__":main()
