"""Upload the review code dataset and push one Kaggle kernel per rerun job.

    uv run python code/kaggle_review_reruns/push.py --work <scratch dir> [--jobs ...] [--skip-dataset]

The code dataset carries code/, experiments/*.py and the small benchmark inputs
(policy + contextual-oracle summary per scale). Checkpoints come from the existing
datasets listed in DATASETS. Nothing here starts work outside Kaggle's own quota;
check `uv run kaggle kernels status softkleenex/<slug>` afterwards.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USER = "softkleenex"
CODE_DATASET = f"{USER}/elasticmla-review-reruns-code"
DATASETS = [CODE_DATASET, f"{USER}/elasticmla-basis-ablation-inputs",
            f"{USER}/elasticmla-250m-checkpoint"]
BENCHMARK_INPUTS = {
    "30m": "experiments/contextual_router_30m/joint_lambda_0p4.pt",
    "122m": "experiments/contextual_router_122m/fine_lambda_0p8.pt",
}
JOBS = ("spectrum250", "basis250", "benchmark", "publickv")


def kaggle(*args):
    cmd = ["uv", "run", "kaggle", *args]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_dataset(work):
    stage = work / "dataset"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(ROOT / "code" / "elastic_mla", stage / "code" / "elastic_mla",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "experiments").mkdir(parents=True)
    for script in (ROOT / "experiments").glob("*.py"):
        shutil.copy(script, stage / "experiments" / script.name)
    for scale, policy in BENCHMARK_INPUTS.items():
        target = stage / "inputs" / scale
        target.mkdir(parents=True)
        shutil.copy(ROOT / policy, target / Path(policy).name)
        shutil.copy(ROOT / f"experiments/contextual_router_{scale}/contextual_oracle_v1/"
                    "contextual_oracle_v1_summary.json", target)
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "ElasticMLA Review Reruns Code", "id": CODE_DATASET,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))
    return stage


def push_kernel(work, job):
    folder = work / "kernels" / job
    folder.mkdir(parents=True, exist_ok=True)
    template = (Path(__file__).parent / "runner_template.py").read_text()
    script = template.replace('JOB = "__JOB__"', f'JOB = "{job}"')
    (folder / "runner.py").write_text(script)
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/elasticmla-review-{job}",
        "title": f"ElasticMLA Review {job}",
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": DATASETS,
        "machine_shape": "NvidiaTeslaT4",
    }, indent=2))
    kaggle("kernels", "push", "-p", str(folder))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--jobs", nargs="*", choices=JOBS, default=list(JOBS),
                   help="pass no names to upload the dataset only")
    p.add_argument("--skip-dataset", action="store_true")
    p.add_argument("--new-dataset", action="store_true", help="create instead of version")
    p.add_argument("--message", default="review reruns")
    args = p.parse_args()
    if not args.skip_dataset:
        stage = build_dataset(args.work)
        if args.new_dataset:
            kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "zip")
        else:
            kaggle("datasets", "version", "-p", str(stage), "-m", args.message, "--dir-mode", "zip")
    for job in args.jobs:
        push_kernel(args.work, job)


if __name__ == "__main__":
    main()
