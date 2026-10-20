"""Kaggle runner for the reruns forced by the October 2026 pre-submission code review.

One template, one kernel per JOB (push.py substitutes the JOB line):
  spectrum250  analyze_risk_capacity_spectrum.py at 250M on the gap-free rank grid:
               the published v5 run (H=32), the h=128 extension (A) and the
               position-matched H=32 control (B), same seeds as the published runs
  basis250     analyze_basis_dependence.py at 250M on the gap-free rank grid
  benchmark    benchmark_cache_memory_latency.py at 30M and 122M with the harness fix
               (every path drops its prefill logits), 3 repeats, published settings
  publickv     analyze_public_kv_basis.py on gpt2 and gpt2-medium with provenance hashes

Inputs are attached datasets: the review code dataset (code/, experiments/, inputs/),
the 30M/122M inputs dataset and the 250M checkpoint dataset. Each result JSON is also
printed between RESULT markers so it survives in the kernel log.
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

JOB = "__JOB__"

WORK = Path("/kaggle/working")
STAGE = WORK / "inputs"
ROOT = WORK / "repo"
INPUT = Path("/kaggle/input")


def sh(cmd):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    print(result.stdout[-4000:], flush=True)
    if result.returncode != 0:
        print("FAILED\n" + result.stderr[-6000:], flush=True)
    return result.returncode == 0


def pin_torch_for_p100():
    """The account is sometimes given a P100 (sm_60) that the preinstalled torch lacks."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"],
        capture_output=True, text=True,
    )
    print("gpu probe:", probe.stdout.strip(), probe.stderr.strip()[-300:], flush=True)
    if "(6, 0)" in probe.stdout or "no kernel image" in probe.stderr:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1",
                        "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)


def find_dir_containing(*names):
    """Mount points vary; locate a directory by what it holds, extracting zips."""
    for candidate in [INPUT] + sorted(p for p in INPUT.rglob("*") if p.is_dir()):
        present = {p.name for p in candidate.iterdir()}
        if all(n in present or n + ".zip" in present for n in names):
            return candidate
    raise SystemExit(f"no attached dataset holds {names}")


def stage(src, name):
    target = STAGE / name
    if (src / name).is_dir():
        shutil.copytree(src / name, target, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(src / (name + ".zip")) as zf:
            zf.extractall(target)
    return target


def setup_repo():
    STAGE.mkdir(parents=True, exist_ok=True)
    src = find_dir_containing("code", "experiments", "inputs")
    for name in ("code", "experiments", "inputs"):
        stage(src, name)
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("code", "experiments"):
        shutil.copytree(STAGE / name, ROOT / name, dirs_exist_ok=True)


def scale_dir(folder):
    """exp0_30m / exp1_122m live in the inputs dataset; the 250M dataset is flattened."""
    for candidate in [INPUT] + sorted(p for p in INPUT.rglob("*") if p.is_dir()):
        if (candidate / folder / "ckpt" / "latest.pt").is_file():
            return candidate / folder
    if folder == "exp2_250m":
        for candidate in sorted(p for p in INPUT.rglob("*") if p.is_dir()):
            if (candidate / "ckpt" / "latest.pt").is_file() and "250m" in str(candidate).lower():
                return candidate
    raise SystemExit(f"checkpoint for {folder} not attached")


def emit(tag, path):
    print(f"RESULT_{tag}_BEGIN", flush=True)
    print(json.dumps(json.loads(Path(path).read_text())), flush=True)
    print(f"RESULT_{tag}_END", flush=True)


def spectrum250():
    d = scale_dir("exp2_250m")
    script = ROOT / "experiments" / "analyze_risk_capacity_spectrum.py"
    common = ["--checkpoint", d / "ckpt" / "latest.pt", "--data", d / "data" / "val.bin",
              "--device", "cuda"]
    runs = {
        "v5": ["--bootstrap-draws", "10000"],
        "A_extended": ["--future-horizon", "128", "--bootstrap-draws", "2000"],
        "B_position_matched": ["--future-horizon", "32", "--position-horizon", "128",
                               "--bootstrap-draws", "2000"],
    }
    for tag, extra in runs.items():
        out = WORK / f"spectrum250_{tag}"
        if sh([sys.executable, "-B", script, *common, *extra, "--output-dir", out]):
            emit(f"spectrum250_{tag}", out / "risk_capacity_v5_summary.json")


def basis250():
    d = scale_dir("exp2_250m")
    out = WORK / "basis_250m.json"
    if sh([sys.executable, "-B", ROOT / "experiments" / "analyze_basis_dependence.py",
           "--checkpoint", d / "ckpt" / "latest.pt", "--data", d / "data" / "val.bin",
           "--output", out, "--device", "cuda"]):
        emit("basis250", out)


def benchmark():
    settings = {
        "30m": ("exp0_30m", "192"),
        "122m": ("exp1_122m", "256"),
    }
    for scale, (folder, prefill) in settings.items():
        d = scale_dir(folder)
        inputs = STAGE / "inputs" / scale
        policy = next(inputs.glob("*.pt"))
        for repeat in range(1, 4):
            out = WORK / f"benchmark_{scale}_rep{repeat}.json"
            if sh([sys.executable, "-B", ROOT / "experiments" / "benchmark_cache_memory_latency.py",
                   "--checkpoint", d / "ckpt" / "latest.pt", "--data", d / "data" / "val.bin",
                   "--policy", policy,
                   "--contextual-summary", inputs / "contextual_oracle_v1_summary.json",
                   "--output", out, "--device", "cuda", "--batch-size", "8",
                   "--prefill-length", prefill, "--decode-steps", "128"]):
                emit(f"benchmark_{scale}_rep{repeat}", out)


def publickv():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=False)
    for model in ("gpt2", "gpt2-medium"):
        out = WORK / f"public_kv_basis_{model.replace('-', '_')}.json"
        if sh([sys.executable, "-B", ROOT / "experiments" / "analyze_public_kv_basis.py",
               "--model", model, "--output", out, "--device", "cuda"]):
            emit(f"publickv_{model}", out)


if __name__ == "__main__":
    pin_torch_for_p100()
    setup_repo()
    {"spectrum250": spectrum250, "basis250": basis250,
     "benchmark": benchmark, "publickv": publickv}[JOB]()
    print("ALL DONE", flush=True)
