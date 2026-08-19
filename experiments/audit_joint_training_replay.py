"""Verify that a deterministic policy replay exactly reproduces a frozen policy."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from elastic_mla import ContextualElasticMLAGPT, MLAGPT


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(tensor.dtype).encode()); h.update(bytes(tensor.numpy()))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=("30m", "122m"), required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--split-source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--contextual-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original = torch.load(args.original, map_location="cpu", weights_only=False)
    replay = torch.load(args.replay, map_location="cpu", weights_only=False)
    split_source = torch.load(args.split_source, map_location="cpu", weights_only=False)
    contextual = json.load(open(args.contextual_summary))
    # Reconstruct the exact pre-training state from the authenticated seed and
    # constructor sequence used by train_joint_rollout_router.py.
    torch.manual_seed(int(original["seed"]))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = MLAGPT(**checkpoint["config"]).eval(); base.load_state_dict(checkpoint["model"])
    reconstructed_model = ContextualElasticMLAGPT(
        base, [torch.tensor(order) for order in contextual["layer_channel_orders"]],
        original["tiers"],
    ).freeze_base()
    if not original.get("random_init"):
        reconstructed_model.router.load_state_dict(split_source["router"])
    reconstructed_initial_sha = state_sha(reconstructed_model.router.state_dict())
    keys = sorted(original["router"])
    tensors_exact = keys == sorted(replay["router"]) and all(
        torch.equal(original["router"][key], replay["router"][key]) for key in keys
    )
    hyperparameter_keys = (
        "rank_lambda", "temperature", "tiers", "random_init", "seed",
        "epochs", "learning_rate",
    )
    hyperparameters_exact = all(original.get(key) == replay.get(key) for key in hyperparameter_keys)
    splits_exact = original.get("split_sequences") == replay.get("split_sequences")
    split_source_authenticated = (
        replay.get("split_source_sha256") == file_sha(args.split_source)
        and replay.get("split_source_objective") == "max"
        and split_source.get("objective") == "max"
        and replay.get("oracle_summary_sha256") == split_source.get("oracle_summary_sha256")
        and replay.get("oracle_records_sha256") == split_source.get("oracle_records_sha256")
        and replay.get("split_sequences") == split_source.get("split_sequences")
        and replay.get("oracle_summary_sha256") == file_sha(args.contextual_summary)
        and original.get("checkpoint_sha256") == file_sha(args.checkpoint)
        and split_source.get("checkpoint_sha256") == original.get("checkpoint_sha256")
    )
    expected_random_init = args.scale == "122m"
    scale_mode_correct = original.get("random_init") is expected_random_init
    mode_correct = replay.get("initialization_mode") == (
        "seeded_random" if expected_random_init else "router_max_weights"
    )
    initial_state_recorded = (
        isinstance(replay.get("initial_router_state_sha256"), str)
        and replay.get("initial_router_state_sha256") == reconstructed_initial_sha
    )
    if original.get("random_init"):
        weight_source_correct = replay.get("weight_initializer_sha256") is None
    else:
        weight_source_correct = (
            replay.get("weight_initializer_sha256") == file_sha(args.split_source)
            and replay.get("initial_router_state_sha256") == state_sha(split_source["router"])
        )
    checks = {
        "router_tensors_bit_exact": tensors_exact,
        "training_history_exact": original.get("history") == replay.get("history"),
        "best_score_exact": original.get("best_score") == replay.get("best_score"),
        "split_sequences_exact": splits_exact,
        "training_hyperparameters_exact": hyperparameters_exact,
        "split_source_authenticated": split_source_authenticated,
        "scale_specific_random_init_correct": scale_mode_correct,
        "initialization_mode_correct": mode_correct,
        "initial_router_state_hash_recorded": initial_state_recorded,
        "weight_source_correct": weight_source_correct,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "scale": args.scale,
        "original_policy_sha256": file_sha(args.original),
        "replay_policy_sha256": file_sha(args.replay),
        "split_source_sha256": file_sha(args.split_source),
        "original_router_state_sha256": state_sha(original["router"]),
        "replay_router_state_sha256": state_sha(replay["router"]),
        "initial_router_state_sha256": replay.get("initial_router_state_sha256"),
        "independently_reconstructed_initial_state_sha256": reconstructed_initial_sha,
        "initialization_mode": replay.get("initialization_mode"),
        **checks,
        "oracle_summary_sha256": replay.get("oracle_summary_sha256"),
        "oracle_records_sha256": replay.get("oracle_records_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output, "w"), indent=2)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
