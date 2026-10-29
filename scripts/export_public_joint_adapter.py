"""Exact compact inference backup for the public-normal residual joint model.

The archive excludes optimizer/RNG state. ``restore`` creates an inference-only
OpenPI checkpoint from the official frozen base and the saved LoRA tensors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = "joint_contract29"
FORMAT = "pi05-public-joint-lora-v1"
FILES = {"adapter.npz", "norm_stats.json", "dataset-manifest.json"}
MODEL = dict(pi05=True, discrete_state_input=True, action_horizon=10, action_dim=8,
             state_dim=9, control_hz=20, representation="residual",
             joint_residual_anchor="measured current joints for every future target",
             gripper="positive close, negative open; no joint residual transform",
             paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_norm(value):
    stats = value["norm_stats"]
    if set(stats) != {"state", "actions"}:
        raise ValueError("Expected measured joint state and joint-action normalization")
    result = {}
    for name, size in (("state", 9), ("actions", 8)):
        item = {key: entry for key, entry in stats[name].items()
                if not (key in ("q01", "q99") and entry is None)}
        if set(item) != {"mean", "std"}:
            raise ValueError("Expected z-score statistics without quantile transforms")
        mean, std = np.asarray(item["mean"]), np.asarray(item["std"])
        if (mean.shape != (size,) or std.shape != (size,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Invalid joint normalization shape or values")
        result[name] = item
    return result


def validate_contract(metadata, manifest):
    if (metadata["format"] != FORMAT or metadata["model"] != MODEL
            or metadata["official_base"] != "gs://openpi-assets/checkpoints/pi05_libero"
            or metadata["purpose"] != "inference only; optimizer and RNG state excluded"
            or metadata["manifest_sha256"] != sha(manifest)
            or set(metadata["files"]) != FILES):
        raise ValueError("Adapter model, dataset or artifact contract mismatch")


def helpers():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(ROOT / "third_party/openpi/src"))
    from export_recovery_adapter import load, tensor_record, tree_identity
    from training_config import PARENT
    return load, tensor_record, tree_identity, PARENT


def reconstruct(adapters, records, keep_arrays=False):
    import ml_dtypes
    load, tensor_record, _, parent = helpers()
    if len(adapters) != 20 or any("lora" not in name or value.dtype != np.float32
            or not np.isfinite(value).all() for name, value in adapters.items()):
        raise ValueError("Expected twenty finite float32 LoRA tensors")
    base = load(parent / "params")
    if any("lora" in name for name in base):
        raise ValueError("Expected the non-LoRA official frozen parent")
    actual, arrays = {}, {}
    for name in list(base):
        value = base.pop(name).astype(ml_dtypes.bfloat16)
        actual[name] = tensor_record(value)
        if keep_arrays:
            arrays[name] = value
    actual.update({name: tensor_record(value) for name, value in adapters.items()})
    if actual != records:
        raise ValueError("Official base plus adapter differs from trained parameter tree")
    if keep_arrays:
        arrays.update(adapters)
    return arrays


def read_archive(adapter, manifest):
    metadata = json.loads((adapter / "metadata.json").read_text())
    validate_contract(metadata, manifest)
    for name, expected in metadata["files"].items():
        if sha(adapter / name) != expected:
            raise ValueError(f"Adapter artifact changed: {name}")
    if sha(adapter / "dataset-manifest.json") != sha(manifest):
        raise ValueError("Embedded dataset manifest differs")
    norm = json.loads((adapter / "norm_stats.json").read_text())
    canonical_norm(norm)
    with np.load(adapter / "adapter.npz", allow_pickle=False) as saved:
        adapters = {name: saved[name] for name in saved.files}
    return metadata, adapters


def save_params(output, arrays):
    from flax.traverse_util import unflatten_dict
    import orbax.checkpoint as ocp
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(output / "params", {"params": unflatten_dict(arrays, sep="/")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "verify", "restore"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Fresh inference checkpoint directory for restore")
    args = parser.parse_args()
    adapter, manifest = args.adapter.resolve(), args.manifest.resolve()
    data = json.loads(manifest.read_text())
    if (data.get("public_joint_schema") != "public-normal-human-recovery-v1" or not data["ready"]
            or data["action_representation"] != "current-state anchored future absolute joint residual + grip"):
        raise ValueError("Expected a ready public-normal residual joint dataset")
    load, tensor_record, tree_identity, _ = helpers()
    if args.mode == "export":
        if args.checkpoint is None or adapter.exists():
            raise ValueError("Export requires a checkpoint and a fresh adapter directory")
        norm = args.checkpoint / "assets" / ASSET / "norm_stats.json"
        source_norm = manifest.parent / "assets" / ASSET / "norm_stats.json"
        if canonical_norm(json.loads(norm.read_text())) != canonical_norm(json.loads(source_norm.read_text())):
            raise ValueError("Checkpoint and train-only normalization differ")
        original = load(args.checkpoint / "params")
        records = {name: tensor_record(value) for name, value in original.items()}
        adapters = {name: value for name, value in original.items() if "lora" in name}
        del original
        reconstruct(adapters, records)
        adapter.mkdir(parents=True)
        np.savez_compressed(adapter / "adapter.npz", **adapters)
        shutil.copyfile(norm, adapter / "norm_stats.json")
        shutil.copyfile(manifest, adapter / "dataset-manifest.json")
        metadata = dict(format=FORMAT, model=MODEL,
            purpose="inference only; optimizer and RNG state excluded",
            official_base="gs://openpi-assets/checkpoints/pi05_libero",
            checkpoint=str(args.checkpoint.resolve()), manifest_sha256=sha(manifest),
            original_parameters=records, tree_sha256=tree_identity(records),
            files={name: sha(adapter / name) for name in sorted(FILES)})
        (adapter / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        metadata, adapters = read_archive(adapter, manifest)
        records = metadata["original_parameters"]
        if tree_identity(records) != metadata["tree_sha256"]:
            raise ValueError("Original tree identity changed")
        if args.mode == "restore" and (args.output is None or args.output.exists()):
            raise ValueError("Restore requires a fresh output directory")
        arrays = reconstruct(adapters, records, keep_arrays=args.mode == "restore")
        if args.mode == "restore":
            output = args.output.resolve()
            output.mkdir(parents=True)
            save_params(output, arrays)
            del arrays
            restored = load(output / "params")
            if {name: tensor_record(value) for name, value in restored.items()} != records:
                raise ValueError("Persisted inference parameters differ from original trained checkpoint")
            assets = output / "assets" / ASSET
            assets.mkdir(parents=True)
            shutil.copyfile(adapter / "norm_stats.json", assets / "norm_stats.json")
            (output / "INFERENCE_ONLY.json").write_text(json.dumps(dict(adapter=str(adapter),
                tree_sha256=metadata["tree_sha256"], optimizer_state=False, model=MODEL), indent=2) + "\n")
    print(json.dumps(dict(status="PASS", mode=args.mode, tensors=len(records),
        lora_tensors=len(adapters), tree_sha256=metadata["tree_sha256"])), flush=True)


if __name__ == "__main__":
    main()
