"""Compact, exact inference backup; verify by reconstructing tensors from the official base."""
# CPU selection and the pinned OpenPI source path must precede model imports.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/openpi/src"))

import ml_dtypes
import numpy as np
from flax.traverse_util import flatten_dict
from openpi.models.model import restore_params

from export_sft_for_ppo import sha256
from training_config import ASSET, NORM_SHA256, PARENT


def tensor_record(array):
    array = np.asarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def tree_identity(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def load(path):
    return flatten_dict(restore_params(path, restore_type=np.ndarray), sep="/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "verify"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    out = args.adapter
    if args.mode == "export":
        if args.checkpoint is None or out.exists():
            raise ValueError("Need a checkpoint and a fresh adapter directory")
        original = load(args.checkpoint / "params")
        records = {k: tensor_record(v) for k, v in original.items()}
        adapters = {k: v for k, v in original.items() if "lora" in k}
        if len(adapters) != 20 or any(v.dtype != np.float32 for v in adapters.values()):
            raise ValueError("Expected twenty float32 LoRA tensors")
        del original
        norm = args.checkpoint / "assets" / ASSET / "norm_stats.json"
        if sha256(norm) != NORM_SHA256:
            raise ValueError("Normalization mismatch")
    else:
        meta = json.loads((out / "metadata.json").read_text())
        if meta["format"] != "pi05-recovery-lora-v1":
            raise ValueError("Unknown backup format")
        for name, expected in meta["files"].items():
            if name not in ("adapter.npz", "norm_stats.json") or sha256(out / name) != expected:
                raise ValueError("Backup file mismatch")
        with np.load(out / "adapter.npz", allow_pickle=False) as saved:
            adapters = {k: saved[k] for k in saved.files}
        records = meta["original_parameters"]
    # Reconstruct every original tensor, including all frozen tensors, from the public base.
    base = load(PARENT / "params")
    if any("lora" in k for k in base):
        raise ValueError("Expected a non-LoRA official base")
    reconstructed = {k: tensor_record(v.astype(ml_dtypes.bfloat16)) for k, v in base.items()}
    del base
    reconstructed.update({k: tensor_record(v) for k, v in adapters.items()})
    if reconstructed != records:
        raise ValueError("Reconstructed parameter tree differs from trained checkpoint")
    if args.mode == "export":
        out.mkdir(parents=True)
        np.savez_compressed(out / "adapter.npz", **adapters)
        shutil.copyfile(norm, out / "norm_stats.json")
        meta = {"format": "pi05-recovery-lora-v1", "purpose": "inference; optimizer state excluded",
                "official_base": "gs://openpi-assets/checkpoints/pi05_libero",
                "checkpoint": str(args.checkpoint.resolve()),
                "model": {"pi05": True, "action_horizon": 10, "discrete_state_input": False,
                          "paligemma_variant": "gemma_2b_lora", "action_expert_variant": "gemma_300m_lora"},
                "original_parameters": records, "tree_sha256": tree_identity(records),
                "files": {name: sha256(out / name) for name in ("adapter.npz", "norm_stats.json")}}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    if sha256(out / "norm_stats.json") != NORM_SHA256 or tree_identity(records) != meta["tree_sha256"]:
        raise ValueError("Backup identity mismatch")
    print(json.dumps({"status": "PASS", "mode": args.mode, "tensors": len(records),
                      "lora_tensors": len(adapters), "tree_sha256": tree_identity(records)}), flush=True)


if __name__ == "__main__":
    main()
