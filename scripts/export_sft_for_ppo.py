"""CPU export of this project's pi05 multitask LoRA checkpoint for RLinf.

Merges adapters before using the pinned OpenPI array mapping. This is a
checkpoint conversion, not evidence of cross-framework policy equivalence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "third_party/openpi/examples/convert_jax_model_to_pytorch.py"


def flatten(tree, prefix=""):
    result = {}
    for key, value in tree.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def merge_lora(params):
    """Return dense flat params and audit; scanned layer axes remain batched.

    Actual fixed variants use alpha=rank, rslora=False. Attention scale is
    therefore one. OpenPI FeedForward._dot independently applies no scale.
    """
    merged = dict(params)
    pairs = []
    for suffix, rank in (("", 16), ("_1", 32)):
        for name in ("q_einsum", "kv_einsum", "attn_vec_einsum"):
            base = f"PaliGemma/llm/layers/attn/{name}{suffix}"
            pairs.append((base + "/w", base + "/lora_a", base + "/lora_b", rank))
        for name in ("gating_einsum", "linear"):
            base = f"PaliGemma/llm/layers/mlp{suffix}/{name}"
            pairs.append((base, base + "_lora_a", base + "_lora_b", rank))
    expected = {key for _, a, b, _ in pairs for key in (a, b)}
    actual = {key for key in params if "lora" in key}
    if actual != expected:
        raise ValueError(f"LoRA key mismatch: missing={expected-actual}, extra={actual-expected}")
    audit = []
    for weight, a_key, b_key, rank in pairs:
        w, a, b = (np.asarray(params[key], dtype=np.float32) for key in (weight, a_key, b_key))
        if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != rank or b.shape[-2] != rank:
            raise ValueError(f"Invalid adapter shapes: {weight}: {a.shape}, {b.shape}")
        if w.shape != (*a.shape[:-1], b.shape[-1]):
            raise ValueError(f"Adapter/base shape mismatch: {weight}")
        if not all(np.isfinite(array).all() for array in (w, a, b)):
            raise ValueError(f"Nonfinite source: {weight}")
        # Actual upstream _make_lora_eqns for BTNH,NHD->BTD produces
        # BTNH,NHL->BTL; BTL,NLD->BTD. N is reduced twice, so B must
        # be summed over heads before contraction with each head's A.
        delta = a @ (b.sum(axis=-3, keepdims=True) if "attn_vec_einsum" in weight else b)
        dense = w + delta
        if not np.isfinite(dense).all():
            raise ValueError(f"Nonfinite merged weight: {weight}")
        merged[weight] = dense
        del merged[a_key], merged[b_key]
        audit.append(dict(key=weight, rank=rank, scale=1.0, shape=list(w.shape),
                          delta_max=float(np.max(np.abs(delta)))))
    return merged, audit


def upstream_mappings():
    """Load only the two pinned pure mapping functions; avoid HF allocation."""
    import torch

    tree = ast.parse(CONVERTER.read_text())
    names = {"slice_paligemma_state_dict", "slice_gemma_state_dict"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("Pinned converter functions missing")
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CONVERTER), "exec"), namespace)
    return namespace


def convert_dense(params):
    import torch

    mappings = upstream_mappings()
    vision = SimpleNamespace(hidden_size=1152, num_hidden_layers=27)
    text = SimpleNamespace(hidden_size=2048, num_hidden_layers=18, num_attention_heads=8, head_dim=256)
    expert = SimpleNamespace(width=1024, depth=18, num_heads=8, head_dim=256)
    backbone = {key.removeprefix("PaliGemma/"): value for key, value in params.items()
                if key.startswith("PaliGemma/")}
    state, expert_state = mappings["slice_paligemma_state_dict"](
        backbone, SimpleNamespace(vision_config=vision, text_config=text))
    state.update(mappings["slice_gemma_state_dict"](
        expert_state, expert, num_expert=1, checkpoint_dir="pi05", pi05=True))
    # RLinf's official-key loader takes the tied embedding from lm_head.
    # The upstream HF save step creates this alias; direct array export must
    # supply it explicitly without duplicating a 2 GiB tensor.
    state["paligemma_with_expert.paligemma.lm_head.weight"] = state.pop(
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight")
    projection_keys = {f"{name}/{kind}" for name in
                       ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
                       for kind in ("kernel", "bias")}
    actual = {key for key in params if not key.startswith("PaliGemma/")}
    if actual != projection_keys:
        raise ValueError(f"Unexpected projection keys: {actual ^ projection_keys}")
    for key in sorted(projection_keys):
        name, kind = key.split("/")
        array = np.asarray(params[key])
        state[f"{name}.{'weight' if kind == 'kernel' else 'bias'}"] = torch.from_numpy(
            array.T if kind == "kernel" else array)
    # Mapping functions pass through unknown JAX leaves; reject every leftover.
    if any("/" in key or "lora" in key for key in state):
        raise ValueError("Unmapped source leaves remain in export")
    # Fixed architecture: vision439 + VLM164 + expert200 + projections8.
    if len(state) != 811:
        raise ValueError(f"Expected 811 mapped tensors, got {len(state)}")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Nonfinite exported tensor")
    return state


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import sys
    sys.path.insert(0, str(ROOT / "third_party/openpi/src"))
    from openpi.models.model import restore_params
    from safetensors.torch import save_file

    checkpoint = args.checkpoint.resolve()
    assets = checkpoint / "assets"
    if not assets.is_dir() or not list(assets.rglob("norm_stats.json")):
        raise ValueError(f"Checkpoint normalization assets missing: {assets}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    params = flatten(restore_params(checkpoint / "params", restore_type=np.ndarray, dtype="float32"))
    if len(params) != 71:
        raise ValueError(f"Expected 71 source leaves, got {len(params)}")
    source_shapes = {key: list(value.shape) for key, value in params.items()}
    metadata = json.loads((checkpoint / "params/_METADATA").read_text())["tree_metadata"]
    recorded_shapes = {"/".join(ast.literal_eval(key)[1:-1]): value["value_metadata"]["write_shape"]
                       for key, value in metadata.items()}
    if source_shapes != recorded_shapes:
        raise ValueError("Restored source keys/shapes disagree with checkpoint metadata")
    dense, audit = merge_lora(params)
    del params
    state = convert_dense(dense)
    del dense
    args.output.mkdir(parents=True)
    save_file({key: value.contiguous() for key, value in state.items()}, str(args.output / "model.safetensors"))
    shutil.copytree(assets, args.output / "assets")
    config = dict(pi05=True, action_dim=32, action_horizon=10, max_token_len=200,
                  discrete_state_input=False, paligemma_variant="gemma_2b",
                  action_expert_variant="gemma_300m", precision="float32")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    record = dict(source=str(checkpoint), source_metadata_sha256=sha256(checkpoint / "params/_METADATA"),
                  converter_sha256=sha256(CONVERTER), source_shapes=source_shapes, merges=audit,
                  output_shapes={key: list(value.shape) for key, value in state.items()},
                  output_sha256=sha256(args.output / "model.safetensors"),
                  assets={str(path.relative_to(assets)): sha256(path) for path in assets.rglob("*") if path.is_file()},
                  policy_equivalence_verified=False)
    (args.output / "export_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), tensors=len(state), merged_pairs=len(audit))))


if __name__ == "__main__":
    main()
