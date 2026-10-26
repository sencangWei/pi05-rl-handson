"""Strict CPU RLinf backbone load gate; this does not establish numeric parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=root / "evidence/ppo-export-load.json")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    sys.path[:0] = [str(root / "third_party/RLinf"), str(root / "third_party/openpi/src")]
    directory = args.export.resolve()
    manifest = json.loads((directory / "export_manifest.json").read_text())
    if manifest.get("policy_equivalence_verified") is not False:
        raise ValueError("Export must explicitly retain policy_equivalence_verified=false")
    weights_path = directory / "model.safetensors"
    weights_sha = sha256(weights_path)
    if weights_sha != manifest["output_sha256"]:
        raise ValueError("Export weights hash mismatch")
    source_metadata = Path(manifest["source"]) / "params/_METADATA"
    if sha256(source_metadata) != manifest["source_metadata_sha256"]:
        raise ValueError("Source checkpoint metadata hash mismatch")
    asset_hashes = {str(path.relative_to(directory / "assets")): sha256(path)
                    for path in (directory / "assets").rglob("*") if path.is_file()}
    if not asset_hashes or asset_hashes != manifest["assets"]:
        raise ValueError("Export asset paths/hashes mismatch")
    config = json.loads((directory / "config.json").read_text())
    expected = dict(pi05=True, action_dim=32, action_horizon=10, max_token_len=200,
                    discrete_state_input=False, paligemma_variant="gemma_2b",
                    action_expert_variant="gemma_300m", precision="float32")
    if config != expected:
        raise ValueError(f"Unexpected exported configuration: {config}")

    import torch
    from safetensors.torch import load_file
    from rlinf.models.embodiment.openpi.checkpoint import _convert_official_openpi_keys
    from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config

    set_torch_compile(False)
    cfg = Pi0Config(pi05=True, action_dim=32, action_horizon=10, max_token_len=200,
                    discrete_state_input=False, paligemma_variant="gemma_2b",
                    action_expert_variant="gemma_300m", dtype="float32")
    with torch.device("meta"):
        model = Pi0(cfg)
    raw = load_file(str(weights_path), device="cpu")
    if len(raw) != 811 or {key: list(value.shape) for key, value in raw.items()} != manifest["output_shapes"]:
        raise ValueError("Export tensor count/shapes disagree with manifest")
    state = _convert_official_openpi_keys(raw, weights_path)
    del raw
    target = model.state_dict()
    missing = sorted(target.keys() - state.keys())
    unexpected = sorted(state.keys() - target.keys())
    shape_errors = {key: [list(target[key].shape), list(state[key].shape)]
                    for key in target.keys() & state.keys() if target[key].shape != state[key].shape}
    if missing or unexpected or shape_errors:
        raise ValueError(json.dumps(dict(missing=missing, unexpected=unexpected, shape_errors=shape_errors)))
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("Strict gate requires float32 exported tensors")
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Strict load returned incompatible keys")
    model.eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Backbone contains non-CPU parameters")
    result = dict(status="PASS", export=str(directory), source=manifest["source"],
                  output_sha256=weights_sha, source_metadata_sha256=manifest["source_metadata_sha256"],
                  export_tensor_count=811, backbone_tensor_count=len(target), missing=[], unexpected=[],
                  shape_errors={}, strict_load=True, asset_hashes_verified=True,
                  config=config, device="cpu", torch_version=torch.__version__,
                  policy_equivalence_verified=False,
                  limitation="Strict keys, shapes and hashes only; no network numeric parity or simulation inference.")
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
