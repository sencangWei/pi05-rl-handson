"""One-step CPU full-network parity: original JAX LoRA versus RLinf dense.

Synthetic model-ready input isolates network conversion from preprocessing and
simulation. Each framework runs in a separate process, never concurrently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
ATOL = 1e-3
RTOL = 1e-3


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_fixture(path):
    rng = np.random.default_rng(42)
    fixture = {key: rng.uniform(-1, 1, (1, 224, 224, 3)).astype(np.float32) for key in CAMERAS}
    fixture[CAMERAS[-1]] = np.zeros((1, 224, 224, 3), np.float32)
    fixture.update(state=rng.uniform(-1, 1, (1, 32)).astype(np.float32),
                   noise=rng.standard_normal((1, 10, 32)).astype(np.float32),
                   tokens=np.zeros((1, 200), np.int32), mask=np.zeros((1, 200), bool))
    fixture["tokens"][0, :8] = [2, 100, 200, 300, 400, 500, 600, 1]
    fixture["mask"][0, :8] = True
    np.savez(path, **fixture)


def observation_data(fixture, convert):
    return dict(images={key: convert(fixture[key]) for key in CAMERAS},
                image_masks={key: convert(np.array([index < 2], bool)) for index, key in enumerate(CAMERAS)},
                state=convert(fixture["state"]), tokenized_prompt=convert(fixture["tokens"]),
                tokenized_prompt_mask=convert(fixture["mask"]))


def jax_probe(args, fixture):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation, restore_params
    from openpi.models.pi0_config import Pi0Config

    config = Pi0Config(pi05=True, action_horizon=10, action_dim=32, max_token_len=200,
                       discrete_state_input=False, dtype="float32",
                       paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    model = config.load(restore_params(args.checkpoint.resolve() / "params", restore_type=np.ndarray, dtype="float32"))
    observation = Observation(**observation_data(fixture, jnp.asarray))
    actions = model.sample_actions(jax.random.key(42), observation, num_steps=1,
                                   noise=jnp.asarray(fixture["noise"]))
    np.save(args.work / "jax-actions.npy", np.asarray(actions))


def torch_probe(args, fixture):
    import torch
    from safetensors.torch import load_file
    from rlinf.models.embodiment.openpi.checkpoint import _convert_official_openpi_keys
    from rlinf.models.embodiment.openpi.modules.model import Observation
    from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config

    torch.set_num_threads(args.threads)
    set_torch_compile(False)
    config = Pi0Config(pi05=True, action_horizon=10, action_dim=32, max_token_len=200,
                       discrete_state_input=False, dtype="float32",
                       paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    with torch.device("meta"):
        model = Pi0(config)
    path = args.export / "model.safetensors"
    state = _convert_official_openpi_keys(load_file(str(path), device="cpu"), path)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval()
    observation = Observation(**observation_data(fixture, torch.from_numpy))
    with torch.inference_mode():
        actions = model.sample_actions(observation, num_steps=1, noise=torch.from_numpy(fixture["noise"]))
    np.save(args.work / "torch-actions.npy", actions.cpu().numpy())


def compare(args):
    jax_actions = np.load(args.work / "jax-actions.npy")
    torch_actions = np.load(args.work / "torch-actions.npy")
    if jax_actions.shape != (1, 10, 32) or torch_actions.shape != jax_actions.shape:
        raise ValueError("Unexpected one-step output shape")
    finite = bool(np.isfinite(jax_actions).all() and np.isfinite(torch_actions).all())
    difference = torch_actions.astype(np.float64) - jax_actions.astype(np.float64)
    passed = finite and np.allclose(torch_actions, jax_actions, atol=ATOL, rtol=RTOL)
    manifest = json.loads((args.export / "export_manifest.json").read_text())
    if Path(manifest["source"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Export lineage does not match JAX checkpoint")
    if sha256(args.export / "model.safetensors") != manifest["output_sha256"]:
        raise ValueError("Export weights hash mismatch")
    if sha256(args.checkpoint / "params/_METADATA") != manifest["source_metadata_sha256"]:
        raise ValueError("Source checkpoint metadata hash mismatch")
    record = dict(status="PASS" if passed else "FAIL", check="one_step_full_network_float32_cpu",
                  checkpoint=str(args.checkpoint.resolve()), export=str(args.export.resolve()),
                  weights_sha256=manifest["output_sha256"], fixture_sha256=sha256(args.work / "input.npz"),
                  jax_output_sha256=sha256(args.work / "jax-actions.npy"),
                  torch_output_sha256=sha256(args.work / "torch-actions.npy"),
                  atol=ATOL, rtol=RTOL, finite=finite, shape=list(jax_actions.shape),
                  max_abs=float(np.max(np.abs(difference))), rms=float(np.sqrt(np.mean(difference ** 2))),
                  first7_max_abs=float(np.max(np.abs(difference[..., :7]))),
                  runtime_source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (
                      ROOT / "third_party/openpi/src/openpi/models/gemma.py",
                      ROOT / "third_party/openpi/src/openpi/models/siglip.py",
                      ROOT / "third_party/RLinf/rlinf/models/embodiment/openpi/modules/gemma.py",
                      ROOT / "third_party/RLinf/rlinf/models/embodiment/openpi/modules/siglip.py",
                      ROOT / "third_party/RLinf/rlinf/models/embodiment/openpi/modules/utils.py")},
                  num_steps=1, precision="float32", synthetic_observation=True,
                  scope="Original trained JAX LoRA versus merged RLinf dense; fixed noise, t=1 velocity via Euler step.",
                  limitation="Does not validate tokenizer, normalization, BF16, ten-step sampling, or simulation success.")
    (args.work / "parity.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "evidence/ppo-export-parity")
    parser.add_argument("--stage", choices=("all", "fixture", "jax", "torch", "compare"), default="all")
    parser.add_argument("--jax-python", type=Path, default=ROOT / ".venv-train/bin/python")
    parser.add_argument("--torch-python", type=Path, default=ROOT / ".venv-ppo/bin/python")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    sys.path[:0] = [str(ROOT / "third_party/RLinf"), str(ROOT / "third_party/openpi/src")]
    args.work.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "fixture"):
        if (args.work / "input.npz").exists():
            raise FileExistsError("Use a fresh work directory to avoid mixing parity runs")
        make_fixture(args.work / "input.npz")
    if args.stage == "all":
        for stage, python in (("jax", args.jax_python), ("torch", args.torch_python)):
            subprocess.run([str(python), str(Path(__file__).resolve()), "--stage", stage,
                            "--checkpoint", str(args.checkpoint.resolve()), "--export", str(args.export.resolve()),
                            "--work", str(args.work.resolve()), "--threads", str(args.threads)], check=True, timeout=1800)
        compare(args)
    elif args.stage in ("jax", "torch"):
        with np.load(args.work / "input.npz") as fixture:
            (jax_probe if args.stage == "jax" else torch_probe)(args, fixture)
    elif args.stage == "compare":
        compare(args)


if __name__ == "__main__":
    main()
