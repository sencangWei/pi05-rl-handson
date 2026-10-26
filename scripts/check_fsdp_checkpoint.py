"""Tiny CUDA test of RLinf DCP resume and streamed full-weight checkpoint saves."""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/RLinf"))
from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy


def assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_equal(a, b)
    else:
        assert actual == expected, (actual, expected)


def draws():
    return (random.random(), float(np.random.random()), torch.rand(4),
            torch.rand(4, device="cuda"))


def exercise(directory, cpu_offload):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    cfg = OmegaConf.load(ROOT / "evidence/ppo-composed.json").actor
    cfg.fsdp_config.wrappers = {"linears": {"module_classes": ["Linear"]}}
    cfg.fsdp_config.mixed_precision_rules = {"linears": "fp32"}
    cfg.fsdp_config.mixed_precision_policies = {"fp32": {"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"}}
    cfg.fsdp_config.mixed_precision.param_dtype = "fp32"
    module = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2)).cuda()
    module[0].requires_grad_(False)
    model = FSDPStrategy(cfg, world_size=1).wrap_model(module, None)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    model(torch.ones(2, 8, device="cuda")).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    expected_model = {k: v.cpu().clone() for k, v in get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=False)).items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    observations = []
    original = Checkpoint.state_dict

    def observe(checkpoint):
        state = original(checkpoint)
        devices = sorted({tensor.device.type for tensor in state["model"].values()})
        assert checkpoint.opts.cpu_offload is cpu_offload
        observations.append(devices)
        return state

    # Omit the option for the old case to exercise the backward-compatible default.
    save_options = {} if cpu_offload else {"checkpoint_cpu_offload": False}
    with patch.object(Checkpoint, "state_dict", observe):
        FSDPStrategy.save_checkpoint(model, optimizer, scheduler, str(directory),
                                     **save_options)
    assert observations == [["cpu" if cpu_offload else "cuda"]], observations
    expected_draws = draws()
    full_path = directory / "model_state_dict/full_weights.pt"
    full_weights = torch.load(full_path, map_location="cpu", weights_only=True)
    plain = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))
    result = plain.load_state_dict(full_weights, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert_equal(full_weights, expected_model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                value.add_(7)
    optimizer.param_groups[0]["lr"] = 123
    scheduler.step()
    draws()
    FSDPStrategy.load_checkpoint(model, optimizer, scheduler, str(directory))
    restored = get_model_state_dict(model, options=StateDictOptions(
        full_state_dict=True, cpu_offload=False))
    assert_equal(restored, expected_model)
    assert_equal(optimizer.state_dict(), expected_optimizer)
    assert_equal(scheduler.state_dict(), expected_scheduler)
    assert_equal(draws(), expected_draws)
    return {"checkpoint_cpu_offload": cpu_offload,
            "save_model_devices": observations[0], "model_tensors": len(full_weights),
            "model_optimizer_scheduler_rng_exact": True,
            "strict_full_weights_load": True,
            "full_weights_sha256": hashlib.sha256(full_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/ppo-fsdp-checkpoint.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA; CPU fallback does not validate FSDP saves")
    os.environ["LOCAL_RANK"] = "0"
    with tempfile.TemporaryDirectory(prefix="pi05-checkpoint-probe-") as temporary:
        dist.init_process_group("nccl", init_method=f"file://{temporary}/group", rank=0, world_size=1)
        try:
            cases = [exercise(Path(temporary) / str(flag), flag) for flag in (True, False)]
        finally:
            dist.destroy_process_group()
    sources = ["third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/base.py",
               "third_party/RLinf/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
               "third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/checkpoint.py",
               "third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/fsdp.py",
               "configs/rlinf-checkpoint-memory.patch", "configs/ppo_goal_overrides.yaml",
               "scripts/check_fsdp_checkpoint.py"]
    record = {"status": "PASS", "torch": torch.__version__, "world_size": 1,
              "scope": "tiny actual CUDA RLinf FSDP strategy; no full-model memory-fit claim",
              "cases": cases, "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                         for name in sources}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
