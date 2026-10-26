"""Verify single-rank adapter resume without invoking any full-model state dict."""
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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/RLinf"))
from check_fsdp_checkpoint import assert_equal, draws  # noqa: E402
from rlinf.hybrid_engines.fsdp.strategy.adapter_checkpoint import read_adapter_model  # noqa: E402
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy  # noqa: E402

BASE = "4526e30c6aa11108a3331f7f4ae7931394629961df42736deedd7169ec154d5f"


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(8, 8)
        self.frozen.requires_grad_(False)
        self.adapter = nn.Linear(8, 2)
        self.register_buffer("running", torch.ones(2))
        self.register_buffer("temporary", torch.zeros(2), persistent=False)

    def forward(self, x):
        return self.adapter(self.frozen(x)) + self.running


def main():
    os.environ["LOCAL_RANK"] = "0"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with tempfile.TemporaryDirectory(prefix="pi05-adapter-checkpoint-") as temporary:
        dist.init_process_group("nccl", init_method=f"file://{temporary}/group", rank=0, world_size=1)
        try:
            cfg = OmegaConf.load(ROOT / "evidence/ppo-composed.json").actor
            cfg.fsdp_config.wrappers = {"linears": {"module_classes": ["Linear"]}}
            cfg.fsdp_config.mixed_precision_rules = {"linears": "fp32"}
            cfg.fsdp_config.mixed_precision_policies = {"fp32": {
                "param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"}}
            cfg.fsdp_config.mixed_precision.param_dtype = "fp32"
            module = Tiny().cuda()
            initial_frozen = module.frozen.weight.detach().clone()
            module.frozen = checkpoint_wrapper(module.frozen)
            module.adapter = checkpoint_wrapper(module.adapter)
            model = FSDPStrategy(cfg, world_size=1).wrap_model(module, None)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
            model(torch.ones(2, 8, device="cuda")).square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_scheduler = copy.deepcopy(scheduler.state_dict())
            options = {"checkpoint_format": "adapter", "checkpoint_base_sha256": BASE}
            # Any accidental full-model state-dict path must fail this test.
            with patch.object(FSDP, "state_dict", side_effect=AssertionError("full state dict forbidden")):
                FSDPStrategy.save_checkpoint(model, optimizer, scheduler, temporary, **options)
                payload = torch.load(Path(temporary) / "adapter_checkpoint.pt", map_location="cpu", weights_only=False)
                assert set(payload["model"]) == {"adapter.weight", "adapter.bias", "running"}
                assert all(v.device.type == "cpu" for v in payload["model"].values())
                assert payload["model"]["adapter.weight"].dtype == torch.float32
                assert not (Path(temporary) / "model_state_dict/full_weights.pt").exists()
                expected_draws = draws()
                with torch.no_grad():
                    module.adapter.weight.add_(1)
                    module.adapter.bias.add_(1)
                    module.running.add_(1)
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.add_(7)
                optimizer.param_groups[0]["lr"] = 123
                scheduler.step()
                draws()
                FSDPStrategy.load_checkpoint(model, optimizer, scheduler, temporary, **options)
                assert_equal(optimizer.state_dict(), expected_optimizer)
                assert_equal(scheduler.state_dict(), expected_scheduler)
                assert_equal(draws(), expected_draws)
                assert_equal(module.frozen.weight, initial_frozen)
                assert_equal(module.adapter.weight, payload["model"]["adapter.weight"])
                assert_equal(module.adapter.bias, payload["model"]["adapter.bias"])
                assert_equal(module.running, payload["model"]["running"])
            plain = Tiny().bfloat16()
            read_adapter_model(plain, payload, BASE, allow_dtype_cast=True)
            assert_equal(plain.adapter.weight, payload["model"]["adapter.weight"].bfloat16())
            rejected = []
            for label in ("base", "key", "shape", "nonfinite", "dtype"):
                bad = copy.deepcopy(payload)
                if label == "base":
                    bad["base_weights_sha256"] = "0" * 64
                elif label == "key":
                    bad["model"].pop("adapter.bias")
                elif label == "shape":
                    bad["model"]["adapter.bias"] = torch.zeros(3)
                elif label == "nonfinite":
                    bad["model"]["adapter.bias"][0] = float("nan")
                try:
                    read_adapter_model(plain, bad, BASE, allow_dtype_cast=label != "dtype")
                except ValueError:
                    rejected.append(label)
                else:
                    raise AssertionError(f"Failed to reject {label}")
            # Resume rejects the same bad base before copying parameters.
            with FSDP.summon_full_params(model, writeback=False):
                restored = module.adapter.weight.detach().clone()
            try:
                FSDPStrategy.load_checkpoint(model, optimizer, scheduler, temporary,
                                             checkpoint_format="adapter", checkpoint_base_sha256="0" * 64)
            except ValueError:
                pass
            else:
                raise AssertionError("Resume accepted wrong base")
            assert_equal(module.adapter.weight, restored)
        finally:
            dist.destroy_process_group()
    sources = ["third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/base.py",
               "third_party/RLinf/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
               "third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/adapter_checkpoint.py",
               "scripts/check_fsdp_adapter_checkpoint.py", "scripts/check_fsdp_checkpoint.py",
               "configs/rlinf-adapter-checkpoint.patch"]
    record = {"status": "PASS", "torch": torch.__version__, "world_size": 1,
              "model_optimizer_scheduler_rng_exact": True, "full_state_dict_forbidden": True,
              "frozen_weights_unchanged": True, "inference_bf16_cast_verified": True,
              "rejected": rejected, "model_keys": list(payload["model"]),
              "scope": "tiny actual CUDA FSDP strategy; full PPO save not yet verified",
              "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    destination = ROOT / "evidence/ppo-fsdp-adapter-checkpoint.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
