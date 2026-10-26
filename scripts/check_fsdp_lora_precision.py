"""Reproduce PEFT's input cast crossing a nested FSDP precision boundary."""
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/RLinf"))


def exercise(cast_inputs):
    model = nn.Sequential(nn.Linear(16, 16)).cuda()
    model = inject_adapter_in_model(
        LoraConfig(r=2, lora_alpha=2, target_modules=["0"]), model
    )
    # Actor owns FP32 storage; auto-wrapped trainable LoRA leaves compute BF16.
    for adapters in (model[0].lora_A, model[0].lora_B):
        adapters["default"] = FSDP(
            adapters["default"], device_id=0, use_orig_params=True,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                buffer_dtype=torch.float32, cast_forward_inputs=cast_inputs,
                cast_root_forward_inputs=False,
            ),
        )
    model = FSDP(model, device_id=0, use_orig_params=True,
                 sharding_strategy=ShardingStrategy.NO_SHARD)
    x = torch.ones((2, 16), device="cuda", dtype=torch.float32)
    out = model(x)
    out.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max().item() > 0 for g in gradients)
    return {"output_dtype": str(out.dtype), "finite_gradients": True}


def main():
    with tempfile.TemporaryDirectory() as temp:
        dist.init_process_group("nccl", init_method=f"file://{temp}/group", rank=0, world_size=1)
        try:
            try:
                exercise(False)
            except RuntimeError as error:
                if "same dtype" not in str(error):
                    raise
                failure = str(error)
            else:
                raise AssertionError("Expected the previous mixed-precision recipe to fail")
            fixed = exercise(True)
            # Exercise the real patched strategy, including root-owned parameters.
            from omegaconf import OmegaConf
            from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
            cfg = OmegaConf.load(ROOT / "evidence/ppo-composed.json").actor
            cfg.fsdp_config.wrappers = {}
            cfg.fsdp_config.mixed_precision_rules = {}
            cfg.fsdp_config.mixed_precision_policies = {}
            os.environ["LOCAL_RANK"] = "0"
            module = nn.Sequential(nn.Linear(16, 16)).cuda()
            module = inject_adapter_in_model(
                LoraConfig(r=2, lora_alpha=2, target_modules=["0"]), module
            )
            seen = []
            module.register_forward_pre_hook(lambda _, args: seen.append(args[0].dtype))
            model = FSDPStrategy(cfg, world_size=1).wrap_model(module, None)
            assert model._handle is not None, "Probe must cover a root owning parameters"
            # Root weights compute BF16, so the probe supplies BF16 to the Linear
            # itself while a separate floating-point kwarg tests root preservation.
            # A root pre-hook sees kwargs before nn.Sequential rejects them: use a
            # standalone hook that removes this sentinel after checking its dtype.
            sentinel_dtypes = []
            def inspect_root(_, args, kwargs):
                sentinel_dtypes.append(kwargs.pop("chains").dtype)
                return args, kwargs
            module.register_forward_pre_hook(inspect_root, with_kwargs=True)
            y = model(torch.ones(2, 16, device="cuda", dtype=torch.bfloat16),
                      chains=torch.full((2, 16), 0.1234567, device="cuda"))
            y.float().square().mean().backward()
            assert sentinel_dtypes == [torch.float32], sentinel_dtypes
            assert any(p.grad is not None and p.grad.abs().max() > 0
                       for p in model.parameters() if p.requires_grad)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.requires_grad and p.grad is not None)
            record = {"status": "PASS", "old_config_error": failure,
                      "cast_forward_inputs_true": fixed,
                      "actual_strategy_root_chains_dtype": str(sentinel_dtypes[0]),
                      "strategy_source_sha256": __import__("hashlib").sha256(
                          (ROOT / "third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/fsdp.py").read_bytes()).hexdigest(),
                      "scope": "actual nested FSDP + PEFT CUDA forward/backward and real RLinf strategy; tiny models"}
            Path("evidence/ppo-fsdp-lora-precision.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record))
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
