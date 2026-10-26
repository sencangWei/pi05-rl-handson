"""Actual CUDA FSDP regression: shared flat/local shard moves once per direction."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/RLinf"))
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy


def main():
    os.environ["LOCAL_RANK"] = "0"
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as temporary:
        dist.init_process_group("nccl", init_method=f"file://{temporary}/group", rank=0, world_size=1)
        try:
            cfg = OmegaConf.load(ROOT / "evidence/ppo-composed.json").actor
            cfg.fsdp_config.wrappers = {}
            cfg.fsdp_config.mixed_precision_rules = {}
            cfg.fsdp_config.mixed_precision_policies = {}
            cfg.fsdp_config.mixed_precision.param_dtype = "fp32"
            strategy = FSDPStrategy(cfg, world_size=1)
            model = strategy.wrap_model(nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 8)).cuda(), None)
            x = torch.randn(2, 128, device="cuda")
            expected = model(x).detach().clone()
            model(x).square().mean().backward()
            expected_params = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            expected_grads = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
            transfers = []
            move = strategy._move_tensor
            def traced(tensor, device):
                if tensor is not None and tensor.device != torch.device(device):
                    transfers.append((str(tensor.device), str(device), tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.numel()))
                return move(tensor, device)
            with patch.object(strategy, "_move_tensor", side_effect=traced):
                for _ in range(3):
                    strategy.offload_param_and_grad(model, True)
                    for name, p in model.named_parameters():
                        assert p.device.type == "cpu"
                        torch.testing.assert_close(p, expected_params[name], rtol=0, atol=0)
                        torch.testing.assert_close(p.grad, expected_grads[name], rtol=0, atol=0)
                    strategy.onload_param_and_grad(model, torch.device("cuda:0"), True)
                    for name, p in model.named_parameters():
                        torch.testing.assert_close(p.grad.cpu(), expected_grads[name], rtol=0, atol=0)
                    # Training calls zero_grad before the next forward.
                    model.zero_grad(set_to_none=True)
                    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
                    model(x).square().mean().backward()
            # Within each phase a source storage must not be copied twice.
            # Use six movement phases identified by device direction.
            phases = []
            for entry in transfers:
                if not phases or phases[-1][0][:2] != entry[:2]:
                    phases.append([])
                phases[-1].append(entry)
            assert len(phases) == 6, phases
            for phase in phases:
                assert len(phase) == len(set(phase)), f"Duplicate source storage transfer: {phase}"
            source = "third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/fsdp.py"
            result = {"status": "PASS", "cycles": 3, "transfers_per_phase": [len(p) for p in phases],
                      "parameters_gradients_outputs_exact": True, "source_sha256": hashlib.sha256((ROOT / source).read_bytes()).hexdigest(),
                      "scope": "tiny real CUDA FSDP; full-model capacity requires separate test"}
            (ROOT / "evidence/ppo-fsdp-offload-alias.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result))
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
