"""Tiny CUDA proof of initial-full and subsequent adapter-only FSDP sync."""
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/RLinf"))
from check_fsdp_adapter_checkpoint import BASE, Tiny
from rlinf.hybrid_engines.fsdp.strategy.adapter_checkpoint import _selected
from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.hybrid_engines.weight_syncer.patch_syncer import PatchWeightSyncer
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)


async def verify(model, module, strategy, cfg, snapshot_device):
    sender = PatchWeightSyncer(snapshot_device=snapshot_device, transport_device="cuda", init_sync_enabled=True)
    receiver = PatchWeightSyncer(snapshot_device=snapshot_device, transport_device="cuda", init_sync_enabled=True)
    target = Tiny().cuda()
    to_sender, to_receiver = asyncio.Queue(), asyncio.Queue()
    async def send_actor(data):
        await to_receiver.put(data)
    async def send_rollout(data):
        await to_sender.put(data)
    with FSDP.summon_full_params(model, writeback=False):
        parameters, buffers = _selected(model)
        selected_names = list(parameters) + list(buffers)
    calls = []
    def full_state(**kwargs):
        calls.append(kwargs)
        return strategy.get_model_state_dict(model, **kwargs)
    actor = SimpleNamespace(cfg=OmegaConf.create({"actor": {"checkpoint_format": "adapter", "checkpoint_base_sha256": BASE}}),
                            model=model, weight_syncer=sender, param_names_need_sync=selected_names,
                            get_model_state_dict=full_state)
    initial = EmbodiedFSDPActor.get_rollout_state_dict(actor)
    await asyncio.gather(sender.init_sender(initial, selected_names, send_actor, to_sender.get),
                         receiver.init_receiver(target.state_dict(), to_receiver.get, send_rollout))
    for name, value in initial.items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    del initial
    assert len(calls) == 1
    frozen = {name: value.detach().clone() for name, value in target.frozen.state_dict().items()}
    rejected = []
    with patch.object(FSDP, "state_dict", side_effect=AssertionError("full state dict after init forbidden")):
        for version in (1, 2):
            with torch.no_grad():
                module.adapter.weight.add_(0.125)
                module.adapter.bias.add_(0.25)
                module.running.add_(0.5)
            selected = EmbodiedFSDPActor.get_rollout_state_dict(actor)
            assert set(selected) == set(selected_names)
            assert set(selected).isdisjoint({"frozen.weight", "frozen.bias", "temporary"})
            with FSDP.summon_full_params(model, writeback=False):
                parameters, buffers = _selected(model)
                for name, value in {**parameters, **buffers}.items():
                    assert selected[name].data_ptr() != value.data_ptr()
            await sender.sync(selected, send_actor, version)
            assert await receiver.apply(target, to_receiver.get) == version
            for name, value in selected.items():
                torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
            for name, value in target.frozen.state_dict().items():
                torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        for label in ("missing", "extra", "shape"):
            bad = dict(selected)
            if label == "missing":
                bad.pop("adapter.bias")
            elif label == "extra":
                bad["extra"] = torch.ones(1, device="cuda")
            else:
                bad["adapter.bias"] = torch.ones(3, device="cuda")
            try:
                sender.create_patch(bad, 3)
            except ValueError:
                rejected.append(label)
            else:
                raise AssertionError(f"Accepted {label}")
        actor.param_names_need_sync = selected_names[:-1]
        try:
            EmbodiedFSDPActor.get_rollout_state_dict(actor)
        except ValueError:
            rejected.append("actor_contract")
        else:
            raise AssertionError("Actor accepted selected key mismatch")
    return {"snapshot_device": snapshot_device, "repeated_updates": 2, "rejected": rejected,
            "selected_keys": selected_names, "full_state_dict_calls": len(calls)}


def main():
    os.environ["LOCAL_RANK"] = "0"
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix="pi05-adapter-sync-") as temporary:
        dist.init_process_group("nccl", init_method=f"file://{temporary}/group", rank=0, world_size=1)
        try:
            cfg = OmegaConf.load(ROOT / "evidence/ppo-composed.json").actor
            cfg.fsdp_config.wrappers = {"linears": {"module_classes": ["Linear"]}}
            cfg.fsdp_config.mixed_precision_rules = {"linears": "fp32"}
            cfg.fsdp_config.mixed_precision_policies = {"fp32": {
                "param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"}}
            cfg.fsdp_config.mixed_precision.param_dtype = "fp32"
            strategy = FSDPStrategy(cfg, world_size=1)
            results = []
            for snapshot_device in ("cpu", "cuda"):
                module = Tiny().cuda()
                module.frozen = checkpoint_wrapper(module.frozen)
                module.adapter = checkpoint_wrapper(module.adapter)
                model = strategy.wrap_model(module, None)
                results.append(asyncio.run(verify(model, module, strategy, cfg, snapshot_device)))
        finally:
            dist.destroy_process_group()
    sources = ["third_party/RLinf/rlinf/workers/actor/embodied_fsdp_actor_worker.py",
               "third_party/RLinf/rlinf/hybrid_engines/weight_syncer/patch_syncer.py",
               "third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/adapter_checkpoint.py",
               "scripts/check_fsdp_adapter_sync.py", "configs/rlinf-adapter-sync.patch"]
    record = {"status": "PASS", "torch": torch.__version__, "world_size": 1, "results": results,
              "scope": "tiny actual CUDA FSDP and actual actor extraction / patch sync; full PPO not verified",
              "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}}
    (ROOT / "evidence/ppo-fsdp-adapter-sync.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
