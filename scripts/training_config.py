"""Fixed official LIBERO parent, independent outputs, LoRA-only OpenPI config."""
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/openpi"
REVISION = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
ASSET = "physical-intelligence/libero"
PARENT = Path("/home/robot/voice-arm-sim/models/pi05_libero")
NORM_SHA256 = "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"


def verify_inventory(root, records):
    root = root.resolve()
    for relative, expected in records.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Inventory path escapes root")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Content checksum mismatch: " + str(path))


def make_config(experiment, steps, batch, lr):
    if not 1 <= steps <= 100000 or batch not in (1, 2, 4) or not math.isfinite(lr) or not 0 < lr <= .001:
        raise ValueError("Invalid bounded training parameters")
    if json.loads((SOURCE / "SNAPSHOT.json").read_text())["openpi"] != REVISION:
        raise ValueError("OpenPI snapshot revision changed")
    source_manifest = json.loads((ROOT / "evidence/openpi-source.json").read_text())
    if source_manifest["revision"] != REVISION:
        raise ValueError("Source inventory revision mismatch")
    verify_inventory(SOURCE, source_manifest["files"])
    if (PARENT / "manifest.json").read_bytes() != (ROOT / "evidence/parent-manifest.json").read_bytes():
        raise ValueError("Parent manifest identity changed")
    manifest = json.loads((PARENT / "manifest.json").read_text())
    if manifest["source"] != "gs://openpi-assets/checkpoints/pi05_libero":
        raise ValueError("Expected official LIBERO parent")
    for record in manifest["objects"]:
        path = (PARENT / record["relative_path"]).resolve()
        if not path.is_relative_to(PARENT) or path.stat().st_size != int(record["size"]):
            raise ValueError("Parent file inventory mismatch")
    verify_inventory(PARENT, {record["relative_path"]: record["sha256"] for record in manifest["objects"]})
    if hashlib.sha256((PARENT / "assets" / ASSET / "norm_stats.json").read_bytes()).hexdigest() != NORM_SHA256:
        raise ValueError("Parent normalization changed")
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "packages/openpi-client/src")]
    from flax import nnx
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training import config, optimizer, weight_loaders
    model = Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
                      action_horizon=10, discrete_state_input=False)
    return dataclasses.replace(config.get_config("pi05_libero"), name="pi05_multitask40_lora", exp_name=experiment,
        project_name="pi05-rl-handson", model=model,
        freeze_filter=nnx.All(nnx.Param, nnx.Not(PathRegex(".*lora.*"))), ema_decay=None,
        data=config.LeRobotLiberoDataConfig(repo_id=ASSET, assets=config.AssetsConfig(
            assets_dir=str(PARENT / "assets"), asset_id=ASSET),
            base_config=config.DataConfig(prompt_from_task=False), extra_delta_transform=False),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(PARENT / "params")),
        batch_size=batch, num_workers=0, seed=42, num_train_steps=steps,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=min(20, steps), peak_lr=lr,
            decay_steps=max(steps, 21), decay_lr=lr), optimizer=optimizer.AdamW(clip_gradient_norm=1.),
        checkpoint_base_dir=str(ROOT / "checkpoints"), assets_base_dir=str(ROOT / "assets"),
        save_interval=min(2000, steps), keep_period=None, log_interval=10,
        wandb_enabled=False, resume=False, overwrite=False)
