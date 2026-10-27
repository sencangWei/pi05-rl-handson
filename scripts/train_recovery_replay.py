"""Bounded recovery LoRA fine-tuning from our SFT parent, with old-task replay."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import re
import time

from export_sft_for_ppo import sha256
from train_multitask import free_gpu_mib

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "checkpoints/pi05_multitask40_lora/multitask40_lora_v1/9999"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="human28_replay2000_v1")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.experiment):
        parser.error("Invalid experiment name")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    if not args.dry_run and free_gpu_mib() < 24576:
        raise RuntimeError("Need 24 GiB free; other GPU processes will not be stopped")
    import training_config as helper
    from recovery_replay_dataset import RecoveryReplayDataset

    cfg = helper.make_config(args.experiment, args.steps, 1, args.learning_rate)
    from openpi.training import data_loader, weight_loaders
    cfg = dataclasses.replace(cfg, name="pi05_recovery_replay_lora",
        weight_loader=weight_loaders.CheckpointWeightLoader(str(PARENT / "params")))
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(f"No overwrite or implicit resume: {cfg.checkpoint_dir}")
    human = ROOT / "evidence/recovery-human28-manifest.json"
    replay = ROOT / "evidence/dataset-manifest.json"
    dataset = RecoveryReplayDataset(human, replay)
    validation = json.loads((ROOT / "evidence/data-validation.json").read_text())
    if not validation["passed"] or validation["manifest_sha256"] != sha256(replay):
        raise ValueError("Public replay validation is missing or stale")
    # Full checksums here prevent silent changes after the original dataset audit.
    replay_records = [r for r in dataset.replay.manifest["records"] if r["split"] == "train"]
    helper.verify_inventory(dataset.replay.root, {r["path"]: r["sha256"] for r in replay_records})
    parent_files = {str(p.relative_to(PARENT)): sha256(p)
                    for directory in (PARENT / "params", PARENT / "assets")
                    for p in sorted(directory.rglob("*")) if p.is_file()}
    if not parent_files or not any(p.startswith("params/") for p in parent_files):
        raise ValueError("SFT parent missing")
    norms = list((PARENT / "assets").rglob("norm_stats.json"))
    if len(norms) != 1 or sha256(norms[0]) != helper.NORM_SHA256:
        raise ValueError("Own SFT parent normalization differs from replay normalization")
    transformed = data_loader.transform_dataset(dataset, cfg.data.create(cfg.assets_dirs, cfg.model))
    for index in (0, 1):  # Both human and public pathways must satisfy the same official transforms.
        if transformed[index]["actions"].shape != (10, 32):
            raise ValueError("Official transformed action window mismatch")
    dataset.fetched.clear()
    matrix = ROOT / "evidence/recovery-sft28-evaluation-plan.json"
    if not matrix.is_file():
        raise ValueError("Freeze evaluation plan before training")
    plan = json.loads(matrix.read_text())
    if (plan["experiment"], plan["train"]["updates"], plan["train"]["lr"],
        plan["human_manifest_sha256"]) != (args.experiment, args.steps, args.learning_rate, sha256(human)):
        raise ValueError("Requested training differs from the frozen evaluation plan")
    compatibility = ROOT / "evidence/recovery-human28-task-compatibility.json"
    compatible = json.loads(compatibility.read_text())
    if not compatible["passed"] or any(sha256(Path(compatible[k]["path"])) != compatible[k]["expected"]
                                      for k in ("bddl", "init")):
        raise ValueError("Original human task identity changed")
    source_names = ["scripts/train_recovery_replay.py", "scripts/recovery_replay_dataset.py",
                    "scripts/multitask_dataset.py", "scripts/training_config.py"]
    record = dict(status="DRY_RUN_READY" if args.dry_run else "RUNNING", actual_updates=0,
        experiment=args.experiment, requested_updates=args.steps, batch_size=1,
        seed=42, learning_rate=args.learning_rate, warmup_updates=min(20, args.steps),
        parent=str(PARENT), parent_files=parent_files, parent_optimizer_resumed=False,
        optimizer="fresh AdamW, clip=1; LoRA parameters only; no EMA",
        human_manifest_sha256=sha256(human), replay_manifest_sha256=sha256(replay),
        human_episodes=len(dataset.human.records), human_frames=len(dataset.human),
        replay_train_episodes=len(replay_records), replay_tasks=40,
        sampling="shuffle seed 42; 1 human slot per 3 public replay slots; public task balanced",
        effective_sampling_slots=len(dataset), evaluation_plan_sha256=sha256(matrix),
        task_compatibility_sha256=sha256(compatibility),
        checkpoint_dir=str(cfg.checkpoint_dir), openpi_revision=helper.REVISION,
        normalization_sha256=helper.NORM_SHA256,
        source_sha256={name: sha256(ROOT / name) for name in source_names},
        validation="full parent/source/public-train/human checksums; official transform both branches")
    status = ROOT / "evidence" / (args.experiment + "-status.json")
    status.write_text(json.dumps(record, indent=2) + "\n")
    (ROOT / "evidence" / (args.experiment + "-config.txt")).write_text(repr(cfg) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "parent_files"}), flush=True)
    if args.dry_run:
        return 0
    if free_gpu_mib() < 24576:
        raise RuntimeError("GPU availability changed during checksum verification")
    spec = importlib.util.spec_from_file_location("recovery_upstream_train", helper.SOURCE / "scripts/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = data_loader.create_torch_dataset

    def create(config, horizon, model):
        if config.repo_id != helper.ASSET or horizon != 10:
            raise ValueError("Unexpected training data request")
        return dataset

    data_loader.create_torch_dataset = create
    started = time.monotonic()
    try:
        trainer.main(cfg)
        record.update(status="COMPLETE", actual_updates=args.steps)
    except BaseException as error:
        record.update(status="FAILED", actual_updates=None, error=repr(error))
        raise
    finally:
        data_loader.create_torch_dataset = original
        record.update(train_wall_seconds=time.monotonic() - started,
                      fetched_samples_including_prefetch=dict(dataset.fetched))
        status.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
