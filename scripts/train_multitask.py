"""Bounded, real OpenPI LoRA updates; data and accelerator preflight before launch."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def free_gpu_mib():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return int(text.splitlines()[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="multitask40_lora_v1")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.experiment):
        parser.error("Invalid experiment name")
    record = dict(status="PREFLIGHT", actual_updates=0, seed=42, **vars(args))
    record_path = ROOT / "evidence" / f"{args.experiment}-status.json"
    # GPU guard runs before importing JAX or constructing a model. Never stop other processes.
    if not args.dry_run:
        free = free_gpu_mib()
        if free < 24576:
            record.update(status="BLOCKED_GPU", free_gpu_mib=free, required_free_mib=24576)
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            return 75
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    import training_config as helper
    from multitask_dataset import MultitaskDataset
    parent = helper.PARENT
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    cfg = helper.make_config(args.experiment, args.steps, args.batch_size, args.learning_rate)
    if cfg.checkpoint_dir.exists():
        raise ValueError("Experiment already exists; no overwrite or implicit resume")
    manifest_path = ROOT / "evidence/dataset-manifest.json"
    dataset = MultitaskDataset(manifest_path)
    validation = json.loads((ROOT / "evidence/data-validation.json").read_text())
    if not validation["passed"] or validation["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("Dataset validation missing or stale")
    from openpi.training import data_loader
    sample = data_loader.transform_dataset(dataset, cfg.data.create(cfg.assets_dirs, cfg.model))[0]
    if sample["actions"].shape != (10, 32):
        raise ValueError("Official action transform shape mismatch")
    record.update(status="DRY_RUN_READY" if args.dry_run else "RUNNING", parent=str(parent),
        openpi_revision=helper.REVISION, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        checkpoint_dir=str(cfg.checkpoint_dir), transformed_action_shape=list(sample["actions"].shape),
        effective_sampling_slots=len(dataset), train_episodes=dataset.manifest["train_episodes"],
        holdout_episodes=dataset.manifest["holdout_episodes"], optimizer_resumed=False,
        parameter_filter="LoRA only; official pi05_libero parent", learning_rate_schedule="20-step warmup then constant")
    if json.loads((helper.SOURCE / "SNAPSHOT.json").read_text())["openpi"] != helper.REVISION:
        raise ValueError("OpenPI snapshot revision mismatch")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    (ROOT / "evidence" / f"{args.experiment}-config.txt").write_text(repr(cfg) + "\n")
    print(json.dumps(record), flush=True)
    if args.dry_run:
        return 0
    # Full parent hashing can take minutes; recheck availability just before model construction.
    free = free_gpu_mib()
    if free < 24576:
        record.update(status="BLOCKED_GPU", free_gpu_mib=free, required_free_mib=24576)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        return 75
    spec = importlib.util.spec_from_file_location("multitask_upstream_train", helper.SOURCE / "scripts/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = data_loader.create_torch_dataset
    def create(config, horizon, model):
        if config.repo_id != helper.ASSET or horizon != 10:
            raise ValueError("Unexpected dataset request")
        return dataset
    data_loader.create_torch_dataset = create
    try:
        trainer.main(cfg)
        record.update(status="COMPLETE", actual_updates=args.steps)
    except BaseException as error:
        record.update(status="FAILED", error=str(error), actual_updates=None)
        raise
    finally:
        data_loader.create_torch_dataset = original
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
