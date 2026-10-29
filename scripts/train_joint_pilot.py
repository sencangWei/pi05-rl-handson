"""Bounded, isolated joint-interface LoRA pilot with a frozen evaluation plan."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import time

from joint_dataset import ASSET, JointDataset, make_config
from replay_mixed_correction import ROOT, SOURCES, sha, write_json
from train_multitask import free_gpu_mib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--experiment", default="joint29_pilot200_v1")
    p.add_argument("--freeze-plan", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.manifest, a.plan = a.manifest.resolve(), a.plan.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", a.experiment):
        p.error("Invalid experiment name")
    for key, value in dict(XLA_PYTHON_CLIENT_PREALLOCATE="false", HF_HUB_OFFLINE="1", WANDB_MODE="disabled").items():
        os.environ.setdefault(key, value)
    steps, lr = 200, 5e-5
    dataset = JointDataset(a.manifest)
    holdout = JointDataset(a.manifest, "holdout")
    files = [a.manifest, a.manifest.parent / "assets" / ASSET / "norm_stats.json"]
    files += [ROOT / "scripts" / n for n in ("train_joint_pilot.py", "joint_dataset.py", "training_config.py",
                "joint_action_contract.py", "motion_recording.py", "record_joint_pairs.py",
                "evaluate_joint_pilot.py")]
    from record_osc_pairs import SNAPSHOTS
    from verify_osc_conversion import source_data
    for source in SOURCES:
        _, _, _, start = source_data(source)
        files += [SNAPSHOTS / source / f"snapshot-{start}.npz", SNAPSHOTS / source / "identity.json"]
    plan = dict(schema="pi05-joint-pilot-plan-v1", experiment=a.experiment,
        train=dict(updates=steps, batch=1, seed=42, lr=lr, warmup=20, optimizer="fresh AdamW clip1, LoRA only, no EMA",
                   windows=len(dataset), horizon=10, action_dim=8, state_dim=9,
                   train_source=dataset.record["source"], holdout_source=holdout.record["source"]),
        evaluation=dict(sources=list(SOURCES), max_controls=600, milestone=300, replan_steps=1,
                        success="native goal and no finger contact for 20 consecutive controls",
                        max_target_joint_jump_rad=.35, seed=42, render_backend="osmesa",
                        gripper="saturate normalized output to [-1,1]; record both raw and executed",
                        controller="ABSOLUTE_JOINT_POSITION", expert_fallback=False,
                        noise="numpy.default_rng(42 + control_index).standard_normal((10,32)), float32",
                        claim="Train-episode vs whole-episode holdout interface pilot; no general recovery claim"),
        inputs_sha256={str(f.relative_to(ROOT)): sha(f) for f in files})
    if a.freeze_plan:
        if a.plan.exists():
            raise FileExistsError(a.plan)
        a.plan.parent.mkdir(parents=True, exist_ok=True)
        write_json(a.plan, plan)
        print(json.dumps(plan), flush=True)
        return
    if json.loads(a.plan.read_text()) != plan:
        raise ValueError("Inputs/config changed after plan freeze")
    cfg = make_config(a.experiment, a.manifest, steps, lr)
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(cfg.checkpoint_dir)
    from openpi.training import data_loader
    transformed = data_loader.transform_dataset(dataset, cfg.data.create(cfg.assets_dirs, cfg.model))
    for i in (0, len(dataset) - 1):
        item = transformed[i]
        if (item["actions"].shape != (10, 32) or item["state"].shape != (32,)
                or not np_finite(item["actions"]) or not np_finite(item["state"])):
            raise ValueError("Official transforms failed joint contract")
    dataset.fetched_indices.clear()
    import torch
    loader = torch.utils.data.DataLoader(range(len(dataset)), batch_size=1, shuffle=True,
        num_workers=0, drop_last=True, generator=torch.Generator().manual_seed(42))
    preview = [int(i[0]) for i in loader][:steps + 1]
    if len(preview) != steps + 1:
        raise ValueError("Pilot needs 201 distinct windows including upstream final prefetch")
    out = a.plan.parent
    write_json(out / "sampling-preview.json", dict(gradient_frames=preview[:steps], unused_prefetch=preview[-1]))
    (out / "config.txt").write_text(repr(cfg) + "\n")
    status = dict(status="DRY_RUN_READY" if a.dry_run else "RUNNING", actual_updates=0,
                  plan_sha256=sha(a.plan), checkpoint_dir=str(cfg.checkpoint_dir),
                  parent="official pi05_libero, fresh optimizer", holdout_gradient_samples=0,
                  normalization="train-only population zscore, std floor0.001",
                  new_execution_human_review=False, ui_changed=False)
    write_json(out / "train-status.json", status)
    print(json.dumps(status), flush=True)
    if a.dry_run:
        return
    if free_gpu_mib() < 24576:
        raise RuntimeError("Need 24GiB free; no other processes will be stopped")
    import training_config
    spec = importlib.util.spec_from_file_location("joint_upstream_train", training_config.SOURCE / "scripts/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = data_loader.create_torch_dataset

    def create(config, horizon, model):
        if config.repo_id != ASSET or horizon != 10:
            raise ValueError("Unexpected joint dataset request")
        return dataset

    data_loader.create_torch_dataset = create
    started = time.monotonic()
    try:
        trainer.main(cfg)
        if dataset.fetched_indices != preview:
            raise ValueError("Actual sampling differs from frozen seeded sampler")
        status.update(status="COMPLETE", actual_updates=steps, sampling_order_exact=True)
        write_json(out / "consumed-samples.json", dict(gradient_frames=dataset.fetched_indices[:steps],
                   unused_prefetch=dataset.fetched_indices[-1], holdout_samples=0))
    except BaseException as error:
        status.update(status="FAILED", actual_updates=None, error=repr(error))
        raise
    finally:
        data_loader.create_torch_dataset = original
        status["train_wall_seconds"] = time.monotonic() - started
        write_json(out / "train-status.json", status)


def np_finite(value):
    import numpy as np
    return np.isfinite(value).all()


if __name__ == "__main__":
    main()
