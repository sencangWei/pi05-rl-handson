"""Frozen absolute/residual comparison using approved unified human episodes."""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from joint_dataset import ASSET, make_config
from replay_mixed_correction import ROOT, sha, write_json
from unified_joint_dataset import UnifiedJointDataset


def config(plan, manifest):
    cfg = make_config(plan["experiment"], manifest, plan["train"]["updates"], plan["train"]["lr"])
    return replace(cfg, name="pi05_unified_joint_" + plan["representation"] + "_lora")


def make_plan(manifest, representation, experiment):
    dataset = UnifiedJointDataset(manifest, representation=representation)
    holdout = UnifiedJointDataset(manifest, "holdout")
    paths = [manifest, manifest.parent / "assets" / ASSET / "norm_stats.json",
             ROOT / "evidence/parent-manifest.json", ROOT / "evidence/openpi-source.json"]
    paths += [ROOT / "scripts" / n for n in ("unified_joint_dataset.py", "train_unified_joint.py",
              "evaluate_unified_joint.py", "joint_dataset.py", "joint_residual.py", "training_config.py",
              "joint_action_contract.py", "motion_recording.py", "evaluate_joint_pilot.py")]
    for record in dataset.manifest["records"]:
        paths.append(ROOT / record["path"])
        if record["kind"] == "human":
            paths += [ROOT / record[key] for key in ("raw_metadata", "snapshot", "replay")]
        else:
            from record_osc_pairs import SNAPSHOTS
            from verify_osc_conversion import source_data
            _, _, _, start = source_data(record["source"])
            paths += [SNAPSHOTS / record["source"] / f"snapshot-{start}.npz",
                      SNAPSHOTS / record["source"] / "identity.json"]
    return dict(schema="pi05-unified-joint-plan-v1", representation=representation, experiment=experiment,
        train=dict(updates=1000, batch=1, seed=42, lr=2e-5, warmup=20, horizon=10,
            state_dim=9, action_dim=8, sampler_slots=len(dataset), holdout_windows=len(holdout),
            train_sources=[e["source"] for e in dataset.episodes], holdout_sources=[e["source"] for e in holdout.episodes],
            optimizer="fresh AdamW clip1; LoRA only; no EMA", control_hz=20),
        evaluation=dict(sources=[r["source"] for r in dataset.manifest["records"] if r["kind"] == "human"
                               or r["split"] == "holdout"], max_controls=1200, milestone=600, replan_steps=1,
            max_target_joint_jump_rad=.35, required_streak=20,
            success="native goal + negative gripper command + width>=.06 + no contact + eef clear>=.10 for20",
            expert_fallback=False, seed=42, holdout_previously_used_for_development=True),
        inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})


def train(args, plan):
    dataset = UnifiedJointDataset(args.manifest, representation=plan["representation"])
    cfg = config(plan, args.manifest)
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(cfg.checkpoint_dir)
    from openpi.training import data_loader
    transformed = data_loader.transform_dataset(dataset, cfg.data.create(cfg.assets_dirs, cfg.model))
    boundaries = [i for i in range(len(dataset)) if i == 0 or dataset.slots[i][0] != dataset.slots[i - 1][0]]
    for index in boundaries + [len(dataset) - 1]:
        item = transformed[index]
        if (item["actions"].shape != (10, 32) or item["state"].shape != (32,)
                or not np.isfinite(item["actions"]).all() or not np.isfinite(item["state"]).all()):
            raise ValueError("Official transform does not preserve joint contract")
    dataset.fetched_indices.clear()
    import torch
    steps = plan["train"]["updates"]
    loader = torch.utils.data.DataLoader(range(len(dataset)), batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, generator=torch.Generator().manual_seed(42))
    preview = [int(i[0]) for i in loader][:steps + 1]
    if len(preview) != steps + 1:
        raise ValueError("Insufficient full-window sampling slots")
    from collections import Counter
    source_counts = Counter(dataset.episodes[dataset.slots[i][0]]["source"] for i in preview[:steps])
    out = args.plan.parent
    write_json(out / "sampling-preview.json", dict(gradient_slots=preview[:steps], unused_prefetch=preview[-1],
               gradient_source_counts=dict(source_counts), holdout_samples=0))
    (out / "config.txt").write_text(repr(cfg) + "\n")
    status = dict(status="DRY_RUN_READY" if args.dry_run else "RUNNING", actual_updates=0,
        plan_sha256=sha(args.plan), checkpoint_dir=str(cfg.checkpoint_dir), representation=plan["representation"],
        parent="official pi05_libero fresh optimizer", holdout_gradient_samples=0, ui_changed=False)
    write_json(out / "train-status.json", status)
    print(json.dumps(status), flush=True)
    if args.dry_run:
        return
    from train_multitask import free_gpu_mib
    if free_gpu_mib() < 24576:
        raise RuntimeError("Need24GiB free; will not stop other jobs")
    import training_config
    spec = importlib.util.spec_from_file_location("unified_upstream_train", training_config.SOURCE / "scripts/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = data_loader.create_torch_dataset

    def create(data_config, horizon, model):
        if data_config.repo_id != ASSET or horizon != 10:
            raise ValueError("Unexpected dataset request")
        return dataset

    data_loader.create_torch_dataset = create
    started = time.monotonic()
    try:
        trainer.main(cfg)
        if dataset.fetched_indices != preview:
            raise ValueError("Actual sampler differs from frozen preview")
        status.update(status="COMPLETE", actual_updates=steps, sampling_order_exact=True)
        write_json(out / "consumed-samples.json", dict(gradient_slots=dataset.fetched_indices[:steps],
            unused_prefetch=dataset.fetched_indices[-1], gradient_source_counts=dict(source_counts), holdout_samples=0))
    except BaseException as error:
        status.update(status="FAILED", actual_updates=None, error=repr(error))
        raise
    finally:
        data_loader.create_torch_dataset = original
        status["wall_seconds"] = time.monotonic() - started
        write_json(out / "train-status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--representation", choices=("absolute", "residual"), required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--freeze-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.manifest, args.plan = args.manifest.resolve(), args.plan.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.experiment):
        parser.error("Invalid experiment")
    for key, value in dict(XLA_PYTHON_CLIENT_PREALLOCATE="false", HF_HUB_OFFLINE="1", WANDB_MODE="disabled").items():
        os.environ.setdefault(key, value)
    plan = make_plan(args.manifest, args.representation, args.experiment)
    if args.freeze_plan:
        if args.plan.exists():
            raise FileExistsError(args.plan)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.plan, plan)
        print(json.dumps(dict(status="FROZEN", plan=str(args.plan), representation=args.representation)), flush=True)
        return
    if json.loads(args.plan.read_text()) != plan:
        raise ValueError("Frozen input/config changed")
    train(args, plan)


if __name__ == "__main__":
    main()
