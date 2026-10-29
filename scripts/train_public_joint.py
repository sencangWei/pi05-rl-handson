"""Independent public-normal + human-recovery state-conditioned pi05 LoRA run."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import train_unified_joint as baseline
from joint_dataset import ASSET, make_config
from public_joint_dataset import PublicJointDataset
from replay_mixed_correction import ROOT, sha, write_json
from train_unified_joint_state import state_input_canary


def config(plan, manifest):
    cfg = make_config(plan["experiment"], manifest, plan["train"]["updates"], plan["train"]["lr"])
    from openpi.training.optimizer import CosineDecaySchedule
    return replace(cfg, name="pi05_public_joint30_residual_lora",
        model=replace(cfg.model, discrete_state_input=True),
        checkpoint_base_dir=plan["train"]["checkpoint_base_dir"],
        save_interval=plan["train"]["updates"],
        lr_schedule=CosineDecaySchedule(warmup_steps=100, peak_lr=plan["train"]["lr"],
            decay_steps=plan["train"]["updates"], decay_lr=plan["train"]["lr"] * .1))


def make_plan(manifest, experiment, updates):
    train = PublicJointDataset(manifest)
    holdout = PublicJointDataset(manifest, "holdout")
    paths = [manifest, manifest.parent / "assets" / ASSET / "norm_stats.json",
             ROOT / "evidence/parent-manifest.json", ROOT / "evidence/openpi-source.json"]
    paths += [ROOT / "scripts" / name for name in ("train_public_joint.py", "public_joint_dataset.py",
        "public_joint_replay.py", "fetch_public_joint_source.py", "train_unified_joint.py",
        "train_unified_joint_state.py", "unified_joint_dataset.py", "joint_dataset.py", "joint_residual.py",
        "joint_action_contract.py", "motion_recording.py", "training_config.py", "evaluate_unified_joint.py",
        "evaluate_public_joint.py", "simulation_snapshot.py", "verify_simulation_snapshot.py",
        "verify_osc_conversion.py", "record_osc_pairs.py", "baseline.py")]
    for record in train.manifest["records"]:
        paths += [ROOT / record[key] for key in ("path", "snapshot", "replay")]
        if record["kind"] == "human":
            paths.append(ROOT / record["raw_metadata"])
        else:
            paths += [Path(record["snapshot"]).parent / name for name in
                      ("model.xml", "identity.json", "initial_state.npy", "protocol.json")]
    return dict(schema="pi05-public-joint-plan-v1", experiment=experiment, representation="residual",
        train=dict(updates=updates, batch=1, seed=42, lr=5e-5, warmup=100, horizon=10,
            state_dim=9, action_dim=8, state_input="all9 measured joints/fingers as normalized pi05 tokens",
            sampler_slots=len(train), holdout_windows=len(holdout), public_normal_mass=.8, human_recovery_mass=.2,
            optimizer="fresh AdamW clip1, LoRA only, no EMA", control_hz=20,
            train_sources=[e["source"] for e in train.episodes],
            holdout_sources=[e["source"] for e in holdout.episodes],
            checkpoint_base_dir="/dev/shm/pi05-public-joint30/checkpoints",
            checkpoint_persistence="volatile full training checkpoint; export exact compact LoRA to project exports"),
        evaluation=dict(sources=[r["source"] for r in train.manifest["records"] if r["kind"] == "human"
            or r["split"] == "holdout"], max_controls=600, milestone=600, replan_steps=1,
            max_target_joint_jump_rad=.35, required_streak=20, expert_fallback=False, seed=42,
            success="native goal + negative gripper command + width>=.06 + no finger contact for20",
            eef_retreat_0_10m="optional diagnostic, not required by public demonstrations or task success",
            recovery_initials_used_in_training=True, public_holdout_episodes_never_in_gradient_or_stats=True),
        inputs_sha256={str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha(path) for path in paths})


def install():
    baseline.config = config
    baseline.UnifiedJointDataset = PublicJointDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--freeze-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 200 <= args.updates < 40000:
        parser.error("Expected200..39999 bounded updates with complete frozen sampler epoch")
    args.manifest, args.plan = args.manifest.resolve(), args.plan.resolve()
    for key, value in dict(XLA_PYTHON_CLIENT_PREALLOCATE="false", HF_HUB_OFFLINE="1", WANDB_MODE="disabled").items():
        os.environ.setdefault(key, value)
    plan = make_plan(args.manifest, args.experiment, args.updates)
    if args.freeze_plan:
        if args.plan.exists():
            raise FileExistsError(args.plan)
        write_json(args.plan, plan)
        print(json.dumps(dict(status="FROZEN", plan=str(args.plan))), flush=True)
        return
    if json.loads(args.plan.read_text()) != plan:
        raise ValueError("Frozen public training inputs changed")
    cfg = config(plan, args.manifest)
    write_json(args.plan.parent / "state-input-canary.json", state_input_canary(cfg, args.manifest))
    install()
    baseline.train(args, plan)


if __name__ == "__main__":
    main()
