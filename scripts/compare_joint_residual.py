"""Bounded residual-target comparison against the frozen absolute-target pilot."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

from joint_dataset import ASSET, JointDataset
from joint_residual import ResidualDataset, residual_config, transform_chunk
from replay_mixed_correction import ROOT, SOURCES, sha, write_json


def prepare(args):
    """Preserve raw pairs; derive train-only statistics and freeze exact comparison."""
    out = args.output.resolve()
    replay = ROOT / "reports/joint-choice-20261029/expert-replay.json"
    if not json.loads(replay.read_text())["passed"]:
        raise ValueError("Expert residual physics replay must pass before preparation")
    out.mkdir(parents=True, exist_ok=False)
    original = ROOT / "data/joint-pilot29-v3/manifest.json"
    manifest = out / "manifest.json"
    manifest.write_bytes(original.read_bytes())
    train = ResidualDataset(manifest)
    actions = np.concatenate([train[i]["actions"] for i in range(len(train))])
    stats = json.loads((original.parent / "assets" / ASSET / "norm_stats.json").read_text())
    stats["norm_stats"]["actions"] = dict(mean=actions.astype(np.float64).mean(axis=0).tolist(),
        std=np.maximum(actions.astype(np.float64).std(axis=0), .001).tolist())
    norm = out / "assets" / ASSET / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    write_json(norm, stats)
    roundtrips = []
    for split in ("train", "holdout"):
        absolute, residual = JointDataset(manifest, split), ResidualDataset(manifest, split)
        errors = []
        for i in range(len(absolute)):
            reference, encoded = absolute[i], residual[i]
            decoded = transform_chunk(encoded["actions"], encoded["state"][:7], decode=True)
            errors.append(float(np.max(np.abs(decoded - reference["actions"]))))
        if max(errors) > 1e-6:
            raise ValueError("Float32 training-label roundtrip failed")
        roundtrips.append(dict(split=split, windows=len(absolute), max_error_rad=max(errors)))
    write_json(out / "roundtrip.json", roundtrips)
    baseline = ROOT / "reports/joint-pilot-20261029/v1/evaluation-v2/plan.json"
    baseline_plan = json.loads(baseline.read_text())
    paths = [manifest, norm, original, baseline, replay]
    paths += [ROOT / "scripts" / n for n in ("compare_joint_residual.py", "joint_residual.py",
              "joint_dataset.py", "joint_action_contract.py", "training_config.py", "evaluate_joint_pilot.py",
              "verify_joint_residual_replay.py")]
    paths += [ROOT / r["path"] for r in train.manifest["records"]]
    plan = dict(schema="pi05-joint-residual-comparison-v1", experiment=args.experiment,
        train=baseline_plan["train"], evaluation=baseline_plan["evaluation"],
        representation="future joint targets minus ONE current observed joint anchor; absolute gripper",
        baseline_plan=str(baseline.relative_to(ROOT)),
        normalization="state identical; train-only residual full-window population zscore, floor0.001",
        raw_labels="executed reconstructed candidate targets, not original historical human commands",
        velocity="not tested: no matched velocity command/controller contract",
        inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths})
    # Carry snapshot/identity and historical source checksums into the new plan.
    plan["inputs_sha256"].update({n: v for n, v in baseline_plan["inputs_sha256"].items()
                                 if n not in plan["inputs_sha256"]})
    write_json(out / "plan.json", plan)
    print(json.dumps(dict(manifest=str(manifest), plan=str(out / "plan.json"), roundtrips=roundtrips)), flush=True)


def train(args, plan):
    from train_multitask import free_gpu_mib
    from openpi.training import data_loader
    import torch
    import training_config
    data = ResidualDataset(args.manifest)
    cfg = residual_config(plan["experiment"], args.manifest)
    if cfg.checkpoint_dir.exists():
        raise FileExistsError(cfg.checkpoint_dir)
    transformed = data_loader.transform_dataset(data, cfg.data.create(cfg.assets_dirs, cfg.model))
    for i in (0, len(data) - 1):
        item = transformed[i]
        if item["actions"].shape != (10, 32) or not np.isfinite(item["actions"]).all():
            raise ValueError("Residual official transform failed")
    data.fetched_indices.clear()
    loader = torch.utils.data.DataLoader(range(len(data)), batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, generator=torch.Generator().manual_seed(42))
    expected = [int(i[0]) for i in loader][:201]
    baseline = json.loads((ROOT / "reports/joint-pilot-20261029/v1/consumed-samples.json").read_text())
    if expected != baseline["gradient_frames"] + [baseline["unused_prefetch"]]:
        raise ValueError("Residual sampling does not match absolute baseline")
    out = args.plan.parent
    (out / "config.txt").write_text(repr(cfg) + "\n")
    write_json(out / "sampling-preview.json", dict(gradient_frames=expected[:200], unused_prefetch=expected[-1]))
    if free_gpu_mib() < 24576:
        raise RuntimeError("Need 24GiB free; do not terminate other jobs")
    spec = importlib.util.spec_from_file_location("residual_upstream_train", training_config.SOURCE / "scripts/train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = data_loader.create_torch_dataset

    def create(config, horizon, model):
        if config.repo_id != ASSET or horizon != 10:
            raise ValueError("Unexpected residual dataset request")
        return data

    data_loader.create_torch_dataset = create
    status = dict(status="RUNNING", actual_updates=0, checkpoint_dir=str(cfg.checkpoint_dir),
                  parent="official pi05_libero", holdout_gradient_samples=0, ui_changed=False,
                  plan_sha256=sha(args.plan))
    write_json(out / "train-status.json", status)
    tick = time.monotonic()
    try:
        trainer.main(cfg)
        if data.fetched_indices != expected:
            raise ValueError("Actual residual sampler differs from frozen baseline")
        write_json(out / "consumed-samples.json", dict(gradient_frames=data.fetched_indices[:200],
                   unused_prefetch=data.fetched_indices[-1], holdout_samples=0))
        status.update(status="COMPLETE", actual_updates=200, sampling_order_exact=True)
    except BaseException as error:
        status.update(status="FAILED", actual_updates=None, error=repr(error))
        raise
    finally:
        data_loader.create_torch_dataset = original
        status["wall_seconds"] = time.monotonic() - tick
        write_json(out / "train-status.json", status)


def serve(args, plan):
    from evaluate_joint_pilot import same_norm_stats
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    cfg = residual_config(plan["experiment"], args.manifest)
    checkpoint = cfg.checkpoint_dir / "199"
    norm = checkpoint / "assets" / ASSET / "norm_stats.json"
    if not same_norm_stats(json.loads((args.manifest.parent / "assets" / ASSET / "norm_stats.json").read_text()),
                           json.loads(norm.read_text())):
        raise ValueError("Residual checkpoint normalization mismatch")
    policy = create_trained_policy(cfg, checkpoint)
    metadata = dict(plan_sha256=sha(args.plan), checkpoint=str(checkpoint), norm_sha256=sha(norm),
                    controller="ABSOLUTE_JOINT_POSITION", action_dim=8, state_dim=9,
                    learned_representation=plan["representation"],
                    checkpoint_files_sha256={str(p.relative_to(checkpoint)): sha(p)
                        for p in sorted(checkpoint.rglob("*")) if p.is_file()})

    class SeededResidualPolicy:
        def infer(self, observation):
            observation = dict(observation)
            seed = int(observation.pop("_probe_noise_seed"))
            anchor = np.asarray(observation["observation/state"])[:7].copy()
            noise = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
            result = policy.infer(observation, noise=noise)
            result["residual_actions"] = np.asarray(result["actions"]).copy()
            result["actions"] = transform_chunk(result["actions"], anchor, decode=True)
            return result

    write_json(args.plan.parent / "policy-server.json", metadata)
    print("JOINT_RESIDUAL_READY " + json.dumps(metadata), flush=True)
    WebsocketPolicyServer(SeededResidualPolicy(), host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "train", "serve", "evaluate"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--experiment", default="joint29_residual200_v1")
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--port", type=int, default=9878)
    args = p.parse_args()
    for key, value in dict(XLA_PYTHON_CLIENT_PREALLOCATE="false", HF_HUB_OFFLINE="1", WANDB_MODE="disabled").items():
        os.environ.setdefault(key, value)
    if args.mode == "prepare":
        if args.output is None:
            p.error("prepare requires --output")
        return prepare(args)
    if args.manifest is None or args.plan is None:
        p.error("requires --manifest and --plan")
    args.manifest, args.plan = args.manifest.resolve(), args.plan.resolve()
    from evaluate_joint_pilot import check_plan, evaluate
    plan = check_plan(args.plan)
    if plan["schema"] != "pi05-joint-residual-comparison-v1":
        raise ValueError("Wrong residual plan")
    if str(args.manifest.relative_to(ROOT)) not in plan["inputs_sha256"]:
        raise ValueError("Manifest is outside frozen residual plan")
    if args.mode == "train":
        # Config imports pin the source before importing upstream trainer modules.
        residual_config(plan["experiment"], args.manifest)
        return train(args, plan)
    if args.mode == "serve":
        residual_config(plan["experiment"], args.manifest)
        return serve(args, plan)
    if args.source is None or args.output is None:
        p.error("evaluate requires --source and --output")
    return evaluate(args, plan)


if __name__ == "__main__":
    main()
