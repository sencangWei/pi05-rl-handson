"""Paired learned-policy recovery evaluation; no scripted recovery controls."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party/RLinf")]


def payload(obs, prompt, image_tools):
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[key][::-1, ::-1]), 224, 224))
        for key in ("agentview_image", "robot0_eye_in_hand_image")]
    quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
    w = np.clip(quat[3], -1., 1.)
    den = np.sqrt(1. - w * w)
    rotation = np.zeros(3) if math.isclose(den, 0.) else quat[:3] * 2. * math.acos(w) / den
    state = np.r_[obs["robot0_eef_pos"], rotation, obs["robot0_gripper_qpos"]]
    return {"observation/image": images[0], "observation/wrist_image": images[1],
            "observation/state": state, "prompt": prompt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--port", type=int, help="Use original JAX policy service instead of Torch conversion")
    parser.add_argument("--expected-server-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/recovery-bowl-v3/manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "evidence/ppo-recovery-composed.json")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--split", choices=("train", "dev", "heldout"), default="heldout")
    parser.add_argument("--category", choices=("empty_grasp", "object_offset", "placement_failure"))
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--record-training-rollouts", action="store_true",
                        help="Save successful learned rollouts with pre-action RGB/state; train split only")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1000:
        raise ValueError("Invalid evaluation bound")
    if args.record_training_rollouts and args.split != "train":
        raise ValueError("Only training initial states may become demonstrations")
    if args.port and (args.checkpoint or not args.expected_server_checkpoint):
        raise ValueError("JAX service requires expected checkpoint identity and no Torch adapter")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    os.environ.setdefault("LIBERO_TYPE", "standard")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".local/numba"))
    from baseline import configure_libero
    configure_libero()
    import imageio.v2 as imageio
    import torch
    from evaluate_native_ppo import NativePolicy, sha256
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from omegaconf import OmegaConf
    from openpi_client import image_tools
    from rlinf.envs.sim.libero.recovery import RecoveryEnv
    from rlinf.hybrid_engines.fsdp.strategy.adapter_checkpoint import read_adapter_model
    from rlinf.models import get_model

    source_paths = [Path(__file__), ROOT / "scripts/evaluate_native_ppo.py", ROOT / "scripts/recovery_recording.py",
                    ROOT / "third_party/RLinf/rlinf/envs/sim/libero/recovery.py"]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    cfg = OmegaConf.load(args.config)
    model_cfg = OmegaConf.merge(cfg.actor.model, cfg.rollout.model)
    model_cfg.num_action_chunks = model_cfg.openpi.action_chunk = 5
    model_cfg.num_steps = model_cfg.openpi.num_steps = 10
    server_metadata = None
    if args.port:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
        server_metadata = client.get_server_metadata()
        if Path(server_metadata["checkpoint"]).resolve() != args.expected_server_checkpoint.resolve():
            raise ValueError("Wrong policy server checkpoint")
        model, base_hash = None, None
    else:
        torch.manual_seed(42)
        model = get_model(model_cfg)
        base_hash = sha256(Path(model_cfg.model_path) / "model.safetensors")
        if args.checkpoint:
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
            read_adapter_model(model, saved, base_hash, allow_dtype_cast=True)
            del saved
        model = model.to("cuda").eval()
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(8)
    raw = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                            task.problem_folder / task.bddl_file),
                            camera_heights=256, camera_widths=256)
    env = RecoveryEnv(raw, {"manifest": str(args.manifest), "split": args.split})
    if args.category:
        env.entries = [entry for entry in env.entries if entry["category"] == args.category]
        if not env.entries:
            raise ValueError("No recovery entries match the selected category")
    control_hz = raw.env.control_freq
    identity = {"base_sha256": base_hash,
                "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "manifest_sha256": sha256(args.manifest), "config_sha256": sha256(args.config),
                "split": args.split, "category": args.category,
                "model_config": None if args.port else OmegaConf.to_container(model_cfg, resolve=True),
                "jax_server": server_metadata,
                "success": "native goal AND no finger contact for 5 consecutive controls",
                "action_execution": f"learned model only; observe every 5 controls; max {args.max_steps} controls",
                "control_hz": control_hz, "video_fps": control_hz,
                "noise": f"numpy.default_rng({args.noise_seed}+call), reset per episode; 10 flow steps",
                "record_training_rollouts": args.record_training_rollouts,
                "sources": source_hashes}
    if source_hashes != {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}:
        raise RuntimeError("Evaluation sources changed during model loading")
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    reports = []
    try:
        for index, entry in enumerate(env.entries):
            env.seed(42)
            env.reset()
            obs = env.restore_recovery(index)
            if args.port:
                from baseline import SeededPolicy
                policy = SeededPolicy(client, args.noise_seed)
            else:
                policy = NativePolicy(model, {"noise_seed": args.noise_seed})
            recorder = None
            if args.record_training_rollouts:
                from recovery_recording import TrainingRecorder
                recorder = TrainingRecorder(args.output, entry, index, task.language)
            report = {key: entry[key] for key in ("id", "category", "state_sha256", "source_initial_id")}
            started = time.monotonic()
            success = False
            native_first = None
            exception = None
            step = 0
            video_name = entry["id"] + ".mp4"
            with imageio.get_writer(args.output / video_name, fps=control_hz, codec="libx264") as video, \
                    (args.output / (entry["id"] + ".jsonl")).open("x") as trace:
                def frame(current_obs):
                    video.append_data(np.concatenate([current_obs[key][::-1, ::-1] for key in
                                      ("agentview_image", "robot0_eye_in_hand_image")], axis=1))
                frame(obs)
                try:
                    while step < args.max_steps and not success:
                        with torch.inference_mode():
                            actions = policy.infer(payload(obs, task.language, image_tools))["actions"]
                        for action in actions[:min(5, args.max_steps - step)]:
                            if recorder is not None:
                                recorder.add(obs, action, raw.get_sim_state(),
                                             payload(obs, task.language, image_tools)["observation/state"])
                            before = obs["robot0_eef_pos"].copy()
                            obs, _, success, info = env.step(action)
                            step += 1
                            if info["native_success"] and native_first is None:
                                native_first = step
                            sim = raw.env
                            bowl = sim.sim.data.body_xpos[sim.obj_body_id["akita_black_bowl_1"]]
                            plate = sim.sim.data.body_xpos[sim.obj_body_id["plate_1"]]
                            record = {"step": step, **info, "action": action.tolist(),
                                      "eef_pos": obs["robot0_eef_pos"].tolist(),
                                      "eef_motion_m": float(np.linalg.norm(obs["robot0_eef_pos"] - before)),
                                      "bowl_pos": bowl.tolist(), "plate_pos": plate.tolist(),
                                      "bowl_plate_xy_m": float(np.linalg.norm((bowl - plate)[:2]))}
                            trace.write(json.dumps(record) + "\n")
                            frame(obs)
                            if success:
                                break
                except Exception as error:  # noqa: BLE001 - record failed trials without hiding them
                    exception = repr(error)
            report.update(success=bool(success), result="exception" if exception else
                          ("success" if success else "timeout"), exception=exception,
                          steps=step, inference_calls=policy.index, native_first_step=native_first,
                          video=video_name, wall_seconds=time.monotonic() - started,
                          reward_return=float(success) - .001 * step)
            reports.append(report)
            if recorder is not None and success and exception is None:
                recorder.save(raw.get_sim_state())
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
            print("RECOVERY_RESULT " + json.dumps(report), flush=True)
    finally:
        env.close()
        if args.port:
            client._ws.close()
    return int(any(report["exception"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
