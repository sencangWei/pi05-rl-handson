"""Frozen human-joint comparison; real policy actions, strict released/retreated goal."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from joint_action_contract import absolute_joint_controller, validate_action
from motion_recording import MotionCapture
from record_osc_pairs import CAMERAS, SNAPSHOTS, digest
from replay_mixed_correction import ROOT, sha, write_json
from simulation_snapshot import identity, restore
from verify_osc_conversion import source_data
from verify_simulation_snapshot import create_env, initialize, measure


def completion_gate(native_goal, finger_contact, command, width, clearance):
    return bool(native_goal and not finger_contact and command < 0 and width >= .06 and clearance >= .10)


def check_plan(path):
    plan = json.loads(path.read_text())
    for name, expected in plan["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return plan


def checked_action(chunk, current, limits, max_jump):
    chunk = np.asarray(chunk, dtype=float)
    if chunk.shape != (10, 8) or not np.isfinite(chunk).all():
        raise ValueError("Expected finite joint-policy action chunk (10,8)")
    action = chunk[0].copy()
    # A normalized continuous gripper is saturated at the actuator API boundary.
    action[-1] = np.clip(action[-1], -1., 1.)
    validate_action(action, limits)
    jump = float(np.max(np.abs(action[:7] - current)))
    if jump > max_jump:
        raise ValueError(f"Joint target jump {jump:.6f} rad exceeds {max_jump}")
    return action, jump


def same_norm_stats(source, checkpoint):
    """Official NormStats serialization adds only absent optional null quantiles."""
    def canonical(value):
        return {key: {k: v for k, v in stats.items()
                      if not (k in ("q01", "q99") and v is None)}
                for key, stats in value["norm_stats"].items()}
    return canonical(source) == canonical(checkpoint)


def serve(args, plan):
    from joint_dataset import ASSET
    from train_unified_joint import config
    from joint_residual import transform_chunk
    cfg = config(plan, args.manifest)
    checkpoint = cfg.checkpoint_dir / str(plan["train"]["updates"] - 1)
    norm = checkpoint / "assets" / ASSET / "norm_stats.json"
    if not same_norm_stats(json.loads((args.manifest.parent / "assets" / ASSET / "norm_stats.json").read_text()),
                           json.loads(norm.read_text())):
        raise ValueError("Checkpoint normalization differs from frozen train-only statistics")
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = create_trained_policy(cfg, checkpoint)

    class SeededPolicy:
        def infer(self, observation):
            observation = dict(observation)
            seed = int(observation.pop("_probe_noise_seed"))
            noise = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
            anchor = np.asarray(observation["observation/state"])[:7].copy()
            result = policy.infer(observation, noise=noise)
            if plan["representation"] == "residual":
                result["residual_actions"] = np.asarray(result["actions"]).copy()
                result["actions"] = transform_chunk(result["actions"], anchor, decode=True)
            return result

    metadata = dict(plan_sha256=sha(args.plan), representation=plan["representation"], checkpoint=str(checkpoint),
                    norm_sha256=sha(norm), action_dim=8, state_dim=9, controller="ABSOLUTE_JOINT_POSITION",
                    checkpoint_files_sha256={str(p.relative_to(checkpoint)): sha(p)
                        for p in sorted(checkpoint.rglob("*")) if p.is_file()})
    write_json(args.plan.parent / "policy-server.json", metadata)
    print("JOINT_POLICY_READY " + json.dumps(metadata), flush=True)
    WebsocketPolicyServer(SeededPolicy(), host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


def evaluate(args, plan):
    import imageio.v2 as imageio
    from unified_joint_dataset import UnifiedJointDataset
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    metadata = client.get_server_metadata()
    if (metadata["plan_sha256"] != sha(args.plan) or metadata["action_dim"] != 8
            or metadata["state_dim"] != 9 or metadata["controller"] != "ABSOLUTE_JOINT_POSITION"):
        raise ValueError("Wrong joint-policy server")
    records = json.loads(args.manifest.read_text())["records"]
    record, = [r for r in records if r["source"] == args.source]
    split = record["split"]
    dataset = UnifiedJointDataset(args.manifest, split)
    episode = next(i for i, e in enumerate(dataset.episodes) if e["source"] == args.source)
    expected = dataset.sample(episode, 0)
    if record["kind"] == "pairs":
        meta, initial, _, start = source_data(args.source)
        snapshot = SNAPSHOTS / args.source / f"snapshot-{start}.npz"
    else:
        snapshot = ROOT / record["snapshot"]
    env = create_env()
    rows, states, calls, inference_seconds = [], [], 0, []
    streak, exception, rejected = 0, None, None
    started = time.monotonic()
    try:
        if record["kind"] == "pairs":
            initialize(env, meta, initial)
        else:
            env.seed(42)
            env.reset()
        restore(env, snapshot)
        if record["kind"] == "pairs" and identity(env) != json.loads(
                (SNAPSHOTS / args.source / "identity.json").read_text()):
            raise ValueError("Snapshot simulator identity mismatch")
        capture = MotionCapture(env)
        with absolute_joint_controller(env) as (controller, limits):
            obs = env.regenerate_obs_from_state(env.get_sim_state())
            motion = capture.read()
            state = np.r_[motion["joint_pos"], motion["gripper_pos"]].astype(np.float32)
            if not np.array_equal(state, expected["state"]):
                raise ValueError("Initial policy joint state differs from recorded pairs")
            for field, camera in CAMERAS:
                if digest(obs[camera][::-1, ::-1]) != digest(expected[field]):
                    raise ValueError("Initial policy RGB differs from recorded pairs")
            if env.check_success():
                raise ValueError("Initial state already satisfies goal")
            states.append(env.get_sim_state().copy())
            write_json(out / "identity.json", dict(plan_sha256=sha(args.plan), server=metadata,
                       split=split, source=args.source, initial_rgb_and_joint_state_exact=True,
                       snapshot_sha256=sha(snapshot),
                       controller=controller.name, limits=limits.tolist()))
            with imageio.get_writer(out / "policy.mp4", fps=20) as video, (out / "trace.jsonl").open("x") as trace:
                video.append_data(measure(env, obs, False, 0)[1])
                try:
                    for index in range(plan["evaluation"]["max_controls"]):
                        motion = capture.read()
                        state = np.r_[motion["joint_pos"], motion["gripper_pos"]].astype(np.float32)
                        request = {"observation/state": state, "prompt": expected["prompt"], "_probe_noise_seed": 42 + index}
                        for field, camera in CAMERAS:
                            request["observation/" + field] = obs[camera][::-1, ::-1].copy()
                        tick = time.monotonic()
                        chunk = np.asarray(client.infer(request)["actions"])
                        inference_seconds.append(time.monotonic() - tick)
                        calls += 1
                        rejected = dict(control=index, predicted_chunk=chunk.tolist())
                        action, jump = checked_action(chunk, motion["joint_pos"], limits,
                                                     plan["evaluation"]["max_target_joint_jump_rad"])
                        hand_before = motion["eef_pos"].copy()
                        obs, _, _, _ = env.step(action.tolist())
                        if not np.array_equal(controller.goal_qpos, action[:7]):
                            raise ValueError("Executed target differs from policy command")
                        row, frame = measure(env, obs, bool(env.check_success()), index + 1)
                        after = capture.read()
                        width = float(np.abs(after["gripper_pos"]).sum())
                        clear = float(np.linalg.norm(after["eef_pos"] - np.asarray(row["positions"])[3:]))
                        complete = completion_gate(row["native_goal"], row["finger_contact"], action[-1], width, clear)
                        streak = streak + 1 if complete else 0
                        row.update(finger_width_m=width, eef_clearance_m=clear, complete_gate=complete)
                        row.update(step=index + 1, joint_state=state.tolist(), raw_action=chunk[0].tolist(),
                                   executed_action=action.tolist(), target_jump_rad=jump, release_streak=streak,
                                   eef_motion_m=float(np.linalg.norm(capture.read()["eef_pos"] - hand_before)),
                                   noise_seed=42 + index, inference_seconds=inference_seconds[-1])
                        rows.append(row)
                        trace.write(json.dumps(row) + "\n")
                        states.append(env.get_sim_state().copy())
                        video.append_data(frame)
                        rejected = None
                        if streak >= 20:
                            break
                        # Match the dataset's current-state observable refresh before the next prediction.
                        obs = env.regenerate_obs_from_state(env.get_sim_state())
                        if (index + 1) % 100 == 0:
                            print(json.dumps(dict(source=args.source, controls=index + 1, streak=streak)), flush=True)
                except Exception as error:
                    exception = repr(error)
        result = dict(source=args.source, split=split, policy_inference=True, expert_fallback=False,
                      controls=len(rows), inference_calls=calls, success=streak >= 20 and exception is None,
                      success_by_600=streak >= 20 and len(rows) <= 600 and exception is None,
                      released_goal_streak=streak, native_goal_controls=sum(r["native_goal"] for r in rows),
                      exception=exception, rejected_before_execution=rejected,
                      wall_seconds=time.monotonic() - started, inference_seconds=inference_seconds,
                      stop_reason="exception" if exception else "released_goal20" if streak >= 20 else "budget1200",
                      initial_rgb_and_joint_state_exact=True)
        np.savez_compressed(out / "states.npz", physical=states)
        write_json(out / "result.json", result)
        print("JOINT_POLICY_RESULT " + json.dumps(result), flush=True)
    finally:
        env.close()
        client._ws.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--serve", action="store_true")
    p.add_argument("--source")
    p.add_argument("--output", type=Path)
    p.add_argument("--port", type=int, default=9877)
    a = p.parse_args()
    a.manifest, a.plan = a.manifest.resolve(), a.plan.resolve()
    if not a.serve and (a.source is None or a.output is None):
        p.error("Evaluation requires --source and --output")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    plan = check_plan(a.plan)
    if str(a.manifest.relative_to(ROOT)) not in plan["inputs_sha256"]:
        raise ValueError("Manifest is outside frozen plan")
    if not a.serve:
        if a.source not in plan["evaluation"]["sources"]:
            raise ValueError("Source outside frozen evaluation matrix")
        record, = [r for r in json.loads(a.manifest.read_text())["records"] if r["source"] == a.source]
        backend = "glx" if record["kind"] == "human" else "osmesa"
        os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = backend
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    return serve(a, plan) if a.serve else evaluate(a, plan)


if __name__ == "__main__":
    main()
