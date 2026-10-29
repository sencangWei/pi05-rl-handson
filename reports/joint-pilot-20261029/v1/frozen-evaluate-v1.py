"""Serve/evaluate the isolated joint pilot; actual policy only, never expert fallback."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from joint_action_contract import absolute_joint_controller, validate_action
from motion_recording import MotionCapture
from record_joint_pairs import check_render_backend
from record_osc_pairs import CAMERAS, PROMPT, SNAPSHOTS, digest
from replay_mixed_correction import ROOT, SOURCES, sha, write_json
from simulation_snapshot import identity, restore
from verify_osc_conversion import source_data
from verify_simulation_snapshot import create_env, initialize, measure


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


def serve(args, plan):
    from joint_dataset import ASSET, make_config
    cfg = make_config(plan["experiment"], args.manifest)
    checkpoint = cfg.checkpoint_dir / "199"
    norm = checkpoint / "assets" / ASSET / "norm_stats.json"
    if sha(norm) != sha(args.manifest.parent / "assets" / ASSET / "norm_stats.json"):
        raise ValueError("Checkpoint normalization differs from frozen train-only statistics")
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy = create_trained_policy(cfg, checkpoint)

    class SeededPolicy:
        def infer(self, observation):
            observation = dict(observation)
            seed = int(observation.pop("_probe_noise_seed"))
            noise = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
            return policy.infer(observation, noise=noise)

    metadata = dict(plan_sha256=sha(args.plan), checkpoint=str(checkpoint),
                    norm_sha256=sha(norm), action_dim=8, state_dim=9, controller="ABSOLUTE_JOINT_POSITION",
                    checkpoint_files_sha256={str(p.relative_to(checkpoint)): sha(p)
                        for p in sorted(checkpoint.rglob("*")) if p.is_file()})
    write_json(args.plan.parent / "policy-server.json", metadata)
    print("JOINT_POLICY_READY " + json.dumps(metadata), flush=True)
    WebsocketPolicyServer(SeededPolicy(), host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


def evaluate(args, plan):
    import imageio.v2 as imageio
    from joint_dataset import JointDataset
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    check_render_backend()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    metadata = client.get_server_metadata()
    if (metadata["plan_sha256"] != sha(args.plan) or metadata["action_dim"] != 8
            or metadata["state_dim"] != 9 or metadata["controller"] != "ABSOLUTE_JOINT_POSITION"):
        raise ValueError("Wrong joint-policy server")
    split = "train" if args.source == SOURCES[0] else "holdout"
    dataset = JointDataset(args.manifest, split)
    expected = dataset[0]
    meta, initial, _, start = source_data(args.source)
    env = create_env()
    rows, states, calls, inference_seconds = [], [], 0, []
    streak, exception, rejected = 0, None, None
    started = time.monotonic()
    try:
        initialize(env, meta, initial)
        restore(env, SNAPSHOTS / args.source / f"snapshot-{start}.npz")
        if identity(env) != json.loads((SNAPSHOTS / args.source / "identity.json").read_text()):
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
                       snapshot_sha256=sha(SNAPSHOTS / args.source / f"snapshot-{start}.npz"),
                       controller=controller.name, limits=limits.tolist()))
            with imageio.get_writer(out / "policy.mp4", fps=20) as video, (out / "trace.jsonl").open("x") as trace:
                video.append_data(measure(env, obs, False, 0)[1])
                try:
                    for index in range(plan["evaluation"]["max_controls"]):
                        motion = capture.read()
                        state = np.r_[motion["joint_pos"], motion["gripper_pos"]].astype(np.float32)
                        request = {"observation/state": state, "prompt": PROMPT, "_probe_noise_seed": 42 + index}
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
                        streak = streak + 1 if row["native_goal"] and not row["finger_contact"] else 0
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
                      success_by_300=streak >= 20 and len(rows) <= 300 and exception is None,
                      released_goal_streak=streak, native_goal_controls=sum(r["native_goal"] for r in rows),
                      exception=exception, rejected_before_execution=rejected,
                      wall_seconds=time.monotonic() - started, inference_seconds=inference_seconds,
                      stop_reason="exception" if exception else "released_goal20" if streak >= 20 else "budget600",
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
    p.add_argument("--source", choices=SOURCES)
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
    return serve(a, plan) if a.serve else evaluate(a, plan)


if __name__ == "__main__":
    main()
