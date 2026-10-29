"""Compare learned policies from approved expert TRAIN snapshots; no expert action replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from record_osc_pairs import CAMERAS, CONVERSION, PROMPT, SNAPSHOTS, digest, observation_state
from replay_mixed_correction import ROOT, SOURCES, sha, write_json

PAIRS = ROOT / "reports/osc-pairs-20261029/v1"
CHECKPOINTS = {
    "parent": "checkpoints/pi05_multitask40_lora/multitask40_lora_v1/9999",
    "candidate": "checkpoints/pi05_recovery_replay_lora/osc_expert_replay2000_v1/1999",
}


def verify_inputs(protocol, weights=False):
    for name, expected in protocol["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    if weights:
        for files in protocol["weight_files"].values():
            for name, expected in files.items():
                if sha(ROOT / name) != expected:
                    raise ValueError(f"Frozen weight changed: {name}")


def freeze(output):
    from verify_osc_conversion import source_data
    output.mkdir(parents=True, exist_ok=False)
    review = json.loads((PAIRS / "human-review.json").read_text())
    if review["status"] != "approved" or {x["source"] for x in review["sources"]} != set(SOURCES):
        raise ValueError("Missing approval for expert sources")
    inputs = [Path(__file__), ROOT / "evidence/osc-sft29-execution-plan.json", PAIRS / "human-review.json"]
    inputs += [ROOT / "scripts" / name for name in (
        "simulation_snapshot.py", "record_osc_pairs.py", "verify_osc_conversion.py",
        "verify_simulation_snapshot.py", "replay_mixed_correction.py", "evaluate_recovery.py",
        "baseline.py", "serve_sft.py", "training_config.py")]
    sources = {}
    for source in SOURCES:
        _, _, records, start = source_data(source)
        snapshot = SNAPSHOTS / source / f"snapshot-{start}.npz"
        inputs += [snapshot, PAIRS / source / "physics.npz", PAIRS / source / "pair-audit.json",
                   CONVERSION / source / "tracking-identity.json"]
        rows = json.loads((PAIRS / source / "pair-audit.json").read_text())
        streak, first_success = 0, None
        for step, row in enumerate(rows, 1):
            post = row["post"]
            streak = streak + 1 if post["native_goal"] and not post["finger_contact"] else 0
            if streak >= 5 and first_success is None:
                first_success = step
        sources[source] = dict(snapshot=str(snapshot.relative_to(ROOT)), start=start,
                               expert_controls=len(records) - start,
                               expert_release_success_step=first_success)
    weights = {}
    for label, checkpoint in CHECKPOINTS.items():
        paths = sorted((ROOT / checkpoint / "params").rglob("*"))
        paths += [ROOT / checkpoint / "assets/physical-intelligence/libero/norm_stats.json"]
        weights[label] = {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}
        if len(weights[label]) < 5:
            raise ValueError("Incomplete model parameters")
    protocol = dict(status="FROZEN_BEFORE_POLICY_ROLLOUT", split="expert_train_diagnostic",
                    heldout=False, seed=42, checkpoints=CHECKPOINTS, sources=sources,
                    control_hz=20, action_horizon=10, replan_controls=5, flow_steps=10,
                    max_controls=600, primary_milestone_controls=300,
                    success="native goal AND no finger contact for 5 consecutive controls",
                    observations="Exact pair-dataset initial refresh; then ordinary env.step observations",
                    actions="Learned 7D OSC only; no scripted recovery, no reference state injection",
                    noise="numpy.default_rng(42+call).standard_normal((10,32)); reset per trial",
                    model_promotion=False, training=False, weight_files=weights,
                    code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs})
    write_json(output / "protocol.json", protocol)


def check_initial(env, obs, source):
    with np.load(PAIRS / source / "physics.npz", allow_pickle=False) as data:
        if not np.array_equal(env.get_sim_state(), data["before"][0]):
            raise ValueError("Initial physics differs from expert pairs")
        if not np.array_equal(observation_state(obs), data["observation_state"][0]):
            raise ValueError("Initial 8D observation differs from expert pairs")
    audit = json.loads((PAIRS / source / "pair-audit.json").read_text())[0]
    for field, key in CAMERAS:
        if digest(obs[key][::-1, ::-1]) != audit[field]:
            raise ValueError(f"Initial camera differs from expert pairs: {field}")
    return dict(physics_exact=True, observation_state_exact=True, both_rgb_exact=True,
                state_sha256=audit["state_sha256"], image_sha256=audit["image"],
                wrist_image_sha256=audit["wrist_image"])


def worker(output, label, source, port):
    import imageio.v2 as imageio
    from baseline import SeededPolicy
    from evaluate_recovery import payload
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from simulation_snapshot import identity, restore
    from verify_osc_conversion import source_data
    from verify_simulation_snapshot import create_env, initialize, measure

    protocol = json.loads((output / "protocol.json").read_text())
    verify_inputs(protocol)
    out = output / label / source
    out.mkdir(parents=True, exist_ok=False)
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    metadata = client.get_server_metadata()
    checkpoint = ROOT / protocol["checkpoints"][label]
    norm = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    if (Path(metadata["checkpoint"]).resolve() != checkpoint.resolve() or
            metadata["norm_stats_sha256"] != sha(norm) or metadata["action_horizon"] != 10):
        raise ValueError("Wrong policy server identity")
    policy = SeededPolicy(client, protocol["seed"])
    env = create_env()
    started = time.monotonic()
    steps, streak, native_first, success_step = 0, 0, None, None
    exception, inference_seconds, states = None, [], []
    try:
        meta, initial, _, _ = source_data(source)
        initialize(env, meta, initial)
        restore(env, ROOT / protocol["sources"][source]["snapshot"])
        expected = json.loads((CONVERSION / source / "tracking-identity.json").read_text())
        if identity(env) != {k: expected[k] for k in identity(env)}:
            raise ValueError("Simulator identity mismatch")
        controller = env.env.robots[0].controller
        for key in ("input_min", "input_max", "output_min", "output_max"):
            if not np.array_equal(getattr(controller, key), expected[key]):
                raise ValueError(f"OSC scaling mismatch: {key}")
        anchor = controller.initial_joint.copy()
        if not np.array_equal(anchor, expected["nullspace_anchor"]):
            raise ValueError("Nullspace anchor mismatch")
        before = env.get_sim_state().copy()
        obs = env.regenerate_obs_from_state(before)
        checks = check_initial(env, obs, source)
        if env.check_success():
            raise ValueError("Expert initial state already satisfies goal")
        initial_measure, frame = measure(env, obs, False, 0)
        states.append(before)
        write_json(out / "identity.json", dict(protocol_sha256=sha(output / "protocol.json"),
                   checkpoint=metadata, initial_checks=checks, initial_measure=initial_measure,
                   simulator=identity(env), nullspace_anchor=anchor.tolist()))
        with imageio.get_writer(out / "policy.mp4", fps=20, codec="libx264") as video, \
                (out / "trace.jsonl").open("x") as trace:
            video.append_data(frame)
            try:
                while steps < protocol["max_controls"] and success_step is None:
                    inference_start = time.monotonic()
                    actions = np.asarray(policy.infer(payload(obs, PROMPT, image_tools))["actions"])
                    inference_seconds.append(time.monotonic() - inference_start)
                    if actions.shape != (10, 7) or not np.isfinite(actions).all():
                        raise ValueError(f"Invalid policy action chunk: {actions.shape}")
                    for action in actions[:min(5, protocol["max_controls"] - steps)]:
                        hand_before = obs["robot0_eef_pos"].copy()
                        obs, _, _, _ = env.step(action.tolist())
                        steps += 1
                        native = bool(env.check_success())
                        row, frame = measure(env, obs, native, steps)
                        streak = streak + 1 if native and not row["finger_contact"] else 0
                        if native and native_first is None:
                            native_first = steps
                        if streak >= 5:
                            success_step = steps
                        bowl = env.sim.data.body_xpos[env.env.obj_body_id["akita_black_bowl_1"]].copy()
                        plate = env.sim.data.body_xpos[env.env.obj_body_id["plate_1"]].copy()
                        row.update(step=steps, action=action.tolist(), inference_call=policy.index,
                                   noise_seed=42 + policy.index - 1, release_streak=streak,
                                   eef_motion_m=float(np.linalg.norm(obs["robot0_eef_pos"] - hand_before)),
                                   bowl_pos=bowl.tolist(), plate_pos=plate.tolist(),
                                   bowl_plate_xy_m=float(np.linalg.norm((bowl - plate)[:2])))
                        trace.write(json.dumps(row) + "\n")
                        video.append_data(frame)
                        states.append(env.get_sim_state().copy())
                        if success_step is not None:
                            break
                    if steps % 100 == 0:
                        print(json.dumps(dict(label=label, source=source, controls=steps)), flush=True)
                if not np.array_equal(controller.initial_joint, anchor):
                    raise ValueError("Policy run changed nullspace anchor")
            except Exception as error:  # noqa: BLE001 - preserve every failed trial
                exception = repr(error)
        np.savez_compressed(out / "states.npz", states=states)
        report = dict(label=label, source=source, split=protocol["split"], steps=steps,
                      success=success_step is not None and exception is None, success_step=success_step,
                      success_by_300=success_step is not None and success_step <= 300 and exception is None,
                      native_first_step=native_first, exception=exception, inference_calls=policy.index,
                      inference_seconds=inference_seconds, wall_seconds=time.monotonic() - started,
                      initial_checks=checks, initial_measure=initial_measure, video="policy.mp4")
        write_json(out / "result.json", report)
        print("EXPERT_POLICY_RESULT " + json.dumps(report), flush=True)
    finally:
        env.close()
        client._ws.close()
    return int(exception is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--label", choices=CHECKPOINTS)
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument("--port", type=int, default=9875)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.freeze:
        freeze(args.output)
        return 0
    if args.label is None or args.source is None:
        parser.error("A worker requires --label and --source")
    return worker(args.output, args.label, args.source, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
