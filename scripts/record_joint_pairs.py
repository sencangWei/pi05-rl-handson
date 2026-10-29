"""Record fresh, physically verified joint-action pairs without relabeling old RGB."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from joint_action_contract import absolute_joint_controller, validate_action
from motion_recording import MotionCapture
from record_osc_pairs import CAMERAS, PROMPT, SNAPSHOTS, digest, validate_dataset
from replay_mixed_correction import ROOT, SOURCES, sha, write_json

EXPERT = ROOT / "reports/joint-contract-20261029/v2"
SCHEMA = "pi05-absolute-joint-pairs-v1"


def check_render_backend():
    # The frozen expert's RGB was rendered with Mesa, not NVIDIA EGL.
    if any(os.environ.get(k) != "osmesa" for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM")):
        raise ValueError("Frozen RGB requires MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa")


def pack_before(obs, state, action, limits, index, original_index, episode, sim_time, hold):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError("Expected joint radians7 + finger positions2")
    action = validate_action(action, limits).copy()
    row = dict(state=state.tolist(), actions=action.tolist(), frame_index=index,
               episode_index=episode, task_index=8, original_control_index=original_index,
               controller="ABSOLUTE_JOINT_POSITION", sim_time_s=float(sim_time),
               timestamp=index * .05, diagnostic_hold=bool(hold), prompt=PROMPT)
    images, hashes = [], {}
    for field, key in CAMERAS:
        raw = np.asarray(obs[key])
        if raw.shape != (256, 256, 3) or raw.dtype != np.uint8:
            raise ValueError("Expected actual RGB uint8 256x256 camera")
        rgb = np.ascontiguousarray(raw[::-1, ::-1])
        stream = BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        row[field] = dict(bytes=stream.getvalue(), path=None)
        hashes[field] = digest(rgb)
        images.append(rgb)
    return row, np.concatenate(images, axis=1), hashes


def worker(output, source):
    import imageio.v2 as imageio
    import pyarrow as pa
    import pyarrow.parquet as pq
    from simulation_snapshot import identity, restore
    from verify_osc_conversion import source_data
    from verify_simulation_snapshot import create_env, initialize, measure

    protocol = json.loads((output / "protocol.json").read_text())
    for path, expected in protocol["inputs_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    expert = EXPERT / source / "velocity_compensated"
    if not json.loads((expert / "result.json").read_text())["passed"]:
        raise ValueError("Expert did not pass the physical gates")
    expected_rows = json.loads((expert / "steps.json").read_text())
    with np.load(expert / "trajectory.npz", allow_pickle=False) as z:
        expected = {k: z[k] for k in z.files}
    meta, initial, records, start = source_data(source)
    out = output / source
    out.mkdir()
    rows, audits, before_states, after_states, states = [], [], [], [], []
    env = create_env()
    try:
        initialize(env, meta, initial)
        restore(env, SNAPSHOTS / source / f"snapshot-{start}.npz")
        restored_identity = identity(env)
        controller_identity = json.loads((expert / "identity.json").read_text())
        if restored_identity != controller_identity["restored_osc_identity"]:
            raise ValueError("Runtime/model mismatch")
        write_json(out / "identity.json", controller_identity)
        capture = MotionCapture(env)
        with absolute_joint_controller(env) as (controller, limits):
            with imageio.get_writer(out / "review.mp4", fps=20) as video:
                for i, action in enumerate(expected["actions"]):
                    before = env.get_sim_state().copy()
                    obs = env.regenerate_obs_from_state(before)
                    motion = capture.read()
                    state = np.r_[motion["joint_pos"], motion["gripper_pos"]]
                    if not np.array_equal(state, expected["before_joint_state"][i]):
                        raise ValueError(f"Pre-joint state changed at {i}")
                    prior = expected_rows[i]
                    row, image, hashes = pack_before(obs, state, action, limits, i, prior["index"],
                        SOURCES.index(source), before[0], prior["diagnostic_hold"])
                    if not np.array_equal(before, env.get_sim_state()):
                        raise ValueError("Recording changed physics")
                    video.append_data(image)
                    obs, _, done, _ = env.step(action.tolist())
                    after = env.get_sim_state().copy()
                    motion_after = capture.read()
                    measured, final_image = measure(env, obs, done, prior["index"])
                    checks = dict(goal=np.array_equal(controller.goal_qpos, action[:7]),
                                  physics=np.array_equal(after, expected["physical_after"][i]),
                                  joints=np.array_equal(np.r_[motion_after["joint_pos"], motion_after["gripper_pos"]],
                                                        expected["after_joint_state"][i]))
                    checks.update({k: measured[k] == prior[k] for k in
                                   ("image_sha256", "native_goal", "grasp", "finger_contact", "contact_count", "positions")})
                    if not all(checks.values()):
                        write_json(out / "divergence.json", dict(index=i, checks=checks,
                                   physical_max_abs=float(np.max(np.abs(after - expected["physical_after"][i]))),
                                   actual=measured, expected=prior))
                        raise ValueError(f"Expert replay diverged at {source}/{i}")
                    rows.append(row)
                    audits.append(dict(index=i, **hashes, state_sha256=digest(state),
                                       action_sha256=digest(action), post=measured))
                    before_states.append(before)
                    after_states.append(after)
                    states.append(state)
                    if (i + 1) % 100 == 0:
                        print(json.dumps(dict(source=source, controls=i + 1, bitwise=True)), flush=True)
                video.append_data(final_image)
        pq.write_table(pa.Table.from_pylist(rows), out / "pairs.parquet", compression="zstd")
        np.savez_compressed(out / "physics.npz", before=before_states, after=after_states,
                            actions=expected["actions"], observation_state=states)
        write_json(out / "audit.json", audits)
        validation = validate_dataset(out / "pairs.parquet", out / "physics.npz", out / "audit.json")
        motion_rows = len(records) - start
        result = dict(source=source, **validation, motion_rows=motion_rows,
                      diagnostic_hold_rows=len(rows) - motion_rows,
                      full_window_anchors=motion_rows - 9, state_dim=9, action_dim=8,
                      expert_physics_and_post_rgb_bitwise=True, policy_inference=False,
                      training=False, files_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
        write_json(out / "result.json", result)
        print(json.dumps(result), flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=SOURCES)
    args = parser.parse_args()
    check_render_backend()
    if args.source:
        return worker(args.output, args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT / "scripts" / p for p in ("record_joint_pairs.py", "record_osc_pairs.py",
              "joint_action_contract.py", "motion_recording.py", "simulation_snapshot.py",
              "verify_simulation_snapshot.py", "verify_osc_conversion.py", "replay_mixed_correction.py")]
    for source in SOURCES:
        from verify_osc_conversion import source_data
        _, _, _, start = source_data(source)
        inputs += [SNAPSHOTS / source / f"snapshot-{start}.npz"]
        inputs += [EXPERT / source / "velocity_compensated" / n for n in
                   ("trajectory.npz", "steps.json", "identity.json", "result.json")]
    write_json(args.output / "protocol.json", dict(schema=SCHEMA, training=False, seed=42,
               state="joint positions7 radians + finger positions2 meters",
               action="absolute joint target7 radians + gripper1 [-1,1]",
               images="fresh actual pre-action RGB; both axes reversed; uint8 256x256",
               control_hz=20, horizon=10, render_backend="osmesa",
               next_state_injection=False, original_images_reused=False,
               inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs}))
    results = []
    for source in SOURCES:
        subprocess.run([sys.executable, "-u", __file__, "--output", str(args.output),
                        "--source", source], check=True)
        results.append(json.loads((args.output / source / "result.json").read_text()))
        write_json(args.output / "manifest.json", dict(schema=SCHEMA, records=results,
                   machine_validated=len(results) == len(SOURCES), training=False,
                   prior_source_human_approval=True, new_execution_human_review=False))


if __name__ == "__main__":
    main()
