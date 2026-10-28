"""Fresh-process contact replay of new snapshots, never repair old recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from replay_mixed_correction import (LIMITS, ROOT, SOURCES, control_blocks,
                                     fixed_joint_controller, load_source, sha, write_json)
from simulation_snapshot import capture, identity, restore

MODES = ("legacy", "native_only", "integration_python", "full")


def create_env():
    from baseline import configure_libero
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    return OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                              task.problem_folder / task.bddl_file),
                              camera_heights=256, camera_widths=256, ignore_done=True)


def initialize(env, meta, initial):
    env.seed(42)
    env.reset()
    env.regenerate_obs_from_state(initial)
    robot = env.env.robots[0]
    robot.controller.update(force=True)
    robot.controller.update_initial_joints(np.asarray(meta["osc_nullspace_initial_joints"]))
    robot.controller.reset_goal()
    robot.gripper.current_action = np.array([-1., 1.])
    for _ in range(2):
        env.regenerate_obs_from_state(env.get_sim_state())


def measure(env, obs, done, index):
    robot = env.env.robots[0]
    bowl = env.env.get_object("akita_black_bowl_1")
    image = np.concatenate([obs[k][::-1, ::-1] for k in
                            ("agentview_image", "robot0_eye_in_hand_image")], axis=1)
    row = dict(index=index, native_goal=bool(done),
               grasp=bool(env.env._check_grasp(robot.gripper, bowl)),
               finger_contact=bool(env.env.check_contact(robot.gripper.contact_geoms, bowl)),
               contact_count=int(env.sim.data.ncon),
               image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
               positions=np.r_[obs["robot0_eef_pos"], obs["akita_black_bowl_1_pos"]].tolist())
    return row, image


def windows(records):
    joint = [b[0]["index"] for b in control_blocks(records)
             if b[0]["controller"] == "JOINT_POSITION"]
    return [(joint[0] - 12, joint[0] + 24), (joint[-1], len(records))]


def execute(env, records, start, stop, callback):
    for block in control_blocks(records):
        i = block[0]["index"]
        if not start <= i < stop:
            continue
        callback("boundary", i, None, None)

        def step(r):
            env.regenerate_obs_from_state(env.get_sim_state())
            obs, _, done, _ = env.step(r["action"].tolist())
            callback("step", r["index"], obs, done)

        if block[0]["controller"] == "JOINT_POSITION":
            with fixed_joint_controller(env, block[0]["target"]):
                for r in block:
                    step(r)
        else:
            step(block[0])


def reference(output, source):
    import imageio.v2 as imageio
    manifest = json.loads((ROOT / "evidence/manual-recovery28-manifest.json").read_text())
    _, meta, initial, records = load_source(source, manifest)
    out = output / source
    out.mkdir()
    env = create_env()
    rows, states = [], []
    try:
        initialize(env, meta, initial)
        write_json(out / "identity.json", identity(env))
        cuts = windows(records)
        write_json(out / "windows.json", cuts)
        with imageio.get_writer(out / "reference.mp4", fps=20) as video:
            def callback(kind, index, obs, done):
                if kind == "boundary":
                    if index in {w[0] for w in cuts}:
                        capture(env, out / f"snapshot-{index}.npz")
                    return
                row, image = measure(env, obs, done, index)
                rows.append(row)
                states.append(env.get_sim_state().copy())
                video.append_data(image)
            execute(env, records, 0, len(records), callback)
        np.savez_compressed(out / "reference-states.npz", after=np.asarray(states))
        write_json(out / "reference-steps.json", rows)
        print(json.dumps(dict(reference=source, controls=len(rows), windows=cuts)), flush=True)
    finally:
        env.close()


def replay(output, source, start, stop, mode):
    import imageio.v2 as imageio
    manifest = json.loads((ROOT / "evidence/manual-recovery28-manifest.json").read_text())
    _, meta, initial, records = load_source(source, manifest)
    out = output / source
    expected_rows = json.loads((out / "reference-steps.json").read_text())
    with np.load(out / "reference-states.npz", allow_pickle=False) as z:
        expected_states = z["after"]
    env = create_env()
    rows, states = [], []
    stem = f"{mode}-{start}-{stop}"
    try:
        initialize(env, meta, initial)
        restore(env, out / f"snapshot-{start}.npz", mode)
        with imageio.get_writer(out / f"{stem}.mp4", fps=20) as video:
            def callback(kind, index, obs, done):
                if kind != "step":
                    return
                row, image = measure(env, obs, done, index)
                actual = env.get_sim_state().copy()
                err = np.abs(actual - expected_states[index])
                nq = env.sim.model.nq
                expected = expected_rows[index]
                metrics = dict(qpos=float(err[1:1+nq].max()), qvel=float(err[1+nq:].max()),
                               cached_position_m=float(np.max(np.abs(
                                   np.asarray(row["positions"]) - expected["positions"]))))
                flags = ("native_goal", "grasp", "finger_contact", "contact_count", "image_sha256")
                row.update(metrics, faithful=all(metrics[k] <= v for k, v in LIMITS.items()),
                           state_bitwise_equal=bool(np.array_equal(actual, expected_states[index])),
                           flags_equal=all(row[k] == expected[k] for k in flags))
                rows.append(row)
                states.append(actual)
                video.append_data(image)
            execute(env, records, start, stop, callback)
        if len(rows) != stop - start:
            raise ValueError("Replay did not execute the entire frozen window")
        write_json(out / f"{stem}-steps.json", rows)
        np.savez_compressed(out / f"{stem}-states.npz", after=np.asarray(states))
        result = dict(source=source, mode=mode, start=start, stop=stop, controls=len(rows),
                      faithful=all(r["faithful"] for r in rows),
                      first_divergent_control=next((r["index"] for r in rows if not r["faithful"]), None),
                      all_state_bitwise_equal=all(r["state_bitwise_equal"] for r in rows),
                      all_flags_and_images_equal=all(r["flags_equal"] for r in rows),
                      max_errors={k: max(r[k] for r in rows) for k in LIMITS},
                      contact_controls=sum(r["finger_contact"] for r in rows),
                      reference_contact_controls=sum(expected_rows[i]["finger_contact"] for i in range(start, stop)),
                      final_goal=rows[-1]["native_goal"], final_contact=rows[-1]["finger_contact"],
                      snapshot_sha256=sha(out / f"snapshot-{start}.npz"))
        write_json(out / f"{stem}-result.json", result)
        print(json.dumps(result), flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", choices=("reference", *MODES))
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.worker == "reference":
            reference(args.output, args.source)
        else:
            replay(args.output, args.source, args.start, args.stop, args.worker)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(sources=list(SOURCES), seed=42, limits=LIMITS, modes=MODES,
                    script_sha256=sha(__file__), snapshot_script_sha256=sha(ROOT / "scripts/simulation_snapshot.py"),
                    windows="12 OSC + first two joint jogs; last joint jog + all final OSC controls",
                    fresh_process_per_replay=True, saved_next_state_injection=False,
                    old_recordings_repaired=False, training_labels_generated=False,
                    claim="New reference execution only; missing old internal states cannot be reconstructed.")
    write_json(args.output / "protocol.json", protocol)
    results = []
    for source in SOURCES:
        base = [sys.executable, "-u", __file__, "--output", str(args.output), "--source", source]
        subprocess.run(base + ["--worker", "reference"], check=True)
        for start, stop in json.loads((args.output / source / "windows.json").read_text()):
            for mode in MODES:
                subprocess.run(base + ["--worker", mode, "--start", str(start), "--stop", str(stop)], check=True)
                results.append(json.loads((args.output / source / f"{mode}-{start}-{stop}-result.json").read_text()))
                write_json(args.output / "summary.json", results)
    return 0 if all(r["faithful"] and r["all_flags_and_images_equal"] for r in results if r["mode"] == "full") else 1


if __name__ == "__main__":
    raise SystemExit(main())
