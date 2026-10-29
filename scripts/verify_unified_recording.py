"""Physical recording/replay acceptance; synthetic smoke takes remain drafts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from unified_joint_recording import load_joint_frames


def check_interface(lab, initial_id, output):
    from correction_data import write_json
    lab.reset(initial_id)
    lab.begin_correction()
    recorder = lab.correction
    lab.manual_hold(1)
    lab.manual_jog(2, 1, .1, 2)
    lab.manual_jog(5, 1, .1, 2)
    q = lab.env.env.robots[0]._joint_positions.copy()
    lab.manual_joint_jog(3, -2.)
    moved = float(np.rad2deg(lab.env.env.robots[0]._joint_positions[3] - q[3]))
    if moved >= -.2:
        raise AssertionError("Elbow did not physically move in requested direction")
    lab.manual_gripper(True, 2)
    lab.manual_hold(2)
    assert lab.correction is recorder and lab.env.env.action_dim == 8
    assert lab.env.env.robots[0].controller.name == "JOINT_POSITION"
    try:
        lab.finish_correction(True)
    except ValueError:
        pass
    else:
        raise AssertionError("Unfinished synthetic recovery was accepted")
    lab.finish_correction(False)
    assert recorder.count == 21 and not recorder.metadata["training_ready"]
    assert lab.env.env.action_dim == 7
    # A second take must capture a valid OSC snapshot after controller restoration.
    lab.begin_correction()
    lab.manual_hold(1)
    lab.finish_correction(False)
    replay_file = output / "fresh-process-replay.json"
    subprocess.run([sys.executable, "-u", __file__, "--directory", str(recorder.directory),
                    "--output", str(replay_file)], check=True)
    replay = json.loads(replay_file.read_text())
    report = dict(ready=True, directory=str(recorder.directory), controls=recorder.count,
                  elbow_actual_degrees=moved, action_dim=8, state_dim=9,
                  all_inputs_one_controller=True, second_take=True, draft=True,
                  training_ready=False, replay=replay)
    write_json(output / "check.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def replay(directory, output):
    from manual_recovery import UI, digest
    source = json.loads((UI.parent / "source_manifest.json").read_text())
    if any(digest(UI / name) != expected for name, expected in source["files"].items()):
        raise ValueError("Frozen UI source checksum differs")
    sys.path.insert(0, str(UI))
    sys.path.insert(0, "/home/robot/voice-arm-sim/.local/openpi/third_party/libero")
    from baseline import configure_libero
    configure_libero()
    from correction_data import digest, write_json
    from joint_action_contract import absolute_joint_controller
    from motion_recording import MotionCapture
    from simulation_snapshot import restore
    from verify_simulation_snapshot import create_env
    from simulation_lab import TARGET
    metadata = json.loads((directory / "metadata.json").read_text())
    frames = load_joint_frames(directory, metadata)
    if not frames:
        raise ValueError("Empty recording")
    segment, = metadata["motion_segments"]
    snapshot = directory / segment["path"]
    if digest(snapshot) != segment["sha256"]:
        raise ValueError("Initial full snapshot checksum differs")
    env = create_env()
    maximum_error = 0.
    try:
        env.seed(42)
        env.reset()
        restore(env, snapshot)
        with absolute_joint_controller(env):
            motion = MotionCapture(env)
            for frame in frames:
                np.testing.assert_allclose(env.get_sim_state(), frame["physics_before"], atol=1e-10, rtol=0)
                obs = env.regenerate_obs_from_state(env.get_sim_state())
                for stored, key in (("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image")):
                    np.testing.assert_array_equal(obs[key][::-1, ::-1], frame[stored])
                before = motion.read()
                for key, value in before.items():
                    np.testing.assert_allclose(value, frame[f"before_{key}"], atol=1e-10, rtol=0)
                env.step(frame["action"].tolist())
                actual = env.get_sim_state()
                error = float(np.max(np.abs(actual - frame["physics_after"])))
                maximum_error = max(maximum_error, error)
                np.testing.assert_allclose(actual, frame["physics_after"], atol=1e-10, rtol=0)
                for key, value in motion.read().items():
                    np.testing.assert_allclose(value, frame[f"after_{key}"], atol=1e-10, rtol=0)
                assert bool(env.check_success()) == bool(frame["done"])
                robot = env.env.robots[0]
                contact = env.env.check_contact(robot.gripper.contact_geoms, env.env.get_object(TARGET))
                assert bool(contact) == bool(frame["finger_contact"])
        result = dict(controls=len(frames), physical_states_match=True, both_cameras_equal=True,
                      all_joint_velocity_eef_and_timestamp_telemetry_matches=True,
                      native_goal_and_finger_contact_flags_match=True,
                      maximum_physics_abs_error=maximum_error, atol=1e-10,
                      fresh_process=True, saved_next_state_injection=False)
        write_json(output, result)
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.directory, args.output)
