"""Two frozen expert trajectories, two joint-target representations, no policy training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from joint_action_contract import KD, KP, absolute_joint_controller, reference_target, validate_action
from motion_recording import MotionCapture
from osc_action_conversion import acceptance, pose_delta
from record_osc_pairs import CONVERSION, SNAPSHOTS
from replay_mixed_correction import ROOT, SOURCES, sha, write_json
from simulation_snapshot import identity, restore
from verify_osc_conversion import HOLD, LIMITS, cabinet_penetration, source_data
from verify_simulation_snapshot import create_env, initialize, measure

MODES = ("next_position", "velocity_compensated")


def check_inputs(protocol):
    for name, expected in protocol["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def worker(output, source, mode):
    import imageio.v2 as imageio

    protocol = json.loads((output / "protocol.json").read_text())
    check_inputs(protocol)
    meta, initial, records, start = source_data(source)
    out = output / source / mode
    out.mkdir(parents=True)
    reference = json.loads((CONVERSION / source / "reference-steps.json").read_text())
    with np.load(CONVERSION / source / "reference-poses.npz", allow_pickle=False) as z:
        poses = {k: z[k] for k in z.files}
    with np.load(CONVERSION / source / "reference-states.npz", allow_pickle=False) as z:
        expected_states = z["after"]
    if len(reference) != len(records) - start + HOLD or len(expected_states) != len(reference):
        raise ValueError("Incomplete reference")
    env = create_env()
    rows, states, pre_joint, post_joint, actions, demands, peaks = [], [], [], [], [], [], []
    try:
        initialize(env, meta, initial)
        restore(env, SNAPSHOTS / source / f"snapshot-{start}.npz")
        restored_identity = identity(env)
        if restored_identity != json.loads((SNAPSHOTS / source / "identity.json").read_text()):
            raise ValueError("Reference runtime/model mismatch")
        before_install = env.get_sim_state().copy()
        robot = env.env.robots[0]
        qindices = 1 + np.asarray(robot._ref_joint_pos_indexes)
        vindices = 1 + env.sim.model.nq + np.asarray(robot._ref_joint_vel_indexes)
        physical_step = env.sim.step

        def probe():
            physical_step()
            peaks.append(cabinet_penetration(env))

        env.sim.step = probe
        capture = MotionCapture(env)
        with absolute_joint_controller(env) as (controller, limits):
            if not np.array_equal(before_install, env.get_sim_state()):
                raise ValueError("Controller installation changed physical state")
            write_json(out / "identity.json", dict(restored_osc_identity=restored_identity,
                       controller=controller.name, controller_class=type(controller).__name__, kp=KP, kd=KD,
                       limits=limits.tolist(), state="joint_pos7 + finger_pos2, radians/meters",
                       action="absolute joint target7 radians + gripper1 [-1,1]"))
            with imageio.get_writer(out / "execution.mp4", fps=20) as video:
                for j, expected in enumerate(reference):
                    # Preserve the reference observation refresh, using ONLY current state.
                    obs_before = env.regenerate_obs_from_state(env.get_sim_state())
                    if j == 0:
                        video.append_data(np.concatenate([obs_before[k][::-1, ::-1] for k in
                                          ("agentview_image", "robot0_eye_in_hand_image")], axis=1))
                    before = capture.read()
                    qref, vref = expected_states[j, qindices], expected_states[j, vindices]
                    # Diagnostic hold uses a stationary final reference, not the drift during OSC hold.
                    if j >= len(records) - start:
                        qref = expected_states[len(records) - start - 1, qindices]
                        vref = np.zeros(7)
                    target, demand = reference_target(qref, vref, limits, mode == "velocity_compensated")
                    grip = records[min(start + j, len(records) - 1)]["action"][-1]
                    action = validate_action(np.r_[target, grip], limits)
                    obs, _, done, _ = env.step(action.tolist())
                    after = capture.read()
                    if not np.array_equal(controller.goal_qpos, target):
                        raise ValueError("Absolute target was rescaled or silently clipped")
                    actual = env.get_sim_state().copy()
                    if not np.isfinite(actual).all() or not np.isclose(
                            after["sim_time"] - before["sim_time"], .05, rtol=0., atol=1e-10):
                        raise ValueError("Invalid physical transition")
                    row, image = measure(env, obs, done, start + j)
                    row.update(position_error_m=float(np.linalg.norm(after["eef_pos"] - poses["after_pos"][j])),
                               rotation_error_rad=float(np.linalg.norm(pose_delta(
                                   after["eef_pos"], after["eef_rotmat"],
                                   poses["after_pos"][j], poses["after_rot"][j])[3:])),
                               joint_error_rad=float(np.max(np.abs(after["joint_pos"] - qref))),
                               cabinet_penetration_m=max(peaks, default=0.),
                               physics_substeps=len(peaks),
                               clipped_axes=int(np.count_nonzero(target != demand)),
                               action=action.tolist(), diagnostic_hold=j >= len(records) - start)
                    rows.append(row)
                    states.append(actual)
                    pre_joint.append(np.r_[before["joint_pos"], before["gripper_pos"]])
                    post_joint.append(np.r_[after["joint_pos"], after["gripper_pos"]])
                    actions.append(action)
                    demands.append(demand)
                    video.append_data(image)
                    peaks.clear()
                    if (j + 1) % 100 == 0:
                        print(json.dumps(dict(source=source, mode=mode, controls=j + 1)), flush=True)
        write_json(out / "steps.json", rows)
        np.savez_compressed(out / "trajectory.npz", physical_after=states, before_joint_state=pre_joint,
                            after_joint_state=post_joint, actions=actions, unclipped_targets=demands)
        result = acceptance(rows, reference, protocol["limits"], HOLD)
        result.update(source=source, mode=mode, controls=len(rows),
                      joint_error_p95_rad=float(np.percentile([r["joint_error_rad"] for r in rows[:-HOLD]], 95)),
                      clipped_controls=sum(r["clipped_axes"] > 0 for r in rows),
                      released_goal_hold_controls=sum(r["native_goal"] and not r["finger_contact"]
                                                      for r in rows[-HOLD:]),
                      first_native_goal_control=next((i + 1 for i, r in enumerate(rows) if r["native_goal"]), None),
                      original_joint_controls=sum(r["controller"] == "JOINT_POSITION" for r in records[start:]),
                      actual_joint_travel_rad=np.abs(np.diff(np.asarray(post_joint)[:, :7], axis=0)).sum(axis=0).tolist(),
                      policy_inference=False, training_ready=False,
                      reason="Action-contract diagnostic, not original commands or approved training pairs")
        write_json(out / "result.json", result)
        check_inputs(protocol)
        print(json.dumps(result), flush=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--mode", choices=MODES)
    a = p.parse_args()
    if a.source:
        if a.mode is None:
            p.error("--source requires --mode")
        worker(a.output, a.source, a.mode)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    paths = [ROOT / "scripts" / n for n in ("verify_joint_actions.py", "joint_action_contract.py",
             "motion_recording.py", "verify_simulation_snapshot.py", "simulation_snapshot.py",
             "verify_osc_conversion.py", "replay_mixed_correction.py", "osc_action_conversion.py",
             "record_osc_pairs.py")]
    for source in SOURCES:
        _, _, _, start = source_data(source)
        paths += [SNAPSHOTS / source / f"snapshot-{start}.npz", SNAPSHOTS / source / "identity.json"]
        paths += [CONVERSION / source / n for n in
                  ("reference-steps.json", "reference-poses.npz", "reference-states.npz")]
    write_json(a.output / "protocol.json", dict(seed=42, source_commit=subprocess.check_output(
               ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               modes=MODES, sources=SOURCES, limits=LIMITS, control_hz=20, hold_controls=HOLD,
               kp=KP, kd=KD, action="absolute target radians7 + gripper1",
               next_position="reference next physical joint angles, not original commands",
               velocity_compensated="next joint angles + kd/kp * next joint velocity; clip to joint limits",
               training=False, policy_inference=False, next_state_injection=False,
               fresh_process_per_execution=True, ui_changed=False,
               inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths}))
    results = []
    for source in SOURCES:
        for mode in MODES:
            subprocess.run([sys.executable, "-u", __file__, "--output", str(a.output),
                            "--source", source, "--mode", mode], check=True)
            results.append(json.loads((a.output / source / mode / "result.json").read_text()))
            write_json(a.output / "summary.json", results)


if __name__ == "__main__":
    main()
