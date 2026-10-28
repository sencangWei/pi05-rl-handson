"""Execute archived OSC / joint controls without injecting saved next states."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("human_1793121358375746008", "human_1793122696228231033")
LIMITS = {"qpos": 1e-6, "qvel": 1e-5, "cached_position_m": 1e-5}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def control_blocks(records):
    """Preserve every 12-control jog's controller construction / OSC reset."""
    blocks, i = [], 0
    while i < len(records):
        joint = records[i]["controller"] == "JOINT_POSITION"
        end = i + (12 if joint else 1)
        block = records[i:end]
        if joint and (len(block) != 12 or any(
                r["controller"] != "JOINT_POSITION" or
                not np.array_equal(r["target"], block[0]["target"]) for r in block)):
            raise ValueError("Incomplete or changing absolute-target joint jog")
        blocks.append(block)
        i = end
    return blocks


def load_source(source, manifest):
    directory = ROOT / "corrections" / source
    audit = next(a for a in manifest["audits"] if a["source"] == source)
    if sha(directory / "metadata.json") != audit["metadata_sha256"]:
        raise ValueError("Metadata identity changed")
    for name, digest in audit["files"].items():
        if sha(directory / name) != digest:
            raise ValueError(f"Raw recording changed: {name}")
    meta = json.loads((directory / "metadata.json").read_text())
    if (meta["status"], meta["split"], meta["task_id"], meta["control_dt"]) != (
            "approved", "train", 8, .05):
        raise ValueError("Wrong task/split/status/control period")
    if sha(ROOT / "scripts/manual_joint_control.py") != meta["joint_control_sha256"]:
        raise ValueError("Original joint implementation changed")
    initial = np.load(directory / "initial_sim_state.npy", allow_pickle=False)
    if hashlib.sha256(initial.tobytes()).hexdigest() != meta["initial_sim_state_sha256"]:
        raise ValueError("Initial state changed")
    records, previous = [], initial
    for i, item in enumerate(meta["control_timeline"]):
        joint = item["controller"] == "JOINT_POSITION"
        if item["controller"] not in ("OSC_POSE", "JOINT_POSITION"):
            raise ValueError("Unknown controller")
        with np.load(directory / item.get("physics", item["path"]), allow_pickle=False) as p:
            before, after, action = p["before"], p["after"], p["action"]
            target = p["target_qpos"] if joint else None
            anchor = None if joint else p["osc_nullspace_initial_joints"]
        if (before.shape != (79,) or after.shape != (79,) or
                action.shape != ((8,) if joint else (7,)) or
                not np.isfinite(np.r_[before, after, action]).all() or
                not np.array_equal(previous, before)):
            raise ValueError("Invalid action or discontinuous physical timeline")
        if joint and (target.shape != (7,) or not np.isfinite(target).all() or
                      np.any(action[:7] != 0)):
            raise ValueError("Joint target is not an OSC action")
        records.append(dict(item, index=i, before=before, after=after, action=action,
                            target=target, anchor=anchor))
        previous = after
    if len(records) != meta["frame_count"] + meta["joint_frame_count"]:
        raise ValueError("Incomplete timeline")
    control_blocks(records)
    return directory, meta, initial, records


@contextmanager
def fixed_joint_controller(env, target):
    """Same controller and finally-reset as the hash-pinned original jog."""
    from robosuite.controllers.joint_pos import JointPositionController
    from robosuite.utils.buffers import DeltaBuffer
    base, robot = env.env, env.env.robots[0]
    osc, old_dim, old_actions = robot.controller, base._action_dim, robot.recent_actions
    if osc.name != "OSC_POSE":
        raise ValueError("Expected original OSC before each jog")

    class FixedTargetController(JointPositionController):
        def set_goal(self, action, set_qpos=None):
            super().set_goal(action, set_qpos=target)

    controller = FixedTargetController(
        sim=env.sim, eef_name=robot.gripper.important_sites["grip_site"],
        joint_indexes={"joints": robot.joint_indexes, "qpos": robot._ref_joint_pos_indexes,
                       "qvel": robot._ref_joint_vel_indexes},
        actuator_range=robot.torque_limits, kp=150, policy_freq=robot.control_freq,
        qpos_limits=env.sim.model.jnt_range[robot.joint_indexes].T)
    try:
        robot.controller = controller
        base._action_dim = robot.action_dim
        robot.recent_actions = DeltaBuffer(dim=robot.action_dim)
        yield
    finally:
        robot.controller, base._action_dim, robot.recent_actions = osc, old_dim, old_actions
        osc.update(force=True)
        osc.update_initial_joints(robot._joint_positions.copy())
        osc.reset_goal()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "evidence/manual-recovery28-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = [load_source(s, manifest) for s in SOURCES]
    for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        os.environ.setdefault(k, "egl")
    from baseline import configure_libero
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio
    packages = {k: importlib.metadata.version(k) for k in ("mujoco", "robosuite", "numpy")}
    if packages != {"mujoco": "3.8.1", "robosuite": "1.4.1", "numpy": "1.26.4"}:
        raise ValueError(f"Wrong original runtime: {packages}")
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    plan = dict(script_sha256=sha(__file__), manifest_sha256=sha(manifest_path),
                joint_source_sha256=sha(ROOT / "scripts/manual_joint_control.py"),
                sources=list(SOURCES), packages=packages, seed=42, control_hz=20,
                fidelity_limits=LIMITS, training_labels_generated=False,
                original_launcher_commit="7b2f7a9e49c1a29adb88632b6966f1634235c700",
                original_launcher_sha256="1233f81994d3ed3f533090295312caa7a53ef34aa2149a64d87f28111049b00b",
                protocol="Initial recorded state once; subsequent refresh uses actual current state. "
                         "Two initial current-state refreshes reproduce the original zero-offset "
                         "scene edit and begin-correction calls. "
                         "Float64 OSC physics actions; absolute joint targets, new controller every "
                         "12 steps, original OSC restored and reanchored after each jog. "
                         "No saved-next-state injection. Failed replay blocks label conversion.")
    write_json(args.output / "protocol.json", plan)
    reports = []
    for directory, meta, initial, records in inputs:
        out = args.output / directory.name
        out.mkdir()
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                                 task.problem_folder / task.bddl_file),
                                 camera_heights=256, camera_widths=256, ignore_done=True)
        rows, states = [], []
        try:
            env.seed(42)
            env.reset()
            env.regenerate_obs_from_state(initial)
            robot, sim = env.env.robots[0], env.sim
            robot.controller.update(force=True)
            robot.controller.update_initial_joints(np.asarray(meta["osc_nullspace_initial_joints"]))
            robot.controller.reset_goal()
            robot.gripper.current_action = np.array([-1., 1.])
            # Recorded launcher refreshed even for a zero-offset scene edit, then
            # begin_correction refreshed again. Observable sampling phase matters.
            for _ in range(2):
                env.regenerate_obs_from_state(env.get_sim_state())
            if not np.array_equal(env.get_sim_state(), initial):
                raise ValueError("Initial physical state restore failed")
            nq = sim.model.nq
            bowl = env.env.get_object("akita_black_bowl_1")
            with imageio.get_writer(out / "executed.mp4", fps=20) as video:
                def step(r):
                    before = env.get_sim_state().copy()
                    # Original recorder's refresh, never teacher forcing a recorded state.
                    env.regenerate_obs_from_state(before)
                    anchor_error = (None if r["anchor"] is None else
                                    float(np.max(np.abs(robot.controller.initial_joint - r["anchor"]))))
                    obs, _, done, _ = env.step(r["action"].tolist())
                    actual = env.get_sim_state().copy()
                    if not np.isfinite(actual).all():
                        raise ValueError("Nonfinite executed state")
                    with np.load(directory / r["path"], allow_pickle=False) as z:
                        pos_error = max(float(np.max(np.abs(obs["robot0_eef_pos"] - z["post_state"][:3]))),
                                        float(np.max(np.abs(obs["akita_black_bowl_1_pos"] - z["post_bowl_pos"]))))
                        recorded_done = bool(z["done"])
                    error = np.abs(actual - r["after"])
                    metrics = dict(qpos=float(error[1:1+nq].max()),
                                   qvel=float(error[1+nq:].max()), cached_position_m=pos_error)
                    rows.append(dict(index=r["index"], controller=r["controller"],
                                     **metrics, faithful=all(metrics[k] <= v for k, v in LIMITS.items()),
                                     before_error=float(np.max(np.abs(before-r["before"]))),
                                     anchor_error=anchor_error, native_goal=bool(done),
                                     recorded_goal=recorded_done,
                                     grasp=bool(env.env._check_grasp(robot.gripper, bowl)),
                                     finger_contact=bool(env.env.check_contact(robot.gripper.contact_geoms, bowl)),
                                     bowl_pos=obs["akita_black_bowl_1_pos"].tolist(),
                                     eef_pos=obs["robot0_eef_pos"].tolist()))
                    states.append(actual)
                    video.append_data(np.concatenate([obs[k][::-1, ::-1] for k in
                                      ("agentview_image", "robot0_eye_in_hand_image")], axis=1))

                for block in control_blocks(records):
                    if block[0]["controller"] == "JOINT_POSITION":
                        with fixed_joint_controller(env, block[0]["target"]):
                            for r in block:
                                step(r)
                    else:
                        step(block[0])
            np.savez_compressed(out / "executed-states.npz", after=np.asarray(states))
            write_json(out / "steps.json", rows)
            failures = [r["index"] for r in rows if not r["faithful"]]
            report = dict(source=directory.name, controls=len(rows),
                          faithful=not failures, first_divergent_control=next(iter(failures), None),
                          max_errors={k: max(r[k] for r in rows) for k in LIMITS},
                          goal_disagreements=sum(r["native_goal"] != r["recorded_goal"] for r in rows),
                          final_native_goal=rows[-1]["native_goal"],
                          final_finger_contact=rows[-1]["finger_contact"],
                          training_labels_generated=False)
            reports.append(report)
            write_json(out / "result.json", report)
            write_json(args.output / "summary.json", reports)
            print(json.dumps(report), flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
