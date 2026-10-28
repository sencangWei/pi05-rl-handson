"""Replay saved model actions with passive physics probes; optionally continue the model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party/RLinf")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contacts(sim):
    import mujoco
    pairs = {}
    for index, contact in enumerate(sim.data.contact[:sim.data.ncon]):
        names = sorted(sim.model.geom_id2name(int(g)) or str(g) for g in
                       (contact.geom1, contact.geom2))
        if not any(n.startswith(("robot0_", "gripper0_", "akita_black_bowl_1")) for n in names):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(sim.model._model, sim.data._data, index, force)
        key = " / ".join(names)
        row = pairs.setdefault(key, {"normal_force_N": 0., "penetration_m": 0.})
        row["normal_force_N"] = max(row["normal_force_N"], abs(float(force[0])))
        row["penetration_m"] = max(row["penetration_m"], max(0., -float(contact.dist)))
    return pairs


def compare(record, original, tolerance):
    errors = {key: float(np.max(np.abs(np.asarray(record[key]) - original[key])))
              for key in ("eef_pos", "bowl_pos", "plate_pos")}
    flags = all(record[key] == original[key] for key in ("native_success", "finger_contact", "released_success"))
    return {"max_pose_error_m": max(errors.values()), "flags_equal": flags,
            "passed": max(errors.values()) <= tolerance and flags}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["extra_controls"] and not args.port:
        raise ValueError("Continuation requires a policy service")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        os.environ.setdefault(name, "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".local/numba"))
    from baseline import configure_libero, SeededPolicy
    configure_libero()
    import imageio.v2 as imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.sim.libero.recovery import RecoveryEnv
    manifest = ROOT / plan["manifest"]
    if sha256(manifest) != plan["manifest_sha256"]:
        raise ValueError("Manifest changed")
    client = None
    if args.port:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from openpi_client import image_tools
        from evaluate_recovery import payload
        client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
        metadata = client.get_server_metadata()
        if Path(metadata["checkpoint"]).resolve() != (ROOT / plan["checkpoint"]).resolve():
            raise ValueError("Wrong continuation checkpoint")
    else:
        metadata = None
    identity = dict(plan_sha256=sha256(args.plan), script_sha256=sha256(__file__),
                    packages={k: importlib.metadata.version(k) for k in
                              ("numpy", "mujoco", "robosuite", "torch")}, server=metadata,
                    limitation="Replay fidelity checks recorded EEF/bowl/plate positions and goal/contact flags, not unrecorded full physics")
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    raw = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                            task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
    env = RecoveryEnv(raw, {"manifest": str(manifest), "split": "dev"})
    sim, robot = raw.env.sim, raw.env.robots[0]
    joint_ids = [sim.model.joint_name2id(name) for name in robot.robot_model.joints]
    joint_ranges = sim.model.jnt_range[joint_ids]
    original_step = sim.step
    probes = []

    def step_probe():
        original_step()
        qpos = sim.data.qpos[robot._ref_joint_pos_indexes]
        lo, hi = robot.torque_limits
        raw_torque = np.asarray(robot.controller.torques)
        probes.append(dict(contacts=contacts(sim),
                           torque_saturation=bool(np.any((raw_torque < lo) | (raw_torque > hi))),
                           min_joint_margin_rad=float(np.min(np.minimum(qpos - joint_ranges[:, 0],
                                                                       joint_ranges[:, 1] - qpos)))))

    reports = []
    try:
        for case in plan["cases"]:
            source = ROOT / case["trace"]
            if sha256(source) != case["sha256"]:
                raise ValueError("Source trace changed")
            saved = [json.loads(line) for line in source.read_text().splitlines()]
            entry_index = next(i for i, e in enumerate(env.entries) if e["id"] == case["id"])
            env.seed(42)
            env.reset()
            obs = env.restore_recovery(entry_index)
            # robosuite reset replaces MjSim and its controller; never retain the old handles.
            sim, robot = raw.env.sim, raw.env.robots[0]
            original_step = sim.step
            joint_ids = [sim.model.joint_name2id(name) for name in robot.robot_model.joints]
            joint_ranges = sim.model.jnt_range[joint_ids]
            initial_contacts = contacts(sim)
            policy = SeededPolicy(client, 42) if client else None
            if policy:
                if len(saved) % 5:
                    raise ValueError("Continuation needs complete original action chunks")
                policy.index = len(saved) // 5
            step, success, failures, max_error = 0, False, [], 0.
            rows = []
            sim.step = step_probe
            with (args.output / (case["label"] + ".jsonl")).open("x") as trace, \
                    imageio.get_writer(args.output / (case["label"] + ".mp4"), fps=20, codec="libx264") as video:
                while step < len(saved) + plan["extra_controls"]:
                    if step < len(saved):
                        actions = [saved[step]["action"]]
                    elif failures or success or policy is None:
                        break
                    else:
                        actions = policy.infer(payload(obs, task.language, image_tools))["actions"][:5]
                    for action in actions:
                        before = obs["robot0_eef_pos"].copy()
                        probes.clear()
                        obs, _, success, info = env.step(np.asarray(action))
                        step += 1
                        bowl = sim.data.body_xpos[raw.env.obj_body_id["akita_black_bowl_1"]].copy()
                        plate = sim.data.body_xpos[raw.env.obj_body_id["plate_1"]].copy()
                        jac = sim.data.get_site_jacp(robot.controller.eef_name).reshape(3, -1)[:, robot._ref_joint_vel_indexes]
                        merged = {}
                        for sample in probes:
                            for key, values in sample["contacts"].items():
                                old = merged.setdefault(key, {"normal_force_N": 0., "penetration_m": 0.})
                                for metric, value in values.items():
                                    old[metric] = max(old[metric], value)
                        record = dict(step=step, phase="replay" if step <= len(saved) else "model_continuation",
                                      **info, action=np.asarray(action).tolist(),
                                      eef_pos=obs["robot0_eef_pos"].tolist(), bowl_pos=bowl.tolist(), plate_pos=plate.tolist(),
                                      eef_motion_m=float(np.linalg.norm(obs["robot0_eef_pos"] - before)),
                                      goal_error_m=float(np.linalg.norm(robot.controller.goal_pos - obs["robot0_eef_pos"])),
                                      translation_jacobian_sigma_min=float(np.linalg.svd(jac, compute_uv=False)[-1]),
                                      grasp=bool(raw.env._check_grasp(robot.gripper, raw.env.get_object("akita_black_bowl_1"))),
                                      gripper_qpos=obs["robot0_gripper_qpos"].tolist(), contacts=merged,
                                      torque_saturated_substeps=sum(p["torque_saturation"] for p in probes),
                                      min_joint_margin_rad=min(p["min_joint_margin_rad"] for p in probes))
                        if step <= len(saved):
                            fidelity = compare(record, saved[step - 1], plan["pose_tolerance_m"])
                            record["fidelity"] = fidelity
                            max_error = max(max_error, fidelity["max_pose_error_m"])
                            if not fidelity["passed"]:
                                failures.append(step)
                        rows.append(record)
                        trace.write(json.dumps(record) + "\n")
                        video.append_data(np.concatenate([obs[k][::-1, ::-1] for k in
                                          ("agentview_image", "robot0_eye_in_hand_image")], axis=1))
                        if success and step >= len(saved):
                            break
                sim.step = original_step
            report = dict(label=case["label"], id=case["id"], replay_controls=len(saved), controls=step,
                          success=bool(success), fidelity_passed=not failures, max_pose_error_m=max_error,
                          fidelity_failed_steps=failures, initial_contacts=initial_contacts,
                          last50=dict(median_motion_mm=float(np.median([r["eef_motion_m"] * 1000 for r in rows[-50:]])),
                                      grasp_controls=sum(r["grasp"] for r in rows[-50:]),
                                      torque_saturated_controls=sum(r["torque_saturated_substeps"] > 0 for r in rows[-50:]),
                                      min_joint_margin_rad=min(r["min_joint_margin_rad"] for r in rows[-50:]),
                                      min_translation_jacobian_sigma=min(r["translation_jacobian_sigma_min"] for r in rows[-50:])))
            reports.append(report)
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps({k: v for k, v in report.items() if k != "initial_contacts"}), flush=True)
    finally:
        sim.step = original_step
        env.close()
        if client:
            client._ws.close()
    return int(any(not r["fidelity_passed"] for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
