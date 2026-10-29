"""Re-execute official LIBERO OSC demonstrations as verified local joint commands.

Generated action8 labels are commands actually executed by the local KP150 joint
controller. They are not native public action8 labels or relabelled source RGB.
Only the initial source state is restored; no recorded next states are injected.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from joint_action_contract import KD, KP, absolute_joint_controller, reference_target, validate_action
from motion_recording import MotionCapture
from record_joint_pairs import SCHEMA, check_render_backend, pack_before
from record_osc_pairs import digest, validate_dataset
from replay_mixed_correction import write_json

SOURCE_REPOSITORY = "yifengzhu-hf/LIBERO-datasets"
SOURCE_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
HOLD = 20
GATES = dict(position_p95_m=.03, position_max_m=.10, joint_p95_rad=.15,
             joint_max_rad=.50, cabinet_extra_penetration_m=.001)


def sha_file(path):
    digest_ = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest_.update(block)
    return digest_.hexdigest()


def split_for_episode(episode):
    if type(episode) is not int or not 0 <= episode < 50:
        raise ValueError("Official source episode must be an integer in [0, 49]")
    return "train" if episode < 40 else "holdout"


def localize_xml(xml, robosuite_root, libero_assets):
    """Localize paths and explicit mesh MIME types; preserve physical geometry."""
    tree = ET.fromstring(xml)
    for element in tree.findall(".//asset/*"):
        filename = element.get("file")
        if not filename:
            continue
        if "/robosuite/" in filename:
            local = Path(robosuite_root) / filename.rsplit("/robosuite/", 1)[1]
        elif "/assets/" in filename:
            local = Path(libero_assets) / filename.split("/assets/", 1)[1]
        else:
            local = Path(filename)
        if not local.is_file():
            raise FileNotFoundError(f"Unresolved demonstration asset: {filename} -> {local}")
        element.set("file", str(local))
        # MuJoCo 3.8 infers MIME types only on a cache miss. Make these
        # explicit so first and subsequent compilations have identical XML.
        if element.tag == "mesh" and "content_type" not in element.attrib:
            mime = {".msh": "model/vnd.mujoco.msh", ".stl": "model/stl",
                    ".obj": "model/obj"}.get(local.suffix.lower())
            if mime:
                element.set("content_type", mime)
    return ET.tostring(tree, encoding="unicode")


def read_source(path, episode):
    import h5py
    split = split_for_episode(episode)
    with h5py.File(path, "r") as source:
        data = source["data"]
        demo = data[f"demo_{episode}"]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        initial = np.asarray(demo["states"][0], dtype=np.float64)
        xml = demo.attrs.get("model_file")
        if isinstance(xml, bytes):
            xml = xml.decode()
        if not isinstance(xml, str):
            raise ValueError("Official demonstration lacks its model XML")
        attrs = {key: str(value) for key, value in data.attrs.items()}
    if (actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 10 or
            not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.00001 or
            initial.ndim != 1 or not np.isfinite(initial).all()):
        raise ValueError("Expected finite normalized OSC7 actions and initial flat state")
    return dict(episode=episode, split=split, actions=actions, initial=initial,
                xml=xml, source_attributes=attrs)


def initialize_source(env, initial, xml):
    """One source-state initialization, before either continuous rollout."""
    env.seed(42)
    env.reset()
    env.reset_from_xml_string(xml)
    env.sim.reset()
    if env.get_sim_state().shape != initial.shape:
        raise ValueError("Source XML/state layout mismatch")
    obs = env.regenerate_obs_from_state(initial)
    robot = env.env.robots[0]
    robot.controller.update(force=True)
    robot.controller.reset_goal()
    return obs


def released_goal(row):
    return bool(row["native_goal"] and not row["grasp"] and not row["finger_contact"])


def assess(reference, converted):
    """Freeze split before this gate; a failure never migrates to another split."""
    if len(reference) != len(converted) or len(reference) < HOLD:
        raise ValueError("Incomplete paired continuous rollouts")
    metrics = {"position_p95_m": float(np.percentile([r["position_error_m"] for r in converted], 95)),
               "position_max_m": max(r["position_error_m"] for r in converted),
               "joint_p95_rad": float(np.percentile([r["joint_error_rad"] for r in converted], 95)),
               "joint_max_rad": max(r["joint_error_rad"] for r in converted),
               "cabinet_extra_penetration_m": max(0., max(r["cabinet_penetration_m"] for r in converted) -
                                                    max(r["cabinet_penetration_m"] for r in reference))}
    checks = {key: value <= GATES[key] for key, value in metrics.items()}
    checks.update(reference_released_goal=all(released_goal(r) for r in reference[-HOLD:]),
                  converted_released_goal=all(released_goal(r) for r in converted[-HOLD:]),
                  finite_physics=all(r["finite_physics"] for r in converted),
                  no_clipped_joint_targets=all(r["clipped_axes"] == 0 for r in converted),
                  control_period=all(r["control_period_ok"] for r in converted))
    return dict(passed=all(checks.values()), checks=checks, metrics=metrics, limits=GATES,
                success_contract="native LIBERO goal + no grasp + no finger contact for final 20 controls",
                strict_eef_clearance_0_10m_required=False,
                strict_eef_clearance_explanation="Public placement demonstrations may not retreat by 10 cm")


def run(path, episode, output):
    import imageio.v2 as imageio
    import pyarrow as pa
    import pyarrow.parquet as pq
    import robosuite
    from libero.libero import get_libero_path
    from simulation_snapshot import capture, identity, restore
    from verify_osc_conversion import cabinet_penetration
    from verify_simulation_snapshot import create_env, measure

    check_render_backend()
    source = read_source(path, episode)
    output.mkdir(parents=True, exist_ok=False)
    xml = localize_xml(source["xml"], Path(robosuite.__file__).parent, get_libero_path("assets"))
    (output / "model.xml").write_text(xml)
    np.save(output / "initial_state.npy", source["initial"], allow_pickle=False)
    protocol = dict(schema=SCHEMA, source_repository=SOURCE_REPOSITORY, source_revision=SOURCE_REVISION,
                    hdf5_path=str(path.resolve()), hdf5_sha256=sha_file(path), episode=episode,
                    split=source["split"], split_rule="demo_0..39 train; demo_40..49 holdout, before outcomes",
                    source_controller="OSC_POSE7", target_controller="ABSOLUTE_JOINT_POSITION8",
                    kp=KP, kd=KD, control_hz=20, gripper="+1 close, -1 open (unchanged source command)",
                    label_origin="actual commands in local continuous joint replay, not source qpos labels",
                    per_step_next_state_injection=False, paired_rgb="new pre-action local rendering",
                    source_controls=len(source["actions"]), diagnostic_hold_controls=HOLD,
                    source_attributes=source["source_attributes"], gates=GATES,
                    script_sha256=sha_file(__file__))
    write_json(output / "protocol.json", protocol)
    reference, reference_motion = [], []
    env = create_env()
    try:
        initialize_source(env, source["initial"], xml)
        write_json(output / "identity.json", identity(env))
        (output / "initialized-model.xml").write_text(env.sim.model.get_xml())
        capture(env, output / "initial-snapshot.npz")
        motion = MotionCapture(env)
        substep_contacts = []
        physical_step = env.sim.step
        def probe_reference():
            physical_step()
            substep_contacts.append(cabinet_penetration(env))
        env.sim.step = probe_reference
        actions = list(source["actions"]) + [np.r_[np.zeros(6), -1.]] * HOLD
        with imageio.get_writer(output / "reference.mp4", fps=20) as video:
            for index, action in enumerate(actions):
                obs, _, _, _ = env.step(action.tolist())
                row, image = measure(env, obs, env.check_success(), index)
                row["cabinet_penetration_m"] = max(substep_contacts, default=0.)
                row["physics_substeps"] = len(substep_contacts)
                substep_contacts.clear()
                reference.append(row)
                reference_motion.append(motion.read())
                video.append_data(image)
        write_json(output / "reference-audit.json", reference)
        np.savez_compressed(output / "reference-motion.npz", **{
            key: np.asarray([item[key] for item in reference_motion]) for key in reference_motion[0]})
    finally:
        env.close()

    # Reconstruct the same XML once, then restore the complete LOCAL snapshot.
    # Source demonstration states are never used again after initialisation.
    env = create_env()
    rows, audit, before_states, after_states, states, executed = [], [], [], [], [], []
    try:
        initialize_source(env, source["initial"], xml)
        write_json(output / "recreated-identity.json", identity(env))
        (output / "recreated-model.xml").write_text(env.sim.model.get_xml())
        restore(env, output / "initial-snapshot.npz")
        motion = MotionCapture(env)
        substep_contacts = []
        physical_step = env.sim.step
        def probe_joint():
            physical_step()
            substep_contacts.append(cabinet_penetration(env))
        env.sim.step = probe_joint
        with absolute_joint_controller(env) as (controller, limits):
            with imageio.get_writer(output / "review.mp4", fps=20) as video:
                for index, ref in enumerate(reference_motion):
                    target, demand = reference_target(ref["joint_pos"], ref["joint_vel"], limits, True)
                    grip = source["actions"][index, -1] if index < len(source["actions"]) else -1.
                    action = validate_action(np.r_[target, grip], limits)
                    before = env.get_sim_state().copy()
                    env._update_observables(force=True)
                    obs = env.env._get_observations()
                    observed = motion.read()
                    state = np.r_[observed["joint_pos"], observed["gripper_pos"]]
                    pair, image, hashes = pack_before(obs, state, action, limits, index, index,
                                                     episode, before[0], index >= len(source["actions"]))
                    if not np.array_equal(before, env.get_sim_state()):
                        raise ValueError("Observation recording mutated physics")
                    video.append_data(image)
                    obs_after, _, _, _ = env.step(action.tolist())
                    after, actual = env.get_sim_state().copy(), motion.read()
                    if not np.array_equal(controller.goal_qpos, action[:7]):
                        raise ValueError("Controller did not execute the saved absolute command")
                    measured, final_image = measure(env, obs_after, env.check_success(), index)
                    measured.update(position_error_m=float(np.linalg.norm(actual["eef_pos"] - ref["eef_pos"])),
                                    joint_error_rad=float(np.max(np.abs(actual["joint_pos"] - ref["joint_pos"]))),
                                    cabinet_penetration_m=max(substep_contacts, default=0.),
                                    physics_substeps=len(substep_contacts),
                                    finite_physics=bool(np.isfinite(after).all()),
                                    clipped_axes=int(np.count_nonzero(target != demand)),
                                    control_period_ok=bool(np.isclose(after[0] - before[0], .05, rtol=0., atol=1e-10)))
                    substep_contacts.clear()
                    rows.append(pair)
                    audit.append(dict(index=index, **hashes, state_sha256=digest(state),
                                      action_sha256=digest(action), post=measured))
                    before_states.append(before)
                    after_states.append(after)
                    states.append(state)
                    executed.append(action)
                    if (index + 1) % 100 == 0:
                        print(json.dumps(dict(episode=episode, controls=index + 1)), flush=True)
                video.append_data(final_image)
        write_json(output / "pair-audit.json", audit)
        np.savez_compressed(output / "physics.npz", before=before_states, after=after_states,
                            actions=executed, observation_state=states)
        result = assess(reference, [r["post"] for r in audit])
        result.update(source=f"public_goal8_demo_{episode:03d}", episode=episode, split=source["split"], controls=len(rows),
                      no_next_state_injection=True, source_sha256=protocol["hdf5_sha256"],
                      controller="ABSOLUTE_JOINT_POSITION", kp=KP, kd=KD,
                      initial_snapshot_sha256=sha_file(output / "initial-snapshot.npz"),
                      first_native_goal_control=next((i + 1 for i, r in enumerate(audit) if r["post"]["native_goal"]), None),
                      motion_rows=len(source["actions"]), paired_rows=len(rows),
                      training_eligible=False, ready=False, human_review="pending")
        # Failed conversions retain diagnostics, never an apparently usable parquet.
        if result["passed"]:
            pq.write_table(pa.Table.from_pylist(rows), output / "pairs.parquet", compression="zstd")
            result["pair_validation"] = validate_dataset(output / "pairs.parquet", output / "physics.npz",
                                                        output / "pair-audit.json")
            result["pairs_sha256"] = sha_file(output / "pairs.parquet")
            result["training_eligible"] = True
        result["files_sha256"] = {name: sha_file(output / name) for name in
                                  ("model.xml", "initial_state.npy", "initial-snapshot.npz", "identity.json",
                                   "protocol.json", "reference-audit.json", "reference-motion.npz",
                                   "pair-audit.json", "physics.npz", "reference.mp4", "review.mp4")}
        if result["passed"]:
            result["files_sha256"]["pairs.parquet"] = result["pairs_sha256"]
        write_json(output / "result.json", result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Must run before importing LIBERO / OpenGL in any downstream module.
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from baseline import APP, configure_libero
    sys.path.insert(0, str(APP / ".local/openpi/third_party/libero"))
    configure_libero()
    return 0 if run(args.hdf5, args.episode, args.output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
