"""Construct disclosed synthetic failure starts, never scripted evaluation actions."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from baseline import configure_libero


def move(env, obs, target, gripper):
    for _ in range(80):
        delta = np.asarray(target) - obs["robot0_eef_pos"]
        if np.linalg.norm(delta) < .008:
            return obs
        action = np.r_[np.clip(delta / .05, -.4, .4), np.zeros(3), gripper]
        obs, _, _, _ = env.step(action)
    raise RuntimeError("Initialization hand positioning did not converge")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".local/numba"))
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(8)
    initial_states = suite.get_task_init_states(8)
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                             task.problem_folder / task.bddl_file),
                             camera_heights=256, camera_widths=256)
    entries = []
    try:
        for split, ids in (("train", range(1, 7)), ("heldout", range(10, 14))):
            for source_id in ids:
                for category in ("empty_grasp", "object_offset", "placement_failure"):
                    env.seed(42)
                    env.reset()
                    obs = env.set_init_state(initial_states[source_id])
                    for _ in range(15):
                        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    rob = env.env
                    bowl = rob.get_object("akita_black_bowl_1")
                    joint = bowl.joints[0]
                    qpos = rob.sim.data.get_joint_qpos(joint).copy()
                    original = qpos[:3].copy()
                    plate = rob.sim.data.body_xpos[rob.obj_body_id["plate_1"]].copy()
                    sign_x = -1 if source_id % 2 else 1
                    sign_y = -1 if (source_id // 2) % 2 else 1
                    magnitude = .04 if split == "train" else .055
                    offset = np.array([sign_x * magnitude, sign_y * magnitude])
                    if category == "empty_grasp":
                        obs = move(env, obs, original + [0, 0, .10], 1)
                        for _ in range(8):
                            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, 1])
                        qpos[:2] += offset
                    elif category == "object_offset":
                        qpos[:2] += offset
                    else:
                        obs = move(env, obs, plate + [0, 0, .25], -1)
                        # A released near miss on the table, away from plate center.
                        qpos[:2] = plate[:2] + offset * 1.7
                    settled_base = env.get_sim_state().copy()
                    # Reject overlaps with distractors: near misses must stay on the table.
                    candidates = [qpos.copy()]
                    if category != "placement_failure":
                        for flip_x, flip_y in ((-1, 1), (1, -1), (-1, -1)):
                            candidate = qpos.copy()
                            candidate[:2] = original[:2] + offset * [flip_x, flip_y]
                            candidates.append(candidate)
                    if category == "placement_failure":
                        directions = [offset / np.linalg.norm(offset)]
                        directions += [np.array([np.cos(t), np.sin(t)])
                                       for t in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
                        candidates = []
                        for direction in directions:
                            candidate = qpos.copy()
                            candidate[:2] = plate[:2] + direction * (.13 if split == "train" else .15)
                            candidates.append(candidate)
                    for candidate in candidates:
                        env.set_init_state(settled_base)
                        rob.sim.data.set_joint_qpos(joint, candidate)
                        rob.sim.data.set_joint_qvel(joint, np.zeros(6))
                        rob.sim.forward()
                        for _ in range(10):
                            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0,
                                                    1 if category == "empty_grasp" else -1])
                        actual = rob.sim.data.get_joint_qpos(joint)[:3]
                        stable = (np.linalg.norm(actual[:2] - candidate[:2]) < .015
                                  and abs(actual[2] - original[2]) < .025)
                        contact = rob.check_contact(rob.robots[0].gripper.contact_geoms, bowl)
                        if stable and not contact and not env.check_success():
                            break
                    else:
                        raise RuntimeError(f"No stable collision-free initial state: {split} {source_id} {category}")
                    state = env.get_sim_state().copy()
                    if not np.isfinite(state).all() or env.check_success():
                        raise RuntimeError("Invalid/already completed failure start")
                    held = bool(rob.check_contact(rob.robots[0].gripper.contact_geoms, bowl))
                    if held:
                        raise RuntimeError("Synthetic initial state unexpectedly contacts bowl")
                    name = f"{split}-{category}-init{source_id:02}"
                    path = args.output / (name + ".npz")
                    np.savez_compressed(path, state=state)
                    Image.fromarray(obs["agentview_image"][::-1, ::-1]).save(args.output / (name + ".png"))
                    entry = {"id": name, "split": split, "category": category,
                             "source_initial_id": source_id, "file": path.name,
                             "requested_offset_xy_m": offset.tolist(),
                             "offset_xy_m": (actual[:2] - (plate[:2] if category == "placement_failure" else original[:2])).tolist(),
                             "source_state_sha256": hashlib.sha256(np.asarray(initial_states[source_id]).tobytes()).hexdigest(),
                             "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "initial_native_success": False, "initial_finger_contact": held,
                             "eef_pos": obs["robot0_eef_pos"].tolist(),
                             "bowl_pos": rob.sim.data.get_joint_qpos(joint)[:3].tolist(),
                             "plate_pos": plate.tolist()}
                    entries.append(entry)
                    print(json.dumps(entry), flush=True)
        manifest = {"format": "libero-recovery-bank-v1", "task_suite": "libero_goal", "task_id": 8,
                    "seed": 42, "construction": "Synthetic physical states using OSC initialization and object pose perturbations; no recovery actions scripted into training/evaluation",
                    "split_protocol": "source initial IDs 1..6 train; 10..13 heldout, larger heldout offsets",
                    "success": "native bowl-on-plate and no finger contact for 5 consecutive controls",
                    "entries": entries}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
