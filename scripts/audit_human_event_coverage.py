"""Query saved human after-states, including excluded joint controls; no motion."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from recovery_events import summarize

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manual = json.loads(args.manifest.read_text())
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        os.environ.setdefault(key, "egl")
    from baseline import configure_libero
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    raw = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                            task.problem_folder / task.bddl_file),
                            camera_heights=256, camera_widths=256)
    report = dict(manifest_sha256=sha(args.manifest), script_sha256=sha(__file__),
                  event_helper_sha256=sha(ROOT / "scripts/recovery_events.py"),
                  packages={k: importlib.metadata.version(k) for k in ("mujoco", "robosuite", "numpy")},
                  protocol="Static forward queries of recorded after-states, no policy or controls. "
                           "Grasp is opposing fingerpad contact, not proof of force closure. "
                           "Run indices zero-based inclusive, timeline includes joint controls. "
                           "Fresh forward positions can differ from cached recorded observations; "
                           "both error and native-goal disagreement are reported, not called exact replay.",
                  sources=[])
    try:
        for audit in manual["audits"]:
            directory = ROOT / "corrections" / audit["source"]
            if sha(directory / "metadata.json") != audit["metadata_sha256"]:
                raise ValueError("Metadata changed")
            meta = json.loads((directory / "metadata.json").read_text())
            if meta["split"] != "train" or meta["task_id"] != 8 or meta["status"] != "approved":
                raise ValueError("Wrong human split/task/status")
            timeline = meta.get("control_timeline") or [
                dict(controller="OSC_POSE", path=f"frame-{i:06d}.npz", physics=f"physics-{i:06d}.npz")
                for i in range(meta["frame_count"])]
            if len(timeline) != meta["frame_count"] + meta.get("joint_frame_count", 0):
                raise ValueError("Timeline count mismatch")
            for name, digest in audit["files"].items():
                if sha(directory / name) != digest:
                    raise ValueError(f"Raw recording changed: {name}")
            raw.seed(42)
            raw.reset()
            sim, robot = raw.env.sim, raw.env.robots[0]
            bowl = raw.env.get_object("akita_black_bowl_1")
            rows, previous, max_error = [], None, 0.
            initial_z = meta["bank_entry"]["bowl_pos"][2]
            for i, item in enumerate(timeline):
                with np.load(directory / item.get("physics", item["path"]), allow_pickle=False) as z:
                    before, after = z["before"], z["after"]
                    if before.shape != (79,) or after.shape != (79,) or not np.isfinite([before, after]).all():
                        raise ValueError("Bad physical state")
                    if previous is not None and not np.array_equal(previous, before):
                        raise ValueError("Discontinuous timeline")
                    previous = after.copy()
                    raw.set_state(after)
                sim.forward()
                b = sim.data.body_xpos[raw.env.obj_body_id["akita_black_bowl_1"]].copy()
                p = sim.data.body_xpos[raw.env.obj_body_id["plate_1"]].copy()
                eef = sim.data.site_xpos[sim.model.site_name2id(robot.controller.eef_name)].copy()
                with np.load(directory / item["path"], allow_pickle=False) as z:
                    error = max(float(np.max(np.abs(b - z["post_bowl_pos"]))),
                                float(np.max(np.abs(eef - z["post_state"][:3]))))
                    native = bool(raw.check_success())
                    recorded_done = bool(z["done"])
                    gripper_command = float(z["action"][-1])
                max_error = max(error, max_error)
                row = dict(timeline_index=i, source_frame=item["path"], controller=item["controller"],
                           included_osc_frame=item["controller"] == "OSC_POSE",
                           grasp=bool(raw.env._check_grasp(robot.gripper, bowl)),
                           finger_contact=bool(raw.env.check_contact(robot.gripper.contact_geoms, bowl)),
                           native_success=native, gripper_command=gripper_command,
                           recorded_done=recorded_done, recorded_pose_error_m=error,
                           bowl_pos=b.tolist(), plate_pos=p.tolist(), eef_pos=eef.tolist())
                rows.append(row)
            clips = []
            for rec in manual["records"]:
                if rec["source"] != audit["source"]:
                    continue
                lo, hi = rec["source_frame_range"]
                clip = [r for r in rows if r["included_osc_frame"] and lo <= r["source_frame"] <= hi]
                if len(clip) != rec["frames"] or sha(ROOT / rec["path"]) != rec["sha256"]:
                    raise ValueError("Training clip identity mismatch")
                clips.append(dict(episode=rec["episode"], terminal=rec["terminal"],
                                  source_frame_range=rec["source_frame_range"],
                                  **summarize(clip, initial_z)))
            out = dict(source=audit["source"], entry=meta["bank_entry"]["id"],
                       max_recorded_pose_error_m=max_error,
                       recorded_goal_disagreements=sum(r["native_success"] != r["recorded_done"] for r in rows),
                       **summarize(rows, initial_z), clips=clips,
                       controller_counts={k: sum(r["controller"] == k for r in rows)
                                          for k in ("OSC_POSE", "JOINT_POSITION")},
                       grasp_by_controller={k: sum(r["controller"] == k and r["grasp"] for r in rows)
                                            for k in ("OSC_POSE", "JOINT_POSITION")},
                       lifted_grasp_by_controller={k: sum(r["controller"] == k and r["grasp"] and
                                                        r["bowl_pos"][2] - initial_z >= .03 for r in rows)
                                                   for k in ("OSC_POSE", "JOINT_POSITION")})
            (args.output / (audit["source"] + ".jsonl")).write_text(
                "".join(json.dumps(r) + "\n" for r in rows))
            report["sources"].append(out)
            print(json.dumps(out), flush=True)
    finally:
        raw.close()
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
