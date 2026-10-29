"""Open a training-bank failure, paused, in the existing Chinese correction UI.

Only manual control is enabled. No policy server, training job, or hardware starts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "support/manual_recovery_ui/scripts"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserve_unshifted_bowl(move_bowl, target):
    """Keep frozen bank physics untouched when reset requests no object edit."""
    def edit(env, obs, dx, dy):
        if dx == 0. and dy == 0.:
            position = obs[f"{target}_pos"].tolist()
            return obs, dict(before_xyz=position, after_xyz=position.copy(), dx_m=dx, dy_m=dy)
        return move_bowl(env, obs, dx, dy)
    return edit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-id", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--category", choices=("placement_failure", "empty_grasp", "object_offset"),
                        default="placement_failure")
    parser.add_argument("--check", action="store_true", help="Restore and verify without opening windows")
    parser.add_argument("--check-joints", action="store_true", help="Headless physical joint/OSC switching smoke test")
    parser.add_argument("--check-takeover", action="store_true", help="Verify OSC/joint/OSC recording without windows")
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "osmesa" if args.check or args.check_joints or args.check_takeover else "glx"
    os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    from baseline import configure_libero
    configure_libero()
    sys.path.insert(0, "/home/robot/voice-arm-sim/.local/openpi/third_party/libero")
    source = json.loads((UI.parent / "source_manifest.json").read_text())
    for name, expected in source["files"].items():
        if digest(UI / name) != expected:
            raise ValueError(f"UI snapshot changed: {name}")
    sys.path.insert(0, str(UI))
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import simulation_lab
    import correction_workbench
    simulation_lab.move_bowl = preserve_unshifted_bowl(simulation_lab.move_bowl, simulation_lab.TARGET)
    from correction_data import write_json
    from correction_workbench import CorrectionWorkbench
    from manual_rotation_guide import guided_classes
    from manual_joint_control import JointSetupMixin, joint_classes
    from motion_recording import MotionCapture, SCHEMA, save_motion
    GuidedPreview, GuidedPanel = guided_classes()
    GuidedPreview, GuidedPanel = joint_classes(GuidedPreview, GuidedPanel)

    bank = ROOT / "data/recovery-bowl-v3"
    manifest = json.loads((bank / "manifest.json").read_text())
    if (manifest["task_suite"], manifest["task_id"]) != ("libero_goal", 8):
        raise ValueError("Expected matched goal task 8 bank")
    entries = {e["source_initial_id"]: e for e in manifest["entries"]
               if e["split"] == "train" and e["category"] == args.category}
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(8)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    simulation_lab.TASK_LANGUAGE = correction_workbench.TASK_LANGUAGE = task.language

    class BankTask:
        layout_count = 7
        metadata = {"suite": "libero_goal", "task_id": 8, "bddl_sha256": digest(bddl),
                    "bank_manifest_sha256": digest(bank / "manifest.json"), "category": args.category}

        def create(self, layout):
            if layout not in entries:
                raise ValueError("本次只采训练布局 1～6；不使用测试布局")
            entry = entries[layout]
            path = bank / entry["file"]
            if digest(path) != entry["sha256"]:
                raise ValueError("Failure state checksum mismatch")
            with np.load(path, allow_pickle=False) as saved:
                state = saved["state"].copy()
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                     camera_widths=256, ignore_done=True)
            try:
                env.seed(42)
                env.reset()
                obs = env.regenerate_obs_from_state(state)
                if simulation_lab.state_digest(env.get_sim_state()) != entry["state_sha256"]:
                    raise ValueError("Restored physics differs from frozen bank state")
                for robot in env.env.robots:
                    robot.controller.update(force=True)
                    robot.controller.reset_goal()
                    # All bank starts have an open gripper; initialize its accumulator accordingly.
                    robot.gripper.current_action = np.array([-1., 1.])
                if env.check_success():
                    raise ValueError("Expected an unfinished task")
                return env, obs
            except Exception:
                env.close()
                raise

    bank_task = BankTask()
    if args.check or args.check_joints:
        env, obs = bank_task.create(args.initial_id)
        try:
            if args.check_joints:
                from manual_joint_control import physical_joint_jog
                checks = []
                for joint, degrees in ((j, d) for j in range(7) for d in (2., -2.)):
                    _, trace = physical_joint_jog(env, joint, degrees, -1.)
                    moved = np.rad2deg(trace["after_qpos"] - trace["before_qpos"])
                    if degrees * moved[joint] <= 0 or abs(moved[joint]) < .2:
                        raise AssertionError(f"J{joint + 1} did not move in requested direction: {moved}")
                    if env.env.action_dim != 7 or env.env.robots[0].controller.name != "OSC_POSE":
                        raise AssertionError("OSC action contract not restored")
                    obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
                    if not np.isfinite(env.get_sim_state()).all():
                        raise AssertionError("Nonfinite OSC physics after joint setup")
                    checks.append({"joint": joint + 1, "requested_degrees": degrees, "actual_degrees": moved.tolist(),
                                   "trace_states": len(trace["states"]), "restored_action_dim": env.env.action_dim})
                normal_step = env.step
                old_controller = env.env.robots[0].controller
                old_buffer = env.env.robots[0].recent_actions
                def failed_step(action):
                    raise RuntimeError("injected joint step failure")
                env.step = failed_step
                try:
                    try:
                        physical_joint_jog(env, 3, 2., -1.)
                    except RuntimeError as error:
                        if str(error) != "injected joint step failure":
                            raise
                    else:
                        raise AssertionError("Expected injected failure")
                    if (env.env.action_dim != 7 or env.env.robots[0].controller is not old_controller
                            or env.env.robots[0].recent_actions is not old_buffer):
                        raise AssertionError("Joint failure left controller/buffer switched")
                finally:
                    env.step = normal_step
                obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
                print(json.dumps({"joint_checks": checks}, indent=2))
            print(json.dumps({"ready": True, "entry": entries[args.initial_id],
                              "native_success": bool(env.check_success()),
                              "control_dt": env.env.control_timestep,
                              "eef": obs["robot0_eef_pos"].tolist()}, indent=2))
        finally:
            env.close()
        return

    class ManualWorkbench(JointSetupMixin, CorrectionWorkbench):
        def begin_correction(self, split="train"):
            if split != "train":
                raise ValueError("训练失败场景只能录为 train，不能混入测试集")
            super().begin_correction(split)
            self.correction.metadata.update(
                task_suite="libero_goal", task_id=8, source="human manual recovery",
                bank_entry=entries[self.layout_id], control_hz=1 / self.view.control_dt,
                launcher_sha256=digest(__file__), physics_files=[], control_timeline=[], joint_frame_count=0,
                rotation_guide_sha256=digest(ROOT / "scripts/manual_rotation_guide.py"),
                joint_control_sha256=digest(ROOT / "scripts/manual_joint_control.py"),
                osc_nullspace_initial_joints=self.env.env.robots[0].controller.initial_joint.tolist(),
                physics_scope="full MuJoCo pre/post state per human control; controller goals reset at load")
            profile = self.correction.metadata["manual_motion_profile"]
            profile.pop("ui_rotation", None)
            profile.update(normal_rotation=.3, precision_rotation=.1, rotation_default="precision",
                           rotation_selection="independent of translation; actual amplitude stored in each action")
            write_json(self.correction.directory / "metadata.json", self.correction.metadata)
            self._begin_motion_segment()

        def _begin_motion_segment(self):
            from simulation_snapshot import capture
            recorder = self.correction
            segments = recorder.metadata.setdefault("motion_segments", [])
            path = recorder.directory / f"motion-start-{len(segments):06d}.npz"
            capture(self.env, path)
            segments.append(dict(path=path.name, sha256=digest(path),
                                 timeline_start=len(recorder.metadata.get("control_timeline", []))))
            self.motion_capture = MotionCapture(self.env)
            recorder.metadata.update(motion_schema=SCHEMA,
                motion_capture_sha256=digest(ROOT / "scripts/motion_recording.py"),
                joint_names=list(self.env.env.robots[0].robot_joints),
                gripper_joint_names=list(self.env.env.robots[0].gripper_joints))
            recorder.metadata.setdefault("motion_files", [])
            write_json(recorder.directory / "metadata.json", recorder.metadata)

        def resume_correction(self, directory):
            import shutil
            source = Path(directory).expanduser().resolve()
            metadata = json.loads((source / "metadata.json").read_text())
            history = [item for key in ("physics_files", "motion_files", "motion_segments")
                       for item in metadata.get(key, [])]
            for item in history:
                if Path(item["path"]).name != item["path"] or digest(source / item["path"]) != item["sha256"]:
                    raise ValueError("续录物理/关节旁档校验失败；原草稿保留")
            super().resume_correction(directory)
            for item in history:
                shutil.copyfile(source / item["path"], self.correction.directory / item["path"])
            self._begin_motion_segment()

        def _append_motion(self, before, controller, action, target=None):
            recorder = self.correction
            timeline = recorder.metadata["control_timeline"]
            path = recorder.directory / f"motion-{len(timeline) - 1:06d}.npz"
            save_motion(path, before, self.motion_capture.read(), controller, action, target)
            timeline[-1]["motion"] = path.name
            recorder.metadata["motion_files"].append(dict(path=path.name, sha256=digest(path)))
            write_json(recorder.directory / "metadata.json", recorder.metadata)

        def _record_joint_step(self, action, target):
            before = self.motion_capture.read()
            result = super()._record_joint_step(action, target)
            self._append_motion(before, "JOINT_POSITION", action, target)
            return result

        def _manual_steps(self, action, repeats):
            if type(repeats) is not int or not 1 <= repeats <= 20:
                raise ValueError("每次执行步数须为 1～20")
            from correction_data import MAX_FRAMES
            if self.correction_active and (self.correction.count +
                    self.correction.metadata.get("joint_frame_count", 0) + repeats > MAX_FRAMES):
                raise ValueError("本段控制步数已达上限，请保存后继续采集")
            for _ in range(repeats):
                motion_before = self.motion_capture.read()
                before = self.env.get_sim_state().copy()
                nullspace = self.env.env.robots[0].controller.initial_joint.copy()
                super()._manual_steps(action, 1)
                path = self.correction.directory / f"physics-{self.correction.count - 1:06d}.npz"
                np.savez_compressed(path, before=before, after=self.env.get_sim_state(), action=action,
                                    osc_nullspace_initial_joints=nullspace)
                self.correction.metadata["physics_files"].append({"path": path.name, "sha256": digest(path)})
                self.correction.metadata["control_timeline"].append({"controller": "OSC_POSE",
                    "path": f"frame-{self.correction.count - 1:06d}.npz", "physics": path.name})
                self._append_motion(motion_before, "OSC_POSE", action)

        def toggle_run(self):
            raise ValueError("这是人工恢复采集窗口，请使用人工纠正页的摇杆和按钮")

        single_step = toggle_run

        def prepare_corrections(self, *args, **kwargs):
            raise ValueError("先录制并审核恢复片段，再准备本项目的匹配场景训练数据")

    if args.check_takeover:
        from contextlib import nullcontext
        class HeadlessPreview:
            def __init__(self, env):
                self.control_dt = env.env.control_timestep
                self.paused, self.single_step, self.running = True, False, True
            def is_running(self):
                return self.running
            def lock(self):
                return nullcontext()
            def sync(self):
                pass
            def show_joint(self, joint):
                pass
            def close(self):
                self.running = False
        output = ROOT / "evaluations" / f"takeover-check-{time.time_ns()}"
        lab = ManualWorkbench(output, bank_task, root=ROOT, viewer_factory=HeadlessPreview)
        try:
            lab.reset(args.initial_id)
            lab.begin_correction()
            recorder = lab.correction
            # Reading FK must not update live physics, controller goals, caches or RNG.
            from simulation_snapshot import capture
            lab.motion_capture.read()
            observed = output / "after-telemetry-read.npz"
            capture(lab.env, observed)
            initial = recorder.directory / recorder.metadata["motion_segments"][0]["path"]
            with np.load(initial, allow_pickle=False) as first, np.load(observed, allow_pickle=False) as second:
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key])
            lab.manual_hold(1)
            q = lab.env.env.robots[0]._joint_positions.copy()
            lab.manual_joint_jog(3, 2.)
            moved = float(np.rad2deg(lab.env.env.robots[0]._joint_positions[3] - q[3]))
            if moved < .2:
                raise AssertionError("Active takeover elbow did not move")
            lab.manual_hold(1)
            if lab.correction is not recorder or not lab.correction_active:
                raise AssertionError("Takeover was ended or replaced during joint control")
            if (recorder.count != 2 or recorder.metadata["joint_frame_count"] != 12
                    or lab.env.env.action_dim != 7):
                raise AssertionError("Mixed recording/action contract differs")
            previous = np.load(recorder.directory / "initial_sim_state.npy")
            motion_files = recorder.metadata["motion_files"]
            if len(motion_files) != 14:
                raise AssertionError("Expected telemetry for every control step")
            robot = lab.env.env.robots[0]
            nq = lab.env.sim.model.nq
            for item in recorder.metadata["control_timeline"]:
                frame = recorder.directory / item.get("physics", item["path"])
                motion = recorder.directory / item["motion"]
                if digest(motion) != next(x["sha256"] for x in motion_files if x["path"] == motion.name):
                    raise AssertionError("Motion archive checksum mismatch")
                with np.load(frame, allow_pickle=False) as data, np.load(motion, allow_pickle=False) as telemetry:
                    np.testing.assert_array_equal(data["before"], previous)
                    previous = data["after"].copy()
                    for prefix in ("before", "after"):
                        physical = data[prefix]
                        np.testing.assert_array_equal(telemetry[f"{prefix}_joint_pos"],
                            physical[1:1 + nq][robot._ref_joint_pos_indexes])
                        np.testing.assert_array_equal(telemetry[f"{prefix}_joint_vel"],
                            physical[1 + nq:][robot._ref_joint_vel_indexes])
                        np.testing.assert_equal(telemetry[f"{prefix}_sim_time"], physical[0])
                    np.testing.assert_array_equal(telemetry["raw_action"], data["action"])
                    if item["controller"] == "JOINT_POSITION":
                        np.testing.assert_array_equal(telemetry["target_qpos"], data["target_qpos"])
                    assert str(telemetry["controller"]) == item["controller"]
            np.testing.assert_array_equal(previous, lab.env.get_sim_state())
            try:
                lab.finish_correction(True)
            except ValueError:
                pass
            else:
                raise AssertionError("Mixed success gate accepted unfinished recovery")
            lab.finish_correction(False)
            report = dict(ready=True, directory=str(recorder.directory), elbow_degrees=moved,
                osc_frames=recorder.count, joint_frames=recorder.metadata["joint_frame_count"],
                continuous_physics=True, same_take=True, restored_osc_dim=lab.env.env.action_dim,
                motion_frames=len(motion_files), motion_readback_matches_physics=True,
                telemetry_read_preserves_full_snapshot=True, full_initial_snapshot=True,
                synthetic_draft=True, training_ready=False)
            # OSC draft continuation creates a new take: its sidecars must follow
            # the inherited image/action prefix rather than refer to missing files.
            lab.begin_correction()
            lab.manual_hold(10)
            draft = lab.finish_correction(False)
            lab.resume_correction(draft)
            lab.manual_hold(1)
            resumed = lab.correction
            if len(resumed.metadata["motion_files"]) != 11 or len(resumed.metadata["motion_segments"]) != 2:
                raise AssertionError("OSC continuation lost its motion history")
            for key in ("physics_files", "motion_files", "motion_segments"):
                for item in resumed.metadata[key]:
                    if digest(resumed.directory / item["path"]) != item["sha256"]:
                        raise AssertionError("Resumed sidecar missing or changed")
            lab.finish_correction(False)
            report.update(osc_resume_sidecars_verified=True, resumed_directory=str(resumed.directory))
            write_json(output / "check.json", report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            lab.close()
        return

    import tkinter as tk
    output = ROOT / "evaluations" / f"manual-recovery-{time.time_ns()}"
    lab = ManualWorkbench(output, bank_task, root=ROOT, viewer_factory=GuidedPreview)
    root = tk.Tk()
    panel = GuidedPanel(root, lab)
    root.title("人工恢复采集 · 碗放盘子 · 训练失败场景")
    for tab in panel.notebook.tabs():
        if panel.notebook.tab(tab, "text") == "模型与训练":
            panel.notebook.hide(tab)
    panel.correction_split_combo.configure(values=("train",))
    panel.notebook.select(panel.correction_page)
    try:
        panel.layout_var.set(str(args.initial_id))
        lab.reset(args.initial_id)
        lab.begin_correction()
        lab.status = "已恢复失败场景并暂停，人工接管录制已开启；等待你的操作"
        panel._refresh()
        print(json.dumps({"ready": True, "paused": lab.view.paused, "entry": entries[args.initial_id]["id"],
                          "correction": lab.correction_state(), "output": str(output)}, ensure_ascii=False), flush=True)
        root.mainloop()
    finally:
        lab.close()


if __name__ == "__main__":
    main()
