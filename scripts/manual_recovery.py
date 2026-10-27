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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-id", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--category", choices=("placement_failure", "empty_grasp", "object_offset"),
                        default="placement_failure")
    parser.add_argument("--check", action="store_true", help="Restore and verify without opening windows")
    parser.add_argument("--check-joints", action="store_true", help="Headless physical joint/OSC switching smoke test")
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "osmesa" if args.check or args.check_joints else "glx"
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
    from correction_data import write_json
    from correction_workbench import CorrectionWorkbench
    from manual_rotation_guide import guided_classes
    from manual_joint_control import JointSetupMixin, joint_classes
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
                launcher_sha256=digest(__file__), physics_files=[],
                rotation_guide_sha256=digest(ROOT / "scripts/manual_rotation_guide.py"),
                joint_control_sha256=digest(ROOT / "scripts/manual_joint_control.py"),
                osc_nullspace_initial_joints=self.env.env.robots[0].controller.initial_joint.tolist(),
                physics_scope="full MuJoCo pre/post state per human control; controller goals reset at load")
            write_json(self.correction.directory / "metadata.json", self.correction.metadata)

        def _manual_steps(self, action, repeats):
            if type(repeats) is not int or not 1 <= repeats <= 20:
                raise ValueError("每次执行步数须为 1～20")
            for _ in range(repeats):
                before = self.env.get_sim_state().copy()
                super()._manual_steps(action, 1)
                path = self.correction.directory / f"physics-{self.correction.count - 1:06d}.npz"
                np.savez_compressed(path, before=before, after=self.env.get_sim_state(), action=action)
                self.correction.metadata["physics_files"].append({"path": path.name, "sha256": digest(path)})
                write_json(self.correction.directory / "metadata.json", self.correction.metadata)

        def toggle_run(self):
            raise ValueError("这是人工恢复采集窗口，请使用人工纠正页的摇杆和按钮")

        single_step = toggle_run

        def prepare_corrections(self, *args, **kwargs):
            raise ValueError("先录制并审核恢复片段，再准备本项目的匹配场景训练数据")

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
