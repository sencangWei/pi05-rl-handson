"""Joint and end-effector takeover, with explicit controller action provenance."""
import json
from pathlib import Path
import time

import numpy as np

JOINT_NAMES = ("基座", "肩部", "上臂", "肘部", "前臂", "腕部", "末端")


def joint_target(qpos, limits, joint, degrees):
    qpos, limits = np.asarray(qpos), np.asarray(limits)
    if qpos.shape != (7,) or limits.shape != (7, 2):
        raise ValueError("只支持当前七关节机械臂")
    if type(joint) is not int or joint not in range(7):
        raise ValueError("请选择 J1～J7")
    if not np.isfinite(degrees) or not .5 <= abs(degrees) <= 5:
        raise ValueError("每次关节转角须为 0.5～5 度")
    if not np.isfinite(qpos).all() or not np.isfinite(limits).all():
        raise ValueError("关节状态无效")
    target = qpos.copy()
    target[joint] = np.clip(qpos[joint] + np.deg2rad(degrees),
                            limits[joint, 0] + .005, limits[joint, 1] - .005)
    return target


def physical_joint_jog(env, joint, degrees, grip, step_callback=None):
    """Run bounded JOINT_POSITION control; always restore OSC, even on failure.

    No qpos writes, disabled contacts, or fake OSC labels. Other joints receive
    their current position as a hold target, but can deflect under contact.
    """
    from robosuite.controllers.joint_pos import JointPositionController
    from robosuite.utils.buffers import DeltaBuffer

    base = env.env
    robot = base.robots[0]
    if len(base.robots) != 1 or robot.controller.name != "OSC_POSE":
        raise ValueError("关节调姿仅用于当前单臂 OSC_POSE 场景")
    before = env.get_sim_state().copy()
    positions = robot._joint_positions.copy()
    target = joint_target(positions, env.sim.model.jnt_range[robot.joint_indexes], joint, degrees)
    osc, old_dim, old_actions = robot.controller, base._action_dim, robot.recent_actions

    class FixedTargetController(JointPositionController):
        def set_goal(self, action, set_qpos=None):
            super().set_goal(action, set_qpos=target)

    controller = FixedTargetController(
        sim=env.sim, eef_name=robot.gripper.important_sites["grip_site"],
        joint_indexes={"joints": robot.joint_indexes, "qpos": robot._ref_joint_pos_indexes,
                       "qvel": robot._ref_joint_vel_indexes},
        actuator_range=robot.torque_limits, kp=150, policy_freq=robot.control_freq,
        qpos_limits=env.sim.model.jnt_range[robot.joint_indexes].T)
    states = [before]
    action = np.r_[np.zeros(7), float(grip)]
    try:
        robot.controller = controller
        base._action_dim = robot.action_dim
        robot.recent_actions = DeltaBuffer(dim=robot.action_dim)
        for _ in range(12):
            obs, _, _, _ = (env.step(action.tolist()) if step_callback is None
                             else step_callback(action, target))
            states.append(env.get_sim_state().copy())
            if not np.isfinite(states[-1]).all():
                raise ValueError("关节调姿出现非有限物理状态，已停止")
    finally:
        robot.controller, base._action_dim, robot.recent_actions = osc, old_dim, old_actions
        osc.update(force=True)
        osc.update_initial_joints(robot._joint_positions.copy())
        osc.reset_goal()
    return obs, {"states": np.asarray(states), "target_qpos": target, "action": action,
                 "before_qpos": positions, "after_qpos": robot._joint_positions.copy()}


class JointSetupMixin:
    def correction_state(self):
        state = super().correction_state()
        joint_frames = self.correction.metadata.get("joint_frame_count", 0) if self.correction else 0
        if joint_frames:
            osc_frames = state["frames"]
            state["frames"] += joint_frames
            state["status"] += f"；末端 {osc_frames} 步，关节 {joint_frames} 步"
        return state

    def _record_joint_step(self, action, target):
        from correction_data import MAX_FRAMES, digest, write_json
        from simulation_lab import TARGET, state_digest
        recorder = self.correction
        meta = recorder.metadata
        if recorder.count + meta.get("joint_frame_count", 0) >= MAX_FRAMES:
            raise ValueError("本段控制步数已达上限，请保存后继续采集")
        before = self.env.get_sim_state().copy()
        if state_digest(before) != self.manual_state_hash:
            raise ValueError("物理状态被外部修改，请先保存当前草稿")
        self.obs = self.env.regenerate_obs_from_state(before)
        state, images = self._state8(self.obs), self.camera_images()
        result = self.env.step(action.tolist())
        obs, _, done, _ = result
        after = self.env.get_sim_state().copy()
        self.obs = obs
        self.manual_state_hash = self.known_state_hash = state_digest(after)
        if not np.isfinite(after).all():
            raise ValueError("关节控制出现非有限物理状态")
        index = meta.get("joint_frame_count", 0)
        path = recorder.directory / f"joint-frame-{index:06d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, state=state, action=action, target_qpos=target,
                post_state=self._state8(obs), post_bowl_pos=obs[f"{TARGET}_pos"],
                done=bool(done), before=before, after=after, **images)
        if not index:
            meta["osc_action_contract"] = meta["action_contract"]
            meta["action_contract"] = "MIXED OSC_POSE / JOINT_POSITION absolute joint target"
            meta["training_ready"] = False
            meta["training_block_reason"] = "需适配混合控制契约；不能直接当作7维OSC训练"
        meta.setdefault("joint_files", []).append({"path": path.name, "sha256": digest(path)})
        meta["joint_frame_count"] = index + 1
        meta["last_joint_at_osc_frame"] = recorder.count
        meta.setdefault("control_timeline", []).append({"controller": "JOINT_POSITION", "path": path.name})
        write_json(recorder.directory / "metadata.json", meta)
        return result

    def manual_joint_jog(self, joint, degrees):
        self._ensure_not_training()
        if self.env is None or self.view is None or not self.view.is_running():
            raise ValueError("请先加载场景")
        from simulation_lab import state_digest
        recording = self.correction_active
        if recording and state_digest(self.env.get_sim_state()) != self.manual_state_hash:
            raise ValueError("物理状态被外部修改，请先保存当前草稿")
        self.view.paused, self.view.single_step, self.step_requested = True, False, False
        with self.view.lock():
            try:
                self.obs, trace = physical_joint_jog(self.env, joint, degrees, self.manual_grip,
                    step_callback=self._record_joint_step if recording else None)
            finally:
                self.known_state_hash = state_digest(self.env.get_sim_state())
        path = self.output / f"joint-setup-{time.time_ns()}.npz"
        np.savez_compressed(path, **trace, training_sample=False, joint=joint,
                            degrees=degrees, contract="JOINT_POSITION absolute target")
        moved = np.rad2deg(trace["after_qpos"][joint] - trace["before_qpos"][joint])
        suffix = "已计入当前接管记录，可直接继续 XYZ、旋转和夹爪操作" if recording else "调姿已留档"
        self.status = f"J{joint + 1} 实际转动 {moved:+.2f}°；{suffix}"
        self.event("joint_takeover" if recording else "joint_setup", joint=joint,
                   degrees=degrees, actual_degrees=float(moved), trace=str(path))
        self.view.show_joint(joint)

    def finish_correction(self, approved):
        if approved and self.correction_active:
            last = self.correction.metadata.get("last_joint_at_osc_frame")
            if last is not None and self.correction.count - last < 10:
                raise ValueError("关节调整后还需完成放置、松爪、抬离，并保持10步验证成功")
        return super().finish_correction(approved)

    def resume_correction(self, directory):
        meta = json.loads((Path(directory) / "metadata.json").read_text())
        if meta.get("joint_frame_count", 0):
            raise ValueError("本段含关节控制，不能用旧OSC重放器续录；原始完整记录已保留")
        return super().resume_correction(directory)


def joint_classes(preview_class, panel_class):
    """Extend the project UI without modifying its frozen upstream snapshot."""
    import tkinter as tk
    from tkinter import ttk

    class JointPreview(preview_class):
        selected_joint = None

        def sync(self):
            if self.selected_joint is None:
                return super().sync()
            import mujoco
            from viewer_controls import paint_world_axes
            with self.handle.lock():
                scene = self.handle.user_scn
                paint_world_axes(scene, self.env.sim.data.site_xpos[self._eef_site_id].copy())
                joint_id = self.env.env.robots[0].joint_indexes[self.selected_joint]
                origin = self.env.sim.data._data.xanchor[joint_id].copy()
                axis = self.env.sim.data._data.xaxis[joint_id].copy()
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), origin,
                                   np.eye(3).ravel(), np.array([1., .6, 0., 1.], dtype=np.float32))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .008,
                                    origin - axis * .12, origin + axis * .16)
                geom.label = f"J{self.selected_joint + 1} (+ right-hand)"
                scene.ngeom += 1
            self.handle.sync()

        def show_joint(self, joint):
            self.selected_joint = joint
            self.sync()

        def show_rotation(self, axis, sign):
            self.selected_joint = None
            super().show_rotation(axis, sign)

    class JointPanel(panel_class):
        def __init__(self, root, lab):
            super().__init__(root, lab)
            self.joint_page = ttk.Frame(self.notebook, padding=12)
            self.notebook.add(self.joint_page, text="关节接管 J1～J7")
            ttk.Label(self.joint_page, text="逐节转动机械臂：J1 在底座，J4 在肘部，J6/J7 靠近夹爪。\n"
                      "指向按钮会标出实际关节轴，橙色箭头指正轴；正转遵循右手规则。\n"
                      "调姿会移动夹爪位置，仍有碰撞；遇阻请换方向，不要连续硬压。",
                      wraplength=690, justify="left").grid(row=0, column=0, columnspan=4, sticky="w", pady=8)
            ttk.Button(self.joint_page, text="继续当前接管（无需结束录制）",
                       command=lambda: self._perform(self._joint_setup)).grid(
                           row=1, column=0, columnspan=4, sticky="w", pady=8)
            self.joint_degrees = tk.StringVar(value="2")
            ttk.Label(self.joint_page, text="每次目标转角（度）").grid(row=2, column=0, sticky="w")
            ttk.Spinbox(self.joint_page, from_=.5, to=5, increment=.5, width=5,
                        textvariable=self.joint_degrees).grid(row=2, column=1, sticky="w")
            self.joint_buttons, self.joint_values = [], []
            for joint, name in enumerate(JOINT_NAMES):
                row = joint + 3
                ttk.Label(self.joint_page, text=f"J{joint + 1} {name}").grid(row=row, column=0, sticky="w", pady=8)
                for column, sign in ((1, -1), (2, 1)):
                    button = ttk.Button(self.joint_page, text=f"J{joint + 1} {'−' if sign < 0 else '+'}", width=8,
                                        command=lambda j=joint, s=sign: self._perform(
                                            lambda: self.lab.manual_joint_jog(j, s * float(self.joint_degrees.get()))))
                    button.grid(row=row, column=column, padx=6)
                    button.bind("<Enter>", lambda event, j=joint: self._show_joint(j))
                    self.joint_buttons.append(button)
                value = tk.StringVar(value="—")
                self.joint_values.append(value)
                ttk.Label(self.joint_page, textvariable=value, width=12).grid(row=row, column=3)
            ttk.Label(self.joint_page, text="录制中可以直接调整关节，再回人工纠正页继续移动和松爪。\n"
                      "关节与末端控制按实际类型完整保存；混合记录需适配后才能训练。",
                      wraplength=690, justify="left").grid(row=10, column=0, columnspan=4, sticky="w", pady=16)
            self.joint_record_button = ttk.Button(self.joint_page, text="从当前姿态开始新采集",
                                                  command=lambda: self._perform(self._record_after_setup))
            self.joint_record_button.grid(row=11, column=0, columnspan=4, sticky="w")
            ttk.Button(self.joint_page, text="保留当前草稿，重载失败场景并重新采集",
                       command=lambda: self._perform(self._restart_recovery)).grid(
                           row=12, column=0, columnspan=4, sticky="w", pady=12)
            self.joint_status = tk.StringVar()
            ttk.Label(self.joint_page, textvariable=self.joint_status, wraplength=690,
                      justify="left").grid(row=13, column=0, columnspan=4, sticky="w")

        def _show_joint(self, joint):
            if self.lab.view is not None:
                self.lab.view.show_joint(joint)

        def _joint_setup(self):
            self.translation_joystick.stop()
            if not self.lab.correction_active:
                self.lab.begin_correction()
            self.lab.status = "当前接管继续录制，直接使用 J1～J7 ±；无需结束或重开"

        def _record_after_setup(self):
            self.lab.begin_correction()
            self.notebook.select(self.correction_page)

        def _restart_recovery(self):
            self.translation_joystick.stop()
            if self.lab.correction_active:
                self.lab.finish_correction(False)
            self.lab.reset(self.lab.layout_id)
            self._record_after_setup()

        def _refresh(self):
            super()._refresh()
            if not hasattr(self, "joint_buttons"):
                return
            ready = self.lab.env is not None and self.lab.view is not None and self.lab.view.is_running()
            for button in self.joint_buttons:
                button.configure(state="normal" if ready else "disabled")
            self.joint_record_button.configure(state="normal" if ready and not self.lab.correction_active else "disabled")
            counts = ""
            if self.lab.correction_active:
                counts = (f"\n当前同一段记录：末端 {self.lab.correction.count} 步，"
                          f"关节 {self.lab.correction.metadata.get('joint_frame_count', 0)} 步")
            self.joint_status.set(self.lab.status + counts)
            if self.lab.env is not None:
                positions = np.rad2deg(self.lab.env.env.robots[0]._joint_positions)
                for value, angle in zip(self.joint_values, positions):
                    value.set(f"{angle:+.1f}°")

    return JointPreview, JointPanel
