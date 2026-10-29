"""Human Cartesian and joint controls sharing one absolute-joint action contract.

Raw commands are float64 radians7 + normalized gripper1. Measured qpos is
recorded separately and is never substituted for the command training label.
"""
import time
from pathlib import Path

import numpy as np

from joint_action_contract import KD, KP, absolute_joint_controller, validate_action
from motion_recording import MotionCapture

SCHEMA = "absolute-joint-human-v1"


def validate_joint_frame(frame, limits):
    for key, shape in (("state", (9,)), ("post_state", (9,)), ("action", (8,)),
                       ("post_bowl_pos", (3,))):
        if frame[key].shape != shape or not np.isfinite(frame[key]).all():
            raise ValueError(f"Invalid joint frame: {key}")
    validate_action(frame["action"], limits)
    for prefix, state in (("before", "state"), ("after", "post_state")):
        for key, shape in (("joint_pos", (7,)), ("joint_vel", (7,)),
                           ("gripper_pos", (2,)), ("gripper_vel", (2,)),
                           ("eef_pos", (3,)), ("eef_rotmat", (3, 3)), ("sim_time", ())):
            value = frame[f"{prefix}_{key}"]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid joint telemetry: {prefix}_{key}")
        if not np.array_equal(frame[state], np.r_[frame[f"{prefix}_joint_pos"], frame[f"{prefix}_gripper_pos"]]):
            raise ValueError("Measured state differs from joint telemetry")
    for key in ("physics_before", "physics_after"):
        if frame[key].ndim != 1 or not np.isfinite(frame[key]).all():
            raise ValueError("Invalid physical state")
    for key in ("done", "finger_contact"):
        if frame[key].shape != () or frame[key].dtype != np.bool_:
            raise ValueError("Expected boolean observation gate")
    for key in ("image", "wrist_image"):
        if frame[key].shape != (256, 256, 3) or frame[key].dtype != np.uint8:
            raise ValueError("Invalid joint camera image")


def cartesian_target(qpos, jacobian, command, limits):
    """Bounded differential IK, world XYZ/meters and rotation/radians.

    This is an operator input adapter, not a learned policy or collision planner.
    Damping keeps an unreachable request finite; the operator still avoids contact.
    """
    qpos, jacobian, command, limits = map(np.asarray, (qpos, jacobian, command, limits))
    if (qpos.shape != (7,) or jacobian.shape != (6, 7) or command.shape != (6,)
            or limits.shape != (7, 2) or not np.isfinite(
                np.r_[qpos, jacobian.ravel(), command, limits.ravel()]).all()
            or np.any(limits[:, 1] - limits[:, 0] <= .01) or np.any(np.abs(command) > 1)):
        raise ValueError("Invalid Cartesian input / seven-joint kinematics")
    delta = command * np.array([.05, .05, .05, .5, .5, .5])
    dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .03 ** 2 * np.eye(6), delta)
    dq /= max(1., float(np.max(np.abs(dq))) / .08)
    return np.clip(qpos + dq, limits[:, 0] + .005, limits[:, 1] - .005)


def load_joint_frames(directory, metadata):
    from correction_data import MAX_FRAMES, digest
    if metadata.get("recording_schema") != SCHEMA:
        raise ValueError("Expected dedicated absolute joint recording, not legacy OSC")
    items = metadata.get("files", [])
    count = metadata.get("frame_count")
    if type(count) is not int or not 0 <= count <= MAX_FRAMES or len(items) != count:
        raise ValueError("Joint recording count differs")
    expected = [f"joint-step-{i:06d}.npz" for i in range(count)]
    if sorted(p.name for p in directory.glob("joint-step-*.npz")) != expected:
        raise ValueError("Joint frame inventory differs")
    result = []
    for name, item in zip(expected, items, strict=True):
        path = directory / name
        if item.get("path") != name or path.is_symlink() or digest(path) != item.get("sha256"):
            raise ValueError("Joint frame checksum differs")
        with np.load(path, allow_pickle=False) as data:
            frame = {k: data[k] for k in data.files}
        validate_joint_frame(frame, metadata["joint_limits"])
        result.append(frame)
    return result


class JointRecorder:
    """Replace an empty legacy recorder; keep its source and initial snapshot."""
    def __init__(self, original, limits):
        from correction_data import digest, write_json
        if original.count:
            raise ValueError("Cannot relabel existing OSC frames")
        self.directory, self.metadata = original.directory, original.metadata
        self.count, self.active = 0, True
        self.metadata.update(
            recording_schema=SCHEMA, action_contract="absolute arm target radians7 + gripper1 (+close/-open)",
            state_contract="measured joint radians7 + two finger positions meters2",
            motion_schema=dict(version=SCHEMA, action="actual absolute joint target radians7 + gripper1",
                               joint_pos="radian", joint_vel="radian/second", gripper_pos="meter",
                               gripper_vel="meter/second", eef_pos="world frame meter",
                               eef_rotmat="world frame, local axes in columns", sim_time="second"),
            unified_sources={name: digest(Path(__file__).parent / name) for name in
                             ("unified_joint_recording.py", "joint_action_contract.py", "motion_recording.py")},
            controller_kp=KP, controller_kd=KD, joint_limits=np.asarray(limits).tolist(),
            joint_frame_count=0, training_ready=False,
            training_gate="dedicated 9D/8D adapter and human review required; never use OSC loader",
            input_adapter=dict(translation_m=.05, rotation_rad=.5, damping=.03,
                               maximum_joint_increment_rad=.08),
            command_source="actual action passed to env.step; not reconstructed from measured motion")
        write_json(self.directory / "metadata.json", self.metadata)

    def append(self, before, after, action, images, physics_before, physics_after, done,
               finger_contact, bowl_pos, kind, operator_input):
        from correction_data import MAX_FRAMES, digest, write_json
        if not self.active or self.count >= MAX_FRAMES:
            raise ValueError("Recording closed or full")
        validate_action(action, self.metadata["joint_limits"])
        frame = dict(action=np.asarray(action, dtype=np.float64),
                     state=np.r_[before["joint_pos"], before["gripper_pos"]],
                     post_state=np.r_[after["joint_pos"], after["gripper_pos"]],
                     physics_before=physics_before, physics_after=physics_after,
                     done=np.asarray(done, dtype=bool), finger_contact=np.asarray(finger_contact, dtype=bool),
                     post_bowl_pos=bowl_pos, operator_kind=np.asarray(kind),
                     operator_input=np.asarray(operator_input), wall_time_ns=np.asarray(time.time_ns()), **images)
        frame.update({f"{prefix}_{k}": v for prefix, motion in (("before", before), ("after", after))
                      for k, v in motion.items()})
        validate_joint_frame(frame, self.metadata["joint_limits"])
        path = self.directory / f"joint-step-{self.count:06d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **frame)
        self.count += 1
        self.metadata["frame_count"] = self.count
        self.metadata["unified_frame_count"] = self.count
        self.metadata["files"].append(dict(path=path.name, sha256=digest(path)))
        write_json(self.directory / "metadata.json", self.metadata)

    def finish(self, approved):
        from correction_data import write_json
        if not self.active or type(approved) is not bool:
            raise ValueError("Expected active recording and explicit approval")
        checks = None
        if approved:
            tail = load_joint_frames(self.directory, self.metadata)[-10:]
            checks = dict(ten_frames=len(tail) == 10,
                          native_goal_last_ten=len(tail) == 10 and all(bool(f["done"]) for f in tail),
                          fingers_open_last_ten=bool(tail) and all(np.abs(f["post_state"][7:]).sum() >= .06 for f in tail),
                          open_command_last_ten=bool(tail) and all(f["action"][-1] < 0 for f in tail),
                          no_finger_contact_last_ten=bool(tail) and all(not bool(f["finger_contact"]) for f in tail),
                          eef_clear=bool(tail) and np.linalg.norm(
                              tail[-1]["after_eef_pos"] - tail[-1]["post_bowl_pos"]) >= .10)
            checks = {k: bool(v) for k, v in checks.items()}
            if not all(checks.values()):
                raise ValueError("尚未完成放置、松爪、抬离并保持10步：" + ", ".join(k for k, v in checks.items() if not v))
        self.metadata.update(status="approved" if approved else "draft", observation_gates=checks,
                             approval_type="manual_user_with_joint_observation_gates" if approved else None,
                             official_success_metric=False)
        write_json(self.directory / "metadata.json", self.metadata)
        self.active = False
        return self.directory


class UnifiedJointMixin:
    unified_joint_mode = True
    _joint_context = None

    def begin_correction(self, split="train"):
        super().begin_correction(split)
        try:
            robot = self.env.env.robots[0]
            limits = self.env.sim.model.jnt_range[robot.joint_indexes].copy()
            self.correction = JointRecorder(self.correction, limits)
            self._joint_context = absolute_joint_controller(self.env)
            _, self._joint_limits = self._joint_context.__enter__()
            self._joint_target = robot._joint_positions.copy()
            self.motion_capture = MotionCapture(self.env)
            self.correction_status = "统一关节录制中：摇杆、旋转、J1～J7 和夹爪均保存真实8维关节指令"
        except Exception:
            self.finish_correction(False)
            raise

    def _execute_joint(self, target, grip, kind, operator_input):
        from correction_data import MAX_FRAMES
        from simulation_lab import TARGET, state_digest
        self._ensure_not_training()
        if not self.correction_active or self.view is None or not self.view.is_running():
            raise ValueError("请先开始统一关节接管录制")
        if self.correction.count >= MAX_FRAMES:
            raise ValueError("本段已达上限；请保存草稿后开始新段")
        action = validate_action(np.r_[target, grip], self._joint_limits)
        self.view.paused, self.view.single_step, self.step_requested = True, False, False
        with self.view.lock():
            physics_before = self.env.get_sim_state().copy()
            if state_digest(physics_before) != self.manual_state_hash:
                raise ValueError("原生窗口修改了物理状态；请保留草稿并重新接管")
            self.obs = self.env.regenerate_obs_from_state(physics_before)
            images = self.camera_images()
            before = self.motion_capture.read()
            self.obs, _, _, _ = self.env.step(action.tolist())
            after = self.motion_capture.read()
            robot = self.env.env.robots[0]
            bowl = self.env.env.get_object(TARGET)
            self.correction.append(before, after, action, images, physics_before,
                self.env.get_sim_state().copy(), bool(self.env.check_success()),
                bool(self.env.env.check_contact(robot.gripper.contact_geoms, bowl)),
                self.obs[f"{TARGET}_pos"].copy(), kind, operator_input)
            self._joint_target = action[:7].copy()
            self.manual_grip = float(grip)
            self.manual_state_hash = self.known_state_hash = state_digest(self.env.get_sim_state())
        self.view.sync()
        self.correction_status = f"统一关节录制：{self.correction.count} 步；含实际指令、双相机和完整运动状态"

    def _manual_steps(self, action, repeats):
        action = np.asarray(action, dtype=float)
        if (type(repeats) is not int or not 1 <= repeats <= 20 or action.shape != (7,)
                or not np.isfinite(action).all() or np.any(np.abs(action) > 1)):
            raise ValueError("Invalid manual input")
        if not self.correction_active:
            raise ValueError("请先开始统一关节接管录制")
        import mujoco
        for _ in range(repeats):
            target = self._joint_target.copy()
            if np.any(action[:6]):
                motion = self.motion_capture.read()
                fk, model = self.motion_capture.fk, self.env.sim.model._model
                mujoco.mj_comPos(model, fk)
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                robot = self.env.env.robots[0]
                mujoco.mj_jacSite(model, fk, jp, jr, robot.eef_site_id)
                jac = np.vstack((jp, jr))[:, robot._ref_joint_vel_indexes]
                target = cartesian_target(motion["joint_pos"], jac, action[:6], self._joint_limits)
            self._execute_joint(target, action[-1], "cartesian" if np.any(action[:6]) else "grip_or_hold", action)

    def manual_joint_jog(self, joint, degrees):
        from manual_joint_control import joint_target
        if not self.correction_active:
            raise ValueError("请先开始统一关节接管录制")
        target = joint_target(self.env.env.robots[0]._joint_positions,
                              self._joint_limits, joint, degrees)
        for _ in range(12):
            self._execute_joint(target, self.manual_grip, "joint", [joint, degrees])
        self.view.show_joint(joint)

    def finish_correction(self, approved):
        try:
            return super().finish_correction(approved)
        finally:
            if not self.correction_active and self._joint_context is not None:
                self._joint_context.__exit__(None, None, None)
                self._joint_context = None
                controller = self.env.env.robots[0].controller
                controller.update(force=True)
                controller.update_initial_joints(self.env.env.robots[0]._joint_positions.copy())
                controller.reset_goal()

    def resume_correction(self, directory):
        raise ValueError("统一关节草稿已完整保存；本版请从当前姿态开始新段，不使用旧OSC续录器")


def unified_panel(panel_class):
    """Change mode hints without changing the checksum-pinned legacy joint UI."""
    class UnifiedPanel(panel_class):
        def __init__(self, root, lab):
            super().__init__(root, lab)
            for label in self.joint_page.grid_slaves(row=10):
                label.configure(text="统一关节模式：摇杆、旋转、调肘和夹爪均执行并记录8维关节目标。\n"
                                "保存双相机、关节和末端状态；审核后使用专用关节训练入口。")

        def _refresh(self):
            super()._refresh()
            if hasattr(self, "correction_resume_button"):
                self.correction_resume_button.configure(state="disabled", text="关节草稿保留，续录待接入")
                self.prepare_corrections_button.configure(state="disabled", text="关节训练转换待接入")
            if hasattr(self, "joint_status") and self.lab.correction_active:
                self.joint_status.set(f"统一关节记录：{self.lab.correction.count} 步（含摇杆、旋转、关节和夹爪）")
    return UnifiedPanel
