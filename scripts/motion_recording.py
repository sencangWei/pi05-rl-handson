"""Read-only physical joint/EEF telemetry; commands retain their controller type."""
import numpy as np

SCHEMA = {
    "version": 1, "joint_pos": "radian", "joint_vel": "radian/second",
    "gripper_pos": "meter", "gripper_vel": "meter/second", "sim_time": "second",
    "eef_pos": "world frame, meter", "eef_rotmat": "world frame, local axes in columns",
    "OSC_POSE": "raw normalized delta xyz3/rotation3/grip1, not measured displacement",
    "JOINT_POSITION": "target_qpos is absolute arm radians; raw_action first7 are placeholders",
    "training": "Measured motion is diagnostic telemetry, not an automatically converted action label"}


class MotionCapture:
    def __init__(self, env):
        import mujoco
        self.env = env
        self.fk = mujoco.MjData(env.sim.model._model)

    def read(self):
        """Compute FK on separate data, leaving live simulation/controller untouched."""
        import mujoco
        env, fk = self.env, self.fk
        robot, data = env.env.robots[0], env.sim.data._data
        mujoco.mj_copyData(fk, env.sim.model._model, data)
        mujoco.mj_kinematics(env.sim.model._model, fk)
        return {
            "joint_pos": data.qpos[robot._ref_joint_pos_indexes].copy(),
            "joint_vel": data.qvel[robot._ref_joint_vel_indexes].copy(),
            "gripper_pos": data.qpos[robot._ref_gripper_joint_pos_indexes].copy(),
            "gripper_vel": data.qvel[robot._ref_gripper_joint_vel_indexes].copy(),
            "eef_pos": fk.site_xpos[robot.eef_site_id].copy(),
            "eef_rotmat": fk.site_xmat[robot.eef_site_id].reshape(3, 3).copy(),
            "sim_time": np.asarray(data.time)}


def save_motion(path, before, after, controller, action, target=None):
    action = np.asarray(action)
    if controller == "OSC_POSE":
        valid = action.shape == (7,) and target is None
    elif controller == "JOINT_POSITION":
        valid = action.shape == (8,) and np.asarray(target).shape == (7,)
    else:
        valid = False
    if not valid:
        raise ValueError("Motion command does not match controller contract")
    fields = {"controller": np.asarray(controller), "raw_action": action}
    if target is not None:
        fields["target_qpos"] = np.asarray(target)
    fields.update({f"{prefix}_{key}": value
                   for prefix, state in (("before", before), ("after", after))
                   for key, value in state.items()})
    if any(not np.isfinite(value).all() for key, value in fields.items() if key != "controller"):
        raise ValueError("Nonfinite motion telemetry")
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **fields)
