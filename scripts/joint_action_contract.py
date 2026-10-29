"""Absolute seven-joint targets plus gripper; distinct from seven-dimensional OSC."""
from contextlib import contextmanager

import numpy as np

KP = 150.
KD = 2 * np.sqrt(KP)


def reference_target(qpos, qvel, limits, velocity_compensation=False):
    """A candidate expert target, NOT the original controller command.

    PD damps velocity even when following a moving reference. kd/kp * velocity
    compensates that term. Contact and acceleration tracking still need testing.
    """
    qpos, qvel, limits = (np.asarray(x, dtype=float) for x in (qpos, qvel, limits))
    if (qpos.shape != (7,) or qvel.shape != (7,) or limits.shape != (7, 2) or
            not np.isfinite(np.r_[qpos, qvel, limits.ravel()]).all() or
            np.any(limits[:, 1] - limits[:, 0] <= .01)):
        raise ValueError("Expected finite seven-joint reference and ordered limits")
    demand = qpos + (KD / KP * qvel if velocity_compensation else 0.)
    return np.clip(demand, limits[:, 0] + .005, limits[:, 1] - .005), demand


def validate_action(action, limits):
    action, limits = np.asarray(action, dtype=float), np.asarray(limits, dtype=float)
    if (action.shape != (8,) or limits.shape != (7, 2) or
            not np.isfinite(np.r_[action, limits.ravel()]).all() or
            np.any(limits[:, 0] >= limits[:, 1]) or abs(action[-1]) > 1 or
            np.any(action[:7] < limits[:, 0]) or np.any(action[:7] > limits[:, 1])):
        raise ValueError("Expected absolute joint radians7 within limits + normalized gripper1")
    return action


@contextmanager
def absolute_joint_controller(env):
    """Diagnostic simulation controller. Never write qpos or disable contacts.

    Discard the environment after use: the original OSC's caches are not reanchored.
    """
    from robosuite.controllers.joint_pos import JointPositionController
    from robosuite.utils.buffers import DeltaBuffer

    base, robot = env.env, env.env.robots[0]
    original, dim, recent = robot.controller, base._action_dim, robot.recent_actions
    if len(base.robots) != 1 or original.name != "OSC_POSE":
        raise ValueError("Expected single-arm OSC simulation")
    limits = env.sim.model.jnt_range[robot.joint_indexes].copy()

    class AbsoluteController(JointPositionController):
        def set_goal(self, action, set_qpos=None):
            target = np.asarray(action, dtype=float)
            validate_action(np.r_[target, 0.], limits)
            super().set_goal(np.zeros(7), set_qpos=target)

    controller = AbsoluteController(
        sim=env.sim, eef_name=robot.gripper.important_sites["grip_site"],
        joint_indexes={"joints": robot.joint_indexes, "qpos": robot._ref_joint_pos_indexes,
                       "qvel": robot._ref_joint_vel_indexes},
        actuator_range=robot.torque_limits, kp=KP, policy_freq=robot.control_freq,
        qpos_limits=limits.T)
    try:
        robot.controller = controller
        base._action_dim = robot.action_dim
        robot.recent_actions = DeltaBuffer(dim=robot.action_dim)
        yield controller, limits
    finally:
        robot.controller, base._action_dim, robot.recent_actions = original, dim, recent
