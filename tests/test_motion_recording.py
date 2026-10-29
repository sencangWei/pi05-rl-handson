import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from motion_recording import save_motion


def pose(joint=0., time=0.):
    return {"joint_pos": np.full(7, joint), "joint_vel": np.zeros(7),
                "gripper_pos": np.zeros(2), "gripper_vel": np.zeros(2),
                "eef_pos": np.ones(3), "eef_rotmat": np.eye(3), "sim_time": np.asarray(time)}


def test_elbow_motion_is_retained_even_when_eef_does_not_move(tmp_path):
    path = tmp_path / "motion.npz"
    action, target = np.r_[np.zeros(7), -1.], np.arange(7) * .1
    save_motion(path, pose(), pose(.2, .05), "JOINT_POSITION", action, target)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["before_eef_pos"], saved["after_eef_pos"])
        np.testing.assert_array_equal(saved["after_joint_pos"], np.full(7, .2))
        np.testing.assert_array_equal(saved["target_qpos"], target)
        np.testing.assert_array_equal(saved["raw_action"], action)
        assert str(saved["controller"]) == "JOINT_POSITION"
    with pytest.raises(FileExistsError):
        save_motion(path, pose(), pose(), "JOINT_POSITION", action, target)


@pytest.mark.parametrize("controller,action,target", [
    ("OSC_POSE", np.zeros(8), None),
    ("JOINT_POSITION", np.zeros(8), None),
    ("OSC_POSE", np.zeros(7), np.zeros(7)),
    ("JOINT_POSITION", np.zeros(7), np.zeros(7)),
])
def test_command_contracts_cannot_be_mixed(tmp_path, controller, action, target):
    with pytest.raises(ValueError):
        save_motion(tmp_path / "bad.npz", pose(), pose(), controller, action, target)
    assert not (tmp_path / "bad.npz").exists()


def test_osc_roundtrip_keeps_its_original_command(tmp_path):
    action = np.array([.1, -.2, .3, 0., 0., .1, -1.])
    path = tmp_path / "osc.npz"
    save_motion(path, pose(), pose(.1, .05), "OSC_POSE", action)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["raw_action"], action)
        assert "target_qpos" not in saved
        assert float(saved["after_sim_time"]) == .05
