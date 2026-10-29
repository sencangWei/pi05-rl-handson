import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support/manual_recovery_ui/scripts"))
from unified_joint_recording import JointRecorder, cartesian_target, load_joint_frames
from correction_data import CorrectionRecorder, load_frames


def test_ik_preserves_uncommanded_elbow_and_limits_increment():
    jac = np.c_[np.eye(6), np.zeros(6)]
    q = np.zeros(7)
    result = cartesian_target(q, jac, np.array([1., 0, 0, 0, 0, 0]), np.tile([-2., 2.], (7, 1)))
    assert 0 < result[0] <= .05
    np.testing.assert_array_equal(result[1:], np.zeros(6))
    np.testing.assert_array_equal(q, np.zeros(7))


def test_singular_jacobian_stays_finite_and_bounded():
    q, limits = np.zeros(7), np.tile([-.1, .1], (7, 1))
    for jac in (np.zeros((6, 7)), np.full((6, 7), .001)):
        target = cartesian_target(q, jac, np.ones(6), limits)
        assert np.isfinite(target).all() and np.max(np.abs(target)) <= .08


@pytest.mark.parametrize("command", [np.zeros(7), np.full(6, np.nan), np.full(6, 1.1)])
def test_invalid_input_rejected(command):
    with pytest.raises(ValueError):
        cartesian_target(np.zeros(7), np.zeros((6, 7)), command, np.tile([-1., 1.], (7, 1)))


@pytest.fixture
def recorder(tmp_path):
    original = CorrectionRecorder(tmp_path / "take", dict(
        task="bowl", scenario_key="test", source_scene="train", split="train", control_dt=.05))
    return JointRecorder(original, np.tile([-2., 2.], (7, 1)))


def append(recorder, done=True, contact=False, opened=True):
    before = dict(joint_pos=np.zeros(7), joint_vel=np.zeros(7), gripper_pos=np.array([.04, -.04]),
                  gripper_vel=np.zeros(2), eef_pos=np.array([0., 0., .2]),
                  eef_rotmat=np.eye(3), sim_time=np.asarray(recorder.count * .05))
    after = dict(before, joint_pos=np.full(7, .02), sim_time=np.asarray((recorder.count + 1) * .05))
    # The target deliberately differs from measured positions: labels must stay commands.
    action = np.r_[np.full(7, .1), -1. if opened else 1.]
    recorder.append(before, after, action,
        {k: np.zeros((256, 256, 3), np.uint8) for k in ("image", "wrist_image")},
        np.zeros(4), np.ones(4), done, contact, np.zeros(3), "joint", [3, 2.])
    return action


def test_native_targets_preserved_and_old_osc_loader_rejected(recorder):
    action = append(recorder)
    frame, = load_joint_frames(recorder.directory, recorder.metadata)
    np.testing.assert_array_equal(frame["action"], action)
    np.testing.assert_array_equal(frame["state"][:7], np.zeros(7))
    np.testing.assert_array_equal(frame["post_state"][:7], np.full(7, .02))
    with pytest.raises(ValueError, match="inventory"):
        load_frames(recorder.directory, recorder.metadata)
    path = recorder.directory / recorder.metadata["files"][0]["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_joint_frames(recorder.directory, recorder.metadata)


@pytest.mark.parametrize("done,contact,opened", [(False, False, True), (True, True, True), (True, False, False)])
def test_failed_or_unreleased_take_cannot_be_approved(recorder, done, contact, opened):
    for _ in range(10):
        append(recorder, done, contact, opened)
    with pytest.raises(ValueError, match="10步"):
        recorder.finish(True)
    assert recorder.active  # Human can continue correcting rather than lose the take.
    recorder.finish(False)
    assert recorder.metadata["status"] == "draft" and not recorder.metadata["training_ready"]


def test_approval_requires_ten_control_frames_and_keeps_training_gate(recorder):
    for _ in range(9):
        append(recorder)
    with pytest.raises(ValueError, match="ten_frames"):
        recorder.finish(True)
    append(recorder)
    recorder.finish(True)
    assert recorder.metadata["status"] == "approved"
    assert not recorder.metadata["training_ready"]
    with pytest.raises(ValueError, match="closed"):
        append(recorder)


def test_existing_osc_recording_cannot_be_relabelled():
    with pytest.raises(ValueError, match="relabel"):
        JointRecorder(SimpleNamespace(count=1), np.tile([-2., 2.], (7, 1)))
