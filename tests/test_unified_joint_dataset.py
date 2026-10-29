"""Joint label semantics, episode sampling, and strict completion regression checks."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_unified_joint import checked_action, completion_gate, same_norm_stats
from joint_residual import transform_chunk
from unified_joint_dataset import UnifiedJointDataset, balanced_slots, norm_statistics


def test_equal_episode_slots_and_full_window_range():
    counts = [281, 326, 997, 482]
    slots = balanced_slots(counts)
    assert Counter(e for e, _ in slots) == {0: 997, 1: 997, 2: 997, 3: 997}
    for episode, start in slots:
        assert 0 <= start < counts[episode]
    for episode in range(4):
        frequencies = Counter(i for e, i in slots if e == episode)
        assert max(frequencies.values()) - min(frequencies.values()) <= 1


def test_residual_one_anchor_preserves_future_targets_and_gripper():
    anchor = np.arange(7, dtype=np.float32) / 10
    actions = np.arange(80, dtype=np.float32).reshape(10, 8) / 100
    encoded = transform_chunk(actions, anchor)
    np.testing.assert_allclose(encoded[9, :7], actions[9, :7] - anchor)
    np.testing.assert_array_equal(encoded[:, 7], actions[:, 7])
    np.testing.assert_allclose(transform_chunk(encoded, anchor, decode=True), actions, atol=1e-7)


def test_labels_use_command_not_measured_state_and_do_not_cross_episode():
    dataset = UnifiedJointDataset.__new__(UnifiedJointDataset)
    dataset.representation = "absolute"
    dataset.episodes = [dict(motion_rows=12, rows=[dict(state=[i] * 9, actions=[i + 100] * 8) for i in range(12)])]
    state, actions = dataset.labels(0, 2)
    assert np.all(state == 2) and np.all(actions[0] == 102) and np.all(actions[-1] == 111)
    with pytest.raises(IndexError):
        dataset.labels(0, 3)


def test_statistics_only_visit_explicit_train_slots():
    dataset = UnifiedJointDataset.__new__(UnifiedJointDataset)
    dataset.representation = "absolute"
    dataset.episodes = [dict(motion_rows=10, rows=[dict(state=[2] * 9, actions=[3] * 8)] * 10),
                        dict(motion_rows=10, rows=[dict(state=[999] * 9, actions=[999] * 8)] * 10)]
    dataset.slots = [(0, 0)]
    stats = norm_statistics(dataset)["norm_stats"]
    assert stats["state"]["mean"] == [2.] * 9
    assert stats["actions"]["mean"] == [3.] * 8
    assert stats["actions"]["std"] == [.001] * 8


@pytest.mark.parametrize("goal,contact,command,width,clear", [
    (False, False, -1, .08, .15), (True, True, -1, .08, .15),
    (True, False, 1, .08, .15), (True, False, -1, .02, .15), (True, False, -1, .08, .05)])
def test_bowl_on_plate_alone_is_not_completion(goal, contact, command, width, clear):
    assert not completion_gate(goal, contact, command, width, clear)


def test_released_retreat_completion():
    assert completion_gate(True, False, -1, .08, .15)


@pytest.mark.parametrize("target", [.36, 3.])
def test_policy_rejects_joint_jump_or_limit_without_clipping(target):
    chunk = np.zeros((10, 8))
    chunk[0, 3] = target
    with pytest.raises(ValueError):
        checked_action(chunk, np.zeros(7), np.tile([-2., 2.], (7, 1)), .35)
    assert chunk[0, 3] == target


def test_policy_gripper_saturation_preserves_predicted_joint_target():
    chunk = np.full((10, 8), .02)
    chunk[:, 7] = -1.2
    action, jump = checked_action(chunk, np.zeros(7), np.tile([-2., 2.], (7, 1)), .35)
    np.testing.assert_array_equal(action[:7], chunk[0, :7])
    assert action[7] == -1 and jump == .02 and chunk[0, 7] == -1.2


def test_checkpoint_normalization_allows_only_optional_null_quantiles():
    source = dict(norm_stats=dict(actions=dict(mean=[1.], std=[.1])))
    saved = dict(norm_stats=dict(actions=dict(mean=[1.], std=[.1], q01=None, q99=None)))
    assert same_norm_stats(source, saved)
    saved["norm_stats"]["actions"]["mean"] = [2.]
    assert not same_norm_stats(source, saved)
