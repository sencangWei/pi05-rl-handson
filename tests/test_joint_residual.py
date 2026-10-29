import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_residual import transform_chunk
from evaluate_joint_pilot import checked_action


def test_one_observation_anchor_for_whole_window_and_gripper_unchanged():
    anchor = np.linspace(-2., 2., 7)
    actions = np.zeros((10, 8))
    actions[:, :7] = anchor + np.arange(10)[:, None] * .01
    actions[:, 7] = np.linspace(-1., 1., 10)
    original = actions.copy()
    encoded = transform_chunk(actions, anchor)
    np.testing.assert_allclose(encoded[:, :7], np.repeat((np.arange(10) * .01)[:, None], 7, axis=1), atol=1e-15)
    np.testing.assert_array_equal(encoded[:, 7], actions[:, 7])
    np.testing.assert_array_equal(actions, original)
    np.testing.assert_allclose(transform_chunk(encoded, anchor, decode=True), actions, atol=1e-15)
    assert encoded[9, 0] == pytest.approx(.09)  # Not ten consecutive .01 deltas.


def test_joint_offset_translation_equivariance_without_cumulative_decode():
    actions = np.full((10, 8), .02)
    anchor = np.linspace(-1., 1., 7)
    decoded = transform_chunk(actions, anchor, decode=True)
    shifted = transform_chunk(actions, anchor + .4, decode=True)
    np.testing.assert_allclose(shifted[:, :7] - decoded[:, :7], .4)
    np.testing.assert_allclose(decoded[9, :7], anchor + .02)
    np.testing.assert_array_equal(shifted[:, 7], decoded[:, 7])


@pytest.mark.parametrize("actions,anchor", [
    (np.zeros((10, 7)), np.zeros(7)), (np.zeros((10, 8)), np.zeros((10, 7))),
    (np.full((10, 8), np.nan), np.zeros(7)), (np.zeros((10, 8)), np.full(7, np.inf)),
])
def test_rejects_wrong_anchor_or_invalid_actions(actions, anchor):
    with pytest.raises(ValueError, match="anchor"):
        transform_chunk(actions, anchor)


def test_decoding_does_not_bypass_joint_limit_or_jump_guard():
    anchor = np.zeros(7)
    residual = np.zeros((10, 8))
    residual[0, 2] = .5
    decoded = transform_chunk(residual, anchor, decode=True)
    with pytest.raises(ValueError, match="jump"):
        checked_action(decoded, anchor, np.tile([-1., 1.], (7, 1)), .35)
    residual[0, 2] = 2.
    with pytest.raises(ValueError, match="within limits"):
        checked_action(transform_chunk(residual, anchor, decode=True), anchor,
                       np.tile([-1., 1.], (7, 1)), .35)
