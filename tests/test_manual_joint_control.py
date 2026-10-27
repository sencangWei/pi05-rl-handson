import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from manual_joint_control import JointSetupMixin, joint_target


def test_only_selected_joint_target_changes():
    q = np.zeros(7)
    result = joint_target(q, np.tile([-2., 2.], (7, 1)), 3, 2.)
    np.testing.assert_allclose(result, np.eye(7)[3] * np.deg2rad(2))
    np.testing.assert_array_equal(q, np.zeros(7))


def test_target_respects_joint_limit():
    q = np.full(7, 1.99)
    result = joint_target(q, np.tile([-2., 2.], (7, 1)), 3, 5.)
    assert result[3] == 1.995


@pytest.mark.parametrize("joint,degrees", [(7, 2), (-1, 2), (True, 2), (3, 0), (3, 6), (3, float("nan"))])
def test_invalid_target_rejected(joint, degrees):
    with pytest.raises(ValueError):
        joint_target(np.zeros(7), np.tile([-2., 2.], (7, 1)), joint, degrees)


def test_recording_blocks_joint_setup_before_physics():
    class Recording(JointSetupMixin):
        def _require_no_correction(self):
            raise ValueError("active recording")

    with pytest.raises(ValueError, match="active recording"):
        Recording().manual_joint_jog(3, 2.)
