"""Loading a frozen failure must not edit or reject its close gripper pose."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from manual_recovery import preserve_unshifted_bowl


def test_zero_offset_preserves_observation_without_calling_editor():
    def reject_near_gripper(*args):
        raise ValueError("gripper too close")

    obs = {"bowl_pos": np.array([.1, .2, .9])}
    fresh, edit = preserve_unshifted_bowl(reject_near_gripper, "bowl")(object(), obs, 0., 0.)
    assert fresh is obs
    assert edit == dict(before_xyz=[.1, .2, .9], after_xyz=[.1, .2, .9], dx_m=0., dy_m=0.)


@pytest.mark.parametrize("dx,dy", [(.01, 0.), (0., -.01), (float("nan"), 0.)])
def test_real_edits_still_use_existing_safety_checks(dx, dy):
    def checked_editor(env, obs, x, y):
        raise ValueError("original safety check")

    with pytest.raises(ValueError, match="original safety check"):
        preserve_unshifted_bowl(checked_editor, "bowl")(object(), {}, dx, dy)
