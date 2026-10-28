import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from osc_action_conversion import acceptance, inverse_scale, pose_delta


@pytest.mark.parametrize("angle", [0., .1, -1.2, np.pi - 1e-8])
def test_world_rotation_left_multiplication_and_inverse_scaling(angle):
    before = Rotation.from_euler("xyz", [.4, -.6, .9]).as_matrix()
    change = Rotation.from_rotvec(np.array([1., 2., 3.]) / np.sqrt(14) * angle).as_matrix()
    target = change @ before
    delta = pose_delta(np.array([1., 2., 3.]), before, np.array([1.01, 1.98, 3.03]), target)
    np.testing.assert_allclose(delta[:3], [.01, -.02, .03], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_rotvec(delta[3:]).as_matrix() @ before, target, atol=1e-7)
    action, demand = inverse_scale(delta, -1, 1, [-.05]*3 + [-4.]*3, [.05]*3 + [4.]*3, -1)
    np.testing.assert_allclose(action[:6], demand)
    np.testing.assert_allclose(action[:6] * np.array([.05]*3 + [4.]*3), delta)
    assert action[-1] == -1


def test_nonsymmetric_scaling_and_saturation_remain_visible():
    action, demand = inverse_scale(np.array([-.5, .25, 2., 0., .75, 1.]), -2, 4, -1, 2, 1)
    np.testing.assert_allclose(demand, [-1, .5, 4, 0, 1.5, 2])
    assert action[-1] == 1
    action, demand = inverse_scale(np.full(6, 3.), -1, 1, -.05, .05, -1)
    assert (action[:6] == 1).all() and (demand == 60).all()


@pytest.mark.parametrize("delta,grip", [(np.zeros(7), 1), (np.full(6, np.nan), 1), (np.zeros(6), 2)])
def test_invalid_actions_rejected(delta, grip):
    with pytest.raises(ValueError):
        inverse_scale(delta, -1, 1, -.05, .05, grip)


def test_hold_requires_every_step_released_and_collision_is_not_success():
    limits = dict(position_p95_m=.02, rotation_p95_rad=.15, cabinet_extra_penetration_m=.001)
    rows = [dict(native_goal=True, finger_contact=False, grasp=True, position_error_m=0.,
                 rotation_error_rad=0., cabinet_penetration_m=0.) for _ in range(30)]
    reference = [dict(r) for r in rows]
    assert acceptance(rows, reference, limits, 20)["passed"]
    rows[-1]["finger_contact"] = True
    assert not acceptance(rows, reference, limits, 20)["gates"]["released_goal_hold"]
    rows[0]["cabinet_penetration_m"] = .01
    assert not acceptance(rows, reference, limits, 20)["gates"]["cabinet_contact"]
    with pytest.raises(ValueError, match="Incomplete"):
        acceptance(rows[:15], reference, limits, 20)
