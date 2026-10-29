import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_action_contract import KD, KP, reference_target, validate_action


def test_moving_target_compensates_pd_damping_without_changing_raw_reference():
    q, v, limits = np.full(7, .1), np.linspace(-.2, .2, 7), np.tile([-2., 2.], (7, 1))
    target, demand = reference_target(q, v, limits, True)
    np.testing.assert_allclose(KP * (target - q) - KD * v, 0., atol=1e-14)
    np.testing.assert_array_equal(target, demand)
    np.testing.assert_array_equal(q, np.full(7, .1))
    np.testing.assert_array_equal(reference_target(q, v, limits)[0], q)


def test_limit_clipping_retains_unclipped_demand_for_audit():
    limits = np.tile([-1., 1.], (7, 1))
    target, demand = reference_target(np.full(7, .99), np.ones(7), limits, True)
    assert np.all(demand > 1.)
    np.testing.assert_array_equal(target, np.full(7, .995))
    validate_action(np.r_[target, -1.], limits)


@pytest.mark.parametrize("action", [np.zeros(7), np.zeros(9), np.r_[np.zeros(7), 2.],
                                   np.r_[np.full(7, np.nan), 0.], np.r_[np.full(7, 3.), 0.]])
def test_rejects_osc_and_invalid_joint_actions(action):
    with pytest.raises(ValueError):
        validate_action(action, np.tile([-2., 2.], (7, 1)))


@pytest.mark.parametrize("q,v,limits", [
    (np.zeros(6), np.zeros(7), np.tile([-1., 1.], (7, 1))),
    (np.zeros(7), np.full(7, np.inf), np.tile([-1., 1.], (7, 1))),
    (np.zeros(7), np.zeros(7), np.tile([1., -1.], (7, 1))),
])
def test_rejects_invalid_reference(q, v, limits):
    with pytest.raises(ValueError):
        reference_target(q, v, limits, True)
