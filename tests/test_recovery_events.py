import pytest

from scripts.recovery_events import paired_divergence, runs, summarize


def row(**kwargs):
    return dict(grasp=False, finger_contact=True, native_success=False,
                bowl_pos=[0., 0., 1.04], plate_pos=[0., 0., 1.],
                eef_pos=[0., 0., 1.1], fidelity={"passed": True}, **kwargs)


def test_contact_and_lift_do_not_imply_grasp():
    result = summarize([row() for _ in range(10)], 1.)
    assert result["lifted_grasp_runs"] == []
    assert result["grasp_controls"] == 0
    # Non-pad finger geometry can carry the bowl: keep this evidence separate.
    assert result["lifted_contact_runs"] == [[0, 9]]


def test_runs_require_consecutive_controls_and_keep_edges():
    assert runs([True] * 4 + [False] + [True] * 5) == [[5, 9]]
    assert runs([True] * 5) == [[0, 4]]


def test_goal_requires_no_finger_contact():
    rows = [row() for _ in range(5)]
    for r in rows:
        r["native_success"] = True
    assert summarize(rows, 1.)["released_goal_runs"] == []
    for r in rows:
        r["finger_contact"] = False
    assert summarize(rows, 1.)["released_goal_runs"] == [[0, 4]]


def test_divergence_sustained_and_one_based():
    a, b = [row() for _ in range(7)], [row() for _ in range(7)]
    for r in b[2:]:
        r["eef_pos"][0] = .02
    assert paired_divergence(a, b) == {"eef_pos": 3, "bowl_pos": None}
    b[3]["fidelity"]["passed"] = False
    with pytest.raises(ValueError, match="fidelity"):
        paired_divergence(a, b)


def test_empty_or_missing_physical_evidence_rejected():
    with pytest.raises(ValueError):
        summarize([], 1.)
    with pytest.raises(KeyError):
        summarize([{}], 1.)
