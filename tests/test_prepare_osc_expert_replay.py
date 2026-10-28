"""Prevent diagnostic holds or synthetic end actions entering expert windows."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_osc_expert_replay import recovery_rows  # noqa: E402
from multitask_dataset import window_indices  # noqa: E402


def rows():
    return [dict(frame_index=i, timestamp=i * .05, diagnostic_hold=i >= 15,
                 controller="OSC_POSE", task_index=8, prompt="test", episode_index=0,
                 actions=[i] * 7) for i in range(35)]


def test_expert_windows_exclude_diagnostics_and_never_pad():
    original = rows()
    kept = recovery_rows(original, 15, 9, "test")
    assert len(kept) == 15 and original[0]["episode_index"] == 0
    assert all(r["episode_index"] == 9 and not r["diagnostic_hold"] for r in kept)
    for anchor in range(len(kept) - 9):
        indices = window_indices(anchor, len(kept))
        assert len(set(indices)) == 10
        assert [kept[i]["actions"][0] for i in indices] == list(range(anchor, anchor + 10))


@pytest.mark.parametrize("field,value", [("diagnostic_hold", True), ("timestamp", 99.),
    ("controller", "JOINT_POSITION"), ("task_index", 36), ("prompt", "other")])
def test_modified_contract_rejected(field, value):
    changed = rows()
    changed[0][field] = value
    with pytest.raises(ValueError, match="contract"):
        recovery_rows(changed, 15, 9, "test")


def test_short_or_missing_episode_rejected():
    for count in (9, 16):
        with pytest.raises(ValueError):
            recovery_rows(rows(), count, 9, "test")
