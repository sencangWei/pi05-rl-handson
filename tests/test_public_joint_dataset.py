"""Sampling must preserve normal task coverage while retaining human recovery."""
from collections import Counter
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_joint_dataset import mixed_slots, train_statistics


def test_group_mass_episode_balance_and_window_bounds():
    episodes = [dict(kind="public", motion_rows=25 + i) for i in range(37)]
    episodes += [dict(kind="human", motion_rows=n) for n in (290, 335, 1006)]
    slots = mixed_slots(episodes)
    counts = Counter(e for e, _ in slots)
    assert len(slots) == 40000
    assert sum(counts[e] for e in range(37)) == 32000
    assert sum(counts[e] for e in range(37, 40)) == 8000
    assert max(counts[e] for e in range(37)) - min(counts[e] for e in range(37)) <= 1
    for e, start in slots:
        assert 0 <= start <= episodes[e]["motion_rows"] - 10
    for e in range(len(episodes)):
        assert {start for episode, start in slots if episode == e} == set(range(episodes[e]["motion_rows"] - 9))


@pytest.mark.parametrize("episodes,size", [([], 100), ([dict(kind="human", motion_rows=20)], 100),
    ([dict(kind="public", motion_rows=20)], 100),
    ([dict(kind="public", motion_rows=9), dict(kind="human", motion_rows=20)], 100),
    ([dict(kind="public", motion_rows=20), dict(kind="human", motion_rows=20)], 99)])
def test_incomplete_or_missing_categories_rejected(episodes, size):
    with pytest.raises(ValueError):
        mixed_slots(episodes, size)


def test_state_scaling_preserves_full_training_range_for_pi05_tokens():
    class Dataset:
        slots = [(0, i) for i in range(12)]

        def labels(self, episode, start):
            return np.full(9, start, np.float32), np.full((10, 8), start, np.float32)

    stats = train_statistics(Dataset())["norm_stats"]
    state = stats["state"]
    values = (np.arange(12)[:, None] - state["mean"]) / state["std"]
    np.testing.assert_allclose(values.min(axis=0), -1)
    np.testing.assert_allclose(values.max(axis=0), 1)
    np.testing.assert_allclose(stats["actions"]["std"], np.arange(12).std())
