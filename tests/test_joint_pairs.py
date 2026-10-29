from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_dataset import JointDataset, JointOutputs
from evaluate_joint_pilot import checked_action, same_norm_stats
from record_joint_pairs import check_render_backend, pack_before
from replay_mixed_correction import sha


def sample():
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    return {"agentview_image": image, "robot0_eye_in_hand_image": image.copy()}


def test_pre_action_images_and_joint_labels_are_copied():
    obs = sample()
    state, action = np.arange(9.) / 100, np.r_[np.arange(7.) / 10, -1.]
    row, image, _ = pack_before(obs, state, action, np.tile([-3., 3.], (7, 1)), 0, 73, 0, 1., False)
    state[:] = 99
    action[:] = 99
    assert row["state"][0] == 0 and row["actions"][-1] == -1
    assert np.array_equal(np.asarray(Image.open(BytesIO(row["image"]["bytes"]))), obs["agentview_image"][::-1, ::-1])
    assert image.shape == (256, 512, 3)


@pytest.mark.parametrize("state,action", [(np.zeros(8), np.zeros(8)), (np.full(9, np.nan), np.zeros(8)),
                                         (np.zeros(9), np.zeros(7)), (np.zeros(9), np.r_[np.zeros(7), 2.])])
def test_rejects_incompatible_or_invalid_contract(state, action):
    with pytest.raises(ValueError):
        pack_before(sample(), state, action, np.tile([-3., 3.], (7, 1)), 0, 0, 0, 0., False)


def test_backend_must_match_frozen_rgb(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    with pytest.raises(ValueError, match="osmesa"):
        check_render_backend()
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    check_render_backend()


@pytest.fixture
def pairs(tmp_path):
    records = []
    for episode, split in enumerate(("train", "holdout")):
        rows = [pack_before(sample(), np.ones(9) * (i + episode * 100), np.r_[np.ones(7) * .01 * i, 1.],
                           np.tile([-3., 3.], (7, 1)), i, i, episode, i * .05, i >= 12)[0] for i in range(32)]
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        records.append(dict(source=split, split=split, path=str(path), sha256=sha(path), motion_rows=12, paired_rows=32))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(schema="pi05-joint-pilot-v1", ready=True, records=records)))
    return path


def test_full_windows_exclude_diagnostic_holds_and_do_not_cross_split(pairs):
    train, holdout = JointDataset(pairs), JointDataset(pairs, "holdout")
    assert len(train) == 3
    item = train[2]
    assert item["actions"].shape == (10, 8) and item["state"].shape == (9,)
    assert item["actions"][-1, 0] == pytest.approx(.11)
    assert holdout[0]["state"][0] == 100
    with pytest.raises(IndexError):
        train[3]
    with pytest.raises(IndexError):
        train[-1]


def test_dataset_detects_on_disk_label_changes(pairs):
    manifest = json.loads(pairs.read_text())
    path = manifest["records"][0]["path"]
    with open(path, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        JointDataset(pairs)


def test_inference_keeps_eighth_gripper_dimension():
    actions = np.arange(320).reshape(10, 32)
    result = JointOutputs()({"actions": actions})["actions"]
    assert result.shape == (10, 8)
    assert np.array_equal(result[:, -1], actions[:, 7])


def test_joint_policy_checks_target_jump_and_saturates_only_gripper():
    chunk = np.ones((10, 8)) * .2
    chunk[0, -1] = 1.1
    action, jump = checked_action(chunk, np.zeros(7), np.tile([-3., 3.], (7, 1)), .35)
    assert jump == .2 and action[-1] == 1.
    assert chunk[0, -1] == 1.1
    with pytest.raises(ValueError, match="jump"):
        checked_action(chunk, np.ones(7), np.tile([-3., 3.], (7, 1)), .35)
    with pytest.raises(ValueError, match="chunk"):
        checked_action(np.zeros((10, 7)), np.zeros(7), np.tile([-3., 3.], (7, 1)), .35)


def test_checkpoint_null_quantiles_preserve_exact_training_statistics():
    source = {"norm_stats": {"state": {"mean": [1., 2.], "std": [.1, .2]},
                             "actions": {"mean": [3.], "std": [.3]}}}
    saved = json.loads(json.dumps(source))
    for entry in saved["norm_stats"].values():
        entry.update(q01=None, q99=None)
    assert same_norm_stats(source, saved)
    saved["norm_stats"]["state"]["mean"][0] += .001
    assert not same_norm_stats(source, saved)
    saved = json.loads(json.dumps(source))
    saved["norm_stats"]["actions"]["q01"] = [0.]
    assert not same_norm_stats(source, saved)
