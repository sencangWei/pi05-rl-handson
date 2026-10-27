"""Training-state isolation and action provenance are required for self-collected data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_recording import TrainingRecorder, validate_training_entry


@pytest.mark.parametrize("entry", [dict(split="dev", source_initial_id=1),
                                  dict(split="heldout", source_initial_id=10),
                                  dict(split="train", source_initial_id=20)])
def test_only_train_bank_ids_allowed(entry):
    with pytest.raises(ValueError, match="training source IDs"):
        validate_training_entry(entry)


def test_records_preaction_rgb_without_speed_scaling(tmp_path):
    import pyarrow.parquet as pq
    from io import BytesIO
    from PIL import Image
    entry = dict(id="train-placement_failure-init01", split="train", source_initial_id=1)
    recorder = TrainingRecorder(tmp_path, entry, 0, "put the bowl on the plate")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[0, 0] = [3, 7, 19]
    obs = dict(agentview_image=image, robot0_eye_in_hand_image=image.copy())
    action = np.array([.3, -.2, .1, 0, 0, 0, -1.], dtype=np.float32)
    recorder.add(obs, action, np.arange(5.), np.arange(8.))
    recorder.save(np.arange(5.) + 1)
    row = pq.read_table(tmp_path / (entry["id"] + ".parquet")).to_pylist()[0]
    np.testing.assert_array_equal(row["actions"], action)
    restored = np.asarray(Image.open(BytesIO(row["image"]["bytes"])))
    np.testing.assert_array_equal(restored[-1, -1], image[0, 0])
    with np.load(tmp_path / (entry["id"] + "-physics.npz")) as data:
        assert data["states"].shape == (2, 5)
        np.testing.assert_array_equal(data["actions"][0], action)


def test_saturated_actions_keep_raw_replay_and_clipped_training_labels(tmp_path):
    import pyarrow.parquet as pq
    entry = dict(id="train-object_offset-init01", split="train", source_initial_id=1)
    recorder = TrainingRecorder(tmp_path, entry, 0, "put the bowl on the plate")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    action = np.array([1.2, -1.3, .1, 0, 0, 0, -1.2])
    recorder.add(dict(agentview_image=image, robot0_eye_in_hand_image=image),
                 action, np.arange(5.), np.arange(8.))
    recorder.save(np.arange(5.) + 1)
    row = pq.read_table(tmp_path / (entry["id"] + ".parquet")).to_pylist()[0]
    np.testing.assert_array_equal(row["actions"], np.clip(action, -1, 1))
    with np.load(tmp_path / (entry["id"] + "-physics.npz")) as data:
        np.testing.assert_array_equal(data["actions"][0], action)
