"""Regression coverage for recovery mixture proportions and episode boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recovery_replay_dataset import HumanClips, RecoveryReplayDataset
from multitask_dataset import MultitaskDataset, window_indices
from prepare_recovery_sft import validate_frame


class ReplayContracts(unittest.TestCase):
    def test_ratio_and_replay_coverage(self):
        data = RecoveryReplayDataset.__new__(RecoveryReplayDataset)
        data.human = range(2050)
        data.replay = range(80)
        routes = [data.route(i) for i in range(len(data))]
        self.assertEqual(sum(name == "human" for name, _ in routes), 80)
        self.assertEqual(sum(name == "replay" for name, _ in routes), 240)
        for index in range(80):
            self.assertEqual(routes.count(("replay", index)), 3)
        with self.assertRaises(IndexError):
            data.route(-1)

    def test_human_boundaries_and_end_padding(self):
        data = HumanClips.__new__(HumanClips)
        data.records = [{"episode": 0, "frames": 2}, {"episode": 1, "frames": 3}]
        data.ends = np.array([2, 5])
        self.assertEqual(data.locate(1), (data.records[0], 1))
        self.assertEqual(data.locate(2), (data.records[1], 0))
        record, frame = data.locate(4)
        np.testing.assert_array_equal(window_indices(frame, record["frames"]), [2] * 10)
        with self.assertRaises(IndexError):
            data.locate(5)

    def test_human_holdout_and_changed_parquet_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            record = dict(split="holdout", path="a.parquet", frames=1, sha256="bad")
            info = dict(schema="pi05-human-replay-v1", ready=True, root=directory, records=[record])
            manifest.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "training episodes"):
                HumanClips(manifest)
            record["split"] = "train"
            (root / "a.parquet").write_bytes(b"changed")
            manifest.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "parquet changed"):
                HumanClips(manifest)

    def test_existing_replay_never_selects_holdout(self):
        data = MultitaskDataset(ROOT / "evidence/dataset-manifest.json")
        for index in range(80):
            record, _ = data.locate(index)
            self.assertEqual(record["split"], "train")
            self.assertEqual(record["task_id"], index % 40)

    def test_invalid_action_and_rgb_contract_rejected(self):
        frame = dict(state=np.zeros(8), action=np.zeros(7), post_state=np.zeros(8),
                     post_bowl_pos=np.zeros(3), done=np.array(False),
                     image=np.zeros((256, 256, 3), dtype=np.uint8),
                     wrist_image=np.zeros((256, 256, 3), dtype=np.uint8))
        validate_frame(frame)
        frame["action"][0] = 1.1
        with self.assertRaisesRegex(ValueError, "bounds"):
            validate_frame(frame)
        frame["action"][0] = 0
        frame["image"] = frame["image"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "image"):
            validate_frame(frame)


if __name__ == "__main__":
    unittest.main()
