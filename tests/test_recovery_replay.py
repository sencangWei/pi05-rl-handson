"""Regression coverage for recovery mixture proportions and episode boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recovery_replay_dataset import BankHumanClips, HumanClips, ModelRecoveryClips, RecoveryReplayDataset  # noqa: E402
from export_sft_for_ppo import sha256  # noqa: E402
from multitask_dataset import MultitaskDataset, window_indices  # noqa: E402
from prepare_recovery_sft import validate_frame  # noqa: E402
from prepare_manual_recovery import osc_runs  # noqa: E402


class ReplayContracts(unittest.TestCase):
    def test_lower_human_ratio_preserves_all_public_tasks_and_indices(self):
        data = RecoveryReplayDataset.__new__(RecoveryReplayDataset)
        data.public_slots = 39
        data.human, data.replay = range(7), range(80)
        routes = [data.route(i) for i in range(len(data))]
        self.assertEqual(sum(name == "human" for name, _ in routes), 80)
        self.assertEqual(sum(name == "replay" for name, _ in routes), 3120)
        for index in range(80):
            self.assertEqual(routes.count(("replay", index)), 39)
        for index in (-1, len(data)):
            with self.assertRaises(IndexError):
                data.route(index)

    def test_invalid_sampling_rejected_before_reading_data(self):
        for slots in (0, -1, 1.5, True):
            with self.subTest(slots=slots), self.assertRaises(ValueError):
                RecoveryReplayDataset("does-not-exist", "does-not-exist", slots)

    def test_joint_interruption_splits_windows(self):
        timeline = [dict(controller=c, path=str(i)) for i, c in enumerate(
            ["OSC_POSE"] * 20 + ["JOINT_POSITION"] * 12 + ["OSC_POSE"] * 15)]
        runs = osc_runs(timeline)
        self.assertEqual([(len(run), terminal) for run, terminal in runs], [(20, False), (15, True)])
        # Last eligible prefix anchor is 10: all ten labels precede the joint interruption.
        data = BankHumanClips.__new__(BankHumanClips)
        data.records = [dict(frames=20, sample_frames=11), dict(frames=15, sample_frames=15)]
        data.ends = np.array([11, 26])
        record, frame = data.locate(10)
        np.testing.assert_array_equal(window_indices(frame, record["frames"]), np.arange(10, 20))
        self.assertEqual(data.locate(11), (data.records[1], 0))

    def test_bank_human_rejects_nonterminal_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "episode.parquet"
            path.write_bytes(b"not-read-by-constructor")
            record = dict(path=path.name, sha256=sha256(path), frames=20, sample_frames=20,
                          terminal=False, task_id=8, controller="OSC_POSE", split="train",
                          entry=dict(split="train", source_initial_id=1))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(schema=BankHumanClips.schema, ready=True,
                                                root=directory, records=[record])))
            with self.assertRaisesRegex(ValueError, "controller boundary"):
                BankHumanClips(manifest)

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

    def test_model_rollouts_are_named_and_sampled_separately(self):
        data = RecoveryReplayDataset.__new__(RecoveryReplayDataset)
        data.source_name = "model_recovery"
        data.human, data.replay = range(7), range(40)
        routes = [data.route(i) for i in range(len(data))]
        self.assertEqual(sum(name == "model_recovery" for name, _ in routes), 40)
        self.assertEqual(sum(name == "replay" for name, _ in routes), 120)

    def test_model_data_rejects_unverified_success_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "a.parquet"
            parquet.write_bytes(b"test-data")
            record = dict(split="train", path=parquet.name, frames=1, sha256=sha256(parquet),
                          independent_replay_verified=False, task_id=8,
                          entry=dict(split="train", source_initial_id=1))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(schema=ModelRecoveryClips.schema, ready=True,
                                               root=directory, records=[record])))
            with self.assertRaisesRegex(ValueError, "independent recovery verification"):
                ModelRecoveryClips(manifest)

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
