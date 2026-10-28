"""Temporal/schema regression checks for the pending-review recovery pairs."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from record_osc_pairs import digest, observation_state, pack_before, validate_dataset


def observation():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]
    image[-1, -1] = [4, 5, 6]
    return dict(robot0_eef_pos=np.array([.1, .2, .3]),
                robot0_eef_quat=np.array([0., 0., 1., 0.]),
                robot0_gripper_qpos=np.array([.02, -.02]),
                agentview_image=image, robot0_eye_in_hand_image=image.copy())


class RecordingTests(unittest.TestCase):
    def test_state_matches_inference_convention_and_signed_rotation(self):
        obs = observation()
        np.testing.assert_allclose(observation_state(obs), [.1, .2, .3, 0, 0, np.pi, .02, -.02])
        obs["robot0_eef_quat"] = np.array([0., 0., np.sqrt(.5), -np.sqrt(.5)])
        self.assertAlmostEqual(observation_state(obs)[5], 1.5 * np.pi)

    def test_before_is_copied_and_two_axis_image_orientation_matches_inference(self):
        obs = observation()
        action = np.array([.2, -.3, 0., .1, 0., 0., -1.])
        row, image, hashes = pack_before(obs, action, 0, 73, 0, "JOINT_POSITION", 3.65)
        obs["agentview_image"][:] = 0
        obs["robot0_eef_pos"][:] = 9
        action[:] = 0
        decoded = np.asarray(Image.open(BytesIO(row["image"]["bytes"])))
        np.testing.assert_equal(decoded[0, 0], [4, 5, 6])
        np.testing.assert_equal(decoded[-1, -1], [1, 2, 3])
        self.assertEqual(digest(decoded), hashes["image"])
        self.assertEqual(image.shape, (256, 512, 3))
        self.assertEqual(row["state"][:3], [.1, .2, .3])
        self.assertEqual(row["actions"][:2], [.2, -.3])
        self.assertEqual(row["controller"], "OSC_POSE")
        self.assertEqual(row["original_controller"], "JOINT_POSITION")

    def test_invalid_actions_are_rejected_without_clipping(self):
        for action in (np.zeros(8), np.r_[np.zeros(6), 1.01], np.r_[np.zeros(6), np.nan]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                pack_before(observation(), action, 0, 0, 0, "OSC_POSE", 0.)

    def test_invalid_camera_and_quaternion_are_rejected(self):
        for key, value in (("agentview_image", np.zeros((224, 224, 3), dtype=np.uint8)),
                           ("robot0_eef_quat", np.zeros(4))):
            obs = observation()
            obs[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pack_before(obs, np.zeros(7), 0, 0, 0, "OSC_POSE", 0.)

    def test_on_disk_validator_rejects_post_state_mispair_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, audit = [], []
            actions = np.zeros((2, 7))
            actions[1, 0] = .1
            states = []
            for i in range(2):
                obs = observation()
                obs["robot0_eef_pos"][0] += .01 * i
                row, _, hashes = pack_before(obs, actions[i], i, i, 0, "OSC_POSE", i * .05)
                rows.append(row)
                states.append(np.asarray(row["state"]))
                audit.append(dict(**hashes, state_sha256=digest(states[-1]), action_sha256=digest(actions[i])))
            before = np.zeros((2, 79))
            after = np.zeros((2, 79))
            before[1, 0] = .05
            after[:, 0] = [.05, .1]
            np.savez(root / "physics.npz", before=before, after=after, actions=actions, observation_state=states)
            (root / "audit.json").write_text(json.dumps(audit))
            pq.write_table(pa.Table.from_pylist(rows), root / "pairs.parquet")
            def check():
                return validate_dataset(root / "pairs.parquet", root / "physics.npz", root / "audit.json")
            self.assertTrue(check()["passed"])
            original_state = rows[0]["state"]
            rows[0]["state"] = rows[1]["state"]  # Typical post-action / next-frame mispair.
            pq.write_table(pa.Table.from_pylist(rows), root / "pairs.parquet")
            with self.assertRaisesRegex(ValueError, "alignment"):
                check()
            rows[0]["state"] = original_state
            row, _, _ = pack_before(observation(), np.ones(7), 0, 0, 0, "OSC_POSE", 0.)
            rows[0]["actions"] = row["actions"]
            pq.write_table(pa.Table.from_pylist(rows), root / "pairs.parquet")
            with self.assertRaisesRegex(ValueError, "alignment"):
                check()


if __name__ == "__main__":
    unittest.main()
