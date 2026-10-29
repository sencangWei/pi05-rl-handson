"""Protect the matched-state comparison from altered observations and inputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_expert_snapshots as subject


class SnapshotGuardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = "expert"
        source = self.root / self.source
        source.mkdir()
        self.physics = np.arange(12, dtype=float)
        self.obs = {
            "robot0_eef_pos": np.array([.1, .2, 1.]),
            "robot0_eef_quat": np.array([0., 0., 0., 1.]),
            "robot0_gripper_qpos": np.array([.02, -.02]),
            "agentview_image": np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
            "robot0_eye_in_hand_image": np.full((4, 4, 3), 42, dtype=np.uint8),
        }
        state = subject.observation_state(self.obs)
        np.savez(source / "physics.npz", before=[self.physics], observation_state=[state])
        audit = dict(state_sha256=subject.digest(state))
        audit.update({field: subject.digest(self.obs[key][::-1, ::-1])
                      for field, key in subject.CAMERAS})
        (source / "pair-audit.json").write_text(json.dumps([audit]))
        self.env = unittest.mock.Mock()
        self.env.get_sim_state.return_value = self.physics.copy()

    def check(self):
        with patch.object(subject, "PAIRS", self.root):
            return subject.check_initial(self.env, self.obs, self.source)

    def test_exact_initial_state_and_both_images_accepted(self):
        self.assertTrue(all(self.check()[key] for key in
                            ("physics_exact", "observation_state_exact", "both_rgb_exact")))

    def test_changed_physics_rejected(self):
        self.env.get_sim_state.return_value[3] += .0001
        with self.assertRaisesRegex(ValueError, "physics"):
            self.check()

    def test_changed_observation_rejected(self):
        self.obs["robot0_eef_pos"][0] += .0001
        with self.assertRaisesRegex(ValueError, "8D observation"):
            self.check()

    def test_either_camera_mutation_rejected(self):
        for _, key in subject.CAMERAS:
            with self.subTest(camera=key):
                self.obs[key][0, 0, 0] ^= 1
                with self.assertRaisesRegex(ValueError, "camera"):
                    self.check()
                self.obs[key][0, 0, 0] ^= 1

    def test_wrong_camera_orientation_rejected(self):
        self.obs["agentview_image"] = self.obs["agentview_image"][::-1, ::-1]
        with self.assertRaisesRegex(ValueError, "camera"):
            self.check()

    def test_frozen_input_and_weight_mutations_rejected(self):
        code, weight = self.root / "code.py", self.root / "params"
        code.write_bytes(b"frozen source")
        weight.write_bytes(b"frozen weights")
        protocol = dict(inputs_sha256={"code.py": subject.sha(code)},
                        weight_files={"parent": {"params": subject.sha(weight)}})
        with patch.object(subject, "ROOT", self.root):
            subject.verify_inputs(protocol, weights=True)
            code.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "Frozen input"):
                subject.verify_inputs(protocol, weights=True)
            code.write_bytes(b"frozen source")
            weight.write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "Frozen weight"):
                subject.verify_inputs(protocol, weights=True)


if __name__ == "__main__":
    unittest.main()
