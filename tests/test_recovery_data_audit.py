"""Prevent pause misclassification, synthetic boundary holds and silent label drift."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_recovery_data import action_metrics, assert_alignment


class RecoveryDataAuditTest(unittest.TestCase):
    def test_rotation_is_not_classified_as_pose_hold(self):
        actions = np.zeros((12, 7))
        actions[:, 3] = .3
        metrics = action_metrics(actions, np.zeros((12, 8)), 12)
        self.assertEqual(metrics["zero_xyz_frames"], 12)
        self.assertEqual(metrics["rotation_only_frames"], 12)
        self.assertEqual(metrics["zero_pose_frames"], 0)

    def test_holds_near_gripper_transition_are_separately_counted(self):
        actions = np.zeros((40, 7))
        actions[:20, 6] = -1
        actions[20:, 6] = 1
        metrics = action_metrics(actions, np.zeros((40, 8)), 40)
        self.assertEqual(metrics["zero_pose_near_gripper_transition"], 21)
        self.assertEqual(metrics["zero_pose_runs"], [40])

    def test_nonterminal_anchor_exclusion_prevents_padding(self):
        actions = np.ones((20, 7))
        self.assertEqual(action_metrics(actions, np.zeros((20, 8)), 11)["padded_label_count"], 0)
        self.assertEqual(action_metrics(actions, np.zeros((20, 8)), 20)["padded_label_count"], 45)

    def test_alignment_rejects_one_step_shift_and_scaling(self):
        actions = np.arange(84, dtype=np.float32).reshape(12, 7)
        states = np.zeros((12, 8), dtype=np.float32)
        assert_alignment(actions, states, actions.copy(), states.copy())
        for changed in (actions * .1, np.roll(actions, 1, axis=0)):
            with self.assertRaises(AssertionError):
                assert_alignment(actions, states, changed, states)


if __name__ == "__main__":
    unittest.main()
