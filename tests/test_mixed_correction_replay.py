"""Mixed jog boundaries must preserve controller resets, never become OSC labels."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replay_mixed_correction import control_blocks


def jog(count=12, target=0):
    return [dict(controller="JOINT_POSITION", target=np.full(7, target)) for _ in range(count)]


class MixedTimelineTests(unittest.TestCase):
    def test_adjacent_same_targets_still_restore_osc_every_twelve(self):
        blocks = control_blocks([dict(controller="OSC_POSE")] + jog(24) + [dict(controller="OSC_POSE")])
        self.assertEqual([len(b) for b in blocks], [1, 12, 12, 1])

    def test_incomplete_jog_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            control_blocks(jog(11))

    def test_changing_target_inside_jog_rejected(self):
        with self.assertRaisesRegex(ValueError, "changing"):
            control_blocks(jog(6) + jog(6, 1))

    def test_osc_inside_joint_jog_rejected(self):
        with self.assertRaises(ValueError):
            control_blocks(jog(6) + [dict(controller="OSC_POSE")] + jog(5))


if __name__ == "__main__":
    unittest.main()
