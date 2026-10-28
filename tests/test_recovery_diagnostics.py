"""Reject physically overlapping starts and mismatched replay evidence."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_recovery_bank import bowl_penetrations  # noqa: E402
from diagnose_recovery import compare  # noqa: E402


class InitialPhysicsTest(unittest.TestCase):
    def sim(self, other, distance, reverse=False):
        names = ["bowl_geom", other]
        ids = (1, 0) if reverse else (0, 1)
        return SimpleNamespace(
            model=SimpleNamespace(geom_id2name=lambda i: names[i]),
            data=SimpleNamespace(ncon=1, contact=[SimpleNamespace(
                geom1=ids[0], geom2=ids[1], dist=distance)]))

    def test_stable_bowl_overlapping_cabinet_is_rejected_in_both_orders(self):
        # The old position-stability and finger-contact checks accept this case.
        for reverse in (False, True):
            collisions = bowl_penetrations(self.sim("cabinet_1_g33", -.009, reverse), ["bowl_geom"])
            self.assertEqual(len(collisions), 1)
            self.assertAlmostEqual(collisions[0]["penetration_m"], .009)

    def test_plate_and_bottle_overlap_are_also_rejected(self):
        for other in ("plate_1_g9", "wine_bottle_1_g3"):
            self.assertTrue(bowl_penetrations(self.sim(other, -.003), ["bowl_geom"]))

    def test_small_table_support_penetration_allowed_but_deep_table_rejected(self):
        self.assertFalse(bowl_penetrations(self.sim("table_collision", -.00015), ["bowl_geom"]))
        self.assertTrue(bowl_penetrations(self.sim("table_collision", -.002), ["bowl_geom"]))

    def test_nonpenetrating_and_internal_contacts_do_not_reject(self):
        self.assertFalse(bowl_penetrations(self.sim("cabinet_1_g33", .0002), ["bowl_geom"]))
        self.assertFalse(bowl_penetrations(self.sim("bowl_geom", -.003), ["bowl_geom"]))


class ReplayFidelityTest(unittest.TestCase):
    def record(self):
        return dict(eef_pos=[0, 0, 1], bowl_pos=[0, .1, .9], plate_pos=[.1, 0, .9],
                    native_success=False, finger_contact=False, released_success=False)

    def test_pose_or_goal_flag_mismatch_cannot_be_used_as_matched_evidence(self):
        reference = self.record()
        self.assertTrue(compare(self.record(), reference, 1e-5)["passed"])
        moved = self.record()
        moved["bowl_pos"] = [0, .10002, .9]
        self.assertFalse(compare(moved, reference, 1e-5)["passed"])
        held = self.record()
        held["finger_contact"] = True
        self.assertFalse(compare(held, reference, 1e-5)["passed"])


if __name__ == "__main__":
    unittest.main()
