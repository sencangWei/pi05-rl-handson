from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_replay_dataset import command_phases, TaskPhaseReplayDataset


class TaskPhaseContracts(unittest.TestCase):
    def test_transition_neighborhood_and_stable_controls(self):
        a = np.zeros((90, 7))
        a[:, 6] = -1
        a[25:60, 6] = 1
        phase = command_phases(a, 90)
        self.assertEqual(phase[:15], ["other"] * 15)
        self.assertEqual(phase[15:36], ["closing"] * 21)
        self.assertEqual(phase[50:71], ["opening"] * 21)
        self.assertEqual(phase[36:50], ["other"] * 14)

    def test_no_fabricated_transition_at_clipped_boundary(self):
        a = np.zeros((20, 7))
        a[:, 6] = -1
        self.assertEqual(command_phases(a, 11), ["other"] * 11)
        self.assertEqual(len(command_phases(a, 11)), 11)

    def test_overlapping_neighborhood_assigns_nearest_earlier_tie(self):
        a = np.zeros((35, 7))
        a[:, 6] = -1
        a[10:20, 6] = 1
        phase = command_phases(a, 35)
        self.assertEqual(phase[15], "closing")
        self.assertEqual(phase[16], "opening")

    def test_invalid_action_labels_rejected(self):
        for a, n in [
            (np.zeros((20, 6)), 20),
            (np.full((20, 7), np.nan), 20),
            (np.zeros((20, 7)), 21),
        ]:
            with self.subTest(n=n), self.assertRaises(ValueError):
                command_phases(a, n)

    def test_80_10_10_routes_cover_all_replay_and_phases(self):
        d = TaskPhaseReplayDataset.__new__(TaskPhaseReplayDataset)
        d.replay = range(80)
        d.phase_pools = {
            s: {p: list(range(7)) for p in ("other", "closing", "opening")}
            for s in ("human", "public_task8")
        }
        rows = [d.route(i) for i in range(len(d))]
        self.assertEqual(len(rows), 2400)
        self.assertEqual(sum(s == "human" for s, _ in rows), 240)
        self.assertEqual(sum(s == "public_task8" for s, _ in rows), 240)
        self.assertEqual(sum(s == "replay" for s, _ in rows), 1920)
        self.assertEqual({i for s, i in rows if s == "replay"}, set(range(80)))
        for s in d.phase_pools:
            self.assertEqual(
                [d.phase_for(i) for i in range(30) if d.route(i)[0] == s],
                ["other", "closing", "opening"],
            )
        for i in (-1, len(d)):
            with self.assertRaises(IndexError):
                d.route(i)


if __name__ == "__main__":
    unittest.main()
