"""Recovery goals cannot count a held bowl or a one-frame near miss as success."""
import importlib.util
import unittest

import numpy as np
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovery", ROOT /
    "third_party/RLinf/rlinf/envs/sim/libero/recovery.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class Wrapped:
    def __init__(self):
        self.native = False
        self.contact = False
        self.env = SimpleNamespace(
            robots=[SimpleNamespace(gripper=SimpleNamespace(contact_geoms=[]))],
            get_object=lambda _: None, check_contact=lambda *_: self.contact)

    def check_success(self):
        return self.native

    def step(self, action):
        return {}, 1., True, {}

    def set_init_state(self, state):
        return state


class RecoveryContractTest(unittest.TestCase):
    def test_requires_release_and_five_consecutive_goal_controls(self):
        env = recovery.RecoveryEnv.__new__(recovery.RecoveryEnv)
        env.wrapped = Wrapped()
        env.streak = 0
        env.wrapped.native = True
        env.wrapped.contact = True
        for _ in range(7):
            self.assertEqual(env.step(None)[1:3], (0., False))
        env.wrapped.contact = False
        for _ in range(4):
            self.assertFalse(env.step(None)[2])
        env.wrapped.native = False
        self.assertFalse(env.step(None)[2])
        self.assertEqual(env.streak, 0)
        env.wrapped.native = True
        for _ in range(4):
            self.assertFalse(env.step(None)[2])
        self.assertEqual(env.step(None)[1:3], (1., True))
        env.set_init_state("next")
        self.assertEqual(env.streak, 0)


class ChunkShapingTest(unittest.TestCase):
    def shaped(self, before, after, done=None, scale=.3, terminal=None):
        before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
        done = np.zeros_like(before, dtype=bool) if done is None else done
        return recovery.shape_chunk_rewards(np.zeros_like(before), before, after, done,
                                            done if terminal is None else terminal, scale, .99)

    def test_hover_cannot_earn_progress_and_retreat_loses_credit(self):
        self.assertLess(self.shaped([[.5]], [[.5]]).sum(), 0.)
        self.assertLess(self.shaped([[.5]], [[.1]]).sum(), 0.)
        self.assertGreater(self.shaped([[.1]], [[.5]]).sum(), 0.)

    def test_discount_is_applied_once_per_chunk(self):
        value = self.shaped([[.1, .2, .3]], [[.2, .3, .4]]).sum()
        self.assertAlmostEqual(value, .3 * (.99 * .4 - .1))

    def test_terminal_cancels_potential_and_masks_late_duplicate_success(self):
        reward = recovery.shape_chunk_rewards(
            [[-.001, .999, .999]], [[.2, .3, .4]], [[.3, .4, .5]],
            [[False, True, True]], [[False, True, True]], .3, .99)
        np.testing.assert_allclose(reward, [[-.001, .939, 0.]])

    def test_timeout_potential_cancels_against_bootstrapped_shaped_value(self):
        reward = self.shaped([[.2]], [[.9]], [[True]], terminal=[[False]]).sum()
        self.assertAlmostEqual(reward, .3 * (.99 * .9 - .2))
        sparse_value = .7
        shaped_value = sparse_value - .3 * .9
        self.assertAlmostEqual(reward + .99 * shaped_value, .99 * sparse_value - .3 * .2)

    def test_discounted_shaping_telescopes_for_success_and_failure(self):
        for path in ((.1, .7, .8), (.1, .8, .2)):
            first = self.shaped([[path[0]]], [[path[1]]]).sum()
            last = self.shaped([[path[1]]], [[path[2]]], [[True]]).sum()
            self.assertAlmostEqual(first + .99 * last, -.3 * path[0])


if __name__ == "__main__":
    unittest.main()
