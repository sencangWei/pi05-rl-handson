"""Recovery goals cannot count a held bowl or a one-frame near miss as success."""
import importlib.util
import unittest
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


if __name__ == "__main__":
    unittest.main()
