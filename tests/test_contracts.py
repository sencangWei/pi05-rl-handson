import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from multitask_dataset import window_indices, MultitaskDataset  # noqa: E402
from prepare_multitask import make_manifest  # noqa: E402
from training_config import verify_inventory  # noqa: E402


class Contracts(unittest.TestCase):
    def test_same_size_parent_or_source_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "artifact"
            path.write_bytes(b"abcd")
            inventory = {"artifact": hashlib.sha256(b"abcd").hexdigest()}
            verify_inventory(root, inventory)
            path.write_bytes(b"abce")
            with self.assertRaisesRegex(ValueError, "Content checksum mismatch"):
                verify_inventory(root, inventory)

    def test_inventory_cannot_escape_source_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "escapes root"):
                verify_inventory(Path(temporary), {"../outside": "unused"})

    def test_frozen_split_has_all_tasks_and_no_leakage(self):
        api = json.loads((ROOT / "evidence/hf-dataset-api.json").read_text())
        a = make_manifest(api, ROOT / "data/libero")
        b = make_manifest(api, ROOT / "data/libero")
        self.assertEqual(a, b)
        self.assertEqual(len(a["records"]), 1693)
        train = {r["episode"] for r in a["records"] if r["split"] == "train"}
        test = {r["episode"] for r in a["records"] if r["split"] == "holdout"}
        self.assertFalse(train & test)
        for split in ("train", "holdout"):
            self.assertEqual({r["task_id"] for r in a["records"] if r["split"] == split}, set(range(40)))

    def test_window_never_crosses_episode_end(self):
        np.testing.assert_array_equal(window_indices(105, 107), [105, 106, 106, 106, 106, 106, 106, 106, 106, 106])
        with self.assertRaises(IndexError):
            window_indices(107, 107)

    def test_balanced_sampling_and_no_holdout_selection(self):
        dataset = MultitaskDataset(ROOT / "evidence/dataset-manifest.json", require_ready=False)
        for index in [*range(80), len(dataset) - 1]:
            record, frame = dataset.locate(index)
            self.assertEqual(record["split"], "train")
            self.assertEqual(record["task_id"], index % 40)
            self.assertTrue(0 <= frame < record["frames"])

    def test_actual_patched_rlinf_reward_ongoing_cost_and_terminal_bonus(self):
        # Execute the actual method body, without importing Ray/CUDA environment infrastructure.
        source = ast.parse((ROOT / "third_party/RLinf/rlinf/envs/sim/libero/libero_env.py").read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "LiberoEnv")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_calc_step_reward")
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        namespace = {}
        exec(compile(module, "actual-rlinf-reward", "exec"), namespace)
        class Config(dict):
            __getattr__ = dict.__getitem__
        env = SimpleNamespace(cfg=Config(step_penalty_coef=.001, reward_coef=1.),
            use_step_penalty=True, use_rel_reward=False)
        fn = namespace["_calc_step_reward"]
        for _ in range(10):
            np.testing.assert_allclose(fn(env, np.array([False, True])), [-.001, .999])
        # Faster success > delayed success > full timeout at this bounded 300-step horizon.
        self.assertGreater(1 - .001 * 100, 1 - .001 * 299)
        self.assertGreater(1 - .001 * 299, -.001 * 300)


if __name__ == "__main__":
    unittest.main()
