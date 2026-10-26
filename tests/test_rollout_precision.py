"""Exercise the native rollout attention dtype contract without a full model."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


class RolloutPrecisionTest(unittest.TestCase):
    def test_unwrapped_attention_requires_bf16_weights_for_bf16_tokens(self):
        root = Path(__file__).resolve().parents[1]
        modules = root / "third_party/RLinf/rlinf/models/embodiment/openpi/modules"
        package = types.ModuleType("_rollout_precision_modules")
        package.__path__ = [str(modules)]
        sys.modules[package.__name__] = package
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.gemma", modules / "gemma.py"
        )
        gemma = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gemma
        spec.loader.exec_module(gemma)
        sys.modules[f"{package.__name__}.utils"].set_torch_compile(False)
        cfg = gemma.Config(width=16, depth=1, mlp_dim=32,
                           num_heads=2, num_kv_heads=1, head_dim=8)
        attention = gemma.Attention([cfg]).eval()
        tokens = torch.ones(1, 2, 16, dtype=torch.bfloat16)
        positions = torch.arange(2)[None]
        mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "same dtype"):
                attention([tokens], positions, mask)
            output, _ = attention.to(torch.bfloat16)([tokens], positions, mask)
        self.assertEqual(output[0].dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output[0]).all())


if __name__ == "__main__":
    unittest.main()
