"""Exercise the pinned runtime GELU functions against original Flax defaults."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import torch


MODULES = Path(__file__).resolve().parents[1] / "third_party/RLinf/rlinf/models/embodiment/openpi/modules"


def load_modules():
    # Load the actual two modules without importing RLinf's Ray worker runtime.
    package = types.ModuleType("_gelu_parity_modules")
    package.__path__ = [str(MODULES)]
    sys.modules[package.__name__] = package
    result = []
    for name in ("utils", "siglip"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", MODULES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result.append(module)
    return result


class GeluParityTest(unittest.TestCase):
    def test_actual_siglip_and_gemma_gelu_match_flax(self):
        utils, siglip = load_modules()
        utils.set_torch_compile(False)
        x = np.linspace(-5, 5, 1001, dtype=np.float32).reshape(-1, 1)
        expected = np.asarray(nn.gelu(jnp.asarray(x)))
        value = torch.from_numpy(x)
        block = siglip.MlpBlock(dim=1, mlp_dim=1).eval()
        with torch.no_grad():
            for layer in (block.fc1, block.fc2):
                layer.weight.fill_(1)
                layer.bias.zero_()
            vision = block(value).numpy()
            gemma = utils.gelu_glu(value, torch.ones_like(value)).numpy()
        for name, result in (("siglip", vision), ("gemma", gemma)):
            with self.subTest(module=name):
                np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
