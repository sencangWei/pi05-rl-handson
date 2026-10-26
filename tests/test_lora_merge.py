"""Numerical checks against the fixed OpenPI equations, without model weights."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "export_sft_for_ppo", Path(__file__).resolve().parents[1] / "scripts/export_sft_for_ppo.py")
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def fixture():
    rng = np.random.default_rng(42)
    params = {}
    for suffix, rank in (("", 16), ("_1", 32)):
        for name, shape in (("q_einsum", (2, 3, 4, 5)),
                            ("kv_einsum", (2, 2, 1, 4, 5)),
                            ("attn_vec_einsum", (2, 3, 5, 4))):
            base = f"PaliGemma/llm/layers/attn/{name}{suffix}"
            params[base + "/w"] = rng.normal(size=shape).astype(np.float32)
            params[base + "/lora_a"] = rng.normal(size=(*shape[:-1], rank)).astype(np.float32)
            params[base + "/lora_b"] = rng.normal(size=(*shape[:-2], rank, shape[-1])).astype(np.float32)
        for name, shape in (("gating_einsum", (2, 2, 4, 7)), ("linear", (2, 7, 4))):
            base = f"PaliGemma/llm/layers/mlp{suffix}/{name}"
            params[base] = rng.normal(size=shape).astype(np.float32)
            params[base + "_lora_a"] = rng.normal(size=(*shape[:-1], rank)).astype(np.float32)
            params[base + "_lora_b"] = rng.normal(size=(*shape[:-2], rank, shape[-1])).astype(np.float32)
    return params


class MergeTests(unittest.TestCase):
    def test_upstream_flax_modules(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path.insert(0, str(export.ROOT / "third_party/openpi/src"))
        from openpi.models import lora

        params = fixture()
        merged, _ = export.merge_lora(params)
        for suffix, rank in (("", 16), ("_1", 32)):
            config = lora.LoRAConfig(rank=rank, alpha=float(rank))
            for name, equation, shape in (("q_einsum", "BTD,NDH->BTNH", (1, 2, 4)),
                                          ("kv_einsum", "BSD,CKDH->CBSKH", (1, 2, 4)),
                                          ("attn_vec_einsum", "BTNH,NHD->BTD", (1, 2, 3, 5))):
                base = f"PaliGemma/llm/layers/attn/{name}{suffix}"
                for layer in range(2):
                    source = {key: params[base + "/" + key][layer] for key in ("w", "lora_a", "lora_b")}
                    x = np.ones(shape, np.float32)
                    original = lora.Einsum(source["w"].shape, lora_config=config).apply(
                        {"params": source}, equation, x)
                    dense = lora.Einsum(source["w"].shape).apply(
                        {"params": {"w": merged[base + "/w"][layer]}}, equation, x)
                    np.testing.assert_allclose(dense, original, rtol=3e-5, atol=3e-5)
            base = f"PaliGemma/llm/layers/mlp{suffix}/"
            for layer in range(2):
                source = {name: params[base + name][layer] for name in (
                    "gating_einsum", "gating_einsum_lora_a", "gating_einsum_lora_b",
                    "linear", "linear_lora_a", "linear_lora_b")}
                x = np.ones((1, 4), np.float32) / 10
                original = lora.FeedForward(4, 7, config).apply({"params": source}, x)
                dense = lora.FeedForward(4, 7).apply({"params": {
                    name: merged[base + name][layer] for name in ("gating_einsum", "linear")}}, x)
                np.testing.assert_allclose(dense, original, rtol=3e-5, atol=3e-5)

    def test_actual_attention_equations_scanned_layers(self):
        params = fixture()
        merged, audit = export.merge_lora(params)
        self.assertEqual(len(audit), 10)
        self.assertFalse(any("lora" in key for key in merged))
        rng = np.random.default_rng(7)
        # These are Attention.__call__ and Einsum._make_lora_eqns equations.
        equations = (("q_einsum", "BTD,NDH->BTNH", "BTD,NDL->BTNL", "BTNL,NLH->BTNH", (1, 2, 4)),
                     ("kv_einsum", "BSD,CKDH->CBSKH", "BSD,CKDL->CBSKL", "CBSKL,CKLH->CBSKH", (1, 2, 4)),
                     ("attn_vec_einsum", "BTNH,NHD->BTD", "BTNH,NHL->BTL", "BTL,NLD->BTD", (1, 2, 3, 5)))
        for suffix in ("", "_1"):
            for name, eq, eq_a, eq_b, shape in equations:
                base = f"PaliGemma/llm/layers/attn/{name}{suffix}"
                x = rng.normal(size=shape).astype(np.float32)
                for layer in range(2):
                    original = np.einsum(eq, x, params[base + "/w"][layer]) + np.einsum(
                        eq_b, np.einsum(eq_a, x, params[base + "/lora_a"][layer]),
                        params[base + "/lora_b"][layer])
                    np.testing.assert_allclose(np.einsum(eq, x, merged[base + "/w"][layer]),
                                               original, rtol=2e-5, atol=2e-5)

    def test_feedforward_dot_and_gating_axes(self):
        params = fixture()
        merged, _ = export.merge_lora(params)
        for suffix in ("", "_1"):
            for name in ("gating_einsum", "linear"):
                base = f"PaliGemma/llm/layers/mlp{suffix}/{name}"
                for index in np.ndindex(params[base].shape[:-2]):
                    w, a, b = (params[key][index] for key in
                               (base, base + "_lora_a", base + "_lora_b"))
                    x = np.arange(w.shape[0], dtype=np.float32)[None] / 10
                    np.testing.assert_allclose(x @ merged[base][index], x @ w + (x @ a) @ b,
                                               rtol=2e-5, atol=2e-5)

    def test_reject_missing_unknown_and_nonfinite_adapters(self):
        for kind in ("missing", "extra", "nan", "rank"):
            params = fixture()
            key = next(key for key in params if key.endswith("lora_a"))
            if kind == "missing":
                del params[key]
            elif kind == "extra":
                params["unknown_lora"] = np.zeros(1)
            elif kind == "nan":
                params[key].flat[0] = np.nan
            else:
                params[key] = params[key][..., :-1]
            with self.assertRaises(ValueError):
                export.merge_lora(params)


if __name__ == "__main__":
    unittest.main()
