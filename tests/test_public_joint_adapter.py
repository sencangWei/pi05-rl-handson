"""Archive semantics are checked without loading the full model."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_public_joint_adapter import FILES, FORMAT, MODEL, canonical_norm, sha, validate_contract


def norm():
    return dict(norm_stats={name: dict(mean=[0.] * size, std=[1.] * size)
        for name, size in (("state", 9), ("actions", 8))})


def test_only_optional_null_quantiles_are_semantically_ignored():
    original = norm()
    saved = deepcopy(original)
    for item in saved["norm_stats"].values():
        item.update(q01=None, q99=None)
    assert canonical_norm(original) == canonical_norm(saved)
    saved["norm_stats"]["actions"]["mean"][0] = .1
    assert canonical_norm(original) != canonical_norm(saved)


@pytest.mark.parametrize("field,value", [("mean", [0.] * 7), ("std", [0.] * 8),
                                        ("std", [np.nan] * 8), ("q01", [0.] * 8)])
def test_invalid_action_normalization_rejected(field, value):
    value_ = norm()
    value_["norm_stats"]["actions"][field] = value
    with pytest.raises(ValueError):
        canonical_norm(value_)


def test_model_and_manifest_contract_rejects_state_or_representation_change(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    metadata = dict(format=FORMAT, model=deepcopy(MODEL),
        official_base="gs://openpi-assets/checkpoints/pi05_libero",
        purpose="inference only; optimizer and RNG state excluded",
        manifest_sha256=sha(manifest), files={name: "unused" for name in FILES})
    validate_contract(metadata, manifest)
    for key, value in (("discrete_state_input", False), ("representation", "absolute"), ("action_dim", 7)):
        altered = deepcopy(metadata)
        altered["model"][key] = value
        with pytest.raises(ValueError):
            validate_contract(altered, manifest)
    manifest.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        validate_contract(metadata, manifest)
