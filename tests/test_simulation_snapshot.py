import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from simulation_snapshot import decode, encode, restore
import simulation_snapshot


def test_native_data_needs_uninitialized_restore():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint/><geom size=".1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    data.qpos[0], data.time = .4, .7
    mujoco.mj_forward(model, data)
    payload = data.__getstate__()
    initialized = mujoco.MjData(model)
    initialized.__setstate__(payload)
    assert initialized.time != data.time  # Reproduces the silent native API pitfall.
    restored = mujoco.MjData.__new__(mujoco.MjData)
    restored.__setstate__(payload)
    mujoco.mj_copyData(initialized, model, restored)
    assert initialized.time == data.time
    np.testing.assert_array_equal(initialized.qpos, data.qpos)
    np.testing.assert_array_equal(initialized.qacc_warmstart, data.qacc_warmstart)


def test_numeric_archive_keeps_rng_tuple_dtype_and_empty_shape():
    state = dict(rng=np.random.get_state(), empty=np.zeros((0, 3)),
                 pixels=np.array([[1, 255]], dtype=np.uint8))
    actual = decode(encode(state))
    assert isinstance(actual["rng"], tuple)
    assert actual["pixels"].dtype == np.uint8
    assert actual["empty"].shape == (0, 3)
    np.testing.assert_array_equal(actual["rng"][1], state["rng"][1])


def test_unknown_python_objects_rejected():
    with pytest.raises(TypeError, match="Unsupported"):
        encode(object())


def test_unknown_restore_mode_rejected_before_mutation():
    with pytest.raises(ValueError, match="Unknown restore mode"):
        restore(None, None, "typo")


def test_wrong_model_rejected_before_native_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation_snapshot, "identity", lambda env: {"model": "expected"})
    path = tmp_path / "snapshot.npz"
    np.savez(path, metadata=np.asarray(json.dumps(dict(format=1, identity={"model": "other"}))))
    with pytest.raises(ValueError, match="identity mismatch"):
        restore(None, path)


def test_corrupted_native_payload_rejected_before_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(simulation_snapshot, "identity", lambda env: {})
    path = tmp_path / "snapshot.npz"
    np.savez(path, metadata=np.asarray(json.dumps(dict(format=1, identity={}, native_sha256="wrong"))),
             native=np.asarray([1], dtype=np.uint8))
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore(None, path)
