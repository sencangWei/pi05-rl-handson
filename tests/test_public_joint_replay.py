from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_joint_replay import HOLD, assess, localize_xml, read_source, split_for_episode


def successful_rows():
    return [dict(native_goal=True, grasp=False, finger_contact=False, position_error_m=.001,
                 joint_error_rad=.002, cabinet_penetration_m=0., finite_physics=True,
                 clipped_axes=0, control_period_ok=True) for _ in range(HOLD + 2)]


def test_split_is_fixed_before_success_filtering():
    assert split_for_episode(0) == split_for_episode(39) == "train"
    assert split_for_episode(40) == split_for_episode(49) == "holdout"
    for episode in (-1, 50, True, 1.0):
        with pytest.raises(ValueError):
            split_for_episode(episode)


def test_success_needs_release_and_tracking_not_only_native_predicate():
    reference, actual = successful_rows(), successful_rows()
    assert assess(reference, actual)["passed"]
    actual[-1]["finger_contact"] = True
    assert not assess(reference, actual)["passed"]
    actual[-1]["finger_contact"] = False
    actual[-1]["position_error_m"] = .2
    assert not assess(reference, actual)["passed"]


@pytest.mark.parametrize("field,value", [("clipped_axes", 1), ("finite_physics", False),
                                         ("control_period_ok", False), ("joint_error_rad", .6)])
def test_bad_physics_or_clipping_rejected(field, value):
    actual = successful_rows()
    actual[0][field] = value
    assert not assess(successful_rows(), actual)["passed"]


def test_reference_failure_and_truncated_rollout_rejected():
    reference = successful_rows()
    reference[-1]["native_goal"] = False
    assert not assess(reference, successful_rows())["passed"]
    with pytest.raises(ValueError):
        assess(reference, successful_rows()[:-1])


def test_xml_relocation_preserves_joint_and_camera_geometry(tmp_path):
    suite, assets = tmp_path / "robosuite", tmp_path / "assets"
    (suite / "models").mkdir(parents=True)
    assets.mkdir()
    (suite / "models/mesh.stl").touch()
    (assets / "bowl.obj").touch()
    xml = '<mujoco><asset><mesh file="/old/robosuite/models/mesh.stl"/><mesh file="/old/libero/assets/bowl.obj"/></asset><worldbody><joint name="joint1" pos="1 2 3"/><camera name="agentview" pos="4 5 6"/></worldbody></mujoco>'
    result = localize_xml(xml, suite, assets)
    assert str(suite / "models/mesh.stl") in result
    assert str(assets / "bowl.obj") in result
    assert 'content_type="model/stl"' in result
    assert 'content_type="model/obj"' in result
    assert 'pos="1 2 3"' in result and 'pos="4 5 6"' in result
    with pytest.raises(FileNotFoundError):
        localize_xml(xml, tmp_path / "missing", assets)


def test_hdf5_reader_takes_only_first_state_and_true_source_action(tmp_path):
    import h5py
    path = tmp_path / "public.hdf5"
    with h5py.File(path, "w") as f:
        demo = f.create_group("data/demo_40")
        demo.attrs["model_file"] = "<mujoco/>"
        demo.create_dataset("states", data=np.arange(20 * 79).reshape(20, 79))
        demo.create_dataset("actions", data=np.zeros((20, 7)))
    result = read_source(path, 40)
    assert result["split"] == "holdout" and result["initial"].shape == (79,)
    assert result["actions"].shape == (20, 7)
    assert "states" not in result
