"""The visual arrow must agree with world-frame OSC rotations for both signs."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from manual_rotation_guide import paint_rotation, rotation_arc


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (-1, 1))
def test_arc_is_centered_at_eef_and_follows_right_hand_rule(axis, sign):
    center = np.array([.2, -.1, 1.1])
    points = rotation_arc(center, axis, sign)
    vectors = points - center
    np.testing.assert_allclose(vectors[:, axis], 0., atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), .095)
    # Directed cross product selects +world axis for +, and its opposite for -.
    assert sign * np.cross(vectors[0], vectors[1])[axis] > 0


def test_annotations_append_without_overwriting_translation_axes():
    import mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scene = mujoco.MjvScene(model, maxgeom=100)
    scene.ngeom = 7
    scene.geoms[0].label = "X+"
    paint_rotation(scene, np.array([0., 0., 1.]), 2, -1)
    assert scene.geoms[0].label == "X+"
    assert scene.geoms[scene.ngeom - 1].label == "Rz- (world Z)"
    assert 7 < scene.ngeom < scene.maxgeom
