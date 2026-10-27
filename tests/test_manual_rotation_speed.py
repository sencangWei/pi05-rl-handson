"""Rotation presets must change recorded control amplitude without moving on selection."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from manual_rotation_guide import RotationSpeedMixin


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Base:
    def _manual_jog(self, axis, sign):
        self.translation = (axis, sign)


class Panel(RotationSpeedMixin, Base):
    def __init__(self):
        self.rotation_mode_var = Value("精细旋转")
        self.jog_repeats_var = Value("5")
        self.jog_scale_var = Value("0.6")
        self.calls = []
        self.stopped = False
        self.lab = SimpleNamespace(manual_jog=lambda *a, **kw: self.calls.append((a, kw)))
        self.translation_joystick = SimpleNamespace(stop=self.stop)

    def stop(self):
        self.stopped = True


@pytest.mark.parametrize("mode,scale", [("普通旋转", .3), ("精细旋转", .1)])
@pytest.mark.parametrize("axis", [3, 4, 5])
@pytest.mark.parametrize("sign", [-1, 1])
def test_rotation_routes_selected_amplitude_and_preserves_repeats(mode, scale, axis, sign):
    panel = Panel()
    panel._rotation_preset(mode)
    assert panel.stopped and panel.calls == []
    assert panel.jog_scale_var.get() == "0.6"
    panel._manual_jog(axis, sign)
    assert panel.calls == [((axis, sign), {"scale": scale, "repeats": 5})]


def test_rotation_mode_leaves_translation_route_unchanged():
    panel = Panel()
    panel._rotation_preset("普通旋转")
    panel._manual_jog(1, -1)
    assert panel.translation == (1, -1)
    assert panel.calls == []


def test_unknown_preset_rejected_without_motion():
    panel = Panel()
    with pytest.raises(ValueError):
        panel._rotation_preset("unknown")
    assert panel.calls == []

