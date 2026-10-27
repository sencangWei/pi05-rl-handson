import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from manual_joint_control import JointSetupMixin, joint_target


def test_only_selected_joint_target_changes():
    q = np.zeros(7)
    result = joint_target(q, np.tile([-2., 2.], (7, 1)), 3, 2.)
    np.testing.assert_allclose(result, np.eye(7)[3] * np.deg2rad(2))
    np.testing.assert_array_equal(q, np.zeros(7))


def test_target_respects_joint_limit():
    q = np.full(7, 1.99)
    result = joint_target(q, np.tile([-2., 2.], (7, 1)), 3, 5.)
    assert result[3] == 1.995


@pytest.mark.parametrize("joint,degrees", [(7, 2), (-1, 2), (True, 2), (3, 0), (3, 6), (3, float("nan"))])
def test_invalid_target_rejected(joint, degrees):
    with pytest.raises(ValueError):
        joint_target(np.zeros(7), np.tile([-2., 2.], (7, 1)), joint, degrees)


def test_active_take_can_jog_joint_and_continue_without_new_recording(tmp_path, monkeypatch):
    import contextlib
    import hashlib
    import types
    import manual_joint_control as control
    ui = Path(__file__).resolve().parents[1] / "support/manual_recovery_ui/scripts"
    monkeypatch.syspath_prepend(str(ui))
    from correction_data import CorrectionRecorder

    def digest(state):
        return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()
    monkeypatch.setitem(sys.modules, "simulation_lab", types.SimpleNamespace(
        state_digest=digest, TARGET="bowl"))

    class Env:
        value = 0.
        def get_sim_state(self):
            return np.array([self.value])
        def regenerate_obs_from_state(self, state):
            return {"state": np.full(8, state[0]), "bowl_pos": np.ones(3)}
        def step(self, action):
            self.value += 1
            return self.regenerate_obs_from_state(self.get_sim_state()), 0., False, {}

    class Base:
        def _require_no_correction(self):
            raise ValueError("active recording")
        def _ensure_not_training(self):
            pass
        def finish_correction(self, approved):
            return self.correction.finish(approved)

    class Recording(control.JointSetupMixin, Base):
        correction_active = True
        _state8 = staticmethod(lambda obs: obs["state"])
        def camera_images(self):
            return {k: np.zeros((256, 256, 3), np.uint8) for k in ("image", "wrist_image")}
        def event(self, *args, **kwargs):
            pass

    lab = Recording()
    lab.env = Env()
    lab.view = types.SimpleNamespace(is_running=lambda: True, lock=contextlib.nullcontext,
        show_joint=lambda j: None, paused=True, single_step=False)
    lab.output = tmp_path
    lab.manual_grip = -1.
    lab.manual_state_hash = lab.known_state_hash = digest(lab.env.get_sim_state())
    lab.correction = CorrectionRecorder(tmp_path / "take", dict(task="test", scenario_key="train",
        source_scene="failure", split="train", control_dt=.05, action_contract="OSC_POSE"))

    def physical(env, joint, degrees, grip, step_callback=None):
        assert step_callback is not None
        states = [env.get_sim_state()]
        for _ in range(12):
            obs, _, _, _ = step_callback(np.r_[np.zeros(7), grip], np.ones(7))
            states.append(env.get_sim_state())
        return obs, dict(states=states, before_qpos=np.zeros(7), after_qpos=np.ones(7))
    monkeypatch.setattr(control, "physical_joint_jog", physical)
    recorder = lab.correction
    lab.manual_joint_jog(3, 2.)
    assert lab.correction is recorder and recorder.active
    assert lab.manual_state_hash == digest(lab.env.get_sim_state())
    assert recorder.metadata["joint_frame_count"] == 12
    assert len(recorder.metadata["control_timeline"]) == 12
    assert recorder.metadata["training_ready"] is False
    for index, item in enumerate(recorder.metadata["joint_files"]):
        with np.load(recorder.directory / item["path"], allow_pickle=False) as frame:
            assert frame["action"].shape == (8,)
            np.testing.assert_array_equal(frame["target_qpos"], np.ones(7))
            np.testing.assert_array_equal(frame["before"], [index])
            np.testing.assert_array_equal(frame["after"], [index + 1])
            assert frame["image"].shape == (256, 256, 3)
    # A normal end-effector command can be appended to the SAME take afterwards.
    before = lab.env.regenerate_obs_from_state(lab.env.get_sim_state())
    obs, _, done, _ = lab.env.step([0, 0, 1, 0, 0, 0, -1])
    recorder.append(before["state"], [0, 0, 1, 0, 0, 0, -1], lab.camera_images(),
                    obs["state"], obs["bowl_pos"], bool(done))
    assert recorder.count == 1 and recorder.active


def test_joint_take_cannot_resume_through_osc_only_loader(tmp_path):
    import json
    (tmp_path / "metadata.json").write_text(json.dumps({"joint_frame_count": 12}))
    class Base:
        def resume_correction(self, directory):
            raise AssertionError("OSC-only loader must not replay mixed controls")
    class Workbench(JointSetupMixin, Base):
        pass
    with pytest.raises(ValueError, match="关节"):
        Workbench().resume_correction(tmp_path)


def test_panel_counts_both_controller_types():
    from types import SimpleNamespace
    class Base:
        def correction_state(self):
            return {"frames": 10, "status": "接管中"}
    class Workbench(JointSetupMixin, Base):
        correction = SimpleNamespace(metadata={"joint_frame_count": 12})
    state = Workbench().correction_state()
    assert state["frames"] == 22
    assert "末端 10 步，关节 12 步" in state["status"]
