"""The corrected config must expose all measured joints to PI05."""
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_unified_joint_state as training


def test_corrected_config_enables_state_without_changing_horizon(monkeypatch):
    @dataclass
    class Model:
        discrete_state_input: bool = False
        action_horizon: int = 10

    @dataclass
    class Config:
        name: str
        model: Model

    original = Config("old", Model())
    monkeypatch.setattr(training, "_original_config", lambda *_: original)
    seen = []
    monkeypatch.setattr(training, "state_input_canary", lambda cfg, path: seen.append(cfg) or {})
    monkeypatch.setattr(training, "write_json", lambda *_: None)
    cfg = training.config(dict(representation="residual"), Path("manifest.json"))
    assert cfg.model.discrete_state_input and cfg.model.action_horizon == 10
    assert seen == [cfg]
    assert not original.model.discrete_state_input
