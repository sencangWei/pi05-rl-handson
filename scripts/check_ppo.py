"""Compose the real upstream PPO configuration without starting CUDA/Ray workers."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    os.environ["EMBODIED_PATH"] = str(ROOT / "third_party/RLinf/examples/embodiment")
    os.environ["PI05_RL_ROOT"] = str(ROOT)
    # Hydra only accepts searchpath in a primary config; materialize the upstream
    # primary config plus our small override, rather than inherit it as a default.
    base = OmegaConf.load(ROOT / "third_party/RLinf/examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml")
    base.defaults.append("_self_")
    override = OmegaConf.load(ROOT / "configs/ppo_goal_overrides.yaml")
    OmegaConf.save(OmegaConf.merge(base, override), ROOT / "configs/ppo_goal_smoke.yaml")
    with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="ppo_goal_smoke")
    value = OmegaConf.to_container(cfg, resolve=True)
    assert value["actor"]["model"]["openpi"]["value_after_vlm"]
    for split in ("train", "eval"):
        assert not value["env"][split]["use_rel_reward"]
        assert value["env"][split]["step_penalty_coef"] == .001
        assert value["env"][split]["reward_coef"] == 1.0
        assert value["env"][split]["max_steps_per_rollout_epoch"] % value["actor"]["model"]["num_action_chunks"] == 0
    assert value["actor"]["micro_batch_size"] == 1
    train = value["env"]["train"]
    rollout_samples = (train["max_steps_per_rollout_epoch"] // value["actor"]["model"]["num_action_chunks"]
                       * train["total_num_envs"] * train["rollout_epoch"])
    assert rollout_samples % value["actor"]["global_batch_size"] == 0
    path = ROOT / "evidence/ppo-composed.json"
    path.write_text(json.dumps(value, indent=2) + "\n")
    print("CONFIG_COMPOSED; not a full RLinf runtime check or training run:", path)


if __name__ == "__main__":
    main()
