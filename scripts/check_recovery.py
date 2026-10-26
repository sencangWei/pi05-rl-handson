"""Compose the recovery run and exercise its actual subprocess reset/step route."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party/RLinf")]


def compose_config():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    os.environ["PI05_RL_ROOT"] = str(ROOT)
    os.environ["EMBODIED_PATH"] = str(ROOT / "third_party/RLinf/examples/embodiment")
    with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="ppo_recovery")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    train = cfg.env.train
    samples = train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
    updates = samples * train.total_num_envs * train.rollout_epoch // cfg.actor.global_batch_size
    assert updates == 15 and cfg.runner.max_epochs * updates == 300
    assert cfg.actor.optim.total_training_steps == cfg.runner.max_epochs == 20
    assert cfg.runner.weight_sync_interval == 1
    from rlinf.config import validate_fsdp_cfg
    validate_fsdp_cfg(cfg.actor)
    manifest = json.loads(Path(train.recovery.manifest).read_text())
    ids = {}
    for split, count in (("train", 18), ("heldout", 12)):
        entries = [e for e in manifest["entries"] if e["split"] == split]
        assert len(entries) == count
        assert {e["category"] for e in entries} == {"empty_grasp", "object_offset", "placement_failure"}
        ids[split] = {e["source_initial_id"] for e in entries}
        for entry in entries:
            path = Path(train.recovery.manifest).parent / entry["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            assert not entry["initial_native_success"] and not entry["initial_finger_contact"]
    assert not ids["train"] & ids["heldout"]
    assert cfg.env.eval.recovery.split == "heldout" and train.recovery.split == "train"
    (ROOT / "evidence/ppo-recovery-composed.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2) + "\n")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    if args.config_only:
        compose_config()
        print("RECOVERY_CONFIG_PASS: 20 rounds / 300 planned optimizer updates")
        return
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    os.environ.setdefault("LIBERO_TYPE", "standard")
    from baseline import configure_libero
    configure_libero()
    cfg = compose_config()
    import numpy as np
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    env = LiberoEnv(cfg.env.train, 1, 0, 1, None)
    try:
        for _ in range(3):
            obs, _ = env.reset()
            assert obs["states"].shape == (1, 8)
            actions = np.zeros((1, 7))
            actions[:, -1] = -1
            env.step(actions)
    finally:
        env.env.close()
    record = {"status": "PASS", "subprocess_resets": 3,
              "train_states": 18, "heldout_states": 12,
              "planned_optimizer_updates": 300, "scheduler_rounds": 20}
    (ROOT / "evidence/ppo-recovery-smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
