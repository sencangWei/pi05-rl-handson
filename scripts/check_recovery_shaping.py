"""Verify frozen recovery comparisons and reward delivery through real subprocesses."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party/RLinf")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    os.environ["PI05_RL_ROOT"] = str(ROOT)
    os.environ["EMBODIED_PATH"] = str(ROOT / "third_party/RLinf/examples/embodiment")
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    os.environ.setdefault("LIBERO_TYPE", "standard")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    configs = {}
    for name in ("ppo_recovery_sparse_v2", "ppo_recovery_shaped_v1"):
        with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "configs")):
            cfg = compose(config_name=name)
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        assert cfg.runner.max_epochs == cfg.actor.optim.total_training_steps == 20
        updates = (cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
                   * cfg.env.train.total_num_envs // cfg.actor.global_batch_size)
        assert updates * cfg.runner.max_epochs == 300
        assert cfg.env.train.recovery.split == "train" and cfg.env.eval.recovery.split == "dev"
        assert cfg.env.train.recovery.mask_after_done
        assert not cfg.env.train.use_rel_reward and not cfg.env.train.ignore_terminations
        if name.endswith("shaped_v1"):
            assert cfg.env.train.recovery.shaping.gamma == cfg.algorithm.gamma
            assert cfg.algorithm.reward_type == "chunk_level"
        ids = {}
        for split, env_cfg in (("train", cfg.env.train), ("dev", cfg.env.eval)):
            path = Path(env_cfg.recovery.manifest)
            manifest = json.loads(path.read_text())
            entries = [e for e in manifest["entries"] if e["split"] == split]
            assert len(entries) == (18 if split == "train" else 12)
            ids[split] = {e["source_initial_id"] for e in entries}
            for entry in entries:
                assert not entry["initial_native_success"] and not entry["initial_finger_contact"]
                state_file = path.parent / entry["file"]
                assert hashlib.sha256(state_file.read_bytes()).hexdigest() == entry["sha256"]
        assert ids["train"] == set(range(1, 7)) and ids["dev"] == set(range(20, 24))
        configs[name] = cfg
        (ROOT / "evidence" / (name + "-composed.json")).write_text(
            json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2) + "\n")
    if args.config_only:
        print("RECOVERY_SHAPING_CONFIG_PASS: paired 300 updates; fresh development IDs 20..23")
        return
    from baseline import configure_libero
    configure_libero()
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    env = LiberoEnv(configs["ppo_recovery_shaped_v1"].env.train, 1, 0, 1, None)
    try:
        env.reset()
        actions = np.zeros((1, 10, 7))
        actions[:, :, -1] = -1
        _, rewards, terminated, truncated, infos = env.chunk_step(actions)
        assert not terminated.any() and not truncated.any()
        start = float(infos[0]["recovery_potential_before"][0])
        finish = float(infos[-1]["recovery_potential_after"][0])
        expected = -.01 + .3 * (.99 * finish - start)
        actual = float(rewards.sum())
        assert np.isclose(actual, expected, atol=1e-6), (actual, expected)
        assert not np.isclose(actual, -.01, atol=1e-7), "Shaping did not reach actual PPO rewards"
        record = {"status": "PASS", "actual_ppo_chunk_reward": actual,
                  "expected_reward": expected, "sparse_reward": -.01,
                  "potential_before": start, "potential_after": finish,
                  "path": "RecoveryEnv subprocess -> LiberoEnv.chunk_step -> PPO reward tensor"}
    finally:
        env.env.close()
    (ROOT / "evidence/recovery27-shaping-smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
