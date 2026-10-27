"""Serve the fixed SFT checkpoint locally for reproducible simulation evaluation."""
import argparse
import dataclasses
import hashlib
import logging
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--parent", action="store_true")
    weights.add_argument("--checkpoint", type=Path)
    parser.add_argument("--port", type=int, default=9875)
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from training_config import make_config, PARENT
    cfg = make_config("multitask40_lora_v1", 10000, 1, 2e-5)
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    checkpoint = ROOT / "checkpoints/pi05_multitask40_lora/multitask40_lora_v1/9999"
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
    if args.parent:
        checkpoint = PARENT
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(
            cfg.model, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"))
    policy = create_trained_policy(cfg, checkpoint)
    norm = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    metadata = dict(backend="official-pi05-libero-vla" if args.parent else "local-pi05-libero-lora",
                    checkpoint=str(checkpoint), image_size=224, action_horizon=10,
                    model_config=repr(cfg.model),
                    discrete_state_input=False, seed_protocol="numpy.default_rng(seed).standard_normal((10,32))",
                    norm_stats_sha256=hashlib.sha256(norm.read_bytes()).hexdigest())

    class SeededPolicy:
        def infer(self, observation):
            observation = dict(observation)
            seed = int(observation.pop("_probe_noise_seed"))
            noise = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
            return policy.infer(observation, noise=noise)

    print("POLICY_READY", metadata, flush=True)
    WebsocketPolicyServer(SeededPolicy(), host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
