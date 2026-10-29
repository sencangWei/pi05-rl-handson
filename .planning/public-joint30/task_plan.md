# Public normal joint-action training (2026-10-30)
Goal: acquire matching public normal demonstrations, validate joint execution, train with approved human recovery replay, evaluate in actual simulation, back up code/evidence.
1. COMPLETE: source/schema/controller/resource audit; pinned official 50 episodes / 4669 frames verified.
2. COMPLETE: all50 continuous physical replays; 37 train/3428 motion frames and10 holdout/958 frames accepted,3 rejected.
3. IN PROGRESS: 10,000-update state-conditioned pi05 LoRA; frozen80/20 public normal/human recovery sampler, fresh official parent, GPU run started.
4. PENDING: normal/failed-grasp/placement evaluation, videos and backup verification.
Constraints: preserve user recordings/checkpoints; no hardware; do not terminate other GPU work. Disk free 1.3GiB; use bounded tmpfs working artifacts with durable adapters/manifests.
Errors: installed specialist models unsupported; direct research fallback. No matching native LIBERO joint-command release verified yet.
