# Public normal joint-action training (2026-10-30)
Goal: acquire matching public normal demonstrations, validate joint execution, train with approved human recovery replay, evaluate in actual simulation, back up code/evidence.
1. COMPLETE: source/schema/controller/resource audit; pinned official 50 episodes / 4669 frames verified.
2. IN PROGRESS: fixed 40/10 split; demo0 physical conversion passed; all50 four-worker validation running.
3. PENDING: train state-conditioned pi05 LoRA using normal data plus recovery.
4. PENDING: normal/failed-grasp/placement evaluation, videos and backup verification.
Constraints: preserve user recordings/checkpoints; no hardware; do not terminate other GPU work. Disk free 1.3GiB; use bounded tmpfs working artifacts with durable adapters/manifests.
Errors: installed specialist models unsupported; direct research fallback. No matching native LIBERO joint-command release verified yet.
