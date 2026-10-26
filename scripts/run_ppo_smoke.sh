#!/usr/bin/env bash
# Two bounded rollout epochs; requires actual conversion and CUDA checks first.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PI05_RL_ROOT="$project_root"
export EMBODIED_PATH="$project_root/third_party/RLinf/examples/embodiment"
export PYTHONPATH="$project_root/third_party/RLinf"
export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export LIBERO_CONFIG_PATH="$project_root/.local/ppo-libero"
export HF_HOME="$project_root/.local/ppo-hf"
export DOWNLOAD_DIR="$project_root/.local/ppo-assets"
export XDG_CACHE_HOME="$project_root/.local/ppo-cache"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
unset VIRTUAL_ENV
.venv-ppo/bin/python - <<'PY'
import json
from pathlib import Path
for name in ('ppo-gpu-check.json', 'ppo-export-load.json', 'ppo-export-parity-v1/parity.json'):
    record = json.loads((Path('evidence') / name).read_text())
    if record['status'] != 'PASS':
        raise SystemExit(f'Required check has not passed: {name}')
print('GPU, strict load and network conversion gates PASS; PPO updates not yet run.')
PY
.venv-ppo/bin/python scripts/check_ppo.py
exec .venv-ppo/bin/python -u \
  third_party/RLinf/examples/embodiment/train_embodied_agent.py \
  --config-path "$project_root/configs" --config-name ppo_goal_smoke
