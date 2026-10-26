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
# A short, isolated path also keeps Ray Unix sockets below the path length limit.
export RAY_TMPDIR="/tmp/pi05-ppo-$UID"
unset VIRTUAL_ENV RAY_ADDRESS
.venv-ppo/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
for name in ('ppo-gpu-check.json', 'ppo-export-load.json', 'ppo-export-parity-v2/parity.json'):
    record = json.loads((Path('evidence') / name).read_text())
    if record['status'] != 'PASS':
        raise SystemExit(f'Required check has not passed: {name}')
parity = record
for name, expected in parity['runtime_source_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Runtime changed since parity check: {name}')
manifest = json.loads(Path(parity['export'], 'export_manifest.json').read_text())
if parity['weights_sha256'] != manifest['output_sha256']:
    raise SystemExit('Parity evidence belongs to different exported weights')
print('GPU, strict load and network conversion gates PASS; PPO updates not yet run.')
PY
.venv-ppo/bin/python scripts/check_ppo_gpu.py
.venv-ppo/bin/python scripts/check_ppo.py
.venv-ppo/bin/python - <<'PY'
from omegaconf import OmegaConf
from rlinf.config import validate_fsdp_cfg
validate_fsdp_cfg(OmegaConf.load('evidence/ppo-composed.json').actor)
print('Actor FSDP configuration validated before starting Ray.')
PY
exec .venv-ppo/bin/python -u \
  third_party/RLinf/examples/embodiment/train_embodied_agent.py \
  --config-path "$project_root/configs" --config-name ppo_goal_smoke
