#!/usr/bin/env bash
# Bounded PPO configurations; requires conversion, CUDA, and checkpoint checks first.
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
export RLINF_LOCAL_RAY_CPUS="${RLINF_LOCAL_RAY_CPUS:-4}"
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
precision = json.loads(Path('evidence/ppo-fsdp-lora-precision.json').read_text())
strategy = Path('third_party/RLinf/rlinf/hybrid_engines/fsdp/strategy/fsdp.py')
if (precision['status'] != 'PASS'
        or precision.get('actual_strategy_root_chains_dtype') != 'torch.float32'
        or hashlib.sha256(strategy.read_bytes()).hexdigest() != precision.get('strategy_source_sha256')):
    raise SystemExit('FSDP/PEFT precision regression gate has not passed for this strategy')
checkpoint = json.loads(Path('evidence/ppo-fsdp-adapter-checkpoint.json').read_text())
if checkpoint['status'] != 'PASS':
    raise SystemExit('Actual checkpoint roundtrip regression gate has not passed')
for name, expected in checkpoint['sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Checkpoint runtime changed since roundtrip check: {name}')
print('Adapter model/optimizer/scheduler/RNG roundtrip gate PASS.')
sync = json.loads(Path('evidence/ppo-fsdp-adapter-sync.json').read_text())
if sync['status'] != 'PASS':
    raise SystemExit('Actual repeated adapter sync regression gate has not passed')
for name, expected in sync['sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Adapter sync runtime changed since regression check: {name}')
print('Repeated adapter-only actor/rollout sync gate PASS.')
base = Path('models/pi05-multitask40-sft-merged/model.safetensors')
digest = hashlib.sha256()
with base.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != manifest['output_sha256']:
    raise SystemExit('Frozen base no longer matches conversion evidence')
alias = json.loads(Path('evidence/ppo-fsdp-offload-alias.json').read_text())
if (alias['status'] != 'PASS' or alias['source_sha256'] !=
        hashlib.sha256(strategy.read_bytes()).hexdigest()):
    raise SystemExit('Repeated offload storage-alias regression gate has not passed')
print('Actual frozen base SHA256 and offload alias gate verified.')
PY
.venv-ppo/bin/python scripts/check_ppo_gpu.py
.venv-ppo/bin/python scripts/check_ppo.py
.venv-ppo/bin/python - <<'PY'
from omegaconf import OmegaConf
from rlinf.config import validate_fsdp_cfg
cfg = OmegaConf.load('evidence/ppo-composed.json').actor
validate_fsdp_cfg(cfg)
if (cfg.fsdp_config.checkpoint_format != 'adapter'
        or cfg.fsdp_config.save_full_model_weights
        or cfg.fsdp_config.checkpoint_base_sha256 !=
           '4526e30c6aa11108a3331f7f4ae7931394629961df42736deedd7169ec154d5f'):
    raise SystemExit('Unexpected adapter/base checkpoint contract')
print('Actor FSDP configuration validated before starting Ray.')
PY
if [[ "${PPO_CONFIG:-ppo_goal_smoke}" == "ppo_recovery" ]]; then
  .venv-ppo/bin/python scripts/check_recovery.py --config-only
fi
if [[ "${PPO_CONFIG:-}" == "ppo_recovery_sparse_v2" || "${PPO_CONFIG:-}" == "ppo_recovery_shaped_v1" ]]; then
  .venv-ppo/bin/python scripts/check_recovery_shaping.py --config-only
fi
exec .venv-ppo/bin/python -u \
  third_party/RLinf/examples/embodiment/train_embodied_agent.py \
  --config-path "$project_root/configs" --config-name "${PPO_CONFIG:-ppo_goal_smoke}" "$@"
