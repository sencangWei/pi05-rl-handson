#!/usr/bin/env bash
# Apply only after the isolated upstream installation has finished.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export UV_CACHE_DIR="$project_root/.local/ppo-uv-cache"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=120
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}download.pytorch.org,mirrors.aliyun.com"
export no_proxy="$NO_PROXY"
unset VIRTUAL_ENV PYTHONPATH UV_TORCH_BACKEND
# Official PyTorch 2.7.1 CUDA 12.8 builds include Blackwell kernels.
uv pip install --python .venv-ppo/bin/python \
  --default-index https://mirrors.aliyun.com/pypi/simple \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'torchaudio==2.7.1+cu128'
.venv-ppo/bin/python -u scripts/check_ppo_gpu.py
uv pip freeze --python .venv-ppo/bin/python > evidence/ppo-requirements.txt
