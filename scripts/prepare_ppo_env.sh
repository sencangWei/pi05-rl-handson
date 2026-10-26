#!/usr/bin/env bash
# Independent upstream-supported PPO runtime; never activates or changes an old venv.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export LIBERO_CONFIG_PATH="$project_root/.local/ppo-libero"
export DOWNLOAD_DIR="$project_root/.local/ppo-assets"
export HF_HOME="$project_root/.local/ppo-hf"
export UV_CACHE_DIR="$project_root/.local/ppo-uv-cache"
export XDG_CACHE_HOME="$project_root/.local/ppo-cache"
export GIT_CONFIG_GLOBAL="$project_root/.local/ppo-gitconfig"
export PYTHONNOUSERSITE=1
export UV_LINK_MODE=copy
export UV_TORCH_BACKEND=cu126
# Scope the measured mirror/proxy workaround to this installer. Keep the proxy
# available for other hosts (notably GitHub and Hugging Face asset downloads).
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}mirrors.aliyun.com,download.pytorch.org"
export no_proxy="${no_proxy:+$no_proxy,}mirrors.aliyun.com,download.pytorch.org"
export UV_HTTP_TIMEOUT=120
export UV_CONCURRENT_DOWNLOADS=4
unset VIRTUAL_ENV PYTHONPATH
mkdir -p "$LIBERO_CONFIG_PATH" "$DOWNLOAD_DIR" "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"
touch "$GIT_CONFIG_GLOBAL"
command -v uv
cd "$project_root/third_party/RLinf"
test "$(git rev-parse HEAD)" = 30e0236981c6a1dc19202de64116834c12673828
bash requirements/install.sh embodied --model openpi --env libero \
  --venv "$project_root/.venv-ppo" --python 3.11.15 --torch 2.7.1 \
  --no-root --no-flash-attn --install-rlinf
uv pip freeze --python "$project_root/.venv-ppo/bin/python" > "$project_root/evidence/ppo-requirements.txt"
echo 'PPO dependencies installed; runtime and rollout still require separate verification.'
