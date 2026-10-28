# 本轮实际命令

项目目录 `/home/robot/pi05-rl-handson`。源码运行版本 `fe28dde81a2218a96737defcdfa02811788b8b24`；Openpi固定 `15a9616a00943ada6c20a0f158e3adb39df2ccac`。父模型、数据和归一化SHA见冻结执行计划。以下是已经执行的历史命令；输出目录及实验名已存在，**不要原样覆盖重跑**。要复现实验，先复制执行计划并同步变更 experiment 字段和 CLI 实验名，再使用全新的输出路径。未提供自动覆盖参数。

## 训练（实际systemd启动及资源限制）

完整 argv 见 [launch.json](../../evidence/recovery-sampling025-28-launch.json)。核心参数如下：

```bash
cd /home/robot/pi05-rl-handson
XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv-train/bin/python -u scripts/train_recovery_replay.py \
  --experiment manual6_sampling025_2000_v1 \
  --public-slots 39 \
  --recovery-manifest evidence/manual-recovery28-manifest.json \
  --evaluation-plan evidence/recovery-sampling025-28-execution-plan.json
```

实际受限于用户systemd unit：MemoryMax40G、MemorySwapMax0、RuntimeMaxSec3600；正常退出0。默认2000更新、lr5e-6、batch1、seed42，完整配置见 [config.txt](../../evidence/manual6_sampling025_2000_v1-config.txt)。不是从上一轮退化模型续训。

## 推理服务（评测使用，已停止）

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv-train/bin/python -u scripts/serve_sft.py \
  --checkpoint checkpoints/pi05_recovery_replay_lora/manual6_sampling025_2000_v1/1999 \
  --port 9875
```

## 同状态开发评测

```bash
PYTHONPATH=/home/robot/voice-arm-sim/.local/openpi/third_party/libero \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-ppo/bin/python -u scripts/evaluate_recovery.py \
  --port 9875 \
  --expected-server-checkpoint /home/robot/pi05-rl-handson/checkpoints/pi05_recovery_replay_lora/manual6_sampling025_2000_v1/1999 \
  --manifest data/recovery-bowl-physicaldev28-v2/manifest.json \
  --split dev --max-steps 300 --noise-seed 42 \
  --output artifacts/20261028-sampling025/candidate
```

模拟器为`.venv-ppo`中的MuJoCo3.8.1；不要用旧`.venv-sim`替代，否则接触动力学不可比。无viewer离线控制20Hz、预测10步执行5步；严格松爪成功条件见正文。

## 推理权重导出与独立恢复

```bash
JAX_PLATFORMS=cpu .venv-train/bin/python scripts/export_recovery_adapter.py export \
  --checkpoint checkpoints/pi05_recovery_replay_lora/manual6_sampling025_2000_v1/1999 \
  --adapter exports/sampling025-20261028/adapter

JAX_PLATFORMS=cpu .venv-train/bin/python scripts/export_recovery_adapter.py verify \
  --adapter /path/to/independently-downloaded/adapter
```

需要官方 `pi05_libero` 基座；本包不含优化器状态，不能冒充可断点续训的完整checkpoint。完整训练checkpoint和train_state保留本机。
