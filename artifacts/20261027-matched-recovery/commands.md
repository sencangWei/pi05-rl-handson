# 本轮命令记录

命令在 `/home/robot/pi05-rl-handson` 执行。需要本项目固定的三套 Python 环境、官方基座、40 任务公开训练集及原 SFT 父检查点。下面是本轮使用的参数；已生成的输出目录不能直接重复使用。复现实验请在干净工作副本恢复输入；若更改实验名，冻结计划中的 experiment 也必须同步更改并重新冻结。

## 收集与校验

```bash
.venv-ppo/bin/python -u scripts/evaluate_recovery.py \
  --config evidence/ppo_recovery_shaped_v1-composed.json \
  --checkpoint exports/pi05-recovery27-shaped_v1-adapter.pt \
  --manifest data/recovery-bowl-v3/manifest.json --split train \
  --output evaluations/matched-recovery-collect-s42-v2 \
  --max-steps 300 --noise-seed 42 --record-training-rollouts
```

collection-identity.json 固定训练集 manifest SHA256；上面的路径已按该哈希核对。恢复示范的 action7 是实际控制范围的标签，同时保留原始策略输出和每步 MuJoCo 状态。两套动作独立重放均通过后才入选。

## 训练

```bash
.venv-train/bin/python -u scripts/train_recovery_replay.py \
  --experiment matched_recovery2000_v1 \
  --recovery-manifest evidence/matched-recovery-manifest-v1.json \
  --evaluation-plan evidence/matched-recovery-evaluation-plan-v1.json \
  --steps 2000 --learning-rate 0.000005
```

实际 systemd 单元限制 MemoryMax=48GiB、MemorySwapMax=0；OMP_NUM_THREADS=4、OPENBLAS_NUM_THREADS=4、XLA_PYTHON_CLIENT_PREALLOCATE=false。完整训练日志为 matched-training.log。

## 导出和实际评测

```bash
JAX_PLATFORMS=cpu .venv-train/bin/python scripts/export_sft_for_ppo.py \
  --checkpoint checkpoints/pi05_recovery_replay_lora/matched_recovery2000_v1/1999 \
  --output models/pi05-matched-recovery2000-sft-merged

.venv-ppo/bin/python -u scripts/evaluate_recovery.py \
  --config evidence/matched-sft-composed.json \
  --manifest data/recovery-bowl-freshdev27/manifest.json --split dev \
  --output evaluations/matched-sft-freshdev \
  --max-steps 300 --noise-seed 42

.venv-ppo/bin/python -u scripts/evaluate_recovery.py \
  --config evidence/matched-sft-composed.json \
  --manifest data/recovery-bowl-dev27/manifest.json --split dev \
  --output evaluations/matched-sft-olddev \
  --max-steps 300 --noise-seed 42

.venv-ppo/bin/python -u scripts/evaluate_native_ppo.py \
  --config evidence/matched-sft-composed.json \
  --output evaluations/matched-sft-retention
```

训练前的三个基线用相同评测器、初态、随机噪声和控制协议，仅将配置换回 evidence/ppo_recovery_shaped_v1-composed.json，并且不传额外 PPO checkpoint。该配置中的 actor/rollout model_path 指向原始 40 任务 SFT 导出模型；文件名含 shaped 并不表示基线加载了 PPO adapter。新配置只修改这两处 model_path，保持原归一化等非权重配置。

## 并列视频与推理权重备份

```bash
.venv-ppo/bin/python scripts/package_recovery_transfer.py \
  --before evaluations/matched-parent-freshdev \
  --after evaluations/matched-sft-freshdev \
  --output artifacts/20261027-matched-recovery/freshdev-comparison

JAX_PLATFORMS=cpu .venv-train/bin/python scripts/export_recovery_adapter.py export \
  --checkpoint checkpoints/pi05_recovery_replay_lora/matched_recovery2000_v1/1999 \
  --adapter exports/matched-recovery2000-lora

# 从公开 Release 独立下载、解压之后核验，路径指向远端恢复副本。
JAX_PLATFORMS=cpu .venv-train/bin/python scripts/export_recovery_adapter.py verify \
  --adapter /路径/远端恢复的/matched-recovery2000-lora
```

LoRA 备份用于推理，不含优化器。需要相同官方 pi05_libero 基座，逐个重构并核对全部参数叶子；不能仅凭上传成功或文件大小声称备份完成。
