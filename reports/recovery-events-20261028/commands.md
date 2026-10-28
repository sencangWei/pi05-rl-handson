# 本轮复现命令

工作目录`/home/robot/pi05-rl-handson`。使用现有`.venv-ppo`（MuJoCo3.8.1），不能换成其他MuJoCo版本。以下输出目录必须尚不存在；重跑请改为新的目录，保留原结果。无需推理服务或训练服务。

```bash
cd /home/robot/pi05-rl-handson
rtk proxy env \
  PYTHONPATH=/home/robot/voice-arm-sim/.local/openpi/third_party/libero \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .venv-ppo/bin/python -u scripts/diagnose_recovery.py \
  --plan evidence/recovery-events-28-plan.json \
  --output artifacts/20261028-recovery-events/replay

rtk proxy env \
  PYTHONPATH=/home/robot/voice-arm-sim/.local/openpi/third_party/libero \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .venv-ppo/bin/python -u scripts/audit_human_event_coverage.py \
  --manifest evidence/manual-recovery28-manifest.json \
  --output artifacts/20261028-recovery-events/human-static-v3

rtk proxy .venv-ppo/bin/python scripts/report_recovery_events.py
rtk proxy .venv-ppo/bin/python -m pytest -q \
  tests/test_recovery_events.py tests/test_recovery_diagnostics.py
rtk proxy .venv-train/bin/ruff check --select E9,F63,F7,F82 \
  scripts/recovery_events.py scripts/audit_human_event_coverage.py \
  scripts/report_recovery_events.py tests/test_recovery_events.py
```

报告脚本仅绑定本轮冻结目录；若诊断另存新目录，须显式调整该脚本的`OUT/HUMAN`再生成另一份报告。报告生成会重写本轮生成的两张图及指标JSON，不改输入录制、轨迹或模型。

复核视频：`artifacts/20261028-recovery-events/replay/{parent,candidate}-{40,41,42,43}.mp4`。20fps，每段含外部和腕部相机，无初始帧；父模型成功的两段分别160／207帧，其余300帧。
