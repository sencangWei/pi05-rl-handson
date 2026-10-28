# 复现

工作目录 `/home/robot/pi05-rl-handson`。使用原 `.venv-ppo` 和相同LIBERO源码，不需要推理服务。原始录制位于 `corrections/`，输入SHA由冻结清单核验。

```bash
cd /home/robot/pi05-rl-handson
rtk proxy env \
  PYTHONPATH=/home/robot/voice-arm-sim/.local/openpi/third_party/libero \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .venv-ppo/bin/python -u scripts/replay_mixed_correction.py \
  --output reports/mixed-replay-20261028/independent-rerun

rtk proxy .venv-ppo/bin/python -m pytest -q \
  tests/test_mixed_correction_replay.py tests/test_manual_joint_control.py \
  tests/test_recovery_events.py tests/test_recovery_diagnostics.py \
  tests/test_recovery_recording.py
rtk proxy .venv-train/bin/ruff check --select E9,F63,F7,F82 \
  scripts/replay_mixed_correction.py tests/test_mixed_correction_replay.py
```

输出目录必须不存在。程序退出0表示诊断执行完毕，**不表示保真通过**；检查各 `result.json` 的 `faithful` 及 `summary.json`。失败时训练标签始终不生成。

当前脚本对应v2；v1、v2各自的 `replay-source.py` 是执行时源码快照，SHA与同目录 `protocol.json` 一致。快照仅用于审计；主脚本的ROOT从所在目录定位仓库，勿直接在报告目录执行快照。实际每帧为对应控制之后的缓存双相机观测，视频帧0对应控制步1。
