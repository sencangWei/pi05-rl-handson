# 实际命令

```sh
cd /home/robot/pi05-rl-handson
systemd-run --user --unit=pi05-taskphase-train-20261028 --property=WorkingDirectory=/home/robot/pi05-rl-handson --property=MemoryMax=40G --property=MemorySwapMax=0 --property=RuntimeMaxSec=3600 --property=StandardOutput=append:/home/robot/pi05-rl-handson/logs/manual6_taskphase_2000_v1-train.log --property=StandardError=append:/home/robot/pi05-rl-handson/logs/manual6_taskphase_2000_v1-train.log --setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false --setenv=OMP_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=4 /home/robot/pi05-rl-handson/.venv-train/bin/python -u scripts/train_recovery_replay.py --experiment manual6_taskphase_2000_v1 --sampling-profile task_phase_v1 --recovery-manifest evidence/manual-recovery28-manifest.json --evaluation-plan evidence/recovery-taskphase-28-execution-plan.json
python3 reports/recovery-taskphase-20261028/evaluate_run.py
```

评测脚本只在训练2000步正常退出、抽样顺序逐项吻合、来源/父权重/评测清单SHA一致后启动独立9875端口服务；结束仅停止自身新服务。现有面板保持原状。

生成配对录像和汇总（评测完成后）：

```sh
python3 reports/recovery-taskphase-20261028/summarize_run.py
JAX_PLATFORMS=cpu .venv-train/bin/python scripts/export_recovery_adapter.py export --checkpoint checkpoints/pi05_recovery_replay_lora/manual6_taskphase_2000_v1/1999 --adapter exports/taskphase-20261028/adapter
```

训练从同一40任务SFT父checkpoint读取参数，优化器重新初始化。运行前先检查冻结清单与数据、父模型、源码SHA，拒绝覆盖已有checkpoint。
