# 冻结训练与配对评测命令

工作目录 `/home/robot/pi05-rl-handson`；以下启动命令由本次实际执行。

```bash
systemd-run --user --unit=pi05-osc-sft29-train --property=WorkingDirectory=/home/robot/pi05-rl-handson --property=MemoryMax=40G --property=MemorySwapMax=0 --property=RuntimeMaxSec=3600 --property=StandardOutput=append:/home/robot/pi05-rl-handson/logs/osc_expert_replay2000_v1-train.log --property=StandardError=append:/home/robot/pi05-rl-handson/logs/osc_expert_replay2000_v1-train.log --setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false --setenv=OMP_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=4 /home/robot/pi05-rl-handson/.venv-train/bin/python -u scripts/train_recovery_replay.py --experiment osc_expert_replay2000_v1 --recovery-manifest evidence/osc-expert-replay29-manifest.json --evaluation-plan evidence/osc-sft29-execution-plan.json
systemd-run --user --unit=pi05-osc-sft29-evaluate --property=WorkingDirectory=/home/robot/pi05-rl-handson --property=RuntimeMaxSec=7200 --property=StandardOutput=append:/home/robot/pi05-rl-handson/logs/osc-sft29-evaluation-runner.log --property=StandardError=append:/home/robot/pi05-rl-handson/logs/osc-sft29-evaluation-runner.log /home/robot/pi05-rl-handson/.venv-ppo/bin/python -u reports/osc-sft-20261029/evaluate_run.py
```

评测服务与模拟器的精确参数见 evaluate_run.py。无UI模型替换。
