# 本次实际执行与复核

工作目录：`/home/robot/pi05-rl-handson`。保留现有结果；执行入口拒绝覆盖已有输出与日志。如下为本轮实际启动记录，不要直接重跑覆盖结果；复现实验应复制执行入口到新实验目录并修改输出、专用unit和日志名称，保持评测参数与权重不变。

```bash
cd /home/robot/pi05-rl-handson
rtk proxy systemd-run --user --unit=pi05-expert-policy29-run \
  --property=WorkingDirectory=/home/robot/pi05-rl-handson \
  --property=RuntimeMaxSec=2400 \
  --property=StandardOutput=append:/home/robot/pi05-rl-handson/logs/expert-policy29-run.log \
  --property=StandardError=append:/home/robot/pi05-rl-handson/logs/expert-policy29-run.log \
  /home/robot/pi05-rl-handson/.venv-ppo/bin/python -u reports/expert-policy-20261029/run.py
```

`run.py`先冻结输入和全部权重文件，再顺序启动两个推理服务，每个模型顺序启动两次独立仿真进程，最后重验输入/权重。变量、端口9875、资源上限、完整子命令均保存在该文件中。只关闭本轮创建的服务；不连接现有交互面板。

无GPU的新结果分析/录像生成：

```bash
rtk proxy .venv-ppo/bin/python reports/expert-policy-20261029/analyze.py
```

它会重生成两个并列视频和分析JSON，不改原始四条模型录像、轨迹、状态、结果和协议；运行结束unit已被systemd清理时，JSON内systemd字段可能为空，应以上次保存的原始完成记录与日志为准。

针对性检查：

```bash
rtk proxy .venv-ppo/bin/python -m unittest discover -s tests -p 'test_*snapshot*.py' -v
rtk proxy .venv-ppo/bin/python -m unittest discover -s tests -p test_record_osc_pairs.py -v
rtk proxy .venv-train/bin/ruff check scripts/evaluate_expert_snapshots.py reports/expert-policy-20261029 tests/test_expert_snapshot_eval.py
rtk proxy .venv-ppo/bin/python -m py_compile scripts/evaluate_expert_snapshots.py reports/expert-policy-20261029/run.py reports/expert-policy-20261029/analyze.py tests/test_expert_snapshot_eval.py
```

播放：

```bash
xdg-open /home/robot/pi05-rl-handson/artifacts/20261029-expert-policy/comparison-human_1793121358375746008.mp4
xdg-open /home/robot/pi05-rl-handson/artifacts/20261029-expert-policy/comparison-human_1793122696228231033.mp4
```
