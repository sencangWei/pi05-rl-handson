# 真实关节图像配对（2026-10-29）

两条已有人审通过来源重新进行物理执行，保存执行前两路实际 RGB、7 个关节角和两个夹爪位置，以及完整 8 维关节指令。保留肘关节/中间关节变化，不转换成不等价的 OSC 标签。

| 来源 | 全部控制 | 有效运动 | 完整未来10步窗口 | 动作前RGB |
| --- | ---: | ---: | ---: | ---: |
| human_1793121358375746008（训练） | 511 | 491 | 482 | 1022 |
| human_1793122696228231033（整条留出） | 454 | 434 | 425 | 908 |

状态9维：7关节角（弧度）+2夹爪位置（米）。动作8维：7绝对目标角（弧度）+归一化夹爪指令。20Hz，每次保留动作前真实两路256×256 RGB。全部965控制的物理状态和动作后RGB与冻结专家执行逐位一致。模型训练排除最后40个诊断保持，不填充不足10步窗口。

[专家执行关键帧](v3/expert-contact-sheet.jpg)与两段 v3/review.mp4 是专家动作执行，不能作为模型成功证据。来源此前已有人审；本次重新执行没有声称新增人工审核。

有效采集位于 v3。v1、diagnostic-v2 因 EGL 与参考 OSMesa 渲染不同而停止，物理状态相同；保留失败产物。改用相同 OSMesa 后全部965步精确匹配。

复现（在项目根目录，输出需使用新的空目录）：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa .venv-ppo/bin/python scripts/record_joint_pairs.py --help
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa .venv-ppo/bin/python scripts/record_joint_pairs.py --output reports/joint-pairs-20261029/reproduction
.venv-ppo/bin/python scripts/joint_dataset.py --pairs reports/joint-pairs-20261029/v3 --output data/joint-pilot29-reproduction
```

配对 Parquet、物理轨迹、审核记录、视频及训练统计以项目相对路径保存在独立 Release 归档，见 [备份清单](../joint-pilot-20261029/backup-manifest.json)。解压到项目根目录后 manifest 的路径保持可用。Git 保存小型报告/元数据/视频；不将这两份约114MiB的PNG Parquet重复提交到 Git。

后续真实模型试训结果见 [独立关节接口试训报告](../joint-pilot-20261029/README.md)。
