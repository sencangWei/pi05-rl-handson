# 两段恢复动作的实际观测重采（待人工审核）

本轮将两段已通过物理检查的关节→OSC转换轨迹重新执行，采集动作执行前的真实相机画面与状态。未训练、未替换面板模型。录像展示专家轨迹执行，不代表 VLA 已学会恢复。

打开 [中文审核页](v1/index.html)，查看两段录像与关键帧。左侧为外部相机，右侧为腕部相机，20fps 原速播放。最后一帧为最后动作执行后的画面；其余帧均为本步动作执行前画面。

| 原录制 ID 尾号 | 动作前画面/状态/动作对数 | 其中诊断保持步 | 视频时长 | 持续放稳并松爪 |
| --- | ---: | ---: | ---: | --- |
| 6008 | 511 | 20 | 25.60秒 | 最后20/20步通过 |
| 1033 | 454 | 20 | 22.75秒 | 最后20/20步通过 |

共965对、1930张无损RGB图。925步恢复动作与40步诊断保持分别标记，保持段不应混作普通动作训练。两段均从已持碗的恢复初态开始，不能当成独立从零抓取演示。

## 配对与复现证据

- 每步先保存真实 RGB、8维状态，再执行原先验证过的7维 OSC 动作；不复用旧录制图像，不使用动作后画面作为动作前输入。
- 相机翻转与状态轴角约定与现有推理入口一致。动作保持 float64，不裁剪；两个相机各256×256，PNG嵌入 Parquet。
- 逐步核对965个执行后状态，与此前转换执行的79维物理状态逐位一致；执行后像素哈希、原生目标、抓持及接触标志也逐步一致。
- 独立从磁盘解码1930张PNG，检查连续物理时间线、动作和状态；四个原始/标注视频共1934帧全解码通过。
- 相关测试35项及5个子测试通过；新增三文件 Ruff 与 Python 编译通过。代码审查子代理因本机配置模型不支持启动失败，已由主代理直接复查，不声称完成独立代理审查。

`v1/protocol.json` 固定输入与源码哈希，`v1/manifest.json` 保持 `ready=false`、`human_review=pending`、`training=false`。各源目录包含逐步物理状态、配对审计和结果，`v1/review-validation.json` 记录独立解码结果。

来源为 `reports/osc-conversion-20261029/v1` 的 tracking 动作与 `reports/full-state-replay-20261028` 的完整初态快照。它们是可复现的新执行，不是对缺少内部状态的旧录制进行逐位重建。不能据此推断任意位置、任意关节恢复或模型泛化成功。

## 命令

在项目根目录、相同依赖版本下执行，输出目录必须尚不存在：

```bash
PYTHONPATH=/home/robot/voice-arm-sim/.local/openpi/third_party/libero \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-ppo/bin/python -u scripts/record_osc_pairs.py --output reports/osc-pairs-20261029/new-run
.venv-ppo/bin/python scripts/render_osc_pairs_review.py reports/osc-pairs-20261029/new-run

.venv-ppo/bin/python -m pytest -q tests/test_record_osc_pairs.py \
  tests/test_osc_action_conversion.py tests/test_simulation_snapshot.py \
  tests/test_mixed_correction_replay.py tests/test_manual_joint_control.py
```

环境快照见 `evidence/osc-pairs29-environment.json`。数据身份与校验由固定版本源码及 protocol 控制；换版本不承诺逐位复现。

## 数据备份

代码、报告、物理状态、审计和录像提交至用户自有公开仓库 `sencangWei/pi05-rl-handson` 的 `experiment/matched-recovery-20261027` 分支。约114MB无损 Parquet 单独存放于 [GitHub Release](https://github.com/sencangWei/pi05-rl-handson/releases/tag/osc-pairs-review-20261029)，避免写入 Git 历史。上传与恢复状态以 `v1/release-data.json` 和后续 `evidence/osc-pairs29-*-restore.json` 为准；准备完成不等于远端备份完成。

Release 压缩包只包含两份 Parquet，按项目相对路径解压。校验完整压缩包及两份文件的 SHA256，结合 Git 恢复出的 physics/audit 再执行 `validate_dataset`，才能验证可恢复。

下一步由用户检查两段动作是否适合学习；本轮不会把技术检查通过自动等同于人工批准训练。
