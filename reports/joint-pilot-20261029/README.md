# 独立关节接口试训：训练完成，闭环策略未通过

本轮检验“能否保留肘部调整并使用关节目标训练π0.5”，不是旧模型升级验收。真实200次LoRA更新完成，模型闭环0/2；未替换面板模型。专家关节动作可执行与模型已学会这些动作是两件事。

| 测试 | 实际控制步 | 推理调用 | 结果与停止原因 |
| --- | ---: | ---: | --- |
| 训练示范初态 | 2 | 3 | 第三次预测 J4=-0.05780rad，超过关节上限约-0.0698rad，在执行前拒绝 |
| 完整留出示范初态 | 0 | 1 | 首次预测 J7=0.82424rad，当前2.24849rad，跳转1.42425rad超过预定0.35rad门限 |

这两次都没有耗尽600步预算，不是运行步数太小。初始两路RGB和9维状态均与配对数据精确匹配；没有专家回放兜底。[实际策略结果](v1/evaluation-v2/summary.json)、[录像关键帧](v1/evaluation-v2/policy-contact-sheet.jpg)、train-eval/policy.mp4（3帧）和holdout-eval/policy.mp4（1帧）保留实际停止状态；没有扩展视频伪装完整动作。

## 数据与模型

两条专家来源共965配对控制、1930张真实动作前PNG；有效运动491训练/434留出，482/425个完整未来10步窗口。整条来源分离，统计只用491训练运动帧、每帧一次计权、population std下限0.001。9维关节状态，8维绝对关节目标。旧OSC数据、原录制和旧模型均保留。

官方 pi05_libero 父权重，pi05_joint_contract_lora 独立配置；全新AdamW，LoRA20张量，梯度裁剪1、无EMA、batch1、seed42、lr5e-5、warmup20、未来10步。真实200更新，含保存耗时83.26秒。200个梯度样本和1个未使用预取与冻结顺序精确匹配，留出梯度数0。检查点 checkpoints/pi05_joint_contract_lora/joint29_pilot200_v1/199。短训练的单样本loss不能证明泛化。

输入与采样见 v1/plan.json、config.txt、sampling-preview.json、consumed-samples.json、train.log、train-status.json；环境见 ../../evidence/joint-pairs29-environment.json。

## 原因边界与下一步

[关节姿态覆盖统计](v1/evaluation-v2/joint-coverage.json)：训练示范J7范围仅0.81962～0.82678rad，留出初态2.24849rad，留出全程处于训练范围之外。模型预测靠近训练姿态。J3/J5也缺少留出姿态覆盖。这说明当前单条训练示范的绝对关节表示试训不可靠，但不能由2例孤立出“训练次数/数据覆盖/表示”各自贡献，也不能用它解释全部旧OSC慢动作。

下一项可验证实验是关节残差目标：以当前观测关节角为共同锚点编码未来10个目标，推理时加回同一锚点。先检验目标重建和真实专家物理回放一致，再冻结新的有界对照训练及完整留出；保留关节限位和跳转检查。此方案尚未实施验证，不能承诺解决泛化或恢复。现阶段无需用户重新录制。

## 故障分类及计划修订

采集v1 EGL画面与OSMesa参考不同，恢复相同渲染后物理/RGB965步精确一致。数据准备的路径解析、父目录缺失已修复，失败目录未删除。

首次模型服务器因官方保存器加入 q01/q99=null 造成字节不同而拒绝；训练mean/std数值完全相同。保留原训练计划、v1/serve.log与frozen-evaluate-v1.py；独立evaluation-v2/plan.json仅允许移除可选null字段后精确比较数值。权重、控制和成功标准没有调整。非null分位数或数值变化仍拒绝。

## 可复现命令

以下在项目根目录执行，环境依赖沿用已固定版本。现有输出不会覆盖；重跑训练先用 --experiment 新名称 --freeze-plan 和新的 --plan 路径冻结计划（原计划刻意保留旧评测源码哈希）；重跑评测使用新的空输出目录。

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv-train/bin/python -u scripts/train_joint_pilot.py --manifest data/joint-pilot29-v3/manifest.json --plan reports/joint-pilot-20261029/v1/plan.json
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 .venv-train/bin/python -u scripts/evaluate_joint_pilot.py --manifest data/joint-pilot29-v3/manifest.json --plan reports/joint-pilot-20261029/v1/evaluation-v2/plan.json --serve --port 9877
# 服务器就绪后，在另一个终端逐条运行：
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-ppo/bin/python -u scripts/evaluate_joint_pilot.py --manifest data/joint-pilot29-v3/manifest.json --plan reports/joint-pilot-20261029/v1/evaluation-v2/plan.json --source human_1793121358375746008 --output evaluations/joint29-reproduce-train --port 9877
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-ppo/bin/python -u scripts/evaluate_joint_pilot.py --manifest data/joint-pilot29-v3/manifest.json --plan reports/joint-pilot-20261029/v1/evaluation-v2/plan.json --source human_1793122696228231033 --output evaluations/joint29-reproduce-holdout --port 9877
```

固定评测：600控制步，每步重规划；原生目标成立且20个连续控制无手指接触才成功；关节跳转上限0.35rad，只饱和夹爪[-1,1]，不裁剪关节冒充成功。固定推理噪声 NumPy default_rng(42+control_index) normal((10,32)) float32。

## 验证及备份

38项针对性测试及5项子测试通过，新增代码Ruff通过。四个策略视频帧全部解码；专家两段共967帧之前已完整解码。无独立子代理复核，不声称独立代码审查。本轮只有仿真，没有真机运动。

[Release](https://github.com/sencangWei/pi05-rl-handson/releases/tag/joint-interface-pilot-20261029) 保存原始配对数据与紧凑推理适配器；[文件清单](backup-manifest.json)记录每个文件和归档SHA256。适配器以官方pi05_libero为父权重，重建全部71参数张量并比对SHA256；完整优化器检查点仅本机。独立远端下载/恢复证明保存于本目录或 evidence/joint-pilot29-remote-restore.json；备份状态以证明文件为准。

恢复适配器验证（先将Release推理归档解压到项目根目录）：

```bash
JAX_PLATFORMS=cpu .venv-train/bin/python - <<'PYVERIFY'
import sys
sys.path.insert(0, "scripts")
import export_recovery_adapter as e
e.ASSET = "joint_contract29"
e.NORM_SHA256 = "156b0f71e2115988cac9ece57e15b576f1e7af9e7ab778f90a2dd74fcd990c12"
sys.argv = ["verify", "verify", "--adapter", "exports/joint29_pilot200_v1"]
e.main()
PYVERIFY
```

该验证逐个比较官方父权重+20个LoRA张量重建出的71张量与训练检查点记录，目标树SHA256为 `8e6d290f01b57e3b4eaa9cb1d733692d79569333e12da902067f3215784800d2`。适配器本身不能替代模型配置、官方基座、关节接口和本次归一化；更不能作为通过的策略部署。
