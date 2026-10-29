# 统一关节动作训练对照（2026-10-29）

**本轮两组训练及八次实际策略评测已完成，均未通过。此版还发现关节状态没有进入神经网络，不能作为完成的统一关节输入方案。修正另见 [状态输入修正版](../unified-joint-state-training-20261029/README.md)。**

本轮接入三条用户新采、已批准且独立重放通过的示范。目标是验证 π0.5 LoRA 能否使用统一的关节动作接口执行抓取/放置恢复；示范重放成功不算模型成功。

## 固定接口与数据

- 数据字段：主视角 RGB、腕部 RGB、英文任务文字、当前七关节角与两个夹爪关节位置（9 维）。**更正：本版 `discrete_state_input=False`，PI05 也没有连续 state 分支，因此网络只消费画面与指令；记录的9维状态没有进入网络。**本轮只比较动作表示，不扩展语音任务。
- 下发动作：七个关节目标角（弧度）＋连续夹爪指令（8 维）；20Hz，PD kp=150，kd=2√150。没有写入下一帧状态或专家兜底。
- 绝对组直接预测七关节目标；相对组预测相对本次观测关节角的目标偏移。整段未来 10 步使用同一个观测锚点，执行前加回该锚点，夹爪不相减/累加。
- 三条新示范：290、335、1006 帧，保留实际下发的关节目标，包括第三条的肘部调整。原始录制文件不改写、不抽帧、不加速。
- 另回放一条旧关节专家示范 491 帧。旧数据是重新执行产生的专家目标，不能描述成原始历史人工关节指令。
- 训练共 2122 帧、2086 个完整 10 步窗口。四条示范等概率：各扩成 997 个均匀分布的抽样槽，共 3988 槽。固定 seed42 无放回打乱，本轮前1000个梯度样本的来源为 227、229、285、259 个。
- 整条旧示范 434 帧留出，不参与归一化或梯度；它曾用于开发检查，不能称全新盲测集。旧示范额外诊断保持帧排除。

## 同条件训练

两组都从官方 `pi05_libero` 父模型重新开始，LoRA，fresh AdamW，clip=1，无 EMA。batch1、1000 次更新、lr2e-5、warmup20、预测10步、seed42。1000次是有界接口对照预算，不是“已经收敛”的判断。较低学习率限制新动作表示适配时的变化；是否够用由真实仿真判断。

父模型本地位置 `/home/robot/voice-arm-sim/models/pi05_libero`；OpenPI固定源码 `15a9616a00943ada6c20a0f158e3adb39df2ccac`。两组的冻结配置、输入 SHA256、实际采样顺序、状态和 checkpoint 索引分别在 `absolute/`、`residual/`。

`data-contract-audit.json` 确认 3988 个槽的状态一致、夹爪不变，float32 相对编码解码最大误差 1.49e-8 rad。此检查只证明数据表示转换正确。

## 预先固定的实际仿真评测

两组都从相同的三条新人类示范初态、同一条旧留出示范初态启动；每个进程恢复完整物理快照，检查第一帧关节状态和两路 RGB 精确相等。每个控制步重新观测，预测10步，仅执行第1步。最大1200控制步（60秒仿真时间），另报告600步以内完成；不是训练步数。允许推理开销导致墙钟时间更长。

成功必须连续20步满足：原任务完成条件、夹爪指令为张开、实测夹爪宽度≥6cm、无碗接触、末端距碗≥10cm。只把碗碰到盘子不算完成。关节目标越物理限位或单步目标相对当前关节变化超过0.35rad会拒绝执行，并保留原始预测与失败视频；不通过静默裁剪关节伪装策略成功。

## 实际结果与配置问题

两组均完成1000次梯度更新，绝对组203.68秒，相对组177.75秒（不含启动前父模型校验）。绝对组 loss 由0.5126到0.0630，相对组0.4166到0.1903，均为训练日志 step0 / step990，不能比较两种标签空间的数值高低。

| 同一快照来源 | 绝对组执行步数 / 原因 | 相对组执行步数 / 原因 |
| --- | --- | --- |
| 新人工1 | 0 / 首目标跳变0.957358rad | 328 / J2超过物理上限 |
| 新人工2 | 0 / 首目标跳变0.566632rad | 166 / J4超过物理上限 |
| 新人工3（含肘部调整） | 0 / 首目标跳变0.810854rad | 134 / J4超过物理上限 |
| 旧留出 | 0 / 首目标跳变1.494395rad | 6 / J4超过物理上限 |

绝对0/4，相对0/4；全部在1200步上限之前被目标校验拦截，因此这次失败不是步数用完。绝对组视频只有初始帧，不能称为运动演示。相对组有实际模型动作；没有专家动作替换。`evaluation-matrix.json` 和每例 `trace.jsonl` / `result.json` / `states.npz` / `policy.mp4` 保存具体证据。

诊断发现：`ModelTransformFactory` 在 PI05 且 `discrete_state_input=False` 时不将状态编码到 token，而 `Pi0.embed_suffix` 的连续状态分支只对非PI05启用。数据形状检查通过，仍不等于模型读到了状态。这是此次实现漏检，不能凭此要求用户重录数据，也不能凭低loss宣布训练有效。

修正版独立使用 `discrete_state_input=True`，训练前逐个改变九维状态并验证真实官方归一化与分词链路的 token 变化，保持其余条件不变。旧版源码、冻结计划和失败记录保留。本轮没有切换用户中文操作面板默认模型，没有碰真机。

## 复现命令

在 `/home/robot/pi05-rl-handson` 中执行；路径须有本地原始数据与父模型缓存。

```bash
.venv-train/bin/python scripts/train_unified_joint.py --manifest data/unified-joint29-v1/absolute/manifest.json --plan reports/unified-joint-training-20261029/absolute/plan.json --representation absolute --experiment human_joint29_absolute1000
.venv-train/bin/python scripts/train_unified_joint.py --manifest data/unified-joint29-v1/residual/manifest.json --plan reports/unified-joint-training-20261029/residual/plan.json --representation residual --experiment human_joint29_residual1000
```

已存在的训练目录会拒绝覆盖，需明确另起实验并重新冻结计划。分别启动一个模型服务：

```bash
.venv-train/bin/python scripts/evaluate_unified_joint.py --manifest data/unified-joint29-v1/absolute/manifest.json --plan reports/unified-joint-training-20261029/absolute/plan.json --serve --port 9877
```

另一个终端实际执行一例：

```bash
.venv-ppo/bin/python scripts/evaluate_unified_joint.py --manifest data/unified-joint29-v1/absolute/manifest.json --plan reports/unified-joint-training-20261029/absolute/plan.json --source human_1793267780353445970 --output evaluations/joint29-new-output --port 9877
```

相对组将两处 `absolute` 改为 `residual`，先停止自己启动的前一个服务。每个输出目录必须新建；默认保留失败结果。

## 验证与备份范围

82项单元检查及5项子检查已通过（包含修正版配置回归），覆盖新数据窗口、同一锚点转换、控制器、物理快照和采集接口。独立代码审阅子代理因账号不支持配置模型而未能运行，本轮由主代理检查并用实际执行补证。

代码、配置、报告将提交到用户拥有的 `sencangWei/pi05-rl-handson` 实验分支，并从远端内容独立恢复核对。原始录制 NPZ、训练 Parquet 和完整模型 checkpoint 仍在本机，不能把代码备份称作已备份这些大文件。
