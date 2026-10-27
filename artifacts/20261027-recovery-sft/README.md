# 完整人工示范＋40 任务回放：恢复能力微调

本轮已完成真实微调和独立仿真。**没有通过晋级条件，不替换现用面板模型。** 开发集成功数由 10/12 降到 7/12；不能称为恢复能力修复。

## 实测结果

| 检查 | 训练前 SFT | 本轮 2000 更新 | 结论 |
| --- | --- | --- | --- |
| 抓空恢复（开发） | 4/4 | 2/4 | 退步 |
| 物体偏移（开发） | 4/4 | 4/4 | 未见成功率变化 |
| 放置失败恢复（开发） | 2/4 | 1/4 | 退步 |
| 开发总计 | 10/12 | 7/12 | 拒绝晋级 |
| 训练分布放置诊断 | 0/6 | 1/6 | 仅局部收益 |
| 四个旧任务、各两个初态 | 8/8 | 8/8 | 小样本保持检查 |

以上是单 seed、小样本结果，不代表完整 40 任务成功率。开发初态已经被观察，不是盲测。原恢复留出 ID10..13 继续隔离，没有用于本轮训练或选择。训练分布诊断指此前 PPO 使用的初态，不是人工示范所在场景。

[逐场结果与验证](evaluation-result.json) · [训练前后同步视频及轨迹](development-comparison/) · [失败分析](failure-analysis.json) · [失败时间拼图](failure-contact-sheet.jpg)

并列视频左侧 before、右侧 after，同一物理初态、20 Hz 仿真时间，未加速；先完成的一侧冻结到另一侧结束。原双相机视频也保留。文件名包含 empty_grasp、object_offset 或 placement_failure，以及初态编号。

## 发现了什么

五个失败都达到 300 个控制步，异常为空。不是少运行几步就被程序提前停下。轨迹和拼图显示多种失败：反复抓持、碗盘错位、倾斜碰撞、在盘子附近继续移动但没有重新抓碗；不能统一解释成“模型认为完成”或“只差松爪”。其中放置 init21、23 全程没有手指接触碗，末尾碗盘 XY 距离仍约 16.1/15.5 cm。empty_grasp init22 曾短暂触发原生判据，却未满足持续松爪要求，正确记为失败。

这里的 placement_failure 初态是**碗已经放偏在桌上、夹爪位于盘子上方**，并非所有场景都在抓着碗等待松手。对应恢复需要重新找到并抓起碗、搬运、释放和撤离。持碗松手的示范不等于覆盖了这个状态族。

## 数据来源与边界

只读复用旧项目中已批准的 **10 条完整人工示范、2050 个控制帧**，逐文件 SHA256 固定。预览见 [human-review](human-review/)。保留真实动作、20 Hz 控制时序、夹持、最终松手和撤离，没有通过加速视频改变训练动作。末段原始记录显示原生成功、夹爪张开与末端离碗至少 10 cm；这是采集记录检查，不是独立物理重放证明。

**人工示范来自 LIBERO spatial 黑碗任务（公开 task36），当前恢复评测来自 LIBERO goal task8。** 机器人、相机、OSC 动作契约兼容，但场景和文字指令不同。本轮是跨场景迁移实验，不把它们冒称为当前失败初态的配对纠正数据，也没有更改原文字指令。负结果不能单独证明人工录制质量差，或单独证明训练次数不足。

草稿、旧慢动作片段、旧项目公开数据划分没有引入。旧项目未修改。回放使用本项目固定公开数据的 1525 条训练 episode，40 任务均衡；168 条公开留出 episode 不用于梯度更新。

## 实际训练

父模型为本项目 40 任务 SFT `multitask40_lora_v1/9999`，继续训练 π0.5 LoRA，重新初始化 AdamW；**本轮没有新增 PPO 更新**。上一轮两组 PPO 奖励对照另见 [报告](../20261027-recovery-stage/README.md)。

- 2000 次更新，batch=1，seed=42，学习率 5e-6，20 步预热，梯度裁剪 1。
- 抽样槽位 25% 人工＋75% 旧任务回放；含预取的样本计数为人工 543、回放 1458，共 2001，不能称为恰好 2000 个已使用样本的统计。
- 正常退出 0，保存 `checkpoints/pi05_recovery_replay_lora/human28_replay2000_v1/1999`。训练调用约 375 秒，另有启动前完整哈希检查；systemd 峰值内存约 31.7 GiB，OOM=0，禁用该单元交换。
- 200 条每十步记录的指标均有限；loss 首条 0.0833、末条 0.0147。loss 降低没有转化为本次恢复成功率提升。

冻结协议、真实运行环境、父模型文件哈希、配置、抽样和退出状态位于根目录 evidence 下的 `recovery-sft28-*`、`human28_replay2000_v1-*`。训练前代码提交为 `0ee727038b73e7a4f9e0500aaaae12672ca8bfdc`。训练环境快照是在此前审计阶段捕获，commit 字段记录捕获时间；实际运行脚本 SHA 以 status 内记录为准。

## 验证与遇到的问题

恢复评测固定相同源代码、初态 SHA、非权重配置、随机噪声、每次执行 5 步后重规划、10 个 flow 步、300 控制步上限。成功要求原生目标成立，且连续 5 控制步无手指接触。旧任务保持检查使用其原生任务成功判据，不额外冒称通过严格松爪检查。

JAX 原始 LoRA 与合并 Torch 权重的 CPU float32 单步 flow 数值对照通过，最大绝对误差 1.43e-6（门限 1e-3）；这不能替代实际 BF16、多步仿真评测。18 个新恢复评测原视频为 20 fps；8 个旧任务保持检查由既有评测器以 10 fps 导出，虽每控制步均保留一帧，但比 20 Hz 仿真时间慢放两倍。训练前后保持检查使用同一导出器，不能把这种播放速度差误认作模型动作慢。26 段均检查帧数为控制步＋1并完整解码；12 段恢复并列视频为 20 fps，也完整解码。

首次全测试未从进程启动就固定 CPU，先导入的 JAX 初始化 GPU 后使 Flax LoRA 测试触发 TF32 精度差异，35 通过、1 失败。固定 `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=` 后，**36 项通过**，没有放宽断言或改动模型。原日志保留。

旧任务历史基线的评测脚本 SHA 与本轮不同，身份检查主动报错。因此保留原记录，并在当前同一脚本下重新运行训练前模型的 8 个初态，再进行严格匹配；不直接忽略源代码差异。独立审查代理因账户不支持其模型未运行，没有声称独立审查通过。

## 复现与备份

```bash
# 已有相同环境、完整公开数据与匹配父模型；训练输出目录必须全新。
JAX_PLATFORMS=cpu .venv-train/bin/python scripts/train_recovery_replay.py --dry-run
.venv-train/bin/python -u scripts/train_recovery_replay.py

# 原始 checkpoint 导出并做数值检查；目标目录必须全新。
.venv-train/bin/python scripts/export_sft_for_ppo.py \
  --checkpoint checkpoints/pi05_recovery_replay_lora/human28_replay2000_v1/1999 \
  --output models/human28_replay2000_v1
.venv-train/bin/python scripts/check_export_parity.py \
  --checkpoint checkpoints/pi05_recovery_replay_lora/human28_replay2000_v1/1999 \
  --export models/human28_replay2000_v1 --work evidence/新数值检查目录
.venv-ppo/bin/python scripts/evaluate_recovery.py --split dev \
  --manifest data/recovery-bowl-dev27/manifest.json \
  --config evidence/human28_replay2000_v1-composed.json --output evaluations/新开发评测目录
```

[公开 Release：人工数据与精确 LoRA 备份](https://github.com/sencangWei/pi05-rl-handson/releases/tag/human-recovery-sft-20261027)。备份包含 20 个 LoRA 张量和归一化；需要相同官方 `pi05_libero` 基座，重构后逐一核对全部 71 个参数叶子的形状、dtype、内容哈希。该备份用于推理，不包含优化器状态，不能冒称完整断点续训包。完整 Orbax 训练检查点仍保留在本机。

已从 Release 独立下载全部 185471769 字节，SHA256 为 `9023c7b5bd1b1a851435665d0e585a9242b324ec4d050b8d2d50ae85cc62ca56`，重构的全部 71 个参数叶子逐一一致：[权重重构证据](../../evidence/recovery-sft28-adapter-restore.json)。下载解压后用以下命令核对重构内容（路径为解压后的 adapter 目录）：

```bash
JAX_PLATFORMS=cpu .venv-train/bin/python scripts/export_recovery_adapter.py verify \
  --adapter exports/human28_replay2000_v1
```

[数据恢复](../../evidence/recovery-sft28-data-restore.json)、[权重下载校验](../../evidence/recovery-sft28-adapter-download.json)、[代码恢复](../../evidence/recovery-sft28-remote-files.json)、[实际 RLinf 算法补丁恢复](../../evidence/recovery-sft28-remote-runtime.json)均保留原始证据。提交 `9c5b6ec` 从用户远端独立恢复，1345 个跟踪文件的工作内容与提交 blob 逐一一致；干净上游重放 9 个补丁，12 个实际运行时文件哈希一致。没有将虚拟环境、官方基座或全部 35 GB 公开数据重复上传；跨机器还需恢复对应依赖和固定版本的仿真工程。

## 下一步应该改变什么

优先从**当前 goal 训练失败状态族**采集完整成功恢复，覆盖抓空后抬起重抓、碗放偏后的重新抓取、最终释放撤离；保留失败起点和正确恢复动作。然后固定回放比例做有界对照，并用新的独立初态检查。不能把已观察开发集混入训练后继续称其为泛化测试。现有证据不支持直接延长这轮跨场景微调或用手写轨迹冒充模型恢复。
