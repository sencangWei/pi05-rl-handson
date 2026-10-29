# 公开正常示范 + 人工恢复：关节 π0.5 实验

目标：补上正常抓取、搬运、放置示范，保留已经验收的人工恢复示范，训练独立的关节动作策略。当前阶段为数据转换与训练准备；结果以 `train-status.json` 和实际评测文件为准。

## 数据来源与范围

- [官方 LIBERO 数据](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a)：`libero_goal/put_the_bowl_on_the_plate_demo.hdf5`，固定 revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`。
- 原始文件 468,246,288 字节，SHA256 `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406`，50 条正常示范，4,669 个运动控制步，20 Hz。50 个初始物理状态全部不同。
- 原始动作是末端 OSC 控制；公开文件中的关节位置不能直接当作关节指令。本实验在真实连续仿真中重新执行关节指令，重新渲染动作前图像并记录实际执行目标。
- 转换前固定源编号 0–39 为训练集、40–49 为留出集。失败转换只排除，不更换分组。最终合格数量见 `data/public-joint30-v1/preparation.json`。
- 3 条已认可的人工恢复示范原样保留，1,631 帧；新采样按正常示范 80%、人工恢复 20%，组内按轨迹平衡、轨迹内覆盖完整动作窗口。人工恢复初态用于训练，因此其评测不计为未见初态泛化。
- 每条公开轨迹最后 20 步仅用于检查放置后稳定松爪，**不进入训练**，避免额外加入静止帧。
- 这是碗放盘子的单任务实验，不能据此声称具有广泛家务、多物体或通用恢复能力。

另查到 [500 条公开关节轨迹](https://huggingface.co/datasets/jiankimr/libero_10_joint_lerobot/tree/2260df157dbdefcbfbc0e3629f63041a13253c66)，其任务是 LIBERO-10，控制增益 KP2000，关节增量与夹爪约定也不同。它不对应当前碗盘任务，未直接混入本轮训练；详见 `source-research.md`。

## 采集、训练、推理契约

输入：主视角 RGB、腕部 RGB、文字指令、7 个实测关节角和两指位置，共 9 维状态。

模型预测：10 步、每步 8 维。7 个关节分量使用“未来绝对目标减去当前实测关节”的残差；窗口内所有目标使用同一个当前锚点。推理时加回锚点后发送绝对关节目标，夹爪正值关闭、负值打开。

底层：KP150、KD=2√150，20 Hz。示范重执行不逐帧注入下一物理状态。逐轨迹验收有限值、目标未裁剪、时间步、关节/末端误差、柜子额外穿透，以及连续 20 步原生目标成立且已经松爪。

状态归一化使用训练集的中点/半范围，把实际训练状态保留在 π0.5 状态分词器的 [-1,1] 范围内；动作使用采样加权的总体均值/标准差。留出集不参与统计。

公开数据评测采用连续物理后的传感器刷新，与新采集方式一致。旧人工录像采用历史状态重生成刷新，其路径保留；两者的摄像机刷新相位差异是数据限制。`fresh-demo0-verification.json` 记录了独立重执行 110 步的逐步物理和画面对照。

## 运行命令

项目根目录 `/home/robot/pi05-rl-handson`，仅仿真，不调用真机。

```bash
# 重执行与配对（源文件先用 fetch_public_joint_source.py 下载并校验）
.venv-ppo/bin/python scripts/public_joint_replay.py --help

# 全部转换完成后：生成独立数据集及训练集统计
.venv-train/bin/python scripts/public_joint_dataset.py \
  --replays /dev/shm/pi05-public-joint30/replays \
  --output data/public-joint30-v1 \
  --human-manifest data/unified-joint29-v1/residual/manifest.json

# 冻结、静态路径验证、实际训练分开运行
.venv-train/bin/python scripts/train_public_joint.py \
  --manifest data/public-joint30-v1/manifest.json \
  --plan reports/public-joint30/run/plan.json \
  --experiment public_normal_recovery_10000 --updates 10000 --freeze-plan

.venv-train/bin/python scripts/train_public_joint.py \
  --manifest data/public-joint30-v1/manifest.json \
  --plan reports/public-joint30/run/plan.json \
  --experiment public_normal_recovery_10000 --updates 10000 --dry-run

.venv-train/bin/python -u scripts/train_public_joint.py \
  --manifest data/public-joint30-v1/manifest.json \
  --plan reports/public-joint30/run/plan.json \
  --experiment public_normal_recovery_10000 --updates 10000
```

配置：官方 `pi05_libero` 父模型、LoRA、关节状态输入开启、batch1、seed42、10,000 次更新、学习率 5e-5，100 步预热后余弦降到 5e-6。单独实验目录，旧数据和 checkpoint 不覆盖。

评测预先固定为通过转换门槛的公开留出轨迹及 3 条人工恢复初态，每次最多 600 控制步、每步重规划。成功要求原生目标成立、发出张爪指令、指宽 ≥6cm、无指尖接触，持续 20 步。10cm 撤离另列诊断，不作为公开示范没有要求的隐含成功标准；无专家动作回退。

## 存储与复现边界

磁盘几乎已满，转换的批量画面/快照和完整训练 checkpoint 暂存 `/dev/shm`，重启会丢失。精确的原始状态、动作、场景 XML 已保存在本目录 `source-actions-states.hdf5`，SHA256 `d7415020d15e6898ff1b7541b87cee6f4ae08c297b91f8d2b386b524af2ee51a`；完整下载来源、环境及转换脚本也保留。

训练完成后使用 `export_public_joint_adapter.py` 导出并恢复验证紧凑 LoRA。该导出用于推理，不包含优化器/RNG，不能冒充完整训练续跑 checkpoint。远端备份及独立恢复证据完成后再记录具体地址和哈希。
