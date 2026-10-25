# π0.5 多任务模仿学习与强化学习实验

独立目录 `/home/robot/pi05-rl-handson`；不改旧训练项目、中文面板或真机。

## 当前状态（2026-10-26）

- 后台服务 `pi05-multitask-night-20261026.service` 已启动，正在下载并逐文件校验数据。
- 新 SFT **实际更新 0 步**。GPU 空闲约978 MiB；队列要求至少24576 MiB，绝不停止其他任务。
- 全部数据下载校验后，先做 CPU 预检，再等显存最多8小时，满足条件才自动启动10000步 LoRA。等待时间不包含下载时间。
- PPO **实际更新 0 步**。已修改实际 RLinf 奖励函数并执行单元测试，配置可由上游 Hydra 组合；运行环境、PyTorch 权重、显存与权重衔接尚未验证。队列不会自动启动未准备好的 PPO。
- 现役 LoRA 四任务各测一个初态，4/4成功；另做碗放盘无手指接触5步检查，94步成功。这是既有模型推理，不是本次训练结果。

## 数据、奖励和机器人互动的关系

视觉语言预训练提供识别与语义基础；机器人示范把图像、指令、状态与动作对应起来；强化学习用环境反馈改善完成率、效率，以及训练分布覆盖的失败恢复。聊天、规划与低层控制通常分工，具体组合因系统而异，并非所有机器人都会在线自主学习。

奖励需要可靠的完成标准，策略还必须实际更新并独立测试。仅添加奖励不会立即产生恢复能力。仿真随机化、失败初态、示范回放和实机验证同样重要。本实验没有真机验证，也没有通用家庭能力的证据。

## 固定数据与动作契约

[Physical Intelligence LIBERO](https://huggingface.co/datasets/physical-intelligence/libero)，CC BY 4.0，版本 `a4336d589d589045d1c56423ffdf3b88a0e19b1f`：40类、1693条、273465帧，来自 spatial/object/goal/10 四组。Parquet 约34.94GB，逐文件核对 SHA-256 与字节数。

比此前单一碗任务覆盖广，但仍是桌面机械臂仿真，不覆盖家庭所有物品和能力。选择它是因为与当前 Panda、相机、动作接口兼容；跨机器人真实数据需要动作、坐标和频率适配，不能直接混训。

- 两路RGB：外部与腕部，256×256，10Hz。
- 状态8维：末端位置3、轴角3、双指位置2；动作7维：末端增量6、夹爪1。
- 窗口10步，官方模型内部补齐动作至32维；不跨示范，尾部重复末动作。
- 每任务按 `sha256("42:episode_id")` 固定划分：训练1525条、留出168条。先均匀采任务，再近似均匀采控制帧。
- 使用官方父模型归一化；不在留出集拟合统计量。父模型可能已见过公开数据，留出仅保证不用于本轮更新，不能称为未见任务泛化。

[dataset-manifest.json](evidence/dataset-manifest.json) 的 ready=true 表示完整下载校验通过；训练还要求 data-validation.json 与该清单哈希相同。未下载完不允许训练。

## SFT 参数与身份

OpenPI 记录版本 `15a9616a00943ada6c20a0f158e3adb39df2ccac`；准确源码快照在 vendor/openpi-source.tar.gz，含许可证。evidence/openpi-source.json 记录压缩包和141个源码文件的哈希。

父模型为只读缓存 `gs://openpi-assets/checkpoints/pi05_libero`，不是现役人工纠错模型，不是从零训练。父模型清单和校验保存在 evidence/parent-*.json。

| 参数 | 有界第一轮 |
| --- | --- |
| 可训练参数 | 仅LoRA，冻结其余 |
| batch | 每次更新1个窗口 |
| 更新次数 | 10000 |
| 学习率 | 2e-5，20步预热后恒定 |
| seed | 42 |
| 保存间隔 | 2000，另存最后一步 |
| 输出 | checkpoints/pi05_multitask40_lora/multitask40_lora_v1 |

10000次采样分散到40类，不足遍历全部训练帧；这是链路与趋势实验，不能用训练时间或训练loss断言能力达标。

## PPO 奖励及尚未完成的部分

[RLinf π0/π0.5 PPO](https://github.com/RLinf/RLinf/blob/main/docs/source-en/rst_source/examples/embodied/pi0.rst)，固定 `30e0236981c6a1dc19202de64116834c12673828`。

实际环境只改一行，将固定每步扣1改为可配置系数：[补丁](configs/rlinf-step-cost.patch)。训练与评测统一：

```text
reward = 1.0 × LIBERO 原始成功标志 - 0.001 / 控制步
use_rel_reward = false
```

普通步-0.001，成功步0.999。关闭相对奖励避免重复时间惩罚被差分抵消。修正上游评测倍率与训练不同的问题。没有额外抓取奖励、碰撞惩罚或学好的恢复策略；额外松爪检查目前仅用于评测，未加入PPO奖励。

配置限定goal中抽屉、灶炉、碗放盘、酒瓶放架四类，单环境，300控制步/rollout，10步动作块，micro batch1/global batch2，CPU卸载，2个rollout epoch。30动作块可被global batch整除。尚未完成Ray/FSDP运行验证，不能认定32GB显存可运行。

需要的 RLinf PyTorch 权重 `RLinf/RLinf-Pi05-LIBERO-SFT` 尚未下载。JAX SFT 到 RLinf PyTorch 的LoRA权重衔接未实现，**两段配置不是已连通的SFT→PPO链路**。评测seed43不保证初态与训练不重合，此PPO小实验不是最终泛化评测。

恢复训练还需加入夹偏、抓空、物体移动后的状态，考察恢复后是否完成任务，并回放常规任务检查遗忘。仅扣时间分不能代替这些实验。

## 实际基线

动作来自现役 counterfactual_v2/7999 的 pi05.sample_actions，端口8766；没有手写抓取轨迹替换模型。只读复用旧工程 b0334d6424d85de642ac06248ddfe2009c36ee45 的rollout，CPU OSMesa渲染，所有新结果写入本项目。

| 任务 | 初态 | 成功判据 | 控制步 |
| --- | --- | --- | --- |
| 抽屉 | 0 | 原始判据成功 | 117 |
| 灶炉 | 0 | 原始判据成功 | 84 |
| 碗放盘 | 0 | 原始判据成功 | 83 |
| 酒瓶放架 | 0 | 原始判据成功 | 159 |
| 碗放盘，另测 | 0 | 原始判据且连续5步无手指接触 | 94 |

视频、动作、推理记录和模型身份在 [demo/](demo/)。视频按控制帧播放，不等于真实耗时。每项只有一个初态，不代表稳定成功率。此基线与新SFT父模型不同，判断训练改善还必须补测同一官方父模型、同初态对照。

## 状态与验证命令

```bash
cd /home/robot/pi05-rl-handson
systemctl --user status pi05-multitask-night-20261026.service --no-pager
cat evidence/night-status.json
cat evidence/download-status.json
tail -n 20 logs/night-queue.log
```

后台顺序：完整下载→行/图像校验→CPU预检→显存等待→有界SFT。阶段失败记录后退出，不覆盖旧实验。机器重启或用户服务管理器退出后，此临时服务不会自动恢复。有服务活动时，不要同时另启任务写同一个实验。

```bash
.venv-train/bin/python -m unittest discover -s tests -v
.venv-train/bin/ruff check scripts tests
.venv-train/bin/python -m compileall -q scripts tests
.venv-sim/bin/python scripts/check_ppo.py
```

## 恢复源码

大型数据、环境、权重和checkpoint不提交。源码快照、补丁、配置、数据身份和小型基线证据提交备份。

```bash
mkdir -p third_party data/libero/meta .local logs
tar -xzf vendor/openpi-source.tar.gz -C third_party
git clone https://github.com/RLinf/RLinf.git third_party/RLinf
git -C third_party/RLinf checkout 30e0236981c6a1dc19202de64116834c12673828
git -C third_party/RLinf apply ../../configs/rlinf-step-cost.patch
cp evidence/dataset-meta/* data/libero/meta/
uv venv --python 3.11.15 .venv-train
uv pip install --python .venv-train/bin/python -r configs/training-requirements.txt
```

这些命令恢复训练源码环境，不提供父模型缓存、完整数据或RLinf PPO运行环境。父模型路径固定在 scripts/training_config.py；换机器需先取得相同身份权重并校验，再明确修改该位置。基线另依赖固定版本旧工程及服务，不能仅凭恢复源码声称完整训练已复现。

上游OpenPI许可证随源码快照保留；LIBERO遵循CC BY 4.0，来源与固定版本见上文；RLinf许可证随固定版本源码提供。
