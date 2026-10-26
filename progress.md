# 进展

2026-10-26：用户批准 VLA + 强化学习路线。读取技能、检查 GPU/磁盘/现有服务，创建独立工程。尚未启动 RL 训练。

凌晨：固定1693条40任务清单，CPU首帧官方变换通过。真实现役LoRA四任务单初态基线4/4，额外碗松爪检查成功。

奖励一行补丁、训练/评测统一倍率、PPO配置组合与4项契约测试通过；JAX到RLinf PyTorch权重衔接仍缺，PPO更新0。

后台用户服务pi05-multitask-night-20261026.service已启动下载，数据完整通过后等待显存。当前GPU空闲978MiB，门槛24576MiB，新SFT更新0，不停止其他任务。后续以evidence/night-status.json及各阶段日志为准。

独立只读审查发现源码与父权重身份只核对标记/大小的缺口；已补逐文件SHA-256和父manifest身份比对，防止同大小内容改变漏检。增加同大小损坏和路径越界回归，6项测试、lint及编译检查通过。父模型16个文件全量哈希已通过。实际模型构建前再次检查显存，避免哈希期间资源变化导致误启动。

代码备份：公开仓库 https://github.com/sencangWei/pi05-rl-handson ，分支experiment/pi05-ppo，算法提交5aa88b3c29fcba64d85678bfac90ce3bd0a86a90。从网络独立克隆，176个跟踪文件逐字节核对、141个OpenPI源码哈希核对、奖励补丁在干净固定RLinf上游重放并核对内容哈希；恢复目录6项测试通过。证据evidence/remote-restore-verification.json。大型数据、环境和权重未上传。

最终状态快照：后台下载本轮445/1693，暂无下载错误；GPU空闲978MiB，SFT/PPO实际更新均0。下载服务仍运行，后续不能把配置或基线录像当作新训练结果。

08:15 更新：上段是凌晨快照。1693/1693已下载，全量行数和首尾相机帧校验通过。仍仅有978MiB空闲显存；旧服务没有卸载接口，遵守不终止其他GPU任务约束。

新增CPU专用有界测速入口，强制隐藏CUDA、最多10次更新。pi05-cpu-probe-20261026.service限制12逻辑CPU、36GiB内存、禁止swap、30分钟。实际3次迭代已于08:14:58保存完成，服务正常退出，训练调用耗时150.42秒（含初始化、编译和保存）；循环72秒，稳定步约23.3秒。首步学习率0、后两步非零。计划10000步约65小时，故未盲目启动长CPU作业。它证明CPU链路可行，不证明能力提升或多任务训练完成。

PPO所需固定上游权重下载及完整哈希校验入口已补，独立pi05-ppo-model-20261026.service正在下载。该权重与JAX LoRA不是同一条训练血缘；LoRA→PyTorch保真转换、PPO环境及显存仍需解决。直接RLinf FSDP路径调用CUDA current_device，不能仅隐藏CUDA就宣称可用CPU跑PPO。

08:21 更新：CPU checkpoint独立恢复通过，train_state.step=3；51个冻结基座张量与只读父模型转换为bf16后完全一致，LoRA有限且已更新，记录在evidence/cpu-probe-checkpoint-validation.json。7项契约测试、ruff、compileall、安装脚本bash语法和Hydra真实配置组合通过。

独立PPO环境安装服务pi05-ppo-env-20261026.service正在运行。固定上游安装器采用Torch2.7.1而非默认2.11，禁用root/可选FlashAttention，使用本项目独立虚拟环境、缓存、LIBERO配置和Git配置。尚未宣称运行验证或PPO更新。权重下载同时进行；两者均不使用GPU。

CPU/PPO准备代码备份：d1d630ecd7118cdd0f39736a5819a9dbbdcdf82d已推送公开仓库experiment/pi05-ppo。从远端获取后，在新分离工作区逐字节核对189个文件，141个OpenPI源码哈希及压缩包哈希一致，干净固定RLinf上游奖励补丁重放一致，恢复后7项测试、lint及bash语法通过。权重和环境本体没有上传，PPO下载/安装仍在运行，不能称为训练完成。
