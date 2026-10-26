# 已知事实

- 2026-10-26：RTX 5090 D 32607 MiB，现有进程占用 31134 MiB；不能假定有训练空间。
- 现有 8766 服务为 π0.5 LoRA counterfactual_v2/7999，不是未微调官方模型。可评测当前基线，不能混称官方基线。
- 官方 π0.5 LIBERO JAX 权重已在旧项目缓存；RLinf 使用其自身 PyTorch 实现，需核对权重和依赖转换，不直接假定 JAX checkpoint 可载入。
- 旧工程 /home/robot/voice-arm-sim 干净，HEAD b0334d6424d85de642ac06248ddfe2009c36ee45；只读复用，不改其面板。
- 可用磁盘约254G，内存62GiB，available43GiB。
- 官方PI LIBERO40类1693条273465帧，约34.94GB，与现役接口兼容；不是家庭通用数据集。
- 每任务固定划分训练1525条、留出168条；父模型可能见过公开数据，不能称未见任务。
- 当前LoRA四个任务单初态成功；另做碗松爪5步检查，94步成功。这是已有模型推理，新训练更新0。
- RLinf相对奖励会抵消重复时间成本，已关闭。实际方法执行测试通过：普通步-0.001、成功步0.999。
- 上游评测奖励倍率与训练不同，已统一1。30动作块无法被batch4整除，改batch2并加检查。
- 新训练环境依赖本项目OpenPI快照；父模型缓存只读复用。首帧官方变换通过10×32动作检查。
- PPO运行环境、PyTorch权重及JAX权重衔接尚未完成。没有已验证的自动PPO训练链路。
- 审查修正：训练前将实际OpenPI源码逐文件与固定快照哈希比对，父manifest与备份身份比对，再逐文件验证权重SHA-256；不再只信版本标记和字节数。

08:15：数据全部1693条273465帧校验通过；GPU仍空闲978MiB，现役JAX/Molmo/Qwen无释放接口。CPU24核，内存可用约45GiB。准备独立有界CPU SFT基准；未终止其他进程，未改旧项目。
- CPU真实测速完成：3迭代循环约72秒，后两步各约23.3秒，连同初始化与checkpoint保存150.42秒。CPU参数能真更新，但10000步约65小时，不满足一晚完成。独立服务正常退出，没有停止其他GPU任务。需要区分优化迭代3与非零学习率更新2。
- RLinf安装器应显式选择Torch2.7.1（匹配rlinf-openpi0.1.1）；默认uv sync会选择2.11。可跳过flash-attn源码编译，但需要独立LIBERO配置目录以免重写旧路径。旧JAX LoRA不能直接由只映射密集权重的转换器接到PPO。

08:26 更新：PPO独立父权重下载服务正常退出0；固定版本45ccfcc4e28634f1576ebf78cab0fbe2fd82432d的3个文件全部通过大小和完整哈希校验，model.safetensors为7473091464字节，SHA-256为4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7。见evidence/rlinf-model-validation.json。独立PPO依赖仍在下载，GPU主训练/PPO更新仍为0，SFT到PPO的LoRA衔接仍未实现。

17:47 更新：用户已自行关闭旧GPU推理服务，启动前空闲31574MiB，JAX GPU矩阵运算通过。独立服务pi05-multitask-sft-20261026.service于17:43启动，真实日志已观测Step 900、loss=0.0113；40任务10000步SFT正在更新，未完成。服务使用原night.lock，限制6小时/48GiB/禁swap。源码、父权重和数据前置核验通过。原夜间队列11:06等待超时，已另存历史；PPO依赖原安装09:17超时，已启动4小时独立重试。PPO更新仍0，JAX LoRA到PPO权重衔接仍未实现。启动环境、服务参数和真实日志节选均已落盘。

21点更正与发现：
- 新SFT已经完成10000次迭代；此前“正在训练”的记录为17:47历史快照。
- PPO代理路径TLS握手全部超时，Aliyun直连8MiB探测206成功约7MB/s。安装器只绕过镜像与PyTorch主机代理，不修改系统/旧项目环境。
- 注意力输出LoRA实际方程BTNH,NHL->BTL; BTL,NLD->BTD在两阶段都约掉head，dense增量应为A@sum_heads(B)，普通逐头A@B不能保真。真实固定Flax模块测试已捕获并验证修复。
- RLinf读官方权重的tied embedding别名lm_head.weight，直接mapping只产出embed_tokens.weight；导出器需显式重命名，严格加载门可捕获此类遗漏。
- 新SFT服务端口8775被已有进程占用；未停止该进程，改9875继续。PPO训练更新仍0。

21:35 仿真更新：新SFT四任务×双初态8/8，碗放盘松爪额外2/2；官方父模型同哈希初态8/8，酒瓶初态0由299降至131步，其余有快有慢。18段真实视频保存artifacts/20261026-sft-simulation。PPO严格加载811→667张量通过；float32整网对照最大误差0.001248未达既定阈值，正定位GELU等数值差异。Torchcu126在5090D实测no kernel image，独立cu128修复进行中。PPO更新仍0。配置已改为本次合并SFT血缘、horizon10/discreteFalse、正确归一化、all_linear适配器与orig_params，真实运行仍待门槛通过。
# 2026-10-26 21:49：PPO启动前实际修复

- 5090D需要sm_120内核，Torch2.7.1+cu126实测不兼容；同版本cu128实际BF16矩阵与SDPA反传通过。
- JAX→RLinf合并权重strict加载通过后，整网固定噪声检查仍失败。RLinf SigLIP和Gemma的GELU默认erf，与Flax tanh近似不同；configs/rlinf-gelu-parity.patch实际修复后，未经放宽的1e-3阈值检查通过（最大误差1.66893e-6）。
- 上游PPO示例默认BF16但未包含当前校验所要求的FP32模块规则；按固定源码docs/source-zh/rst_source/extending/new_model_fsdp.rst的Pi05配方设置FP32参数存储、BF16嵌入及分组规则，包含新增value_head。validate_fsdp_cfg已通过，实际更新尚未验证。

21:57：rollout通过OmegaConf.merge继承actor.model；actor.precision=fp32为FSDP参数存储，不能同时让无FSDP的rollout使用该值。原生Attention将BF16 token直接传nn.Linear，FP32权重时最小测试实际报same dtype，BF16权重时执行通过。rollout明确设置precision=bf16；actor配置不变。此前v2模型加载阶段内存逼近48GiB上限，内存事件max>0但oom_kill=0；未将其误称为GPU训练OOM。

22:13 更新：v3完成300控制步采样，首个训练前向在SigLIP head的PEFT/FSDP边界报FP32/BF16不匹配，实际PPO更新0。PEFT读取未展开权重的FP32 dtype后转换输入，子FSDP随后展开BF16权重，故启用子边界输入转换。Torch2.7也在根_pre_forward执行此转换，另补根策略副本禁用转换，避免PPO chains精度损失；没有原地修改与子节点共享的策略。真实小模型CUDA前后向与真实RLinf包装器的根chains FP32检查PASS。v4在采样/更新前主动停止，保留日志；v5服务22:12启动。独立原生评测入口和5项实际API契约测试通过，待PPO成功后与原生SFT对照。

## 2026-10-26 22:26 CST 保存内存故障与重跑
- v5 完成首轮采样、训练调用和内置评测（任务0两次成功），保存 step1 时触发48GiB CPU cgroup上限，退出oom-kill；没有持久化PPO权重，不声称可交付。证据：evidence/ppo-launch-v5-oom.log、ppo-v5-kernel-oom.log。
- NO_SHARD的DCP仍生成完整state_dict；保存时CPU副本导致额外内存压力。新增checkpoint_cpu_offload=False，仅本配方启用，保留DCP恢复和标准full_weights.pt。小模型实际CUDA保存/恢复的model、optimizer、scheduler及四类RNG逐项一致，原选项为CPU张量，新选项为CUDA张量；不能代替完整模型容量实测。
- 增加PPO_UPDATE日志，在checkpoint前记录有限梯度优化次数和训练均值，避免checkpoint失败掩盖实际训练状态。
- 原生Torch SFT独立仿真8/8成功，控制步119/121/84/80/87/83/129/137；身份、相机录像见artifacts/20261026-ppo-simulation/native-sft-summary.json。与PPO使用同一运行时、初态哈希、噪声及10次流推理/5步重规划协议。
- v6已启动：pi05-ppo-smoke-20261026-v6.service，48GiB RAM上限、禁用swap、45分钟时限；运行中，需等待完整权重保存和独立评测。
- 全部18项单元测试在训练环境和PPO环境均通过（JAX_PLATFORMS=cpu）。未指定CPU时JAX先初始化GPU导致Flax LoRA测试TF32误差，重跑固定CPU后通过，未放宽阈值。
