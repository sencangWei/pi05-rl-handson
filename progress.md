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

08:26 更新：PPO独立父权重下载服务正常退出0；固定版本45ccfcc4e28634f1576ebf78cab0fbe2fd82432d的3个文件全部通过大小和完整哈希校验，model.safetensors为7473091464字节，SHA-256为4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7。见evidence/rlinf-model-validation.json。独立PPO依赖仍在下载，GPU主训练/PPO更新仍为0，SFT到PPO的LoRA衔接仍未实现。

17:47 更新：用户已自行关闭旧GPU推理服务，启动前空闲31574MiB，JAX GPU矩阵运算通过。独立服务pi05-multitask-sft-20261026.service于17:43启动，真实日志已观测Step 900、loss=0.0113；40任务10000步SFT正在更新，未完成。服务使用原night.lock，限制6小时/48GiB/禁swap。源码、父权重和数据前置核验通过。原夜间队列11:06等待超时，已另存历史；PPO依赖原安装09:17超时，已启动4小时独立重试。PPO更新仍0，JAX LoRA到PPO权重衔接仍未实现。启动环境、服务参数和真实日志节选均已落盘。

21点更新：10000次SFT迭代已经18:14正常结束，9999完成保存。PPO依赖重试先保存日志后停止，原因已定位为127.0.0.1代理TLS超时；独立镜像直连下载165包完成，安装仍进行中。新增LoRA合并及严格加载检查；实际Flax层数值测试捕获并修复跨头输出投影合并错误。真实导出与SFT四任务双初态仿真正在执行，结果未出，PPO更新仍0。所有动作仅限新项目。

21:35 仿真更新：新SFT四任务×双初态8/8，碗放盘松爪额外2/2；官方父模型同哈希初态8/8，酒瓶初态0由299降至131步，其余有快有慢。18段真实视频保存artifacts/20261026-sft-simulation。PPO严格加载811→667张量通过；float32整网对照最大误差0.001248未达既定阈值，正定位GELU等数值差异。Torchcu126在5090D实测no kernel image，独立cu128修复进行中。PPO更新仍0。配置已改为本次合并SFT血缘、horizon10/discreteFalse、正确归一化、all_linear适配器与orig_params，真实运行仍待门槛通过。

21:49 更新：独立Torch2.7.1+cu128实际BF16矩阵/SDPA前后向通过。定位数值差异为RLinf的精确GELU与Flax默认tanh近似不同，实际两行补丁后整网原阈值检查PASS，max_abs=1.66893e-6，保留v1失败与v2通过证据。12项测试在JAX_PLATFORMS=cpu通过；同进程未固定CPU时GELU测试先初始化了JAX GPU，导致后续Flax合并测试精度失败，因此明确测试入口固定CPU。首次PPO启动于FSDP精度配置校验退出，更新0；按上游官方Pi05规则补齐后CPU校验通过，独立v2服务21:48启动、正在加载模型，实际更新仍待观察。

21:57 更新：v2由本项目主动停止，实际更新0，原日志保留；定位到rollout继承actor FP32存储，但无FSDP转换且Gemma输入强制BF16。真实原生Attention最小测试复现dtype冲突，采样权重BF16后通过。仅覆盖rollout.model.precision，actor仍按官方FP32存储/BF16计算规则。v3服务已启动；不能以启动成功替代实际训练。算法提交ae1a721已从GitHub独立获取，27个文件逐字节一致，干净RLinf奖励/GELU补丁重放和5个数值对照源码哈希均通过，证据.local/gelu-remote-verification-20261026.json。

22:13 更新：v3完成300控制步采样，首个训练前向在SigLIP head的PEFT/FSDP边界报FP32/BF16不匹配，实际PPO更新0。PEFT读取未展开权重的FP32 dtype后转换输入，子FSDP随后展开BF16权重，故启用子边界输入转换。Torch2.7也在根_pre_forward执行此转换，另补根策略副本禁用转换，避免PPO chains精度损失；没有原地修改与子节点共享的策略。真实小模型CUDA前后向与真实RLinf包装器的根chains FP32检查PASS。v4在采样/更新前主动停止，保留日志；v5服务22:12启动。独立原生评测入口和5项实际API契约测试通过，待PPO成功后与原生SFT对照。

## 2026-10-26 22:26 CST 保存内存故障与重跑
- v5 完成首轮采样、训练调用和内置评测（任务0两次成功），保存 step1 时触发48GiB CPU cgroup上限，退出oom-kill；没有持久化PPO权重，不声称可交付。证据：evidence/ppo-launch-v5-oom.log、ppo-v5-kernel-oom.log。
- NO_SHARD的DCP仍生成完整state_dict；保存时CPU副本导致额外内存压力。新增checkpoint_cpu_offload=False，仅本配方启用，保留DCP恢复和标准full_weights.pt。小模型实际CUDA保存/恢复的model、optimizer、scheduler及四类RNG逐项一致，原选项为CPU张量，新选项为CUDA张量；不能代替完整模型容量实测。
- 增加PPO_UPDATE日志，在checkpoint前记录有限梯度优化次数和训练均值，避免checkpoint失败掩盖实际训练状态。
- 原生Torch SFT独立仿真8/8成功，控制步119/121/84/80/87/83/129/137；身份、相机录像见artifacts/20261026-ppo-simulation/native-sft-summary.json。与PPO使用同一运行时、初态哈希、噪声及10次流推理/5步重规划协议。
- v6已启动：pi05-ppo-smoke-20261026-v6.service，48GiB RAM上限、禁用swap、45分钟时限；运行中，需等待完整权重保存和独立评测。
- 全部18项单元测试在训练环境和PPO环境均通过（JAX_PLATFORMS=cpu）。未指定CPU时JAX先初始化GPU导致Flax LoRA测试TF32误差，重跑固定CPU后通过，未放宽阈值。

## 2026-10-26 22:40 CST adapter 保存路线
- 更正22:26保存内存假设：v6即使关闭CPU offload仍于22:30触发48GiB上限；此前15次有限梯度更新已记录，但未保存可用PPO权重。不能把小模型DCP验证当成大模型容量验证。
- 新增显式单卡FSDP1/NO_SHARD adapter格式，只保存可训练LoRA、价值头、持久buffer、优化器/调度器和RNG，绑定完整冻结基座SHA。绕开全模型state_dict和全量优化器gather；原DCP默认行为保留。
- 小模型实际CUDA恢复及错误拒绝检查通过。v7已22:37启动；完整模型保存与独立八场景评测仍待完成。
- all_linear注入发生在原生freeze_vlm之后，因此本轮实际训练所有注入的LoRA及价值头；冻结基座不更新，不能按train_expert_only字面声称只训练动作专家LoRA。

22:45更正：v7首轮adapter保存成功，但第二轮更新后权重同步再次触发48GiB上限。共观察30次更新，只有首轮15次持久化。已另启独立八场景评测，不能称2epoch完整退出成功。18项本项目测试通过；未限定tests目录时会误收集上游硬件/重复OpenPI测试，最终入口明确为pytest -q tests。

## 2026-10-26 22:55 CST 首轮独立仿真与第二轮同步修复
- v7已保存15次更新权重的独立8场景仿真全部成功；SFT对照同样8/8，步数差异小，不声称PPO明显改善能力。成对录像和完整结果见artifacts/20261026-ppo-simulation/first-saved-15-updates/。
- 第二轮同步仍调用全模型state_dict，引入不必要的大模型副本。显式adapter单卡路径改为首次全量、后续仅同步可训练参数及持久buffer；CPU/CUDA snapshot均以真实actor+PatchWeightSyncer验证，两次重复同步一致、冻结参数不变、缺键/多键/形状错误拒绝；禁止初始化后再调用完整state_dict。
- 七个增量补丁在固定上游干净源码重放后与运行时逐字节一致。启动器加入实际回归证据与源码哈希门禁。v7目录保留为logs/ppo_goal_smoke_v7_second_sync_oom；v8从同一SFT父权重重跑两轮，不与v7累加步数。

22:57更正：集成复查发现adapter同步条件误读actor顶层配置，而真实配置位于actor.fsdp_config，导致v8未启用选参路径。已主动停止v8并保存原日志及运行目录；不是完成训练。真实配置子树用于小模型回归后，旧代码稳定失败（禁止初始化后完整state_dict），修复读取路径后CPU/CUDA两种快照均通过，七补丁重放一致。v9使用修复后的相同训练配方重新开始。

23:00更正：v9实际启用选参同步后，保存首轮15次更新成功，但在下一轮同步时仍触发48GiB cgroup限制；不能称内存问题已修复。保存权重121226557字节，SHA0d0da3e660f03203f3642188eb1f8083633395c5391aa4a9c6d6aae533869916，550可训练张量均有限；原日志、内核OOM和目录已存档。正在回查卸载/装载及共享内存传输路径。对照打包器新增基座、归一化、采样器、初态和63项来源中62项动作运行时来源一致性检查；允许评测器自身新增记账哈希，不改变动作。25项测试、2子测试、Ruff与bash语法通过。


## 2026-10-26 23:14 CST 最终30更新检查点与独立评测
- v10采用max_epochs=2、weight_sync_interval=2、val_check_interval=1。每轮评测先同步新权重，第二轮保留该采样模型，跳过冗余的轮首再次同步；没有跳过训练后的评测同步。
- 两次PPO_UPDATE各15次有限梯度更新。最终global_step_2/actor/adapter_checkpoint.pt原子写入完成，121226493字节，SHA256 0595e247536bc48d5498f5085823a51c595b36cfd14b965f642f81256daaf7f1；550张量有限，优化器状态242个step30/308个step0，scheduler last_epoch2。不能把全部550张量都称已更新。
- 保存后的进程23:12:22仍被48GiB cgroup内存上限终止；service_result=oom-kill，未正常退出。跳过冗余同步只是使最终权重落盘，不是彻底修复内存故障。
- 新进程pi05-native-ppo-final-eval-20261026独立加载最终adapter与匹配SHA基座，执行task0/7/8/9×init0/10；原生SFT使用同初态、运行时、噪声和10次流采样/5步重规划协议。
- 最新25测试和2子测试通过，80条Flax警告；Ruff通过。原始记录evidence/ppo-v10-*.json及ppo-launch-v10-oom.log；仿真结果待实际生成。

最终独立评测完成：服务Result=success、ExecMainStatus=0，8/8触发LIBERO原始成功判据。PPO步骤118/123/86/80/87/83/131/140，均值106；匹配SFT均值105，未见效率或成功率提升。最终对照目录final-saved-30-updates；不把原始任务成功等同额外松爪或恢复。


## 最终公开备份核验（2026-10-26）
- fd1c22f已推送用户公开仓库sencangWei/pi05-rl-handson，experiment/pi05-ppo分支；新克隆370个跟踪文件SHA逐一一致，干净固定上游重放7补丁后9个实际运行文件完全一致，见evidence/ppo-final-remote-restore.json。
- 最终121226493字节adapter发布为ppo-smoke-20261026 Release；从GitHub独立分32段下载，验证每段Content-Range与长度，拼接SHA256为0595e247536bc48d5498f5085823a51c595b36cfd14b965f642f81256daaf7f1，与本机一致。见evidence/ppo-release-verification.json。13GB冻结基座、完整数据和虚拟环境未上传。
- 最终仿真八场景8/8，对照8/8，无PPO提升证据；训练保存后OOM的问题仍未解决，不以备份完成改变此结论。

## 2026-10-27 内存修复与恢复专项
- FSDP重复别名复制根因修复；CUDA三次往返回归通过，完整模型3轮45次有限PPO更新正常退出0、峰值38.13GiB、OOM事件0。
- 30个合成失败初态已生成：18训练/12留出，三类抓空/偏移/放置失败；严格无接触目标连续5步，真实子进程reset/step通过。
- 26 tests及2个subtests通过（JAX_PLATFORMS=cpu），精度/adapter保存恢复/同步回归通过。20轮300次恢复训练执行中，尚无能力提升结论。
- 8补丁在干净固定RLinf上游重放，12个运行文件SHA逐一匹配。

- 00:42：300次有限优化器更新完成，20轮正常退出0；48GiB上限内峰值40.68GiB、OOM0。最终adapter SHA c1cadff1e4a87b5f448fa9dacbaccc885ef5e58ca4ddea92e51cdc110c77f63f，242状态step300/308状态step0。18训练初态全部实际访问，独立12场景三模型评测开始。
- 3bd42eb远端新克隆452个跟踪文件git blob逐一一致；8补丁重放12运行文件哈希通过。

- 00:49：独立三模型共36场完成，服务正常退出0；SFT/旧30/新300均9/12，类别均4/4、4/4、1/4。放置失败init10/11/12三组均超时，新300偏移init13由SFT98步变208步。未见提升，未替换旧面板模型。
- 48段视频完整解码通过，36条轨迹逐一核对成功末5步原生目标及无手指接触、超时300步及无异常。保存逐场三栏、双相机原视频、轨迹、身份哈希与结果，不隐藏失败。
- 独立code-reviewer启动因账户不支持其gpt-5.5模型而失败；没有独立审查结论。已人工检查实际初态恢复位置、严格成功训练/评测共用实现、相同身份及噪声、动作执行来源、视频帧数/时间和checkpoint更新数。

- 00:58最终备份核验：公开远端experiment/pi05-ppo的b3407ff在独立克隆恢复，556个跟踪文件字节和git tree逐一一致；从远端证据在固定干净RLinf上游重放8补丁，12个实际运行文件SHA一致。新Release权重独立32段下载后121226493字节、SHA c1cadff1e4a87b5f448fa9dacbaccc885ef5e58ca4ddea92e51cdc110c77f63f与本机一致。三份核验JSON保存在evidence/ppo-recovery27-{remote-files,remote-runtime,release-verification}.json。没有自动替换现用模型；放置恢复不足和新模型效率退化如实保留。

新轮开始：用户“对”确认上一轮提出的恢复专项方向。当前基线300更新与SFT均9/12，不能认定单纯增加更新有效；在改奖励前先用训练初态诊断。GPU仅537MiB桌面占用、系统可用54GiB；不停止其他任务。

- 阶段奖励六项回归与真实子进程奖励到达检查通过；超时保留 endpoint potential 以匹配标准 bootstrap。新 dev ID20..23 已固定，SFT 开发评测运行中；仅对旧训练放置初态的探索诊断为 0/6。实际算法9个补丁在干净上游恢复，12文件哈希一致。

- 15:13：sparse_v2 完成300次有限更新，20轮正常退出0、峰值42.89GiB、OOM0；独立新dev评测9/12（SFT10/12），没有提升。首次口头更新误报10/12，已依据最终summary立即更正；以逐场结果为准。shaped_v1 已自动顺序启动，面板未替换。GitHub API代理超时，直连验证公开仓库ADMIN可用，sparse权重已上传新Release草稿。

- 15:31 最终主对照：两组均完成20轮/300次有限更新、正常退出0、OOM0，峰值RAM42.89/38.60GiB。开发SFT10/12、sparse9/12、shaped10/12，成功控制步中位数87/97/97.5；共同9成功场景中位数84/97/90。两候选未通过冻结选择，不替换面板。36原视频与12三栏已完整解码/帧数/身份/严格松爪检查通过。
- 后续冻结源码训练初态诊断：SFT放置恢复0/6，shaped2/6，成功init02/03为115/114步。仅是训练分布内结果，不能声称新初态泛化提升，也未对sparse作此诊断。实际train成功视频和动作轨迹单独列清单，未生成新的训练观测动作数据，未启动再训练。
- 亲自检查init21/23三模型原相机时间拼图和逐步轨迹：shaped两场300步原生目标均从未成立，分别碗盘最终XY相距3.06/24.07cm，存在抓持反复调整/推离。sparse init20曾触发原生判据但持续接触，严格成功未误报。
- 两个约121MB新adapter均已上传独立Release并分别从远端重新分段下载，完整SHA256与原检查点一致；当前在汇总最终提交和公开发布。

- 最终备份：公开远端 experiment/pi05-ppo 的 feff505 在独立目录恢复，767 个跟踪文件的 git tree、工作副本和提交 blob 逐一一致；9 个算法补丁在固定干净上游重放，12 个运行文件 SHA 与现役源码一致。两份 adapter 已公开发布为 ppo-recovery-shaping-20261027 预发布，匿名 API 200，完整独立下载 SHA 一致，证据见 recovery27-stage-final-remote-{files,runtime}.json 与 recovery27-stage-public-release.json。
- 评测边界澄清：新开发集也用于训练期间的监控评测，但没有梯度更新；最终固定 300 更新后的独立进程复评仍属于开发评测，不是盲测。没有自动替换面板或启动新一轮训练。

用户再次确认继续恢复示范+旧数据回放路线。启动只读数据审计，GPU空闲约31GiB、RAM可用54GiB、磁盘148GB；不改旧工程、不触碰真机或终止其他任务。
- 17:02 完整示范 10条/2050帧已整理，24项测试和官方 transform dry-run通过。独立systemd单元 pi05-human28-replay-20261027 启动，48GiB内存/禁交换/40分钟总上限，重新核验哈希后训练。预训练代码0ee7270已推远端并独立恢复792文件一致；模型/数据备份与最终评测待完成。

- 17:14 完成人工28回放实验2000次更新，正常退出0、31.7GiB峰值、OOM0、1999检查点。导出与数值对照通过。
- 17:36 完成12开发＋6训练分布诊断＋8旧任务新模型评测；另用当前脚本补跑8场旧模型保持基线。开发10/12→7/12、放置2/4→1/4、训练放置0/6→1/6，旧任务8/8→8/8。不晋级、不换面板。26段新原视频与12并列视频已核对；旧任务录像为10fps慢放、恢复20fps，报告已纠正。
- 最新CPU限定36测试通过；新增导出/视频打包脚本Ruff及py_compile通过。已发布人工数据和精确20LoRA张量备份；数据远端独立下载、14文件字节哈希一致，权重与最终代码恢复核验进行中。

最终备份核验：公开分支 experiment/pi05-ppo 的9c5b6ec从远端独立检出，1345跟踪文件的树、工作内容和提交blob逐一一致；RLinf干净上游重放9补丁，12运行文件一致。Release数据与适配器独立下载SHA通过；远端适配器加官方基座重构71叶子完全一致（20 LoRA）。仅推理备份，不含优化器。原始完整检查点仍在本机。36项CPU测试通过。晚采集已卸载systemd单元的默认值不能证明运行设置，已在证据中明确标注。实验交付完成，不将负结果称为恢复修复，不替换面板模型。

2026-10-27 19:45 用户同意补齐匹配失败场景示范后对照训练。选择先做学习策略自主成功轨迹收集（非手写控制、非人工标签），从当前训练失败初态开始；成功不足则不启动训练。GPU仅桌面占用537MiB；旧项目只读。原human28负结果保留。

- 匹配恢复采集18场13成功（抓空5/6、偏移6/6、放置2/6），1127帧；13条原命令和限制幅度标签均在独立仿真完整重放，所有物理状态最大误差0.0，终态松爪成功。初次录制器将有限越界动作误拒；现保留原命令+实际控制范围标签，失败日志保留。已冻结新dev30..33、回归20..23、保持8场门槛。

- 匹配恢复轮训练前基线完成：新dev30..33为6/12（抓空2/4、偏移3/4、放置1/4），旧dev20..23为10/12（4/4、4/4、2/4），保持8场8/8。源代码及协议冻结，checkpoint与数据逐文件校验后才进入更新。c8baf4d公开远端独立恢复1389跟踪文件一致；上游无关工具软链接导致安全解包拒绝，验证器现只归档manifest声明的运行目录，仍保留安全过滤，9补丁/12文件恢复通过。

## 配对恢复示范 SFT 最终评测（2026-10-27）
13条模型自主恢复示范、1127帧，原始与裁剪标签独立重放均通过。从原40任务父模型完成2000次更新，退出0、OOM0、峰值36.75GiB。新开发6/12→5/12、旧开发10/12→7/12、常规8/8→8/8；未通过晋级，不替换面板。本轮没有新增PPO更新。88段实际录像完整解码通过，配对初态/非权重配置一致；43项项目测试通过。数据远端恢复42文件一致；权重和最终代码恢复核验进行中。

本轮最终备份完成：公开实验分支 c739fb2 独立检出1973跟踪文件，代码、配置、报告及录像逐字节一致；RLinf干净上游9补丁/12运行文件一致，外部模拟器源码远端一致。Release数据独立下载42文件哈希一致；权重独立下载185501122字节且SHA一致，官方基座重构71参数叶子（20 LoRA）全部一致。核验记录另作元数据提交。本轮交付完成，恢复表现退步、候选不晋级、面板未替换。


# Progress
2026-10-28: User approved stopping collection and proceeding with data compatibility + replay + paired comparison. Read Karpathy, RTK, reproducibility and planning skills; inspected git status and six approved metadata. No training launched yet.

2026-10-28 manual6: six approved raw demonstrations preserved; 1630 OSC frames, 1612 eligible window anchors across eight clips, 816 joint frames archived. Nonterminal windows stop before joint interruptions; no fabricated joint-to-OSC targets. Nine targeted tests and Ruff passed. Fresh development IDs40..43 (12 failure states) frozen before training; public replay 75%, human25%.

2026-10-28 02:21: Preflight full public train/parent/human checksums and both official transforms passed; systemd pi05-manual6-train-20261028 launched (48GiB, no swap, 90min bound). 724d96c independently restored from origin, 2011 tracked files exact. Full six-demo data archive 564035222 bytes/SHA9f4a2754251209e9f1a7e086d172649e0323808aaf3e397fbb14b7714e6e7c26 uploading.

2026-10-28 02:30: manual6 2000 updates COMPLETE; checkpoint1999 saved, unit normal exit/no OOM. Six-demo data release independently downloaded and all4226 file hashes verified. Paired JAX parent/new evaluation started; no model promotion.

2026-10-28: manual6 experiment complete: 2000 updates, exit0/OOM0. 64 paired episodes fresh5→6/12, old9→9/12, normal8→8/8. Success-only paired median steps fresh89→116, old90→110, not a speed fix. All88 original/comparison videos fully decoded. Frozen gate passes narrowly, UI unchanged, still6 fresh failures; no broad recovery claim. Release data independently restored4226 files; compact adapter export71 leaves exact, independent download underway.

2026-10-28 backup verification: public origin payload18e2041 independently fetched/restored, 2615 tracked file contents identical; Release data4226 files checked; downloaded adapter full71 leaves/20 LoRA reconstruct exactly. No UI model replacement; own training/evaluation services inactive with exit0.

2026-10-28 started approved layered failure diagnostics; original eval and UI unchanged. Read local evaluation/simulator contracts; no external training launched.

2026-10-28 diagnostic execution complete locally: repaired initial-state overlap validation, generated12validdevstates, paired models10/12→10/12 with9/9sharedsuccesses slower95→114mediancontrols; remaining2nevergrasped. Historical600-step continuation0/6 (5invalidstarts), exactmatchedphysicsreplays kept. Originalbank/report preserved with prominent correction.86tests+2subtests/Ruff/60videosPASS. Own diagnostic services inactive/success; originalmanualGUI PID2124267 alive. Report reports/recovery-diagnostics-20261028/README.md. Remote payload/restore verification in progress; no model promotion/training/UI modification.

2026-10-28 diagnostic backup complete: payload8bd6b6e017b4f58d3778f44a9b37b3656e9bdf10 independently fetched from public sencangWei/pi05-rl-handson branch experiment/matched-recovery-20261027; all2787 tracked file SHA256s and whole git tree identical. Covers code,12 valid states,60 videos, physics traces and reports. Existing external runtime12files match prior restored patches, no new weight/data release. Proof/status follow in metadata commit. No pending diagnostic work; policy recovery remains incomplete.

## 2026-10-28 数据审计及远端恢复核验完成

完成只读物理起点、1630帧对齐及采样重建；诊断库初态1例不合格，未污染本轮6条原录制。90 tests + 2 subtests通过，Ruff通过。下一轮仅改抽样比例的提案已保存，未启动训练；保持用户面板/权重不变。审计代码/提案/证据已提交 d1a0d16 到 sencangWei/pi05-rl-handson 的 experiment/matched-recovery-20261027，独立GitHub克隆恢复后全2796个跟踪文件SHA256一致；证明见 evidence/recovery-data-audit28-remote-restore.json。

## 2026-10-28 Sampling-only ablation results

Fresh 2000-update LoRA run completed, exit0; actual1950 public/50 human windows match seeded preview, one unused prefetch excluded. Valid matched dev: parent10/12, old1:3 mix10/12, new1:39 mix8/12 (empty4/4,offset4/4,placement0/4). Eight shared successes old113.5→new92.5 median controls, parent92.5→new92.5. Gate FAIL, no UI replacement, heldout sealed. All12 raw candidate and24 comparisons fully decoded; representative frames/trace analysis preserved. Four failures each reached300controls without native success, not an early-stop fix. Inference export71 reconstructed leaves exact/20 finite LoRA tensors; report and remote backup verification underway.

2026-10-28 sampling-only ablation delivery complete: result payload c2d71769a6645c2b536d715f8fc70b3a311a833f independently fetched from owned public origin, all2976 tracked file SHA256s and tree18e65384c8106cf07d4bc7c44edf11fc3a9a7074 equal. Release sampling-ablation-20261028 downloaded independently in32 validated ranges; full185506847bytes/SHA4f3a6db8a609372dcf7b2496beffdecfe01a895722479b1c91dd0d45cbe7b960 equal. Reconstructed71 parameter leaves/20 LoRA exactly. Training Openpi imports local vendored15a9616 snapshot, all141 manifest file hashes equal; external b0334d6 tree only provides LIBERO evaluator path. No promotion/new training, no heldout evaluation. Negative result remains8/12 and placement0/4.

## 2026-10-28 Approved task/phase sampling experiment
User approved the next bounded run after sampling025 failed. Audit observable phase labels first; preserve all raw/public data and OSC temporal labels. Freeze task/phase mixture, parent, 2000-update budget and matched dev/retention gates before training. No UI promotion without verified gates. Back up code/results to owned origin and independently restore.

Task-phase preflight: all1612human and7633publictask8 anchors partitioned exhaustively/disjointly. Human pools other1386/closing126/opening100. Public pools6289/672/672. 85 scoped tests passed and error-level Ruff passed. Physical grasp labels unavailable; command changes only. Reviewer role unavailable (unsupported configured model); direct review. No UI promotion this experiment.

## 2026-10-28 Task/phase bounded experiment result
2000 LoRA updates completed, batch1/seed42, fresh optimizer; actual1603 general replay+192 publictask8+205 human samples match frozen preview. Twelve reused physically valid dev scenes: parent10/12, old25%10/12, previous2.5%8/12, taskphase8/12; placement0/4. Common8 median controls parent92.5 vs candidate98; both frozen gates FAIL. All36 paired comparison videos decoded; full raw traces retained. Contact sheet shows tipped plate/init40 and unsuccessful grasp/transport/init41..43; native goal never met in all four timeouts. No UI promotion, no heldout/40-task retention evaluation. Inference export71 leaves/20LoRA exact; optimizer remains local. Result/code/weight remote restore verification pending.

2026-10-28 Task/phase delivery verification COMPLETE: owned public origin branch experiment/matched-recovery-20261027 payload d64a37edd407d90e314b8e4c542e6be445253d79 independently restored all3225 tracked files; rebuilt tree b2b3dfb3fabdaf69e94ab592ed9966ee2f2f6c77 exact. Release taskphase-ablation-20261028 independently downloaded185501727bytes in32 validated ranges; SHA2568cf5a41d4c5df14bfe0403383f87ad78591c37d6b92da472a0ceb2cf484638be exact; official-base reconstruction71 tensors/20LoRA exact. Own train/server inactive with exit0; no UI change. Bounded experiment delivered with negative outcome8/12 and placement0/4; no heldout or40-task retention claim.

Started approved event-level placement recovery diagnosis; preserve trained models/raw data/UI. No new training launched.

## 2026-10-28 放置恢复事件诊断
八条父/候选动作回放记录的EEF/碗/盘位置及目标/手指接触标志误差0.0，视频全解码。父模型placement2/4，候选0/4；候选40未持续提起且碰翻盘，41以非指垫接触抬起但未搬向盘，42/43推走并撞桌、后段饱和和小Jacobian。代理不能当稳定抓持真值。
六条已批准原录制2446控制含1630OSC/816joint，身份及时间线连续性通过；两条混合控制的抬起指垫代理OSC0/55步对joint417/370步，关键搬运未进入现有7D OSC监督，全部原始数据保留。这是控制契约覆盖缺口，不是退化唯一原因的证明。静态after查询和缓存观测最大差19.121mm、5个目标标志差异，首个严格检查失败已显式留证，不能声称人工精确回放。
三项诊断源码、10项针对性测试、完整范围Ruff及编译通过，报告/两张帧图/八视频/逐步JSON已保留。未训练、未改面板/其他任务、封存集未开；下一步先检验原关节搬运是否可用当前控制契约完整执行，不盲目要求重录位置或增加更新。报告reports/recovery-events-20261028/README.md；本轮远端备份正在验证。

2026-10-28 放置恢复事件诊断交付完成：源码、测试、逐步指标、报告和八段录像已提交 6b7d2e6595486252989879124ce8b4e1ca1fe9c0 至用户自有 sencangWei/pi05-rl-handson 的 experiment/matched-recovery-20261027 分支；从远端内容独立恢复全部3274个跟踪文件，SHA256及重建Git树4cb44d4b43fb4fb0c458962d8032ee02d364d3a5一致。证明见evidence/recovery-events-28-remote-restore.json。当前任务仅完成诊断交付；恢复未修好、未训练/未换面板；下一项控制契约实验尚未执行。
