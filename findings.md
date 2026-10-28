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

2026-10-27：v10 cgroup OOM：anon约22.35GiB，shmem约25.18GiB；actor共享RSS约25.16GiB。FSDP offload/onload先移动_local_shard再独立移动flat_param.data（两者常共享同一存储），随后又丢弃第一次移动的shard，是可静态确认的重复复制。需实际回归和大模型验证；Ray默认暴露96 CPU引入大量闲置worker，拟在本项目显式限制4 CPU。未把这两个发现当作已解决OOM。

## 2026-10-27 新恢复对照：奖励路径
- 实际 LiberoEnv.step 丢弃子环境 _reward，使用自身 _calc_step_reward(success)；仅修改 RecoveryEnv 返回值无法影响 PPO。
- 上轮300更新训练成功率并非全零，因此“整个训练没有正反馈”不成立；需要单独诊断 placement_failure 类。
- 原12个heldout初态隔离，不再作为本轮调参开发集；新开发源ID20..23，训练仍1..6。

### 奖励超时边界校正
- 实际 EnvTransition.compute_rewards 在 auto-reset 截断时加入 gamma * V(final_obs)，所以 PBRS 对超时必须保留 Phi(end)，仅真正成功终止置零；新增测试验证与 shaped bootstrap value 抵消。不能把 timeout 与 terminal 混为一谈。
- SFT 训练放置诊断 0/6，均 native_first_step=null；不是仅成功后不松手，策略从这些失败初态未达到原生放置判据。该诊断启动后源码改动，作为探索证据保留，不作冻结源代码对照。

## 2026-10-27 恢复奖励消融最终发现

- **已证实：**原子环境reward被chunk_step重新计算覆盖；在最终奖励张量注入PBRS，并屏蔽首次done后槽位，两组共享该结束屏蔽。6项边界回归和真实子进程reward检验通过。
- **已证实：**两组20轮300有限更新都正常退出、0OOM，峰值42.89/38.60GiB；检查点绑定同一冻结SFT基座。内存问题在此负载得到容量验证，不能保证任意更大配置不OOM。
- **已证实：**新开发12初态SFT10/12、sparse9/12、shaped10/12，成功步中位数87/97/97.5；都未过冻结选择门。奖励未证明改善开发成功率，现用模型未替换。
- **已证实：**训练放置初态单独对照0/6→2/6，显示训练分布内局部收益；未对sparse做该诊断，不能单独归因PBRS。两段train实际成功视频/动作轨迹导出，未导出训练观测动作对或混入开发集。
- **观察：**开发放置init21和23两候选仍300步超时；shaped从未达到原生放置判据，并非仅结束判断；21仍抓持、23把碗推远。仅增加时限是否有效未验证。
- **推断及下一步：**正确恢复行为覆盖/探索可能不足，适合用训练失败状态族的成功恢复示范配合旧数据回放，再用新冻结评测初态验证；不把少量样本认作全部原因，不用无界延长训练替代对照。
- **边界：**单seed、单任务、合成物理初态；未证明40任务保持、中文连续指令、未知物体、真机或消费产品稳定性。

## 2026-10-27 完整人工示范回放实验

- 旧项目只读审计选择 10 个完整已批准人工 episode、2050 帧；保留实际动作和 20 Hz 控制时序。原始 metadata/逐帧 NPZ、输出 Parquet 均做 SHA256。末 10 帧原生成功且张爪，最后末端离碗至少 10 cm。相机和 OSC 契约保持一致。人眼检查全部起止/过程拼图；不把这等同于独立物理重放。
- 必须区分任务：人工来自 spatial 黑碗旁盘任务（公开 task36），恢复开发集为 goal task8。原 spatial BDDL 和 .pruned_init 的 SHA 与采集时相同，但不是 goal 场景等价证明。这是跨场景迁移实验，不能据此假设当前失败已覆盖。
- 冻结方案：从自有 multitask40 SFT/9999 继续 LoRA、全新优化器，2000更新/batch1/lr5e-6/seed42，25%人工+75%旧训练回放；不引入当前 holdout，不改官方归一化。12开发/6训练诊断/8旧任务固定对照；达不到预设门槛不晋级，不自动换面板。
- 24项单测通过；dry-run完整哈希与官方两分支 transform 通过。预检需要顺序读取数十GB文件，期间无梯度日志是预检阶段，不是训练挂住。
- code-reviewer 子代理因当前账户不支持其指定模型而未运行，未当作独立审查通过；本轮依靠本地自审、契约测试与真实训练/仿真验证。

### 完整人工回放的实测结果与更正
- 2000 更新正常退出并保存1999；200条指标有限，loss0.0833→0.0147，峰值31.7GiB、OOM0。实际运行调用约375秒，另有完整输入预检；不能用训练时间或loss断言成功。
- 相同非权重契约/初态/噪声下，开发10/12→7/12：抓空4→2、偏移4→4、放置2→1；训练分布放置0/6→1/6。未通过预设晋级门，不替换面板；本轮没有新增PPO更新。
- 五个失败全都300控制步且无异常，存在反复抓持、偏位/碰撞及未重新触碗。placement_failure构造为碗放偏在桌上，并非一直抓住碗。init21/23全程无手指触碗，末尾碗盘XY仍16.1/15.5cm。不能把所有失败叫不松手或步数不足。
- 旧任务历史基线源代码身份不同，被审计拦下。保留历史记录、在当前脚本补跑同权重基线，严格匹配后均8/8。未修改冻结历史哈希。
- 视频时间更正：18个新恢复原视频和12段对照为20fps；8个新旧任务保持录像为既有导出器10fps、每控制步一帧，相比20Hz仿真慢放两倍。不是策略速度结论；所有26段新原视频帧数/解码验证通过。
- CPU float32单flow步导出对照max_abs1.43e-6，实际BF16闭环另测。首次全测35/36是JAX先初始化GPU引发TF32误差，进程起始固定CPU后36/36，未放宽断言；该问题以前也记录过，原日志继续保留。
- 不把负结果归咎于人工质量：10条示范来自spatial，而失败评测为goal；场景和初始状态不配对。下一次应优先取得当前goal训练失败状态族的完整恢复示范，并使用新的独立初态对照。


# Findings
Project /home/robot/pi05-rl-handson, origin sencangWei/pi05-rl-handson, branch experiment/matched-recovery-20261027.
Six approved recordings: four OSC-only (465,270,245,245 frames) and two mixed OSC/joint (120+444 and285+372). Mixed recordings explicitly training_ready=false. Do not relabel joint targets as OSC.
Latest recording human_1793124070194277437 approved,245 OSC, all5 observation gates true,490 checksums validated. Gates are demonstration checks, not official benchmark results.
Previous manual SFT crossed spatial->goal and regressed; previous matched PPO-teacher SFT also failed promotion. Need pinned parent and fresh paired protocol.
Unrelated untracked evidence/recovery-environment-20261027.json and evidence/recovery-sft28-environment.json must be preserved.

Manual6 compatibility: four complete OSC demos and two mixed demos. Mixed OSC segments can train only the labels actually recorded; intervening joint maneuvers remain untrained. Full raw 79-value states are continuous. Independent replay of empty-grasp demo first diverged above 1e-5 at control34 on finger-contact velocity; final native goal still true. Exact deterministic trajectory equivalence is NOT established. All raw files and metadata retained.

2026-10-28 diagnostic baseline: manual6 model still fails six fresh dev cases (placement40; all categories42; empty43; offset43). Last50 command norm0.217..0.565 with actual median1.587..7.277mm/step. Parent offset43 has command0.815 vs actual0.252mm/step. Existing traces lack force/limits/Jacobian, so collision or singularity is unproven. Reproduce original commands first; no assumed action-size fix.

2026-10-28 final diagnostic findings: old synthetic dev bank6/12 has initial bowl/cabinet/plate/bottle penetration, including successes; old5→6/12 cannot support improvement. Strict insertion+settled collision guard fixed; appended equal-radius candidates under pre-outcome amendment, no outcome selection; separate process restoration newbank12/12 valid. Parent/new10/12 each, placement42 gain/41 regression, all9 shared successes slower95→114median controls. Newfailures40/41 never grasp0/300, final50 torque saturation0, translation sigma0.203/0.189; policy not reacquiring bowl despite motions near plate. Six historical600-control trials0/6, five invalidinitials; valid40 proves budget not sole cause. WrongMuJoCo3.2.3 replay rejected; matched3.8.1 eight historical+six continuation prefixes+two newfailures zero recorded pose error. Fullstate/pixel equality unproven.86 tests+2subtests/Ruff/60video decodesPASS. External12 runtime hashes and simulator unchanged vs prior verified backups. No new training, UI/model unchanged.

## 2026-10-28 人工恢复数据审计

- 6录制/8 OSC片段/1630帧 raw→Parquet状态、动作及两路像素全部一致；1612可采窗口起点。180帧位姿全零，最长10控制步=0.5秒；405帧仅旋转，不可当停顿删除。
- 同任务8内纠错抽样集中：seed42索引重建2000次人工543、公开任务8仅33；含1次预取计数与原状态一致。93.02%理论、94.27%重建任务内人工比例；尚非因果证明。
- 训练库17/18通过；placement_failure-init06穿插但未用于6条录制。精确原录制起点6/6通过。816关节控制保留，当前7D OSC监督不能学习完整关节接管。
- 冻结仅改采样1:3→1:39的提案，未训练/未换UI；保留全量数据。报告 reports/recovery-data-audit-20261028/README.md。

## 2026-10-28 Approved task/phase sampling experiment
User approved the next bounded run after sampling025 failed. Audit observable phase labels first; preserve all raw/public data and OSC temporal labels. Freeze task/phase mixture, parent, 2000-update budget and matched dev/retention gates before training. No UI promotion without verified gates. Back up code/results to owned origin and independently restore.

Task-phase preflight: all1612human and7633publictask8 anchors partitioned exhaustively/disjointly. Human pools other1386/closing126/opening100. Public pools6289/672/672. 85 scoped tests passed and error-level Ruff passed. Physical grasp labels unavailable; command changes only. Reviewer role unavailable (unsupported configured model); direct review. No UI promotion this experiment.

## 2026-10-28 Task/phase bounded experiment result
2000 LoRA updates completed, batch1/seed42, fresh optimizer; actual1603 general replay+192 publictask8+205 human samples match frozen preview. Twelve reused physically valid dev scenes: parent10/12, old25%10/12, previous2.5%8/12, taskphase8/12; placement0/4. Common8 median controls parent92.5 vs candidate98; both frozen gates FAIL. All36 paired comparison videos decoded; full raw traces retained. Contact sheet shows tipped plate/init40 and unsuccessful grasp/transport/init41..43; native goal never met in all four timeouts. No UI promotion, no heldout/40-task retention evaluation. Inference export71 leaves/20LoRA exact; optimizer remains local. Result/code/weight remote restore verification pending.

2026-10-28 Task/phase delivery verification COMPLETE: owned public origin branch experiment/matched-recovery-20261027 payload d64a37edd407d90e314b8e4c542e6be445253d79 independently restored all3225 tracked files; rebuilt tree b2b3dfb3fabdaf69e94ab592ed9966ee2f2f6c77 exact. Release taskphase-ablation-20261028 independently downloaded185501727bytes in32 validated ranges; SHA2568cf5a41d4c5df14bfe0403383f87ad78591c37d6b92da472a0ceb2cf484638be exact; official-base reconstruction71 tensors/20LoRA exact. Own train/server inactive with exit0; no UI change. Bounded experiment delivered with negative outcome8/12 and placement0/4; no heldout or40-task retention claim.

## 2026-10-28 放置恢复事件诊断
八条父/候选动作回放记录的EEF/碗/盘位置及目标/手指接触标志误差0.0，视频全解码。父模型placement2/4，候选0/4；候选40未持续提起且碰翻盘，41以非指垫接触抬起但未搬向盘，42/43推走并撞桌、后段饱和和小Jacobian。代理不能当稳定抓持真值。
六条已批准原录制2446控制含1630OSC/816joint，身份及时间线连续性通过；两条混合控制的抬起指垫代理OSC0/55步对joint417/370步，关键搬运未进入现有7D OSC监督，全部原始数据保留。这是控制契约覆盖缺口，不是退化唯一原因的证明。静态after查询和缓存观测最大差19.121mm、5个目标标志差异，首个严格检查失败已显式留证，不能声称人工精确回放。
三项诊断源码、10项针对性测试、完整范围Ruff及编译通过，报告/两张帧图/八视频/逐步JSON已保留。未训练、未改面板/其他任务、封存集未开；下一步先检验原关节搬运是否可用当前控制契约完整执行，不盲目要求重录位置或增加更新。报告reports/recovery-events-20261028/README.md；本轮远端备份正在验证。

2026-10-28 放置恢复事件诊断交付完成：源码、测试、逐步指标、报告和八段录像已提交 6b7d2e6595486252989879124ce8b4e1ca1fe9c0 至用户自有 sencangWei/pi05-rl-handson 的 experiment/matched-recovery-20261027 分支；从远端内容独立恢复全部3274个跟踪文件，SHA256及重建Git树4cb44d4b43fb4fb0c458962d8032ee02d364d3a5一致。证明见evidence/recovery-events-28-remote-restore.json。当前任务仅完成诊断交付；恢复未修好、未训练/未换面板；下一项控制契约实验尚未执行。
