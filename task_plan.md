# π0.5 + PPO 仿真实验计划

目标：真实π0.5示范微调、真实PPO更新、保存权重、独立MuJoCo/LIBERO仿真与同初态对照；无手写轨迹替换。
约束：独立目录与环境，不改旧工程、不碰真机、不停止其他GPU任务、不使用付费资源。

1. [完成] 固定源码、父模型、40任务1693条273465帧数据；训练1525条、留出168条。
2. [完成] 10000次GPU SFT更新，正常保存9999；四任务双初态8/8，额外松爪2/2。
3. [完成] JAX→Torch LoRA合并、严格加载、数值对照；CUDA/FSDP精度与adapter保存/同步回归。
4. [历史] v10真实30次PPO有限梯度更新已原子保存，checkpoint可读、张量有限、优化器step=30；保存后进程OOM，未正常退出。当时内存故障未解决；10月27日修复结果见下。
5. [完成] 最终30更新权重独立八场景仿真（8/8，SFT同为8/8，平均步数105→106，无提升证据），逐场对照原生Torch SFT；录制相机与并列视频。历史15更新权重与SFT均8/8，无提升证据。
6. [完成] 提交本轮证据及录像到公开仓库；从远端恢复所有跟踪文件、干净上游重放7个补丁核对9个运行时文件；最终adapter以Release发布，32段独立下载的范围与长度检查通过，合并后SHA与本机完全一致。

不把PPO试跑称为能力提升，不把LIBERO原始判据称为额外松爪检查；尚未验证40任务遗忘、失败恢复、连续家庭指令或真机。
历史尝试与更正见progress.md；当前指标以最终逐场summary和checkpoint证据为准。

## 2026-10-27 内存修复与恢复专项
1. [完成] 修复卸载/装载重复存储复制，CUDA别名回归通过；Ray CPU worker预算4。
2. [完成] 完整模型3轮45次验证及20轮300次专项均正常退出0；后者峰值40.68GiB、OOM0，检查点optimizer step=300、scheduler=20、550模型张量有限。
3. [完成] 18训练/12留出物理初态生成且真实子进程reset/step通过；三模型各12场严格松爪对照完成，均9/12，放置失败3场仍超时，新模型一场明显变慢。36段原视频、12段三栏对照与逐步轨迹均保留。
4. [完成] 汇总结果与失败边界；b3407ff远端556个跟踪文件逐一一致，干净上游恢复8补丁/12运行文件，Release权重独立下载121226493字节且SHA一致。证据随最终核验记录提交。

## 2026-10-27 恢复训练第二轮（用户确认继续）
1. [完成] 训练放置初态探索诊断0/6；新开发初态ID20..23共12个，原12个留出隔离。
2. [完成] 修复真实奖励张量路径和结束后重复奖励；6项PBRS/终止边界回归、真实子进程奖励检验通过。仅从训练初态导出实际成功的自主恢复片段；无成功则如实报告，开发集不用于训练。
3. [完成] 同父模型/seed42的两组各300有限更新，均正常退出、OOM0；dev SFT10/12、sparse9/12、shaped10/12，无整体提升。训练初态诊断0/6→2/6，只导出实际train成功视频/轨迹，不当作新训练数据。不替换面板。
4. [完成] 共66段开发/训练诊断视频完整验证，报告与失败轨迹已提交。feff505远端767个跟踪文件逐一一致；干净上游恢复9补丁/12运行文件。两份adapter公开发布且独立下载完整SHA一致；最终核验记录另作提交。

## 恢复示范与旧数据回放（用户确认继续）
1. [完成] 只读审计并整理10条完整批准示范、2050帧；排除草稿和旧慢片段，固定真实20Hz时序与动作契约。
2. [完成] 固定25%人工/75%40任务回放、SFT父模型、2000更新及独立评测；明确spatial→goal跨场景边界。
3. [完成] 2000更新正常退出0、OOM0；开发10/12→7/12，训练放置0/6→1/6，旧任务8/8→8/8。未通过晋级，不替换面板。
4. [完成] 成功失败原视频、12组同步对照、报告及日志均保留；9c5b6ec远端1345文件逐一恢复一致，干净上游9补丁/12运行文件一致。人工数据和权重均独立下载SHA通过，权重从官方基座重构71叶子完全一致。实验交付完成，恢复能力未修复。

## 2026-10-27 配对恢复示范（用户再次确认继续）
1. [完成] 固定同场景训练失败初态，只从 train 源ID1..6收集已训练策略真正成功的全程RGB/状态/动作；模型自采集明确标记，不冒充人工。每条独立物理重放验收。
2. [完成] 门槛：至少两个不同训练初态的放置失败恢复成功，三类均有示范；不足则不盲目启动微调。开发/留出数据不进入梯度。
3. [完成] 从原40任务SFT父模型、25%配对恢复/75%旧任务回放，2000更新/batch1/lr5e-6/seed42；全新优化器。不续训已退化的human28模型。
4. [完成] 新开发源ID30..33作主要评测，同状态/代码/噪声新旧对照；原dev20..23仅次要回归，原heldout10..13继续隔离；旧任务8例保持。恢复总数提升且各类不退化、保持8/8才晋级。
5. [完成] 保留正负结果、实际录像、数据和配置。c739fb2公开远端独立恢复1973文件一致；干净上游9补丁/12运行文件一致，外部模拟器源码一致。Release数据42文件及权重完整SHA通过，71参数叶子（20 LoRA）重构一致。不晋级，不替换面板。

本轮最终结果：新开发6/12→5/12、旧开发10/12→7/12、常规8/8→8/8；不晋级、不替换面板。2000更新正常退出0，OOM0，峰值36.75GiB。88段录像完整解码通过，43项项目测试通过。


# Manual recovery 2026-10-28
Goal: use the six approved manual demonstrations with existing task replay; evaluate parent vs new policy on frozen unseen recovery states. No more collection now. Preserve original recordings and unrelated jobs.
1. DONE: six recordings frozen, OSC/joint inspected. Raw trajectories preserved; independent contact replay is not bit-exact.
2. DONE: 1612 eligible OSC anchors across eight clips; 816 joint controls archived only. Fresh IDs40..43 protocol frozen.
3. DONE: manual6_replay2000_v1, 2000 LoRA updates, 25% human / 75% old replay, memory monitoring.
4. DONE: 64 matched evaluations; fresh5/12→6/12, regression9/12→9/12, retention8/8→8/8. Motion efficiency worse; 88 videos decoded, no automatic UI replacement.
5. DONE: payload18e2041 restored independently from user-owned GitHub origin; all2615 tracked files identical. Release data4226 file hashes match; downloaded adapter reconstructs71 leaves exactly (20 LoRA). Evidence committed.
Stop: report measured paired results, no unsupported generalization claim; do not replace panel model unless promotion passes.

## 2026-10-29 用户批准：关节恢复转换为7维OSC的物理验收
1. [完成] 保留原录制和旧快照；固定两个新参考的完整剩余恢复段、动作缩放/旋转约定及验收门。只读FK副本，不向执行仿真注入参考状态。
2. [完成] 同一完整快照：原混合动作基准、仅关节段转换的位姿增量、仅关节段按参考位姿闭环跟踪。保持20Hz/原OSC指令/夹爪指令，不偷用关节目标更新OSC零空间锚点。
3. [完成] 验收实际抓取、全物理子步柜体接触、末端误差、末尾20控制的放置释放；保留失败和左右对照。转换未通过则不生成训练集、不训练、不换面板。
4. [完成] 30项测试、运行检查、8视频3860帧解码；代码与结果推自有远端，全部3461个跟踪文件独立恢复比对一致，恢复目录内9项新测试通过。

实测：两条参考逐状态位级一致；直接差分0/2、按实际位姿修正2/2通过，位置误差P95为1.20/0.85厘米；释放保持20/20，未训练/未换面板。30项测试、lint、编译及8录像3860帧已通过；载荷9eec59001dd9bd6daa69779332566ce3d5317611已推至自有公开仓库sencangWei/pi05-rl-handson的experiment/matched-recovery-20261027分支；全部3461个跟踪文件恢复内容一致，重建Git树84671b2b0123f9874ac1a553d20d22d11768b4be一致。证明evidence/osc-conversion29-remote-restore.json。

## 2026-10-28 用户同意：慢动作/恢复失败分层诊断
1. DONE: original trace hashes/protocol frozen; same MuJoCo3.8.1 matched eight trajectories exactly on recorded positions/flags. Wrong-version replay archived and excluded.
2. DONE: original dev bank audit6/12 invalid from bowl overlap; generator now rejects insertion/settled overlaps and records candidates. Separate restoration audit new bank12/12 valid; original states preserved.
3. DONE: six historical failures continued300→600 controls,0/6success (five invalid starts); new valid paired bank parent10/12 vs manual6 10/12, all9 shared successes slower95→114 median controls. Two remaining failures never grasped bowl, each exact300-control replay complete.
4. DONE: report/correction,86 tests+2subtests, Ruff and60 full video decodes passed. No training/UI change. Code/evidence payload8bd6b6e independently restored from user-owned GitHub origin, all2787 tracked files identical; proof recorded in evidence/recovery-diagnostics28-remote-restore.json.
Stop: diagnosis delivered; do not claim recovery repaired or promote unchanged model. Training-state audit and targeted relabeling are a proposed next experiment, not executed in this turn.

## 2026-10-28 用户继续同意：训练数据与采样审计
1. DONE: preserve manifests/raw recordings; audit train initial states with the existing penetration gate, without using dev/heldout as training data.
2. DONE: verify raw→Parquet action/state alignment; quantify stationary controls, terminal padding, window cancellation and actual mixture weights against public task8 train data.
3. DONE: document evidence-backed next experiment, distinguishing sampling/label hypotheses from causal proof; do not launch training or replace UI during this audit.
4. DONE: CPU tests (90 + 2 subtests) and lint pass; audit payload d1a0d16 pushed to user-owned origin and independently restored, all2796 tracked files SHA256-identical.
Stop: deliver measured data audit and concrete training recommendation; preserve original human demonstrations and all unrelated jobs.

## 2026-10-28 Approved sampling-only ablation
1. DONE: configurable 1:39 sampling, preserve default1:3 and all data; seeded preview1950 public/50 human.92tests+6subtests and scoped Ruff passed; code fe28dde independently restored from remote patch, whole tree/all2802 tracked files identical.
2. DONE: fresh parent LoRA2000 updates, batch1/seed42/lr5e-6; finalized1999, normal exit0, exact consumed1950public/50human and excluded1prefetch.
3. DONE: same valid12 dev states: parent10/12, old mix10/12, candidate8/12; placement0/4. Shared-success median old113.5→new92.5, parent92.5→new92.5. Gate failed; no UI promotion, heldout sealed.
4. DONE: report, 24 comparisons/12 original videos decoded and published. Remote payload c2d7176 independently restored: all2976 tracked files identical. Release adapter independently downloaded185506847bytes/SHA exact; official-base reconstruction71 leaves/20 LoRA exact. No UI replacement; unrelated jobs preserved.

## 2026-10-28 Approved task/phase sampling experiment
User approved the next bounded run after sampling025 failed. Audit observable phase labels first; preserve all raw/public data and OSC temporal labels. Freeze task/phase mixture, parent, 2000-update budget and matched dev/retention gates before training. No UI promotion without verified gates. Back up code/results to owned origin and independently restore.

Task-phase preflight: all1612human and7633publictask8 anchors partitioned exhaustively/disjointly. Human pools other1386/closing126/opening100. Public pools6289/672/672. 85 scoped tests passed and error-level Ruff passed. Physical grasp labels unavailable; command changes only. Reviewer role unavailable (unsupported configured model); direct review. No UI promotion this experiment.

## 2026-10-28 Task/phase bounded experiment result
2000 LoRA updates completed, batch1/seed42, fresh optimizer; actual1603 general replay+192 publictask8+205 human samples match frozen preview. Twelve reused physically valid dev scenes: parent10/12, old25%10/12, previous2.5%8/12, taskphase8/12; placement0/4. Common8 median controls parent92.5 vs candidate98; both frozen gates FAIL. All36 paired comparison videos decoded; full raw traces retained. Contact sheet shows tipped plate/init40 and unsuccessful grasp/transport/init41..43; native goal never met in all four timeouts. No UI promotion, no heldout/40-task retention evaluation. Inference export71 leaves/20LoRA exact; optimizer remains local. Result/code/weight remote restore verification pending.

2026-10-28 Task/phase delivery verification COMPLETE: owned public origin branch experiment/matched-recovery-20261027 payload d64a37edd407d90e314b8e4c542e6be445253d79 independently restored all3225 tracked files; rebuilt tree b2b3dfb3fabdaf69e94ab592ed9966ee2f2f6c77 exact. Release taskphase-ablation-20261028 independently downloaded185501727bytes in32 validated ranges; SHA2568cf5a41d4c5df14bfe0403383f87ad78591c37d6b92da472a0ceb2cf484638be exact; official-base reconstruction71 tensors/20LoRA exact. Own train/server inactive with exit0; no UI change. Bounded experiment delivered with negative outcome8/12 and placement0/4; no heldout or40-task retention claim.

## 2026-10-28 放置恢复事件诊断（用户同意继续）
1. [完成] 冻结现有父/候选轨迹与六条原始人工录制身份；八条放置回放全部记录位置/标志精确一致，定义代理事件及限制。
2. [完成] 四个放置初态逐场配对；区分提起未搬运/推走/撞桌；2446个人工保存状态检查确认两条关键搬运位于未入OSC监督的关节段。
3. [完成] 报告、两张帧图、八段录像及下一项控制契约方案已交付；10项针对性测试、完整Ruff、编译、八视频全解码通过。载荷6b7d2e6已推自有远端，独立恢复全部3274跟踪文件及Git树一致。未训练/未换面板；恢复未修好。

## 2026-10-28 用户批准：混合关节接管动作的可执行性验证
1. [完成] 两条原录制SHA和1221控制时间线固定；原关节源码相同，原启动器按录制SHA从Git历史找回。
2. [完成] v1/v2各实际执行两条原OSC→关节→OSC控制。最终释放门均通过，严格轨迹保真均失败；补齐两次初始化刷新只改观测相位、不改物理轨迹。原状态只含time/qpos/qvel，具体偏差来源未完全隔离。
3. [条件未满足，未执行] 原动作严格保真未通过，故本轮不转换关节→OSC、不生成标签、不重训。下一步先补全状态保存恢复并做短接触验证，非重复采集更多位置。
4. [完成] 30个相关测试、错误级Ruff和四录像2442帧解码通过。载荷5048f3e已推自有远端，全部3307个跟踪文件从远端独立恢复，SHA256与重建Git树f36ef5616a70b282946c958c3944a7c277e939e4一致；证明evidence/mixed-replay-28-remote-restore.json。未改面板/模型或其他任务。
备注：explore角色因安装模型不支持启动失败，本轮由主代理直接检查。

## 2026-10-28 完整状态恢复契约（用户确认继续）
1. [完成] 新增离线 OSC 边界快照：MuJoCo 原生数据、控制器目标/缓存、夹爪累积、观测相位、机器人缓冲、计数和 RNG；固定运行库与模型身份，禁止恢复时 forward/reset_goal。
2. [完成] 两条旧动作各生成一次新的参考执行，在 OSC→joint 前和最后 joint→OSC 前各保存快照；独立新进程恢复后执行剩余动作，不注入下一状态。对照旧79维、原生物理单独恢复、完整恢复。严格门沿用 qpos1e-6/qvel1e-5/缓存位置1e-5m，并记录像素/目标/接触一致性。
3. [完成] 完整恢复及 integration+Python 均4/4完全一致，旧79维0/4；40项测试、22录像2126帧通过。载荷 db2c707e989648f7a5e974aadcc2a45ae459e416 已推至自有公开仓库 sencangWei/pi05-rl-handson 的 experiment/matched-recovery-20261027 分支；全部 3406 个跟踪文件从远端独立恢复，内容 SHA256 与重建 Git 树 cefb8715771eab09f1d00ab06e0ad47043fa817d 一致。4份快照经远端分段下载和 Git blob SHA 校验。此实验无法补回旧录制丢失的内部状态，未转换训练标签、未训练或替换面板模型。


## 2026-10-29 批准：两条通过转换轨迹的实际观测重采与审核
1. [完成] 固定通过的tracking动作、完整初态快照和源码身份；不改原录制/训练/面板。
2. [完成] 新进程逐步采集执行前RGB/8D状态与7D动作，逐位核对执行后物理状态及像素。
3. [完成] 保存待审核Parquet、全程20fps视频、中文审核页与校验证据；ready=false，不启动训练。
4. [完成] 针对性测试、全量数据/视频校验、可视检查后提交自有远端并独立恢复验证。

2026-10-29 重采交付完成：用户明确确认“两段没毛病确实放上去了”，human-review.json 已记录审批与视频哈希；965对含40诊断保持步，未训练/换面板。载荷 b05024ae09d27643780c7380e057d20f39bfbe47 已推至自有公开 sencangWei/pi05-rl-handson 的 experiment/matched-recovery-20261027；独立恢复3487跟踪文件，SHA256和Git树6be2b363c8f7e5a10811f3fc796d116cb2e2f5b0精确一致，恢复树上35测试及5子测试/Ruff再次通过。Release osc-pairs-review-20261029 的113996547字节数据包通过32段远端下载、整体/文件SHA核对及1930张PNG独立配对校验。证明 evidence/osc-pairs29-remote-restore.json 与 evidence/osc-pairs29-data-restore.json。后续训练格式/抽样预检尚未执行，ready仍false；两段为专家新执行，非模型恢复能力验收。


## 2026-10-29 已批准：OSC专家补充监督的有界微调
1. 合并既有全部1612个OSC人工起点与两条已审核专家执行；排除40个诊断保持步及新片末9个不足完整窗口的起点，不改变20Hz动作。
2. 同一40任务父模型、fresh AdamW、2000更新/batch1/seed42/lr5e-6；75%公开回放/25%恢复槽，旧人工与新专家按有效起点分配。冻结源码、数据、父权重与12个既有dev场景；不打开封存集。
3. 实际训练、核对梯度样本顺序，独立仿真对比父模型；逐场报告恢复/耗步/时延。无UI自动晋升。
4. 测试、训练配置、结果与推理权重备份自有远端，独立恢复比对。成功仅指完成有证据实验，不预设能力改善。

2026-10-29 00:55 CST: actual bounded OSC expert replay training launched as pi05-osc-sft29-train; preflight passed. Frozen first2000 preview: public1457 + oldhuman345 + expert198; 1 final prefetch excluded. Matched 12+12 evaluation runner waits for verified completion. No UI change.


## 2026-10-29 OSC专家补充微调完成（能力门未通过）
实际2000次LoRA更新、batch1、seed42，公开1457/旧人工345/新增专家198，全部2000样本顺序与冻结预览一致；未删除任何原始纠错。训练与保存正常退出，无OOM。重新配对12个既有开发初态：父9/12、新6/12；抓空4→2/4、偏移4→4/4、放置1→0/4。共同成功6例控制步中位数92.5→166.5，预设两门均失败，未替换面板，未评测封存集或40任务保留能力。
四个放置场景均未到原生目标；前三个没有手指接触且碗高度完全不变，不能归因于成功后未松手。新增示范从持碗开始，阶段分布偏向只是待验证假设，不是已证明因果。23项针对性测试+9子测试、Ruff通过；60视频文件12201帧全解码通过。父评测后端口TIME-WAIT导致误报，保留失败日志，修可用性探测并仅接续候选；策略算法和冻结训练源不变。结果、原始轨迹、左右对比见reports/osc-sft-20261029/README.md；远端最终恢复核验进行中。

2026-10-29 交付核验完成：载荷 b2ed8691448d2b94e1e7589c148d481662efee90 已备份自有公开 sencangWei/pi05-rl-handson 的 experiment/matched-recovery-20261027 分支；独立远端恢复3644个跟踪文件，SHA256及重建Git树 bdd46e4ec5c46162a75de517e0470cea7e347f37 一致。Release数据四文件与适配器三文件均远端独立恢复一致，官方基座重建71张量/20LoRA精确匹配。证明 evidence/osc-sft29-final-remote-restore.json、osc-sft29-data-restore.json、osc-sft29-adapter-restore.json。仅推理权重有远端备份，完整优化器checkpoint仍在本机。负面能力结果不变，未替换面板；后续同专家完整初态诊断尚未执行。

## 2026-10-29 批准：专家完整初态父/新策略对照诊断
1. [完成] 固定两条已审核专家的完整快照、父/候选权重和动作/成功契约；环境静态检查。
2. [完成] 独立恢复同一初态，父/候选各跑两例；先校验观测/状态，再真实闭环策略执行，保留失败与录像。
3. [完成] 对比持碗放置能力与既有未抓取开发场景，说明证据边界；不训练、不换面板、不打开封存集。
4. [验证完成，备份进行中] 针对性检查、结果/录像/代码提交自有远端并独立恢复验证。
探索子代理因已安装spark模型不受当前账户支持而启动失败，主代理直接检查，不声称独立审查。

四次真实闭环执行完成：父模型600步预算0/2，新模型118/562步成功2/2；预先冻结300步里程碑新模型1/2。这是两个训练初态的诊断，不能当恢复泛化率。第二条401–500步平均末端位移0.729毫米/步，全程接触，慢动作仍存在。初始完整物理状态、8维状态和两路实际RGB与审核数据精确匹配；前后输入与权重哈希不变。6项新增守卫测试和5项既有契约测试、Ruff、编译通过；4条1880控制步与6录像3086帧全解码通过。报告及左右同步录像见reports/expert-policy-20261029/README.md；未训练、未换面板、未触碰封存集/真机/其他任务。
