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
