# 最终已保存PPO权重：30次更新的独立仿真

独立进程实际恢复最终adapter，在MuJoCo/LIBERO运行八个场景，正常退出0。SFT与PPO均8/8触发原始任务成功判据；平均控制步为105与106，PPO在此小样本上没有改善效率，也没有成功率提升证据。

## 逐场对照

|任务|初态|SFT步数|PPO步数|同初态并列录像|
|---|---|---|---|---|
|打开抽屉|0|119|118|[播放](comparison-goal-00-init-00.mp4)|
|打开抽屉|10|121|123|[播放](comparison-goal-00-init-10.mp4)|
|打开灶炉|0|84|86|[播放](comparison-goal-07-init-00.mp4)|
|打开灶炉|10|80|80|[播放](comparison-goal-07-init-10.mp4)|
|碗放盘|0|87|87|[播放](comparison-goal-08-init-00.mp4)|
|碗放盘|10|83|83|[播放](comparison-goal-08-init-10.mp4)|
|酒瓶放架|0|129|131|[播放](comparison-goal-09-init-00.mp4)|
|酒瓶放架|10|137|140|[播放](comparison-goal-09-init-10.mp4)|

左侧SFT、右侧PPO，每侧同时显示外部与腕部相机。10fps对应控制帧，未加速；较早完成一侧冻结末帧。视频时长不能当成真实推理耗时。

## 模型与测试协议

- 血缘：官方π0.5 LIBERO → 40任务、10000次LoRA SFT → 合并冻结基座 → 新LoRA/价值头30次PPO优化；没有用手写抓放轨迹替代模型动作。
- 最终adapter格式rlinf-fsdp1-single-rank-adapter-v1，121226493字节；SHA256 `0595e247536bc48d5498f5085823a51c595b36cfd14b965f642f81256daaf7f1`。550个可训练张量均有限；242个优化器状态step30，308个step0，不能称全部注入LoRA都被更新。
- 冻结合并基座SHA256 `4526e30c6aa11108a3331f7f4ae7931394629961df42736deedd7169ec154d5f`；必须匹配，约13GB基座未上传。adapter通过GitHub Release提供。
- 两策略使用同一原生Torch/BF16运行时、初态字节哈希、归一化、seed42、逐调用固定噪声。62个动作/模拟运行时源码哈希匹配，见identity-validation.json。
- 独立评测10次流采样，预测10步、每次执行5步后重新观测，最多300控制步；训练采样5次流采样、每10步重规划。评测协议两侧一致，但不同于训练采样。
- 原始结果见ppo-summary.json与native-sft-summary.json；身份见ppo-identity.json。该八场景不是40任务留出集评测，不能称未见任务泛化。

## 遇到的问题与处理

1. JAX→Torch输出注意力LoRA合并需跨头求和；修正转换。两处GELU近似不一致已修正，float32固定输入整网数值误差1.66893e-6，原阈值1e-3通过。该数值检查不证明BF16整轨迹完全相同。
2. 5090D不支持旧cu126算子路径；独立环境改cu128，真实BF16矩阵和SDPA前后向通过。
3. PEFT输入转换与FSDP子边界精度不一致；子边界启用输入转换，根边界保留FP32，真实小模型回归通过。
4. 全模型保存超48GiB内存上限；改为绑定基座SHA的adapter格式，保存可训练张量、价值头、优化器、调度器与RNG。小模型CUDA恢复与拒绝错误基座等检查通过，大模型最终文件也独立加载成功。
5. 重复权重同步仍耗内存；首次全量、后续选参同步，且每轮评测已同步更新权重，跳过第二轮开头的冗余同步。v10最终30更新权重因此成功落盘。
6. **内存问题仍未彻底解决**：v10最终原子保存后进程仍被48GiB内存上限终止，service_result=oom-kill。不能称训练正常结束、可稳定长训，不能把小模型回归当成完整模型容量修复。独立评测进程已正常退出。

## 边界与后续

成功使用LIBERO原始任务判据，不额外验证碗已松爪、退开或持续稳定；最终画面不代表这些额外条件已达成。尚未检验失败恢复、40任务遗忘、中文连续指令或真实家庭物品。30次PPO更新只是训练链路试跑，不能把稀疏奖励当成已经学会通用恢复。下一步应先解决完整训练内存退出，再扩大训练与独立评测矩阵；不得以本结果替代产品验收。

## 复跑（本机现有基座与固定仿真依赖）

```bash
cd /home/robot/pi05-rl-handson
JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  .venv-ppo/bin/python -u scripts/evaluate_native_ppo.py \
  --checkpoint exports/pi05-ppo-20261026-30updates-adapter.pt \
  --output evaluations/another_final_ppo_test
```

输出目录必须全新；依赖只读复用的固定版本/home/robot/voice-arm-sim，不能仅克隆本仓库就称已完整复现运行环境。可用配置与身份哈希随仓库；13GB基座、完整数据与虚拟环境不在Git备份中。

实际帧抽查：[八场景过程截图](ppo-timelines.png)、[四任务终帧对照](comparison-endpoints.png)。24段MP4均完整解码，帧率与哈希见video-validation.json。
