# 人工恢复采集：关节调姿

原来的 Rx/Ry/Rz 控制末端绕世界轴的姿态，OSC 协调多个关节，不能指定某一个中间关节。新增“关节调姿 J1～J7”页用于逐节调姿：J1 基座、J4 肘部、J6/J7 靠近夹爪。鼠标指向按钮显示该实际关节的正轴，正转遵循右手规则。

1. 点“结束本段为草稿，进入关节调姿”，旧录制保留。
2. 用 J1～J7 ± 微调，默认每次目标 2°，允许 0.5～5°。其他关节以当前角度为保持目标，接触时仍可能偏转。
3. 点“从当前姿态开始新采集”，回人工纠正页使用 XYZ、Rx/Ry/Rz、夹爪和保持按钮。
4. 若要重来，点“保留当前草稿，重载失败场景并重新采集”。

关节调姿走真实 JOINT_POSITION 物理控制，每次12个20Hz控制步，保留碰撞，没有自动避障保证。结束或异常后恢复 OSC_POSE、7维动作空间及其缓冲区。

## 数据边界

JOINT_POSITION 的8维输入与现有 OSC 的7维动作不是同一契约。关节调姿在 evaluations/manual-recovery-*/joint-setup-*.npz 单独保存完整状态、目标关节角及实际前后角度，training_sample=False。录制期间禁止关节调姿，不伪造7维标签。

新采集从调姿后的物理状态开始，元数据保存 OSC nullspace initial joints。关节调姿动作本身不会被当前策略学到；训练纯关节恢复需要匹配关节动作模型和数据契约。

## 验证

- `.venv-train/bin/python -m pytest -q tests/test_manual_joint_control.py tests/test_manual_rotation_guide.py tests/test_recovery_recording.py`：21 passed。仿真环境缺少 pyarrow，所以包含Parquet的录制测试用训练环境执行。
- `.venv-sim/bin/python scripts/manual_recovery.py --initial-id 4 --check-joints`：J1～J7 各 ±2°，实际约 ±1.94～1.98°；14次切换均恢复7维OSC并可继续物理控制。
- 注入 env.step 异常后，控制器、动作维度、缓冲区恢复，随后OSC控制状态有限。
- GUI 点击 J4+ 实际约+1.98°，轴标记显示于肘部；从当前姿态重新采集后10帧动作与10对物理状态正常保存。验证录制保留为草稿。
- `compileall` 与 `git diff --check` 通过。
- 证据：`evidence/manual-joint-control-20261028.json`。

启动：

```bash
cd /home/robot/pi05-rl-handson
.venv-sim/bin/python scripts/manual_recovery.py --initial-id 4
```
