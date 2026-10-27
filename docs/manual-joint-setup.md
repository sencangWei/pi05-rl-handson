# 人工恢复采集：同段关节接管

原来的 Rx/Ry/Rz 控制末端绕世界轴的姿态，不能指定某一个中间关节。现在“关节接管 J1～J7”页允许在人工录制中直接逐节调整：J1 基座、J4 肘部、J6/J7 靠近夹爪。鼠标指向按钮显示实际关节正轴，正转遵循右手规则。

1. 保持当前人工接管录制，切到“关节接管 J1～J7”。无需结束当前段。
2. 用 J1～J7 ± 微调，默认每次目标 2°，允许 0.5～5°。其他关节保持当前角度目标，接触时仍可能偏转。
3. 回人工纠正页继续 XYZ、Rx/Ry/Rz、夹爪与保持。记录目录、时间顺序及物理状态连续。
4. 若要重来，点“保留当前草稿，重载失败场景并重新采集”；旧段保存为草稿。

关节控制走真实 JOINT_POSITION 物理仿真，每次12个20Hz控制步，保留碰撞。它会移动夹爪位置，没有自动避障保证；遇阻请换方向。完成或异常后恢复 OSC_POSE、7维动作空间及动作缓冲。

## 数据与训练边界

OSC 帧保留原有7维动作；关节帧单独保存8维控制输入、实际生效的7关节绝对目标、动作前双相机RGB、状态和前后完整物理状态。metadata 的 control_timeline 保存混合顺序，界面同时显示两类步数。OSC 物理伴随帧还保存该步的 nullspace initial joints。

混合段标记 training_ready=false，不能直接交给旧7维OSC导出器训练，避免把关节目标误当末端动作。这次修复的是接管操作和完整数据留存，尚未实现混合动作策略训练及关节段续录重放。旧OSC续录器会明确拒绝混合段，原文件保留。关节调整后至少再完成10步OSC验证，且仍须通过原有放置、松爪和抬离成功检查，才能批准。

加载失败库时，零偏移重置原样保留物理状态，不调用挪碗操作；否则抓空初态会因为夹爪距碗小于12厘米而被错误拦截。非零偏移仍经过原有距离、开爪与碰撞检查。冻结的上游界面文件未修改。

## 验证

- `.venv-train/bin/python -m pytest -q tests/test_manual_joint_control.py tests/test_manual_rotation_guide.py tests/test_recovery_recording.py`：覆盖同段录制、控制类型、状态哈希、混合续录拦截与计数。
- `.venv-sim/bin/python scripts/manual_recovery.py --initial-id 4 --check-takeover`：真实物理仿真完成 OSC → J4 → OSC；同一段2步OSC、12步关节，肘部实际+1.98°，逐步物理状态连续，恢复7维OSC。测试段保存为草稿。
- GUI实际点击J4+，再回人工纠正页保持10步：同一目录记录12步关节、10步OSC。
- 既有 `--check-joints` 验证14次J1～J7正负切换及异常恢复，详见历史证据 `evidence/manual-joint-control-20261028.json`。
- 本次证据：`evidence/manual-joint-takeover-20261028.json`。
- 抓空布局1：`--initial-id 1 --category empty_grasp --check-takeover` 实测通过重置和OSC→关节→OSC，肘部+1.97°、物理状态逐步连续；零偏移及非零偏移检查包含在 `tests/test_manual_bank_reset.py`，相关测试共27项通过。

启动：

```bash
cd /home/robot/pi05-rl-handson
.venv-sim/bin/python scripts/manual_recovery.py --initial-id 4
```
