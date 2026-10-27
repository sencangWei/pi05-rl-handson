# 人工恢复采集窗口

入口：`scripts/manual_recovery.py`。使用本项目 `.venv-sim`，依赖当前机器已有的
`voice-arm-sim/.local/openpi/third_party/libero` 资源和 `data/recovery-bowl-v3` 失败状态库。

```sh
.venv-sim/bin/python scripts/manual_recovery.py --check
.venv-sim/bin/python scripts/manual_recovery.py --initial-id 1 --category placement_failure
```

默认恢复 `train-placement_failure-init01`：这是冻结的合成放置失败初态，
此前自主教师从此初态运行 300 步未成功；并非该运行第 300 步的终态。
加载时校验完整物理状态的 SHA256，重设控制器目标与开爪累积量。
仅使用训练初态 1–6；不采留出/开发集。默认任务为 **LIBERO goal 8**，
文字指令以当前官方任务资源为准，不能与旧 spatial 场景混用。

窗口启动时已暂停并开启人工接管。等待时间不产生数据。
鼠标摇杆对应三维窗口视角，Z 按钮升降；普通幅度用于搬运、精细幅度用于抓放。
完成后松爪、撤手至少 10cm、保持 10 步，再点“保存为合格”。
未完成请存草稿；关闭时保留草稿。合格也仍需后续人工审核。
模型推理和训练入口在本采集窗口禁用；不连接真机。

产物位于本项目 `corrections/human_*/`，包含双相机、8 维状态、7 维动作、
每次控制前后的完整物理状态及哈希、来源场景/任务/控制周期，标注为人工数据。
20Hz 为仿真控制周期；不会把按钮间停顿作为保持动作。
新数据尚未自动加入训练，需审核、建立清单并与旧数据回放合并。

`scripts/` 是从用户已有 `.pi05-handson/scripts` 逐字复制的中文面板依赖快照，
具体文件 SHA256 见 `source_manifest.json`，入口启动时逐一校验。
原项目未修改。完整面板代码保留以复用摇杆、相机、坐标轴和纠正数据验证；
入口隐藏模型/训练页，并将任务身份与输出根目录切换到本项目。
