# 恢复专项：同初态三模型对照

左：40任务SFT；中：旧30更新PPO；右：新300更新PPO。全部三类各4个、共12个留出初态。每个视频按20Hz仿真时间同步，先完成的一列冻结末帧。

**三模型都是9/12，没有恢复成功率提升证据。** 新模型偏移init13由98步变208步，效率退化；放置失败init10/11/12三组均到300控制步上限。

| 初态 | SFT控制步 | 旧30控制步 | 新300控制步 | 三栏视频 |
| --- | --- | --- | --- | --- |
| heldout-empty_grasp-init10 | 77 成功 | 77 成功 | 79 成功 | [播放](comparison-heldout-empty_grasp-init10.mp4) |
| heldout-object_offset-init10 | 90 成功 | 90 成功 | 93 成功 | [播放](comparison-heldout-object_offset-init10.mp4) |
| heldout-placement_failure-init10 | 300 超时 | 300 超时 | 300 超时 | [播放](comparison-heldout-placement_failure-init10.mp4) |
| heldout-empty_grasp-init11 | 90 成功 | 88 成功 | 85 成功 | [播放](comparison-heldout-empty_grasp-init11.mp4) |
| heldout-object_offset-init11 | 88 成功 | 88 成功 | 88 成功 | [播放](comparison-heldout-object_offset-init11.mp4) |
| heldout-placement_failure-init11 | 300 超时 | 300 超时 | 300 超时 | [播放](comparison-heldout-placement_failure-init11.mp4) |
| heldout-empty_grasp-init12 | 79 成功 | 77 成功 | 77 成功 | [播放](comparison-heldout-empty_grasp-init12.mp4) |
| heldout-object_offset-init12 | 86 成功 | 84 成功 | 84 成功 | [播放](comparison-heldout-object_offset-init12.mp4) |
| heldout-placement_failure-init12 | 300 超时 | 300 超时 | 300 超时 | [播放](comparison-heldout-placement_failure-init12.mp4) |
| heldout-empty_grasp-init13 | 90 成功 | 90 成功 | 90 成功 | [播放](comparison-heldout-empty_grasp-init13.mp4) |
| heldout-object_offset-init13 | 98 成功 | 109 成功 | 208 成功 | [播放](comparison-heldout-object_offset-init13.mp4) |
| heldout-placement_failure-init13 | 112 成功 | 99 成功 | 110 成功 | [播放](comparison-heldout-placement_failure-init13.mp4) |

每个模型子目录保存双相机原视频、逐控制步JSONL、summary和模型身份： [SFT](sft/) / [旧PPO](old30/) / [新PPO](new300/)。

严格成功要求原生目标成立且目标碗无手指接触，连续5控制步；结果来自实际策略动作。合成初态的构造动作不参与恢复执行。

完整训练、判据、限制及权重恢复见 [专项报告](../README.md)。
