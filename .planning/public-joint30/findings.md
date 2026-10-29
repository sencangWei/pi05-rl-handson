# Evidence
DROID official schema provides commanded joint_position7 + gripper_position1, unlike LIBERO OSC actions7. DROID real imagery/gripper/frequency differs from current sim; cannot mix unchanged.
https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md
https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/droid_rlds_dataset.py
LIBERO official author dataset has individual HDF5 files with states/actions; same-task put_the_bowl_on_the_plate_demo.hdf5 is ~468MB. Need inspect actual downloaded labels; joint_states alone is not joint commands. Can re-execute converted targets under existing validated joint controller and gate physical success.
https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/main/libero_goal
Previous training: 3 recovery episodes + 1 reexecuted trajectory, 2122 frames, no extra public joint-action normal data; cannot represent broad task coverage.

Native public joint source verified: jiankimr/libero_10_joint_lerobot revision2260df157dbdefcbfbc0e3629f63041a13253c66, 500 episodes/137752 frames, 410 success; 377 after no-clipping/drift filtering. Does not contain current goal8, KP2000 vs local150; author code unavailable404. Current mainline chosen: official same-task50episodes via validated physical joint re-execution, not pretend native labels. Research agent owns scripts/public_joint_replay.py only.
Source download pinned f13aa24a3da8c43c7225569f28c562979fa0e35a; full HDF5 download slow. Evaluating HTTP range access to download only state/actions.

Official source complete: sha256 e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406. 50 episodes4669frames, train40/3711frames, holdout10/958frames before filtering. Native joint release found (500episodes) but wrong task and kp2000 delta control; use real physical KP150 re-execution to match approved human commands. First demo nativegoal+release20 passed, maxEEF error5.89mm. Runtime must use .venv-ppo MuJoCo3.8.1; .venv-sim3.2.3 cannot restore snapshots. Explicit mesh MIME fixes cache-dependent XMLidentity. Public observations must refresh sensors without state regeneration.
