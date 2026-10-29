# Public joint-action data investigation (2026-10-30)

## Decision

Use the official LIBERO_goal `put_the_bowl_on_the_plate_demo.hdf5` (50 normal task demonstrations) as the **source of reference motion**, then generate local action8 commands by continuous physical re-execution under our existing KP150 absolute-joint controller. Keep the three previously approved human recovery recordings as separate training replay. Original OSC7 controls and original images must not be labelled as native joint8 data.

The public source is [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a), revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`. The exact task is the existing LIBERO_goal task8 (black bowl next to plate onto plate), with approximately 468 MB compressed HDF5. Preserve the downloaded artifact checksum in every conversion protocol. Original HDF5 actions are OSC_POSE7, not arm-joint7 + gripper.

## A real public joint-action alternative

[jiankimr/libero_10_joint_lerobot](https://huggingface.co/datasets/jiankimr/libero_10_joint_lerobot/tree/2260df157dbdefcbfbc0e3629f63041a13253c66) revision `2260df157dbdefcbfbc0e3629f63041a13253c66` contains 500 episodes, 137752 frames, 10 LIBERO-10 long-horizon tasks and approximately 994 MB. Metadata: [info.json](https://huggingface.co/datasets/jiankimr/libero_10_joint_lerobot/resolve/2260df157dbdefcbfbc0e3629f63041a13253c66/meta/info.json); the dataset README and build report describe conversion and success filtering.

- State9 is Panda joint positions7 in radians + gripper finger positions2 in metres.
- Action8 is **joint delta radians7 + gripper**, 20 Hz; gripper is 1=open, 0=close.
- The author describes actual JOINT_POSITION re-execution with kp=2000, delta bound0.2 rad. Desired delta is `clip(q_reference[t+1] - q_current_measured, -0.2, +0.2)`; simulator input is `delta / 0.2`. Gripper simulator command is `1 - 2 * gripper`.
- Main and wrist images are 256x256 RGB, already flipped on both image axes.
- 410/500 episodes succeeded (111329 frames). The published dataset deliberately retains 90 failures. A stricter success + no clipping + maximum drift<=0.1 filter leaves 377 episodes.
- Its tasks do **not** include our exact normal bowl-next-to-plate task. Task8 is both moka pots on the stove. Training that task under our unchanged bowl prompt would be incorrect.
- No corresponding author-published `libero_goal_joint_lerobot` or `libero_spatial_joint_lerobot` was found in the HF dataset listings during the bounded search.
- The README points to `jiankimr/h100vast-backup` and `jiankimr/rsec` for source, but those GitHub repositories/raw paths returned404 during this investigation. Controller semantics above are documented dataset claims; the conversion implementation could not be independently source-audited.

Do not silently mix this kp2000 delta representation with our kp150 absolute joint targets. Changing control gains also requires replaying the approved human recovery data. These are compatibility constraints, not a reason to ignore public normal data.

## Local conversion acceptance

`.venv-ppo/bin/python scripts/public_joint_replay.py --hdf5 <official HDF5> --episode <0..49> --output <new directory>`:

1. Freeze demo0..39=train and demo40..49=holdout **before** observing outcomes. Never move a failed holdout demonstration into train.
2. Restore only the demonstration XML and initial state once, localizing asset paths and making existing extension-derived mesh MIME types explicit for this host; preserve geometry, joint order and camera placement.
3. Execute the original OSC actions in continuous physics; record actual qpos/qvel and end-effector trajectories. Preserve a complete locally captured initial snapshot.
4. Restore that same local initial snapshot in a fresh environment, install the existing KP150 absolute controller, and execute candidate targets `q_reference_next + (kd/kp) * qvel_reference_next`. No per-step saved next-state injection.
5. Record each actual command8 with its **pre-command** joint9 state and freshly rendered main/wrist RGB. Verify continuous time, target equality, image/action hashes and on-disk parquet alignment.
6. Require normal native LIBERO success and release (no grasp/no finger contact) for the final20 diagnostic controls, finite dynamics, no joint-limit clipping, trajectory tracking gates, and no added cabinet penetration. Failed conversions retain diagnostics but do not emit training parquet.
7. This normal-data gate does not require the newer recovery evaluation's 0.10 m end-effector retreat. Placement source demonstrations may stop before retreat. Record this difference explicitly rather than describing the gates as identical.

[robosuite upstream issue781](https://github.com/ARISE-Initiative/robosuite/issues/781) independently illustrates why observed joint positions are not automatically valid JOINT_POSITION action labels: absolute-vs-delta controller configuration and real execution matter.

## Evidence boundary

A successful source conversion proves a usable local expert trajectory for that episode. It does not prove VLA training success, autonomous recovery, generalization, or that a public OSC dataset originally contained joint8 action labels. Aggregate converted successes, conversion rejection reasons, train/holdout coverage and downstream policy evaluations separately.

## Runtime proof and reproducibility fixes

The snapshot contract requires the already installed `.venv-ppo`: MuJoCo3.8.1, robosuite1.4.1, NumPy1.26.4. `.venv-sim` has MuJoCo3.2.3 and is incompatible with these full snapshots. No MuJoCo version was upgraded.

MuJoCo3.8 infers mesh `content_type` on an asset-cache miss but omits the inferred attribute on a cache hit. The XML serializer therefore initially produced different textual model hashes in two physically equivalent environments. The converter now explicitly sets the extension-derived MIME metadata (`.msh`, `.stl`, `.obj`) before either compilation. Full snapshot identity checks remain enabled and pass across fresh environments.

The first exact-task source demo0 (90 original controls) physically succeeded in OSC replay. Its continuous KP150 joint replay, including20 diagnostic open-gripper controls, also passed all frozen gates: end-effector tracking p95=0.005317m, max=0.005893m; maximum joint error=0.025594rad; no clipped commands or added cabinet penetration; stable native goal with release. The110 rows/220 decoded RGB images passed pre-action alignment and continuous-physics validation. This initial proof used a losslessly extracted one-episode source subset, so its source hash differs from the full official download; production conversion is separately run against the complete verified source.

The complete official468246288-byte source has SHA256 `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406`, matching the pinned HF LFS metadata. `source-actions-states.hdf5` is a durable7,110,383-byte lossless subset preserving all50 full state/action arrays and XML/initial-state attributes; every array and XML was compared against the complete source. Its SHA256 is `d7415020d15e6898ff1b7541b87cee6f4ae08c297b91f8d2b386b524af2ee51a`. Source RGB is intentionally excluded because valid joint replay records new local RGB.

Evaluation must use the converter observation path: `env._update_observables(force=True)` followed by `env.env._get_observations()`, after one initialization and full snapshot restore. Direct tests at actual joint controls1,50,70,75 found `regenerate_obs_from_state(current_state)` changes both camera byte arrays and derived `qacc`; the initial snapshot alone happened to produce equal images. Thus using per-control regeneration in evaluation would introduce a train/evaluation observation mismatch even when flattened qpos/qvel/time appear unchanged. The converter records/checks unchanged physical state around observation capture and never calls this setter per control.

A fresh independent process restored production demo0 and re-executed all110 recorded joint commands. Every pre- and post-command physics state was bitwise equal to the recorded rollout, and both RGB camera hashes matched for every control. Evidence: `fresh-demo0-verification.json`. Targeted converter tests:9 passed.

## Complete 50-episode conversion

All50 official exact-task episodes were continuously re-executed with the frozen converter. 47 passed; rejected IDs: [13, 21, 36]. Train: 37/40 accepted, 3428 source-motion rows. Holdout: 10/10 accepted, 958 source-motion rows. Holdout IDs remain excluded from training regardless of outcomes. Remove the20 diagnostic hold rows per accepted episode from training; they exist only to test stable release.

Episode13 reaches native goal but has one late finger contact and clipped targets; episode21 misses converted native goal; episode36 also fails stable release and clipping. No rejected episode emits training parquet. All 597 artifact hashes and all protocol converter hashes were freshly verified. Full structured evidence is `conversion-summary.json`. Bulk replay artifacts are still in RAM disk and must be persisted before reboot; durable compact source is sufficient to regenerate them. Automated acceptance does not claim user review or VLA recovery performance.
