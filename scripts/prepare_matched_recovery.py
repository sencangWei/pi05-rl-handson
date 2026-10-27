"""Freeze a bounded matched-scene experiment after independently verified collection."""
import json
from pathlib import Path

from export_sft_for_ppo import sha256
from recovery_replay_dataset import ModelRecoveryClips

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "evidence/matched-recovery-manifest-v1.json"
    data = ModelRecoveryClips(manifest)
    bank = ROOT / "data/recovery-bowl-v3/manifest.json"
    if sha256(bank) != data.manifest["bank_manifest_sha256"]:
        raise ValueError("Training bank changed after independent replay")
    from baseline import configure_libero
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    base = Path(get_libero_path("bddl_files")) / task.problem_folder
    init = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    files = {"bddl": base / task.bddl_file, "init": init}
    compatibility = dict(passed=True, task_suite="libero_goal", task_id=8,
                         prompt=task.language, bank_manifest_sha256=sha256(bank),
                         **{key: dict(path=str(path), expected=sha256(path)) for key, path in files.items()})
    fresh = ROOT / "data/recovery-bowl-freshdev27/manifest.json"
    old = ROOT / "data/recovery-bowl-dev27/manifest.json"
    plan = dict(status="FROZEN_BEFORE_TRAINING", experiment="matched_recovery2000_v1",
                recovery_manifest_sha256=sha256(manifest),
                train=dict(parent="multitask40_lora_v1/9999", updates=2000, lr=5e-6, batch=1,
                           seed=42, recovery_probability=.25, old_replay_probability=.75,
                           recovery_episodes=len(data.records), recovery_frames=len(data)),
                development=dict(manifest=str(fresh.relative_to(ROOT)), manifest_sha256=sha256(fresh),
                                 source_ids=list(range(30, 34)), trials=12, controls_max=300,
                                 replan=5, flow_steps=10, noise_seed=42,
                                 success="native goal AND no finger contact for 5 consecutive controls",
                                 label="fresh development; not training or blind final test"),
                regression=dict(manifest=str(old.relative_to(ROOT)), manifest_sha256=sha256(old),
                                source_ids=list(range(20, 24)), trials=12),
                retention=dict(tasks=[0, 7, 8, 9], initial_states=[0, 10], trials=8, max_controls=300),
                selection="Fresh development total strictly improves over freshly rerun SFT parent, no category decreases; old development total/category do not decrease; retention 8/8. No automatic panel replacement.",
                quarantine="Heldout source IDs 10..13 excluded from training and selection",
                provenance="Learned PPO teacher autonomous successful trajectories; no human or scripted action labels. This round is LoRA SFT, not additional PPO.")
    for filename, info in (("matched-recovery-task-compatibility.json", compatibility),
                           ("matched-recovery-evaluation-plan-v1.json", plan)):
        path = ROOT / "evidence" / filename
        with path.open("x") as stream:
            json.dump(info, stream, indent=2)
            stream.write("\n")
    print(json.dumps(plan), flush=True)


if __name__ == "__main__":
    main()
