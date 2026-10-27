"""Snapshot ten previously approved, complete human demonstrations; never edit sources."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from export_sft_for_ppo import sha256

ROOT = Path(__file__).resolve().parents[1]
SELECTION_SHA = "7200c875b761a56f6dd084dbdbd2e2638477b55e152345e720a47e20ec869548"
TASK = "pick up the black bowl next to the plate and place it on the plate"


def validate_frame(frame):
    for key, shape in (("state", (8,)), ("action", (7,)), ("post_state", (8,)), ("post_bowl_pos", (3,))):
        if frame[key].shape != shape or not np.isfinite(frame[key]).all():
            raise ValueError(f"Invalid {key}")
    if np.any(np.abs(frame["action"]) > 1):
        raise ValueError("Action out of bounds")
    for key in ("image", "wrist_image"):
        if frame[key].shape != (256, 256, 3) or frame[key].dtype != np.uint8:
            raise ValueError(f"Invalid {key}")
    if frame["done"].shape != () or frame["done"].dtype != np.bool_:
        raise ValueError("Invalid done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/robot/.pi05-handson"))
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    source = args.source.resolve()
    selection = source / "evidence/curation-edge-recovery-v1-selection.json"
    if sha256(selection) != SELECTION_SHA:
        raise ValueError("Reviewed selection changed")
    clips = json.loads(selection.read_text())["clips"][4:]
    if len(clips) != 10:
        raise ValueError("Expected ten complete later demonstrations")
    output = ROOT / "data/recovery-human28"
    output.mkdir(parents=True, exist_ok=False)
    media = ROOT / "artifacts/20261027-recovery-sft/human-review"
    media.mkdir(parents=True, exist_ok=False)
    replay = ROOT / "evidence/dataset-manifest.json"
    replay_info = json.loads(replay.read_text())
    task_ids = {r["task_id"] for r in replay_info["records"] if r["prompt"] == TASK}
    if task_ids != {36}:
        raise ValueError("Public task identity changed")
    records, audits = [], []
    sheets = []
    for episode, clip in enumerate(clips):
        directory = source / clip["source_directory"]
        meta_path = directory / "metadata.json"
        if sha256(meta_path) != clip["metadata_sha256"]:
            raise ValueError("Human metadata changed")
        meta = json.loads(meta_path.read_text())
        if (meta["status"] != "approved" or meta["split"] != "train" or meta["task"] != TASK
                or meta["approval_type"] != "manual_user_with_observation_gates"
                or not meta["human_actions_only"] or not meta["pauses_excluded"]
                or meta["control_dt"] != .05 or clip["start"] != 0
                or clip["stop"] != meta["frame_count"]
                or meta["camera_orientation"] != "same double-axis flip as original course inference"
                or meta["action_contract"] != "OSC_POSE normalized world delta xyz/axis-angle rotation; +1 close, -1 open"):
            raise ValueError("Approved whole-episode recording contract mismatch")
        if len(meta["files"]) != meta["frame_count"]:
            raise ValueError("Frame inventory mismatch")
        rows, tail, actions, previous = [], [], [], None
        discrepancy = 0.
        frames_for_review = {}
        keyframes = np.linspace(0, meta["frame_count"] - 1, 6, dtype=int)
        with imageio.get_writer(media / f"human-{episode:02d}.mp4", fps=20, codec="libx264") as video:
            for index, item in enumerate(meta["files"]):
                path = directory / f"frame-{index:06d}.npz"
                if item["path"] != path.name or path.is_symlink() or sha256(path) != item["sha256"]:
                    raise ValueError("Raw human frame checksum mismatch")
                with np.load(path, allow_pickle=False) as stored:
                    frame = {key: stored[key] for key in stored.files}
                validate_frame(frame)
                if previous is not None:
                    discrepancy = max(discrepancy, float(np.max(np.abs(previous - frame["state"]))))
                previous = frame["post_state"]
                row = dict(state=frame["state"].tolist(), actions=frame["action"].tolist(),
                           episode_index=episode, task_index=36, frame_index=index)
                for key in ("image", "wrist_image"):
                    encoded = BytesIO()
                    Image.fromarray(frame[key]).save(encoded, format="PNG")
                    row[key] = {"bytes": encoded.getvalue(), "path": None}
                rows.append(row)
                actions.append(frame["action"])
                tail.append(frame)
                tail = tail[-10:]
                paired = np.concatenate([frame["image"], frame["wrist_image"]], axis=1)
                video.append_data(paired)
                if index in keyframes:
                    frames_for_review[index] = Image.fromarray(paired).resize((256, 128))
        if (not all(bool(f["done"]) and np.abs(f["post_state"][6:]).sum() >= .06
                    and f["action"][6] < 0 for f in tail)
                or np.linalg.norm(tail[-1]["post_state"][:3] - tail[-1]["post_bowl_pos"]) < .10):
            raise ValueError("Success/open-gripper/retreat observation gate failed")
        path = output / f"episode-{episode:02d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        action_array = np.asarray(actions)
        records.append(dict(path=str(path.relative_to(ROOT)), frames=len(rows), episode=episode,
                            task_id=36, prompt=TASK, split="train", sha256=sha256(path)))
        audits.append(dict(source_directory=clip["source_directory"], metadata_sha256=sha256(meta_path),
                           source_frame_sha256=[r["sha256"] for r in meta["files"]],
                           source_scenario=meta["scenario_key"], initial_sim_state_sha256=meta.get("initial_sim_state_sha256"),
                           frames=len(rows), control_dt=.05, source_range=[0, len(rows)],
                           observation_gates_passed=True, reason=clip["reason"],
                           translation_norm_median=float(np.median(np.linalg.norm(action_array[:, :3], axis=1))),
                           zero_translation_fraction=float(np.mean(np.linalg.norm(action_array[:, :3], axis=1) < 1e-4)),
                           cached_post_vs_next_refreshed_state_max_abs=discrepancy))
        strip = Image.new("RGB", (1536, 158), "white")
        draw = ImageDraw.Draw(strip)
        draw.text((5, 4), f"Human {episode:02d}: {directory.name}; {len(rows)} controls, 20 Hz", fill="black")
        for column, (index, picture) in enumerate(frames_for_review.items()):
            strip.paste(picture, (column * 256, 30))
            draw.text((column * 256 + 5, 17), f"frame {index}", fill="black")
        sheets.append(strip)
        print(json.dumps(dict(episode=episode, frames=len(rows), approved=True)), flush=True)
    for page in range(2):
        sheet = Image.new("RGB", (1536, 790), "white")
        for row, strip in enumerate(sheets[page * 5:page * 5 + 5]):
            sheet.paste(strip, (0, row * 158))
        sheet.save(media / f"contact-sheet-{page}.jpg", quality=92)
    manifest = dict(schema="pi05-human-replay-v1", ready=True, root=str(ROOT), records=records,
                    human_episodes=len(records), human_frames=sum(r["frames"] for r in records),
                    selection_sha256=SELECTION_SHA, replay_manifest_sha256=sha256(replay),
                    sampling="25% human frames, 75% task-balanced 40-task public TRAIN replay",
                    temporal_policy="Original continuous 20Hz controls, unchanged actions and pre-action RGB/state; no speed scaling, stitching or dropped frames",
                    observation_caveat="Source recorder refreshes pre-action observables with sim.forward; returned post-state may be cached. Never use post_state as policy input.",
                    normalization="Use current multitask parent official LIBERO stats, not old course stats",
                    audits=audits)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (ROOT / "evidence/recovery-human28-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(episodes=len(records), frames=manifest["human_frames"], status="READY")))


if __name__ == "__main__":
    main()
