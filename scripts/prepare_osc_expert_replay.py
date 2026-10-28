"""Append approved, physically verified OSC expert pairs to all old manual clips."""
import json
from pathlib import Path

from export_sft_for_ppo import sha256
from record_osc_pairs import validate_dataset
from recovery_replay_dataset import BankHumanClips

ROOT = Path(__file__).resolve().parents[1]


def recovery_rows(rows, original_controls, episode, prompt):
    """Discard diagnostic holds; require full, unchanged ten-control windows."""
    if original_controls < 10 or len(rows) != original_controls + 20:
        raise ValueError("Expected recovery followed by twenty diagnostic holds")
    for index, row in enumerate(rows):
        if (row["frame_index"] != index or row["timestamp"] != index * .05
                or row["diagnostic_hold"] != (index >= original_controls)
                or row["controller"] != "OSC_POSE" or row["task_index"] != 8
                or row["prompt"] != prompt):
            raise ValueError("OSC temporal/task/hold contract changed")
    return [dict(row, episode_index=episode) for row in rows[:original_controls]]


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    old_path = ROOT / "evidence/manual-recovery28-manifest.json"
    old = BankHumanClips(old_path)
    review_root = ROOT / "reports/osc-pairs-20261029/v1"
    review_path = review_root / "manifest.json"
    review = json.loads(review_path.read_text())
    approval_path = review_root / "human-review.json"
    approval = json.loads(approval_path.read_text())
    approved = {r["source"]: r for r in approval["sources"]}
    if (review["human_review"] != "approved" or approval["status"] != "approved"
            or approval["actor"] != "user" or not approval["exclude_diagnostic_hold_anchors"]
            or len(review["records"]) != 2
            or set(approved) != {r["source"] for r in review["records"]}):
        raise ValueError("Both expert clips require explicit user approval")
    output = ROOT / "data/osc-expert-replay29-v1"
    output.mkdir(parents=True, exist_ok=False)
    records = list(old.records)
    audits = []
    for item in review["records"]:
        source = item["source"]
        directory = review_root / source
        if (not directory.resolve().is_relative_to(review_root.resolve())
                or not item["passed"] or item["human_review"] != "approved"
                or not item["expert_reference_actions"] or item["vla_inference"]):
            raise ValueError("Invalid expert provenance")
        for name, identity in item["files"].items():
            if sha256(directory / name) != identity["sha256"]:
                raise ValueError("Reviewed source changed")
        if sha256(directory / "review-labeled.mp4") != approved[source]["labeled_sha256"]:
            raise ValueError("Approved video changed")
        audit = validate_dataset(directory / "pairs.parquet", directory / "physics.npz",
                                 directory / "pair-audit.json")
        previous = [r for r in old.records if r["source"] == source]
        if not previous or any(r["entry"] != previous[0]["entry"] for r in previous):
            raise ValueError("Missing training-bank provenance")
        rows = recovery_rows(pq.read_table(directory / "pairs.parquet").to_pylist(),
                             item["original_controls"], len(records),
                             "pick up the black bowl next to the plate and place it on the plate")
        path = output / (source + ".parquet")
        pq.write_table(pa.Table.from_pylist(rows), path)
        records.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path),
            frames=len(rows), sample_frames=len(rows) - 9, terminal=False,
            controller="OSC_POSE", episode=len(records), task_id=8,
            prompt=previous[0]["prompt"], split="train", entry=previous[0]["entry"],
            source=source, provenance="expert-derived OSC execution, actual pre-action observations",
            recorded_prompt=rows[0]["prompt"],
            original_pairs_sha256=item["files"]["pairs.parquet"]["sha256"]))
        audits.append(dict(source=source, validation=audit, kept_controls=len(rows),
                           eligible_anchors=len(rows) - 9, excluded_hold_controls=20,
                           excluded_short_window_anchors=9))
    manifest = dict(schema=BankHumanClips.schema, ready=True, root=str(ROOT), records=records,
        bank_manifest_sha256=old.manifest["bank_manifest_sha256"],
        replay_manifest_sha256=old.manifest["replay_manifest_sha256"],
        previous_manifest_sha256=sha256(old_path), expert_review_manifest_sha256=sha256(review_path),
        human_review_sha256=sha256(approval_path), audits=audits,
        old_records_unchanged=records[:len(old.records)] == old.records,
        old_eligible_anchors=len(old), expert_eligible_anchors=sum(a["eligible_anchors"] for a in audits),
        temporal_policy="20Hz unchanged labels. New expert clips exclude all diagnostic holds and last nine incomplete-window anchors; no padding/resampling. Old clips unchanged.",
        provenance="Old human OSC plus two expert-derived executions of human recovery; not autonomous VLA demonstrations.",
        prompt_policy="Validate recorded full bowl-to-plate instruction; use unchanged bank-compatible canonical training prompt 'put the bowl on the plate'. Physical bank/task8 identity unchanged.",
        sampling="75% task-balanced public replay / 25% combined recovery; recovery uniformly over eligible anchors; report seeded actual counts",
        normalization="Unchanged official LIBERO normalization, matching own 40-task parent")
    target = ROOT / "evidence/osc-expert-replay29-manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest.json").write_bytes(target.read_bytes())
    loaded = BankHumanClips(target)
    assert len(loaded) == len(old) + manifest["expert_eligible_anchors"]
    print(json.dumps(dict(ready=True, records=len(records), old_anchors=len(old),
                         expert_anchors=manifest["expert_eligible_anchors"], total_anchors=len(loaded))))


if __name__ == "__main__":
    main()
