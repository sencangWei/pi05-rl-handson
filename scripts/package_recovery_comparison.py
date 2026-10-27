"""Validate and package matched SFT / PPO recovery comparisons."""
import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("sft", "old30", "new300"):
        parser.add_argument("--" + label, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs=3, default=("sft", "old30", "new300"),
                        help="Names for the SFT, middle and right models in artifacts and videos")
    args = parser.parse_args()
    labels = tuple(args.labels)
    if len(set(labels)) != 3 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", label) for label in labels):
        parser.error("labels must be three unique names containing only letters, digits, _ or -")
    sources = dict(zip(labels, (args.sft, args.old30, args.new300), strict=True))
    baseline = labels[0]
    identities = {label: json.loads((path / "identity.json").read_text()) for label, path in sources.items()}
    fields = ("base_sha256", "manifest_sha256", "config_sha256", "model_config", "success",
              "action_execution", "control_hz", "video_fps", "noise", "sources")
    for label in labels:
        assert all(identities[label][key] == identities[baseline][key] for key in fields)
    results = {label: {r["id"]: r for r in json.loads((path / "summary.json").read_text())}
               for label, path in sources.items()}
    assert len(results[baseline]) == 12
    assert all(results[label].keys() == results[baseline].keys() for label in labels)
    args.output.mkdir(parents=True, exist_ok=False)
    for label in labels:
        shutil.copytree(sources[label], args.output / label)
    rows = []
    totals = {label: Counter() for label in labels}
    for name in results[baseline]:
        row = {"id": name, "category": results[baseline][name]["category"],
               "state_sha256": results[baseline][name]["state_sha256"], "models": {}}
        duration = 0
        filters = []
        command = ["ffmpeg", "-nostdin", "-v", "error"]
        for index, label in enumerate(labels):
            result = results[label][name]
            assert not result["exception"]
            assert result["state_sha256"] == row["state_sha256"]
            trace = [json.loads(line) for line in (sources[label] / (name + ".jsonl")).read_text().splitlines()]
            assert len(trace) == result["steps"]
            if result["success"]:
                assert len(trace) >= 5 and trace[-1]["released_success"]
                assert all(sample["native_success"] and not sample["finger_contact"]
                           for sample in trace[-5:])
            longest = current = 0
            for sample in trace:
                current = current + 1 if sample["eef_motion_m"] < .0002 and not sample["native_success"] else 0
                longest = max(longest, current)
            row["models"][label] = {**result, "longest_low_motion_controls": longest}
            totals[label][row["category"] + "/success"] += int(result["success"])
            totals[label][row["category"] + "/trials"] += 1
            fps = identities[label]["video_fps"]
            duration = max(duration, (result["steps"] + 1) / fps)
            command += ["-i", str(sources[label] / result["video"])]
            filters.append(f"[{index}:v]crop=256:256:0:0,setpts=PTS-STARTPTS,"
                           "tpad=stop_mode=clone:stop_duration=60,"
                           "drawtext=fontcolor=white:box=1:boxcolor=black@0.7:fontsize=13:"
                           f"text='{label} {result['result']} {result['steps']} controls'[v{index}]")
        filters.append("[v0][v1][v2]hstack=inputs=3[v]")
        destination = args.output / ("comparison-" + name + ".mp4")
        subprocess.run(command + ["-filter_complex", ";".join(filters), "-map", "[v]", "-t", str(duration),
                                  "-c:v", "libx264", "-threads", "2", "-crf", "23",
                                  "-pix_fmt", "yuv420p", str(destination)], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(destination),
                        "-f", "null", "-"], check=True)
        row["comparison_video"] = destination.name
        rows.append(row)
    report = {"identity_validation": {"status": "PASS", "equal_fields": fields,
                                      "matched_states": len(rows)},
              "categories": {label: dict(counts) for label, counts in totals.items()},
              "totals": {label: {"success": sum(r["success"] for r in results[label].values()),
                                 "trials": len(results[label])} for label in labels},
              "video": (f"left {labels[0]} / middle {labels[1]} / right {labels[2]}; "
                        "simulation time, shorter result freezes"),
              "rows": rows}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"]))


if __name__ == "__main__":
    main()
