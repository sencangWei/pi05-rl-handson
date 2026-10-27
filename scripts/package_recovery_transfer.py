"""Package two different SFT weights under an otherwise identical recovery protocol."""
import argparse
import copy
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {"before": args.before, "after": args.after}
    identities = {k: json.loads((v / "identity.json").read_text()) for k, v in sources.items()}
    fields = ("manifest_sha256", "config_sha256", "split", "category", "success", "action_execution",
              "control_hz", "video_fps", "noise", "sources")
    for field in fields:
        if identities["before"][field] != identities["after"][field]:
            raise ValueError("Unmatched evaluation contract: " + field)
    servers = [identity.get("jax_server") for identity in identities.values()]
    if any(servers):
        if not all(servers):
            raise ValueError("Cannot compare JAX and Torch backends")
        for field in ("backend", "model_config", "image_size", "action_horizon",
                      "discrete_state_input", "seed_protocol", "norm_stats_sha256"):
            if servers[0][field] != servers[1][field]:
                raise ValueError("Unmatched JAX policy contract: " + field)
        if servers[0]["checkpoint"] == servers[1]["checkpoint"]:
            raise ValueError("Expected different parent and candidate checkpoints")
    else:
        configs = []
        for identity in identities.values():
            config = copy.deepcopy(identity["model_config"])
            config.pop("model_path")
            configs.append(config)
            if identity["checkpoint_sha256"] is not None:
                raise ValueError("Expected SFT without an additional PPO adapter")
        if configs[0] != configs[1]:
            raise ValueError("Non-weight model configuration changed")
    results = {k: {r["id"]: r for r in json.loads((v / "summary.json").read_text())}
               for k, v in sources.items()}
    if len(results["before"]) != 12 or results["before"].keys() != results["after"].keys():
        raise ValueError("Expected twelve matched development states")
    args.output.mkdir(parents=True, exist_ok=False)
    for label, source in sources.items():
        shutil.copytree(source, args.output / label)
    rows, totals = [], {label: Counter() for label in sources}
    for name, before in results["before"].items():
        row = {"id": name, "state_sha256": before["state_sha256"], "models": {}}
        command = ["ffmpeg", "-nostdin", "-v", "error"]
        filters, duration = [], 0
        for index, (label, source) in enumerate(sources.items()):
            result = results[label][name]
            if result["exception"] or result["state_sha256"] != row["state_sha256"]:
                raise ValueError("Unmatched state or rollout exception: " + name)
            trace = [json.loads(line) for line in (source / (name + ".jsonl")).read_text().splitlines()]
            if len(trace) != result["steps"]:
                raise ValueError("Incomplete control trace")
            if result["success"] and not (len(trace) >= 5 and trace[-1]["released_success"]
                    and all(t["native_success"] and not t["finger_contact"] for t in trace[-5:])):
                raise ValueError("Success lacks released-object evidence")
            longest = current = 0
            for sample in trace:
                current = current + 1 if sample["eef_motion_m"] < .0002 and not sample["native_success"] else 0
                longest = max(longest, current)
            low_motion = sum(sample["eef_motion_m"] < .0002 and not sample["native_success"]
                             for sample in trace)
            row["models"][label] = {**result, "longest_low_motion_controls": longest,
                                     "low_motion_controls": low_motion,
                                     "low_motion_fraction": low_motion / max(1, len(trace))}
            totals[label][result["category"] + "/success"] += int(result["success"])
            totals[label][result["category"] + "/trials"] += 1
            duration = max(duration, (result["steps"] + 1) / identities[label]["video_fps"])
            command += ["-i", str(source / result["video"])]
            filters.append(f"[{index}:v]crop=256:256:0:0,setpts=PTS-STARTPTS,"
                           "tpad=stop_mode=clone:stop_duration=60,"
                           "drawtext=fontcolor=white:box=1:boxcolor=black@0.7:fontsize=12:"
                           f"text='{label} {result['result']} {result['steps']} controls'[v{index}]")
        filters.append("[v0][v1]hstack=inputs=2[v]")
        destination = args.output / ("comparison-" + name + ".mp4")
        subprocess.run(command + ["-filter_complex", ";".join(filters), "-map", "[v]", "-t", str(duration),
                                  "-c:v", "libx264", "-threads", "2", "-crf", "23",
                                  "-pix_fmt", "yuv420p", str(destination)], check=True)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(destination), "-f", "null", "-"], check=True)
        row["comparison_video"] = destination.name
        rows.append(row)
    report = {"status": "PASS", "equal_protocol_fields": fields, "non_weight_config_equal": True,
              "weight_sha256": {k: v["base_sha256"] for k, v in identities.items()},
              "jax_servers": {k: v.get("jax_server") for k, v in identities.items()},
              "matched_states": len(rows), "categories": {k: dict(v) for k, v in totals.items()},
              "totals": {k: {"success": sum(r["success"] for r in v.values()), "trials": len(v)}
                         for k, v in results.items()},
              "video": "left before / right after; simulation time at 20 Hz; shorter result freezes",
              "rows": rows}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"]))


if __name__ == "__main__":
    main()
