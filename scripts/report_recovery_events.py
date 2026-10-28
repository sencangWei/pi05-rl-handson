"""Build paired event metrics and evidence sheets from frozen diagnostic outputs."""
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recovery_events import paired_divergence, summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/20261028-recovery-events"
REPORT = ROOT / "reports/recovery-events-20261028"
HUMAN = OUT / "human-static-v3"


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "evidence/recovery-events-28-plan.json").read_text())
    manifest = json.loads((ROOT / plan["manifest"]).read_text())
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    sheet = Image.new("RGB", (2048, 4 * 284), "#17202b")
    draw = ImageDraw.Draw(sheet)
    results = []
    # Selected frames illustrate observed divergence and a later transport/failure stage.
    late = {40: 150, 41: 155, 42: 250, 43: 180}
    for index, initial in enumerate(range(40, 44)):
        entry = next(e for e in manifest["entries"] if e["id"] == f"dev-placement_failure-init{initial}")
        traces = {k: load(OUT / f"replay/{k}-{initial}.jsonl") for k in ("parent", "candidate")}
        result = dict(id=initial, divergence=paired_divergence(traces["parent"], traces["candidate"]))
        for label, rows in traces.items():
            result[label] = summarize(rows, entry["bowl_pos"][2])
            table_contacts = [(v["normal_force_N"], r["step"], k)
                              for r in rows for k, v in r["contacts"].items()
                              if "table_collision" in k and ("robot0_" in k or "gripper0_" in k)]
            if table_contacts:
                force, control, geoms = max(table_contacts)
                result[label]["peak_robot_table_contact"] = dict(normal_force_N=force, control=control, geoms=geoms)
            result[label]["last50"] = dict(
                torque_saturated_controls=sum(r["torque_saturated_substeps"] > 0 for r in rows[-50:]),
                min_translation_jacobian_sigma=min(r["translation_jacobian_sigma_min"] for r in rows[-50:]),
                min_joint_margin_rad=min(r["min_joint_margin_rad"] for r in rows[-50:]),
                median_motion_mm=float(np.median([1000 * r["eef_motion_m"] for r in rows[-50:]])),
            )
        frame_records = []
        first = result["divergence"]["eef_pos"] or 1
        for column, (step, label) in enumerate([(first, "parent"), (first, "candidate"),
                                                (late[initial], "parent"), (late[initial], "candidate")]):
            video_path = OUT / f"replay/{label}-{initial}.mp4"
            reader = imageio.get_reader(video_path)
            try:
                frame = reader.get_data(step - 1)  # Probe video has no initial frame.
            finally:
                reader.close()
            x, y = column * 512, index * 284
            sheet.paste(Image.fromarray(frame), (x, y + 28))
            draw.text((x + 6, y + 5), f"init{initial} | {label} | control {step}", font=font, fill="white")
            frame_records.append(dict(label=label, step=step, video=str(video_path.relative_to(ROOT)),
                                      video_frame_index=step - 1))
        result["illustrated_frames"] = frame_records
        results.append(result)
    sheet.save(REPORT / "paired-events.png")
    human = json.loads((HUMAN / "summary.json").read_text())
    human_sheet = Image.new("RGB", (2560, 2 * 284), "#17202b")
    hd = ImageDraw.Draw(human_sheet)
    human_frames = []
    for row, (source, steps) in enumerate([
            ("human_1793121358375746008", [84, 100, 300, 528, 533]),
            ("human_1793122696228231033", [234, 250, 450, 606, 613])]):
        timeline = load(HUMAN / f"{source}.jsonl")
        for column, index in enumerate(steps):
            item = timeline[index]
            with np.load(ROOT / "corrections" / source / item["source_frame"], allow_pickle=False) as z:
                frame = np.concatenate([z["image"], z["wrist_image"]], axis=1)
            x, y = column * 512, row * 284
            human_sheet.paste(Image.fromarray(frame), (x, y + 28))
            hd.text((x + 6, y + 5), f"source {source[-4:]} | t{index} | {item['controller']}",
                    font=font, fill="white")
            human_frames.append(dict(source=source, timeline_index=index, source_frame=item["source_frame"],
                                     controller=item["controller"], image_timing="before control"))
    human_sheet.save(REPORT / "human-control-boundaries.png")
    output = dict(plan_sha256=hashlib.sha256((ROOT / "evidence/recovery-events-28-plan.json").read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  event_helper_sha256=hashlib.sha256((ROOT / "scripts/recovery_events.py").read_bytes()).hexdigest(),
                  run_indices="zero-based inclusive; divergence and illustrated model controls one-based",
                  pairs=results, human_illustrations=human_frames,
                  human_static_summary_sha256=hashlib.sha256((HUMAN / "summary.json").read_bytes()).hexdigest(),
                  total_human_controls=sum(s["controls"] for s in human["sources"]),
                  human_lifted_grasp_by_controller={k: sum(s["lifted_grasp_by_controller"][k] for s in human["sources"])
                                                    for k in ("OSC_POSE", "JOINT_POSITION")})
    (OUT / "paired-events.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
