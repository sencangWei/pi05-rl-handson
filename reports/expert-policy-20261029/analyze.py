"""Audit four policy traces and render synchronized parent/candidate comparisons."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_expert_snapshots import verify_inputs  # noqa: E402

OUT = ROOT / "artifacts/20261029-expert-policy"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    verify_inputs(protocol)
    summary = json.loads((OUT / "summary.json").read_text())
    assert len(summary) == 4
    results = {}
    for result in summary:
        key = (result["label"], result["source"])
        assert key not in results
        path = OUT.joinpath(*key)
        assert result == json.loads((path / "result.json").read_text())
        rows = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
        assert len(rows) == result["steps"] and result["exception"] is None
        states = np.load(path / "states.npz", allow_pickle=False)["states"]
        assert len(states) == len(rows) + 1 and np.isfinite(states).all()
        np.testing.assert_allclose(np.diff(states[:, 0]), .05, rtol=0, atol=1e-9)
        streak, success, native_first = 0, None, None
        longest, run = 0, 0
        for i, row in enumerate(rows, 1):
            assert row["step"] == row["index"] == i
            assert row["inference_call"] == (i - 1) // 5 + 1
            assert row["noise_seed"] == 42 + (i - 1) // 5
            assert np.isfinite(row["action"]).all() and len(row["action"]) == 7
            streak = streak + 1 if row["native_goal"] and not row["finger_contact"] else 0
            assert row["release_streak"] == streak
            if row["native_goal"] and native_first is None:
                native_first = i
            if streak >= 5 and success is None:
                success = i
            run = run + 1 if row["eef_motion_m"] < .0002 and not row["native_goal"] else 0
            longest = max(longest, run)
        assert success == result["success_step"] and bool(success) == result["success"]
        assert native_first == result["native_first_step"]
        assert result["success_by_300"] == (success is not None and success <= 300)
        assert all(result["initial_checks"][k] for k in
                   ("physics_exact", "observation_state_exact", "both_rgb_exact"))
        timings = result["inference_seconds"]
        metrics = dict(label=key[0], source=key[1], steps=len(rows), success=bool(success),
                       success_step=success, success_by_300=result["success_by_300"],
                       native_first_step=native_first, simulation_seconds=len(rows) / 20,
                       wall_seconds=result["wall_seconds"], native_controls=sum(r["native_goal"] for r in rows),
                       contact_controls=sum(r["finger_contact"] for r in rows),
                       grasp_controls=sum(r["grasp"] for r in rows), longest_low_motion_controls=longest,
                       final_bowl_plate_xy_m=rows[-1]["bowl_plate_xy_m"],
                       first_inference_seconds=timings[0], warm_inference_median_seconds=float(np.median(timings[1:])),
                       initial_measure=result["initial_measure"], blocks=[])
        for start in range(0, len(rows), 100):
            block = rows[start:start + 100]
            metrics["blocks"].append(dict(start=start + 1, end=start + len(block),
                eef_motion_mean_mm=float(np.mean([r["eef_motion_m"] * 1000 for r in block])),
                contact_controls=sum(r["finger_contact"] for r in block),
                native_controls=sum(r["native_goal"] for r in block),
                final_bowl_plate_xy_m=block[-1]["bowl_plate_xy_m"]))
        results[key] = metrics

    comparisons = []
    sheet = Image.new("RGB", (1024, 2240), "#202020")
    draw = ImageDraw.Draw(sheet)
    for si, source in enumerate(protocol["sources"]):
        readers = [imageio.get_reader(OUT / label / source / "policy.mp4")
                   for label in ("parent", "candidate")]
        lengths = [results[(label, source)]["steps"] + 1 for label in ("parent", "candidate")]
        assert all(reader.get_meta_data()["fps"] == 20 for reader in readers)
        target = OUT / f"comparison-{source}.mp4"
        with imageio.get_writer(target, fps=20, codec="libx264") as writer:
            for step in range(max(lengths)):
                frame = Image.new("RGB", (1024, 288), "#202020")
                text = ImageDraw.Draw(frame)
                for side, (label, reader, length) in enumerate(zip(("parent", "candidate"), readers, lengths)):
                    index = min(step, length - 1)
                    frame.paste(Image.fromarray(reader.get_data(index)), (side * 512, 32))
                    status = "SUCCESS" if results[(label, source)]["success"] else "TIMEOUT"
                    suffix = f" FROZEN {status}" if step >= length - 1 else ""
                    text.text((side * 512 + 8, 8), f"{label}  control={index} / {length - 1}{suffix}", fill="white")
                writer.append_data(np.asarray(frame))
        comparisons.append(str(target.relative_to(ROOT)))
        for side, (label, reader, length) in enumerate(zip(("parent", "candidate"), readers, lengths)):
            for fi, control in enumerate((0, 100, 300, length - 1)):
                control = min(control, length - 1)
                sheet.paste(Image.fromarray(reader.get_data(control)), (side * 512, (si * 4 + fi) * 280 + 24))
                draw.text((side * 512 + 4, (si * 4 + fi) * 280 + 4),
                          f"{source[-6:]} {label} control={control}", fill="white")
            reader.close()
    sheet.save(Path(__file__).with_name("contact-sheet.jpg"))
    videos = []
    for path in sorted(OUT.rglob("*.mp4")):
        with imageio.get_reader(path) as reader:
            frames = sum(1 for _ in reader)
        expected = 601 if path.parent == OUT else results[(path.parent.parent.name, path.parent.name)]["steps"] + 1
        assert frames == expected, path
        videos.append(dict(path=str(path.relative_to(ROOT)), frames=frames,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(status="COMPLETE", split=protocol["split"], heldout=False,
                  protocol_sha256=hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest(),
                  trials=list(results.values()), comparisons=comparisons, videos_decoded=videos,
                  trace_physics_success_audit=True, model_promotion=False, training=False,
                  runner_sha256=hashlib.sha256(Path(__file__).with_name("run.py").read_bytes()).hexdigest(),
                  systemd=subprocess.check_output(["systemctl", "--user", "show", "pi05-expert-policy29-run.service",
                      "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus"], text=True))
    (ROOT / "evidence/expert-policy29-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], trials=len(results), videos=len(videos),
                         decoded_frames=sum(v["frames"] for v in videos), systemd=report["systemd"])))


if __name__ == "__main__":
    main()
