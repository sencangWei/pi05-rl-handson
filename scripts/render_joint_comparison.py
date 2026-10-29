"""Validate completed joint diagnostics and render time-aligned reference comparisons."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((args.output / "protocol.json").read_text())
    results = json.loads((args.output / "summary.json").read_text())
    assert len(results) == 4 and all(not r["training_ready"] for r in results)
    receipts = []
    for source in protocol["sources"]:
        selected = [r for r in results if r["source"] == source]
        assert [r["mode"] for r in selected] == protocol["modes"]
        count = selected[0]["controls"]
        paths = [root / "reports/osc-conversion-20261029/v1" / source / "reference.mp4"]
        for result in selected:
            folder = args.output / source / result["mode"]
            rows = json.loads((folder / "steps.json").read_text())
            assert len(rows) == count == result["controls"]
            with np.load(folder / "trajectory.npz", allow_pickle=False) as z:
                assert z["actions"].shape == (count, 8)
                assert z["before_joint_state"].shape == (count, 9)
                assert z["after_joint_state"].shape == (count, 9)
                assert all(len(z[k]) == count and np.isfinite(z[k]).all() for k in z.files)
                np.testing.assert_array_equal(z["actions"], [r["action"] for r in rows])
            paths.append(folder / "execution.mp4")
        videos = [[frame for frame in imageio.get_reader(path)] for path in paths]
        assert [len(v) for v in videos] == [count, count + 1, count + 1]
        # Reference contains post-control images only; candidates also have an initial frame.
        videos = [videos[0], videos[1][1:], videos[2][1:]]
        preview = Image.new("RGB", (1536, 288 * 4), "black")
        checkpoints = np.linspace(0, count - 1, 4, dtype=int).tolist()
        destination = args.output / source / "comparison.mp4"
        with imageio.get_writer(destination, fps=20) as writer:
            for j, frames in enumerate(zip(*videos, strict=True)):
                canvas = Image.new("RGB", (1536, 288), "black")
                draw = ImageDraw.Draw(canvas)
                for i, (frame, title) in enumerate(zip(frames, ("Mixed expert reference", *protocol["modes"]), strict=True)):
                    assert frame.shape == (256, 512, 3)
                    canvas.paste(Image.fromarray(frame), (512 * i, 32))
                    draw.text((512 * i + 8, 9), f"{title} | control {j + 1}/{count}", fill="white")
                writer.append_data(np.asarray(canvas))
                if j in checkpoints:
                    preview.paste(canvas, (0, 288 * checkpoints.index(j)))
        preview.save(args.output / source / "comparison.png")
        actual = sum(1 for _ in imageio.get_reader(destination))
        assert actual == count
        receipts.append(dict(source=source, original_video_frames=[count, count + 1, count + 1],
                             comparison_frames=actual, comparison_fps=20, aligned="post-control frames",
                             finite_trajectory=True, action_rows_exact=True))
    (args.output / "artifact-validation.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(receipts))


if __name__ == "__main__":
    main()
