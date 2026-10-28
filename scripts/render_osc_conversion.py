"""Render measured reference/delta/tracking videos, without changing execution speed."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODES = ("reference", "delta", "tracking")
LABELS = ("原混合动作：参考执行", "直接差分转换：失败", "按参考位姿修正：通过")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.run / "protocol.json").read_text())
    title_font, small_font = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 16)
    validation = []
    for source in protocol["sources"]:
        folder = args.run / source
        results = [json.loads((folder / f"{m}-result.json").read_text()) for m in MODES]
        if [r["passed"] for r in results] != [True, False, True]:
            raise ValueError("Labels require the measured pass/fail ordering")
        rows = [json.loads((folder / f"{m}-steps.json").read_text()) for m in MODES]
        count = len(rows[0])
        if any(len(r) != count for r in rows):
            raise ValueError("Comparison requires equal control counts")
        selected = {0, int(np.argmax([r["position_error_m"] for r in rows[1]])), count - 1}
        sheets = []
        with ExitStack() as stack:
            readers = [stack.enter_context(imageio.get_reader(folder / f"{m}.mp4")) for m in MODES]
            streams = [iter(reader) for reader in readers]
            writer = stack.enter_context(imageio.get_writer(folder / "comparison.mp4", fps=20, macro_block_size=1))
            for index in range(count):
                canvas = Image.new("RGB", (1536, 368), "#171c24")
                draw = ImageDraw.Draw(canvas)
                for column, reader in enumerate(streams):
                    frame = next(reader)
                    if frame.shape != (256, 512, 3):
                        raise ValueError(f"Unexpected frame: {frame.shape}")
                    x = column * 512
                    row = rows[column][index]
                    draw.text((x + 10, 2), LABELS[column], font=title_font, fill="white")
                    draw.text((x + 10, 32), "场景相机                         腕部相机", font=small_font, fill="#bfccdf")
                    canvas.paste(Image.fromarray(frame), (x, 56))
                    draw.text((x + 10, 314), f"控制步 {row['index']} | 仿真 {index / 20:.2f}s | 误差 {row['position_error_m'] * 100:.2f}cm",
                              font=small_font, fill="white")
                    draw.text((x + 10, 339), f"目标成立 {row['native_goal']} | 手指接触 {row['finger_contact']}",
                              font=small_font, fill="#bfccdf")
                writer.append_data(np.asarray(canvas))
                if index in selected:
                    sheets.append(canvas)
            for reader in streams:
                try:
                    next(reader)
                except StopIteration:
                    continue
                raise ValueError("Extra unpaired video frames")
        sheet = Image.new("RGB", (1536, 368 * len(sheets)))
        for index, frame in enumerate(sheets):
            sheet.paste(frame, (0, index * 368))
        sheet.save(folder / "contact-sheet.jpg", quality=92)
        with imageio.get_reader(folder / "comparison.mp4") as reader:
            decoded = sum(1 for _ in reader)
        if decoded != count:
            raise ValueError("Comparison decode incomplete")
        validation.append(dict(source=source, raw_videos=3, raw_frames=count * 3,
                               comparison_frames=decoded, fps=20, speedup=1))
    (args.run / "video-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
