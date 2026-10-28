"""Stream three-column snapshot validation videos at the original 20 Hz."""
import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output / "comparisons"
    out.mkdir(exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
    rows = []
    for source in sorted(args.output.glob("human_*")):
        for start, stop in json.loads((source / "windows.json").read_text()):
            name = f"{source.name[-4:]}-{start}-{stop}"
            dest = out / f"{name}.mp4"
            with ExitStack() as stack:
                readers = [stack.enter_context(imageio.get_reader(path)) for path in (
                    source / "reference.mp4", source / f"legacy-{start}-{stop}.mp4",
                    source / f"full-{start}-{stop}.mp4")]
                reference = itertools.islice(iter(readers[0]), start, stop)
                writer = stack.enter_context(imageio.get_writer(dest, fps=20))
                count = 0
                for i, frames in enumerate(zip(reference, readers[1], readers[2])):
                    canvas = Image.new("RGB", (1536, 288), "#181818")
                    draw = ImageDraw.Draw(canvas)
                    labels = ("新参考执行", "旧79维状态恢复", "完整状态恢复")
                    for col, (frame, label) in enumerate(zip(frames, labels)):
                        canvas.paste(Image.fromarray(frame), (col * 512, 32))
                        draw.text((col * 512 + 8, 3), f"{label}  控制步{start + i}", font=font, fill="white")
                    writer.append_data(np.asarray(canvas))
                    if i == 10:
                        canvas.save(out / f"{name}.png")
                    count += 1
                if count != stop - start:
                    raise ValueError(f"Truncated comparison: {name}: {count}")
            rows.append(dict(video=dest.name, source=source.name, start=start, stop=stop,
                             frames=count, fps=20, columns=list(labels),
                             sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    (out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(videos=len(rows), frames=sum(r["frames"] for r in rows))))


if __name__ == "__main__":
    main()
