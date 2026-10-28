"""Render a Chinese review page from newly recorded pre-action recovery pairs."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq

from record_osc_pairs import validate_dataset
from replay_mixed_correction import sha, write_json

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["ready"] or manifest["training"] or manifest["human_review"] != "pending":
        raise ValueError("Review rendering requires untrained pending candidates")
    font = ImageFont.truetype(FONT, 20)
    small = ImageFont.truetype(FONT, 16)
    sections, validations = [], []
    for episode, result in enumerate(manifest["records"]):
        source = result["source"]
        folder = args.run / source
        for name, info in result["files"].items():
            if sha(folder / name) != info["sha256"]:
                raise ValueError(f"Changed recording: {name}")
        pair_check = validate_dataset(folder / "pairs.parquet", folder / "physics.npz", folder / "pair-audit.json")
        rows = pq.read_table(folder / "pairs.parquet").to_pylist()
        audit = json.loads((folder / "pair-audit.json").read_text())
        selected = set(np.linspace(0, len(rows) - 1, 6, dtype=int).tolist())
        shots = []
        with imageio.get_reader(folder / "review.mp4") as reader, imageio.get_writer(
                folder / "review-labeled.mp4", fps=20, macro_block_size=1) as writer:
            count = 0
            fps = reader.get_meta_data()["fps"]
            for i, frame in enumerate(reader):
                if i > len(rows) or frame.shape != (256, 512, 3):
                    raise ValueError("Unexpected video frame count or shape")
                canvas = Image.new("RGB", (512, 368), "#17202d")
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 2), f"第{episode + 1}段 · 实际重采 · 待你审核", font=font, fill="white")
                draw.text((10, 32), "左：场景相机                 右：腕部相机", font=small, fill="#c9d3e1")
                canvas.paste(Image.fromarray(frame), (0, 56))
                if i < len(rows):
                    row = rows[i]
                    stage = "末尾稳定性检查" if row["diagnostic_hold"] else (
                        "关节恢复转换为OSC" if row["original_controller"] == "JOINT_POSITION" else "保留原OSC动作")
                    draw.text((10, 314), f"动作前 {i}/{len(rows)} · {i / 20:.2f}s · {stage}", font=small, fill="white")
                    post = audit[i]["post"]
                    draw.text((10, 340), f"执行本步后：目标 {post['native_goal']} · 手指接触 {post['finger_contact']}",
                              font=small, fill="#c9d3e1")
                    # Compare lossless saved pairs as well as fully decoding lossy review video.
                    for field in ("image", "wrist_image"):
                        Image.open(BytesIO(row[field]["bytes"])).verify()
                else:
                    draw.text((10, 314), f"最后动作后 · {len(rows) / 20:.2f}s · 放稳检查结束", font=small, fill="white")
                    draw.text((10, 340), "参考动作执行；尚未训练模型", font=small, fill="#c9d3e1")
                writer.append_data(np.asarray(canvas))
                if i in selected:
                    shots.append(canvas)
                count += 1
        if count != len(rows) + 1 or fps != 20:
            raise ValueError("Video temporal contract failed")
        sheet = Image.new("RGB", (1536, 736), "#17202d")
        for i, shot in enumerate(shots):
            sheet.paste(shot, ((i % 3) * 512, (i // 3) * 368))
        sheet.save(folder / "contact-sheet.jpg", quality=92)
        with imageio.get_reader(folder / "review-labeled.mp4") as reader:
            decoded = sum(1 for _ in reader)
        if decoded != count:
            raise ValueError("Labeled video decode incomplete")
        validations.append(dict(source=source, **pair_check, raw_video_frames=count,
                                labeled_video_frames=decoded, fps=fps, speedup=1,
                                duration_s=count / 20, labeled_sha256=sha(folder / "review-labeled.mp4")))
        sections.append(f'''<section><h2>第{episode + 1}段：原录制尾号 {source[-4:]}</h2>
<p>{result['original_controls']}步恢复操作＋20步稳定性检查；全程{count / 20:.2f}秒，原速20fps。</p>
<video controls preload="metadata" src="{source}/review-labeled.mp4"></video>
<p><a href="{source}/review-labeled.mp4">打开录像</a> · <a href="{source}/contact-sheet.jpg">查看六张关键帧</a></p>
<img src="{source}/contact-sheet.jpg" alt="从开始到完成的六张真实执行关键帧">
</section>''')
    write_json(args.run / "review-validation.json", validations)
    (args.run / "index.html").write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>恢复片段重采审核</title>
<style>body{max-width:1100px;margin:24px auto;padding:0 20px;font:18px/1.7 sans-serif;color:#17202d;background:#f1f4f8}
section{background:white;padding:20px;margin:22px 0;border-radius:12px}video{width:100%;max-width:1024px;background:#17202d}
img{width:100%;height:auto}a{color:#175bb6}.note{background:#fff1ca;padding:16px;border-radius:8px}</style>
<h1>两段恢复片段：实际重采审核</h1><p class="note"><b>待你确认，尚未训练。</b>
这是参考动作在新仿真中的执行，不是VLA自主恢复。每个训练候选动作都配对了本次执行前的场景相机、腕部相机和状态。
录像按20fps原速播放，没有加速。末尾20步是放稳检查，保留但不建议作为训练锚点。</p>
<p>请重点看：搬运是否合理、能否正确放到盘子、是否松爪、有无多余或不自然的动作。
这些片段从恢复中途开始，不代表从初始场景独立抓取。全部965步与上轮通过轨迹的物理状态、执行后像素和目标/接触标志精确一致。</p>
''' + "\n".join(sections) + "</html>\n", encoding="utf-8")
    print(json.dumps(validations), flush=True)


if __name__ == "__main__":
    main()
