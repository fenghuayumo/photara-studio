"""Render the filter control and rate/distortion plots from POPSpa review artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path)
    parser.add_argument("physical", type=Path)
    parser.add_argument("mip_audit", type=Path)
    parser.add_argument("classic_audit", type=Path)
    args = parser.parse_args()
    font_path = font_manager.findfont("DejaVu Sans")
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 17)
    tile_w, tile_h = 450, 350
    canvas = Image.new("RGB", (tile_w * 2, tile_h * 2 + 40), "#f1f3f5")
    draw = ImageDraw.Draw(canvas)
    for column, (label, folder) in enumerate((("Classic", args.classic_audit), ("Mip2D", args.mip_audit))):
        for row, degree in enumerate((3, 0)):
            src = Image.open(folder / f"sh{degree}-368.png")
            x, y = column * tile_w, row * tile_h
            draw.text((x + 12, y + 7), f"{label} / SH{degree}", fill="#19212d", font=font)
            canvas.paste(src.crop((298, 40, 555, 340)), (x + 10, y + 40))
            canvas.paste(src.crop((0, 650, 175, 890)), (x + 270, y + 98))
    draw.text((12, tile_h * 2 + 9), "Identical Gaussian parameters. Only filter mode and active SH degree differ.", fill="#19212d", font=small)
    canvas.save(args.current / "filter-audit.png")

    base = read(args.current / "baseline/result.json")
    baseline = {row["view"]: row for row in base["metrics"]}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for root, mode, label, color in (
        (args.physical, "unpruned", "No prune + 10k (physical s3)", "#777777"),
        (args.current, "s3pair20", "POPSpa 20% (normalized s3)", "#1368c4"),
        (args.current, "codecpair", "POPSpa 20% + SOG-XT", "#df7a1d"),
    ):
        rows = read(root / mode / "result.json")["metrics"]
        changes = sorted(row["psnr"] - baseline[row["view"]]["psnr"] for row in rows)
        left.plot([100 * i / (len(changes) - 1) for i in range(len(changes))], changes, label=label, color=color)
    left.axhline(0, color="black", lw=0.8, ls="--")
    left.set(xlabel="Views sorted by PSNR change (%)", ylabel="PSNR change vs original Mip2D PLY (dB)", title="All 739 views: averages hide a regression tail")
    left.legend(fontsize=8)
    left.grid(alpha=0.2)

    for root, mode, label, color in (
        (args.current, "codecbase", "Original + SOG-XT", "#777777"),
        (args.physical, "codec20", "20%, physical s3", "#a1b6cc"),
        (args.physical, "codec40", "40%, physical s3", "#85ad85"),
        (args.current, "codecpair", "20%, normalized s3", "#1368c4"),
    ):
        data = read(root / mode / "result.json")
        rows = [row for row in data["metrics"] if row["holdout"]]
        psnr = sum(row["psnr"] for row in rows) / len(rows)
        mb = data["encoded_bytes"] / 1e6
        right.scatter(mb, psnr, s=65, color=color)
        offset = (5, -16) if mode == "codec20" else (5, 8)
        right.annotate(label, (mb, psnr), textcoords="offset points", xytext=offset, fontsize=8)
    right.set(xscale="log", xlabel="SOG-XT file size (MB)", ylabel="Compaction-holdout PSNR (dB)", title="Same decode/render settings; 93 holdout views")
    right.margins(x=0.3, y=0.2)
    right.grid(alpha=0.2)
    fig.savefig(args.current / "quality-and-size.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
