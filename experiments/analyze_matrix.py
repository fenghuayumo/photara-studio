"""Summarize a run_ab.py matrix: canonical metrics plus per-view photometric headroom.

`aligned_psnr` fits one gain/bias per held-out view against the photograph, so it
measures the reconstruction that remains once acquisition exposure is removed.
Both numbers come from the same saved renders; the aligned value is an upper
bound on what a per-view photometric model could absorb.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr(target, render, mask):
    return -10 * np.log10(np.mean((render - target)[mask] ** 2) + 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    rows = []
    for arm in sorted(p for p in args.matrix.iterdir() if p.is_dir()):
        metrics_path = arm / "evaluation" / "metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open() as stream:
            entries = list(csv.DictReader(stream))
        canonical = {key: float(np.mean([float(e[key]) for e in entries]))
                     for key in ("psnr", "ssim", "full_psnr", "full_ssim")}
        base, aligned, gains = [], [], []
        for target_path in sorted(
                (arm / "evaluation").glob("target_*.png"),
                key=lambda p: int(p.stem.split("_")[1])):
            index = target_path.stem.split("_")[1]
            render_path = arm / "evaluation" / f"view_{index}.png"
            if not render_path.exists():
                continue
            target, render = load(target_path), load(render_path)
            mask = target.max(axis=2) > 0.02
            base.append(psnr(target, render, mask))
            flat_t, flat_r = target[mask], render[mask]
            affine = np.stack([
                np.linalg.lstsq(
                    np.stack([flat_r[:, c], np.ones_like(flat_r[:, c])], axis=1),
                    flat_t[:, c], rcond=None)[0]
                for c in range(3)])
            corrected = np.clip(render * affine[:, 0] + affine[:, 1], 0.0, 1.0)
            aligned.append(psnr(target, corrected, mask))
            gains.append(affine[:, 0])
        gains = np.asarray(gains)
        rows.append({
            "arm": arm.name,
            "psnr": canonical["psnr"],
            "ssim": canonical["ssim"],
            "png_psnr": float(np.mean(base)),
            "aligned_psnr": float(np.mean(aligned)),
            "gain_deviation": float(np.abs(gains - 1.0).mean()),
            "maximum_gain_deviation": float(np.abs(gains - 1.0).max()),
        })
    header = ("arm", "psnr", "ssim", "png_psnr", "aligned_psnr",
              "gain_deviation", "maximum_gain_deviation")
    print(" ".join(f"{h:>16}" for h in header))
    for row in rows:
        print(" ".join(
            f"{row[h]:>16.4f}" if isinstance(row[h], float) else f"{row[h]:>16}"
            for h in header))
    if rows:
        reference = rows[0]
        print("\ndelta vs", reference["arm"])
        for row in rows[1:]:
            print(f"  {row['arm']:>16} psnr={row['psnr'] - reference['psnr']:+.4f} "
                  f"ssim={row['ssim'] - reference['ssim']:+.5f} "
                  f"aligned={row['aligned_psnr'] - reference['aligned_psnr']:+.4f}")
    (args.matrix / "analysis.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
