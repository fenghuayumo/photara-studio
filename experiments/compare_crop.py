"""Side-by-side crops (target | render | amplified difference) for visual QA."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("render", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--box", default="", help="x0,y0,x1,y1 crop; empty keeps the full frame")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gain", type=float, default=6.0, help="difference amplification")
    args = parser.parse_args()

    target = load(args.target)
    render = load(args.render)
    if args.box:
        x0, y0, x1, y1 = (int(v) for v in args.box.split(","))
        target = target[y0:y1, x0:x1]
        render = render[y0:y1, x0:x1]
    difference = np.clip(np.abs(target - render) * args.gain, 0.0, 1.0)
    separator = np.ones((target.shape[0], 4, 3), dtype=np.float32)
    panel = np.concatenate([target, separator, render, separator, difference], axis=1)
    if args.scale != 1.0:
        height = max(1, int(round(panel.shape[0] * args.scale)))
        width = max(1, int(round(panel.shape[1] * args.scale)))
        image = Image.fromarray(np.uint8(np.clip(panel, 0, 1) * 255))
        panel_image = image.resize((width, height), Image.LANCZOS)
    else:
        panel_image = Image.fromarray(np.uint8(np.clip(panel, 0, 1) * 255))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel_image.save(args.output)
    print(f"{args.output} {panel_image.size[0]}x{panel_image.size[1]}")


if __name__ == "__main__":
    main()
