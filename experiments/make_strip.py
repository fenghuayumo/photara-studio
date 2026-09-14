"""Horizontal strip of same-size images for visual comparison of floaters."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--box", default="", help="x0,y0,x1,y1 crop")
    args = parser.parse_args()
    panels = []
    for path in args.images:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        if args.box:
            x0, y0, x1, y1 = (int(v) for v in args.box.split(","))
            image = image[y0:y1, x0:x1]
        panels.append(image)
        panels.append(np.full((image.shape[0], 4, 3), 255, dtype=np.uint8))
    strip = np.concatenate(panels[:-1], axis=1)
    result = Image.fromarray(strip)
    if args.scale != 1.0:
        result = result.resize(
            (max(1, int(result.width * args.scale)),
             max(1, int(result.height * args.scale))), Image.LANCZOS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"{args.output} {result.width}x{result.height}")


if __name__ == "__main__":
    main()
