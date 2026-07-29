#!/usr/bin/env python3
"""Export AetherScan TSDF camera frames as a pygsplat Nerfstudio dataset.

The TSDF frame manifest stores OpenCV world-to-camera extrinsics. Nerfstudio
stores OpenGL camera-to-world matrices, so this script inverts the extrinsic
and flips the camera Y/Z axes. pygsplat flips those axes back while parsing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        path.resolve()
        for path in args.images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    with args.frames_csv.open("r", encoding="utf-8", newline="") as stream:
        camera_rows = list(csv.DictReader(stream))
    if len(image_paths) != len(camera_rows):
        raise ValueError(
            f"image/camera count mismatch: {len(image_paths)} images, "
            f"{len(camera_rows)} camera rows"
        )

    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    frames: list[dict[str, object]] = []
    maximum_roundtrip_error = 0.0
    for index, (row, image_path) in enumerate(zip(camera_rows, image_paths)):
        width = int(row["width"])
        height = int(row["height"])
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"image size mismatch at {index}: {image_path.name} is "
                    f"{image.size}, camera is {(width, height)}"
                )

        world_to_camera = np.eye(4, dtype=np.float64)
        for r in range(3):
            for c in range(3):
                world_to_camera[r, c] = float(row[f"r{r}{c}"])
            world_to_camera[r, 3] = float(row[f"t{r}"])
        camera_to_world_cv = np.linalg.inv(world_to_camera)
        camera_to_world_gl = camera_to_world_cv @ opencv_to_opengl

        # This is exactly the conversion performed by pygsplat's parser.
        parsed_camera_to_world_cv = camera_to_world_gl @ opencv_to_opengl
        roundtrip = np.linalg.inv(parsed_camera_to_world_cv)
        maximum_roundtrip_error = max(
            maximum_roundtrip_error,
            float(np.max(np.abs(roundtrip - world_to_camera))),
        )
        frames.append(
            {
                "file_path": image_path.as_posix(),
                "w": width,
                "h": height,
                "fl_x": float(row["fx"]),
                "fl_y": float(row["fy"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "transform_matrix": camera_to_world_gl.tolist(),
            }
        )

    output = {
        "camera_model": "OPENCV",
        "frames": frames,
        "aetherscan_source_frames": str(args.frames_csv.resolve()),
        "maximum_world_to_camera_roundtrip_error": maximum_roundtrip_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(frames)} cameras to {args.output}")
    print(f"Maximum extrinsic round-trip error: {maximum_roundtrip_error:.3e}")
    print(f"First image: {image_paths[0].name}")


if __name__ == "__main__":
    main()
