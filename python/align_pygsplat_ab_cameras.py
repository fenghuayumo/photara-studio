#!/usr/bin/env python3
"""Express Photara cameras in a pygsplat/COLMAP model's world frame.

The script matches cameras by image name, estimates the Sim(3) mapping from
the COLMAP camera centers to the Photara camera centers, reports center and
orientation residuals, and writes a Nerfstudio transforms file that preserves
the Photara intrinsics while using the pygsplat model's native coordinates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pygsplat-repo", type=Path, required=True)
    parser.add_argument("--colmap-data", type=Path, required=True)
    parser.add_argument("--aether-transforms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def estimate_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = target_zero.T @ source_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ sign @ vt
    variance = float(np.sum(source_zero * source_zero) / len(source))
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = target_center - scale * rotation @ source_center
    return scale, rotation, translation


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.pygsplat_repo.resolve()))
    from datasets.colmap import Parser  # pylint: disable=import-outside-toplevel

    colmap = Parser(
        data_dir=str(args.colmap_data.resolve()),
        normalize=False,
        test_every=0,
    )
    aether = json.loads(args.aether_transforms.read_text(encoding="utf-8"))
    aether_by_name = {
        Path(frame["file_path"]).name: frame for frame in aether["frames"]
    }
    missing = [
        name for name in colmap.image_names if name not in aether_by_name
    ]
    if missing:
        raise ValueError(f"{len(missing)} COLMAP images lack Aether cameras")

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    aether_c2w_cv = []
    ordered_frames = []
    for name in colmap.image_names:
        frame = aether_by_name[name]
        aether_c2w_cv.append(
            np.asarray(frame["transform_matrix"], dtype=np.float64) @ flip
        )
        ordered_frames.append(frame)
    aether_c2w_cv = np.stack(aether_c2w_cv)
    colmap_c2w_cv = np.asarray(colmap.camtoworlds, dtype=np.float64)

    scale, rotation, translation = estimate_similarity(
        colmap_c2w_cv[:, :3, 3],
        aether_c2w_cv[:, :3, 3],
    )
    predicted_centers = (
        scale
        * (rotation @ colmap_c2w_cv[:, :3, 3].T).T
        + translation
    )
    center_errors = np.linalg.norm(
        predicted_centers - aether_c2w_cv[:, :3, 3], axis=1
    )
    orientation_errors = np.asarray(
        [
            rotation_angle_degrees(
                (rotation @ colmap_c2w_cv[i, :3, :3]).T
                @ aether_c2w_cv[i, :3, :3]
            )
            for i in range(len(colmap_c2w_cv))
        ]
    )

    output_frames = []
    for frame, c2w_aether in zip(ordered_frames, aether_c2w_cv):
        c2w_colmap = np.eye(4, dtype=np.float64)
        c2w_colmap[:3, :3] = rotation.T @ c2w_aether[:3, :3]
        c2w_colmap[:3, 3] = (
            rotation.T @ (c2w_aether[:3, 3] - translation) / scale
        )
        converted = dict(frame)
        converted["transform_matrix"] = (c2w_colmap @ flip).tolist()
        output_frames.append(converted)

    result = {
        "camera_model": "OPENCV",
        "frames": output_frames,
        "similarity_colmap_to_aether": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
        "alignment": {
            "center_error_mean": float(center_errors.mean()),
            "center_error_p50": float(np.percentile(center_errors, 50)),
            "center_error_p95": float(np.percentile(center_errors, 95)),
            "center_error_max": float(center_errors.max()),
            "orientation_error_deg_mean": float(orientation_errors.mean()),
            "orientation_error_deg_p50": float(
                np.percentile(orientation_errors, 50)
            ),
            "orientation_error_deg_p95": float(
                np.percentile(orientation_errors, 95)
            ),
            "orientation_error_deg_max": float(orientation_errors.max()),
        },
        "aether_equivalent_tsdf_scale_factor": 1.0 / scale,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(output_frames)} cameras to {args.output}")
    print(f"COLMAP -> Aether scale: {scale:.12g}")
    print(
        "Center error mean/p95/max: "
        f"{center_errors.mean():.6g} / "
        f"{np.percentile(center_errors, 95):.6g} / "
        f"{center_errors.max():.6g}"
    )
    print(
        "Orientation error mean/p95/max (deg): "
        f"{orientation_errors.mean():.6g} / "
        f"{np.percentile(orientation_errors, 95):.6g} / "
        f"{orientation_errors.max():.6g}"
    )


if __name__ == "__main__":
    main()
