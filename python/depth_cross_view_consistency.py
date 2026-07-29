#!/usr/bin/env python3
"""Measure multi-view consistency of exported z-depth frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--world-scale", type=float, default=1.0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument(
        "--thresholds-mm",
        type=float,
        nargs="+",
        default=[1.059, 2.119, 4.237, 8.474],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_frames(directory: Path) -> tuple[list[dict[str, str]], list[np.ndarray]]:
    with (directory / "frames.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    depths = []
    for row in rows:
        height, width = int(row["height"]), int(row["width"])
        depths.append(
            np.fromfile(directory / row["file"], dtype="<u2").reshape(
                height, width
            )
        )
    return rows, depths


def extrinsic(row: dict[str, str]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    for r in range(3):
        for c in range(3):
            matrix[r, c] = float(row[f"r{r}{c}"])
        matrix[r, 3] = float(row[f"t{r}"])
    return matrix


def main() -> None:
    args = parse_args()
    rows, depths_u16 = load_frames(args.frames_dir)
    w2cs = np.stack([extrinsic(row) for row in rows])
    c2ws = np.linalg.inv(w2cs)
    centers = c2ws[:, :3, 3]
    center_distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=-1
    )
    nearest = np.argsort(center_distances, axis=1)[:, 1 : args.neighbors + 1]
    thresholds = np.asarray(args.thresholds_mm, dtype=np.float64) / 1000.0

    aggregate = {
        "projected": 0,
        "target_valid": 0,
        "target_invalid": 0,
        "occluded": np.zeros(len(thresholds), dtype=np.int64),
        "consistent": np.zeros(len(thresholds), dtype=np.int64),
        "behind": np.zeros(len(thresholds), dtype=np.int64),
    }
    residual_samples = []
    per_view = []
    view0_regions = {
        "table": (20, 720, 380, 1000),
        "body": (360, 390, 650, 710),
        "tail": (640, 360, 990, 650),
    }
    roi_accumulator = {
        name: {
            "projected": 0,
            "target_valid": 0,
            "consistent": np.zeros(len(thresholds), dtype=np.int64),
            "behind": np.zeros(len(thresholds), dtype=np.int64),
        }
        for name in view0_regions
    }

    for reference_index, row in enumerate(rows):
        depth_native = (
            depths_u16[reference_index].astype(np.float64) / args.depth_scale
        )
        height, width = depth_native.shape
        yy, xx = np.mgrid[0:height:args.stride, 0:width:args.stride]
        zz = depth_native[:: args.stride, :: args.stride]
        valid = zz > 0.0
        pixel_x = xx[valid].astype(np.float64)
        pixel_y = yy[valid].astype(np.float64)
        z = zz[valid]
        fx, fy = float(row["fx"]), float(row["fy"])
        cx, cy = float(row["cx"]), float(row["cy"])
        points_camera = np.stack(
            [
                (pixel_x - cx) * z / fx,
                (pixel_y - cy) * z / fy,
                z,
            ],
            axis=1,
        )
        points_world = (
            c2ws[reference_index, :3, :3] @ points_camera.T
        ).T + c2ws[reference_index, :3, 3]

        view_counts = {
            "reference_valid": int(len(z)),
            "projected": 0,
            "target_valid": 0,
            "consistent": np.zeros(len(thresholds), dtype=np.int64),
            "behind": np.zeros(len(thresholds), dtype=np.int64),
        }
        for target_index in nearest[reference_index]:
            target_row = rows[int(target_index)]
            camera_target = (
                w2cs[target_index, :3, :3] @ points_world.T
            ).T + w2cs[target_index, :3, 3]
            z_projected = camera_target[:, 2]
            tx = (
                float(target_row["fx"]) * camera_target[:, 0] / z_projected
                + float(target_row["cx"])
            )
            ty = (
                float(target_row["fy"]) * camera_target[:, 1] / z_projected
                + float(target_row["cy"])
            )
            target_depth = depths_u16[int(target_index)]
            target_height, target_width = target_depth.shape
            inside = (
                (z_projected > 0.0)
                & (tx >= 0.0)
                & (ty >= 0.0)
                & (tx < target_width)
                & (ty < target_height)
            )
            source_indices = np.flatnonzero(inside)
            if len(source_indices) == 0:
                continue
            sample_x = np.clip(
                np.rint(tx[inside]).astype(np.int64), 0, target_width - 1
            )
            sample_y = np.clip(
                np.rint(ty[inside]).astype(np.int64), 0, target_height - 1
            )
            sampled_native = (
                target_depth[sample_y, sample_x].astype(np.float64)
                / args.depth_scale
            )
            target_valid = sampled_native > 0.0
            residual = (
                sampled_native[target_valid]
                - z_projected[inside][target_valid]
            ) * args.world_scale

            projected_count = int(len(source_indices))
            target_valid_count = int(np.count_nonzero(target_valid))
            aggregate["projected"] += projected_count
            aggregate["target_valid"] += target_valid_count
            aggregate["target_invalid"] += projected_count - target_valid_count
            view_counts["projected"] += projected_count
            view_counts["target_valid"] += target_valid_count
            if len(residual):
                residual_samples.append(
                    residual[:: max(1, len(residual) // 10000)]
                )
            for threshold_index, threshold in enumerate(thresholds):
                consistent = np.abs(residual) <= threshold
                occluded = residual < -threshold
                behind = residual > threshold
                aggregate["consistent"][threshold_index] += int(
                    np.count_nonzero(consistent)
                )
                aggregate["occluded"][threshold_index] += int(
                    np.count_nonzero(occluded)
                )
                aggregate["behind"][threshold_index] += int(
                    np.count_nonzero(behind)
                )
                view_counts["consistent"][threshold_index] += int(
                    np.count_nonzero(consistent)
                )
                view_counts["behind"][threshold_index] += int(
                    np.count_nonzero(behind)
                )

            if reference_index == 0:
                valid_source_indices = source_indices[target_valid]
                for name, (x0, y0, x1, y1) in view0_regions.items():
                    roi_source = (
                        (pixel_x[source_indices] >= x0)
                        & (pixel_x[source_indices] < x1)
                        & (pixel_y[source_indices] >= y0)
                        & (pixel_y[source_indices] < y1)
                    )
                    roi_valid = (
                        (pixel_x[valid_source_indices] >= x0)
                        & (pixel_x[valid_source_indices] < x1)
                        & (pixel_y[valid_source_indices] >= y0)
                        & (pixel_y[valid_source_indices] < y1)
                    )
                    roi_residual = residual[roi_valid]
                    roi_accumulator[name]["projected"] += int(
                        np.count_nonzero(roi_source)
                    )
                    roi_accumulator[name]["target_valid"] += int(
                        np.count_nonzero(roi_valid)
                    )
                    for threshold_index, threshold in enumerate(thresholds):
                        roi_accumulator[name]["consistent"][
                            threshold_index
                        ] += int(
                            np.count_nonzero(
                                np.abs(roi_residual) <= threshold
                            )
                        )
                        roi_accumulator[name]["behind"][
                            threshold_index
                        ] += int(np.count_nonzero(roi_residual > threshold))

        per_view.append(view_counts)
        print(
            f"[{reference_index + 1:02d}/{len(rows)}] "
            f"valid={view_counts['reference_valid']} "
            f"target_valid={view_counts['target_valid']}/"
            f"{view_counts['projected']}"
        )

    residuals = (
        np.concatenate(residual_samples)
        if residual_samples
        else np.empty(0, dtype=np.float64)
    )

    def threshold_report(container: dict[str, object]) -> list[dict[str, float]]:
        reports = []
        target_valid = int(container["target_valid"])
        for index, threshold in enumerate(thresholds):
            consistent = int(container["consistent"][index])
            behind = int(container["behind"][index])
            visible = consistent + behind
            reports.append(
                {
                    "threshold_mm": float(threshold * 1000.0),
                    "consistent": consistent,
                    "behind": behind,
                    "consistent_over_target_valid": (
                        consistent / target_valid if target_valid else 0.0
                    ),
                    "visible_consistency": (
                        consistent / visible if visible else 0.0
                    ),
                }
            )
        return reports

    report = {
        "frames_dir": str(args.frames_dir.resolve()),
        "world_scale": args.world_scale,
        "stride": args.stride,
        "neighbors": args.neighbors,
        "reference_valid_total": int(
            sum(view["reference_valid"] for view in per_view)
        ),
        "projected": int(aggregate["projected"]),
        "target_valid": int(aggregate["target_valid"]),
        "target_coverage": (
            int(aggregate["target_valid"]) / int(aggregate["projected"])
            if aggregate["projected"]
            else 0.0
        ),
        "thresholds": threshold_report(aggregate),
        "absolute_residual_mm_quantiles": {
            str(q): float(np.percentile(np.abs(residuals), q) * 1000.0)
            for q in [10, 25, 50, 75, 90, 95, 99]
        },
        "view0_rois": {
            name: {
                "projected": int(values["projected"]),
                "target_valid": int(values["target_valid"]),
                "target_coverage": (
                    int(values["target_valid"]) / int(values["projected"])
                    if values["projected"]
                    else 0.0
                ),
                "thresholds": threshold_report(values),
            }
            for name, values in roi_accumulator.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
