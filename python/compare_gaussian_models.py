#!/usr/bin/env python3
"""Compare physical Gaussian parameters from two 3DGS/GGGS PLY files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


QUANTILES = np.asarray([0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pygsplat-repo", type=Path, required=True)
    parser.add_argument("--aether-ply", type=Path, required=True)
    parser.add_argument("--pygsplat-ply", type=Path, required=True)
    parser.add_argument("--pygsplat-world-scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    result = np.percentile(values, QUANTILES)
    return {
        str(float(percentile)): float(value)
        for percentile, value in zip(QUANTILES, result)
    }


def summarize_model(path: Path, world_scale: float) -> dict[str, object]:
    from utils.splat_io import load_splats  # pylint: disable=import-outside-toplevel

    means, logits, scale_logs, _, _, _, degree, filter_3d = load_splats(
        str(path.resolve()), return_filter_3d=True
    )
    if filter_3d is None:
        raise ValueError(f"{path} has no filter_3D")
    means_np = means.numpy().astype(np.float64) * world_scale
    scales = np.exp(scale_logs.numpy().astype(np.float64)) * world_scale
    filters = filter_3d.numpy().astype(np.float64).reshape(-1) * world_scale
    opacity = torch.sigmoid(logits).numpy().astype(np.float64).reshape(-1)

    scale_sq = scales * scales
    filtered_sq = scale_sq + filters[:, None] ** 2
    effective_scales = np.sqrt(filtered_sq)
    filter_coefficient = np.sqrt(
        np.prod(scale_sq, axis=1) / np.prod(filtered_sq, axis=1)
    )
    effective_opacity = opacity * filter_coefficient
    geometric_scale = np.cbrt(np.prod(scales, axis=1))
    effective_geometric_scale = np.cbrt(
        np.prod(effective_scales, axis=1)
    )
    axis_ratio = scales.max(axis=1) / np.maximum(scales.min(axis=1), 1e-30)
    effective_axis_ratio = effective_scales.max(axis=1) / np.maximum(
        effective_scales.min(axis=1), 1e-30
    )
    robust_min = np.percentile(means_np, 1, axis=0)
    robust_max = np.percentile(means_np, 99, axis=0)

    opacity_thresholds = [0.001, 0.0039215686, 0.01, 0.05, 0.1, 0.5]
    return {
        "path": str(path.resolve()),
        "count": int(len(means)),
        "sh_degree": int(torch.as_tensor(degree).reshape(-1)[0]),
        "world_scale": world_scale,
        "opacity": quantiles(opacity),
        "effective_opacity": quantiles(effective_opacity),
        "opacity_fraction_below": {
            str(value): float(np.mean(opacity < value))
            for value in opacity_thresholds
        },
        "effective_opacity_fraction_below": {
            str(value): float(np.mean(effective_opacity < value))
            for value in opacity_thresholds
        },
        "scale_min_axis": quantiles(scales.min(axis=1)),
        "scale_geometric_mean": quantiles(geometric_scale),
        "scale_max_axis": quantiles(scales.max(axis=1)),
        "effective_scale_geometric_mean": quantiles(
            effective_geometric_scale
        ),
        "axis_ratio": quantiles(axis_ratio),
        "effective_axis_ratio": quantiles(effective_axis_ratio),
        "filter_3d": quantiles(filters),
        "filter_to_min_scale": quantiles(
            filters / np.maximum(scales.min(axis=1), 1e-30)
        ),
        "filter_opacity_coefficient": quantiles(filter_coefficient),
        "position_robust_p1_p99": {
            "min": robust_min.tolist(),
            "max": robust_max.tolist(),
            "extent": (robust_max - robust_min).tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.pygsplat_repo.resolve()))
    report = {
        "aether": summarize_model(args.aether_ply, 1.0),
        "pygsplat": summarize_model(
            args.pygsplat_ply, args.pygsplat_world_scale
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for name, model in report.items():
        print(
            name,
            "count",
            model["count"],
            "opacity p50/p10",
            model["opacity"]["50.0"],
            model["opacity"]["10.0"],
            "effective opacity p50/p10",
            model["effective_opacity"]["50.0"],
            model["effective_opacity"]["10.0"],
            "scale geom p50/p90",
            model["scale_geometric_mean"]["50.0"],
            model["scale_geometric_mean"]["90.0"],
            "filter p50/p90",
            model["filter_3d"]["50.0"],
            model["filter_3d"]["90.0"],
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
