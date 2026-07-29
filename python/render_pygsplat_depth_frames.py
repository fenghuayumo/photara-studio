#!/usr/bin/env python3
"""Render exact GGGS median-depth frames from an external PLY with pygsplat."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pygsplat-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--sdf-trunc", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.pygsplat_repo.resolve()))
    from datasets.colmap import Dataset  # pylint: disable=import-outside-toplevel
    from gs2mesh import _render_view_geometry  # pylint: disable=import-outside-toplevel
    from main import load_runner_from_model  # pylint: disable=import-outside-toplevel
    from simple_trainer import Config  # pylint: disable=import-outside-toplevel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        data_dir=str(args.data_dir.resolve()),
        data_type="nerfstudio",
        transforms_path=str(args.transforms.resolve()),
        point_cloud_path=str(args.point_cloud.resolve()),
        test_every=0,
        result_dir=str(args.output_dir.resolve()),
        use_fastergs=True,
        depth_normal_mode="gggs",
        meshing=True,
        use_mask=False,
    )
    runner = load_runner_from_model(cfg, ckpt_path=str(args.ply.resolve()))
    runner.release_training_memory_for_meshing(drop_datasets=True)
    dataset = Dataset(
        runner.parser, split="train", patch_size=None, load_mask=False
    )

    rows: list[dict[str, object]] = []
    valid_counts = []
    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]
            height, width = map(int, data["image"].shape[:2])
            c2w = data["camtoworld"].to(runner.device).unsqueeze(0)
            intrinsic = data["K"].to(runner.device).unsqueeze(0)
            w2c_value = data.get("world2cam")
            w2c = (
                w2c_value.to(runner.device).unsqueeze(0)
                if w2c_value is not None
                else torch.linalg.inv(c2w)
            )
            colors, alphas, depth, _ = _render_view_geometry(
                runner,
                c2w,
                intrinsic,
                width,
                height,
                w2c,
                True,
                # gs2mesh's current acos(abs(dot)) > 100-degree test cannot
                # reject a pixel because acos(abs(dot)) is at most 90 degrees.
                # Avoid retaining the otherwise unused per-frame normal buffer.
                want_normal=False,
            )
            depth[
                (alphas[0, :, :, 0] < 0.5)
                | (depth <= 0.0)
                | (depth > args.max_depth)
            ] = 0.0
            depth_u16 = (
                depth.detach().float().cpu().numpy() * args.depth_scale
            ).astype(np.uint16)
            # Match gs2mesh's synchronization/lifetime pattern. The GGGS CUDA
            # extension otherwise retains output allocations across views on
            # some Windows builds when RGB is never consumed.
            _ = colors[0].detach().float().cpu().numpy()
            filename = f"depth_{index:04d}.u16"
            depth_u16.tofile(args.output_dir / filename)
            valid_counts.append(int(np.count_nonzero(depth_u16)))

            w2c_np = w2c[0].detach().double().cpu().numpy()
            k_np = intrinsic[0].detach().double().cpu().numpy()
            row: dict[str, object] = {
                "file": filename,
                "image": runner.parser.image_names[index],
                "width": width,
                "height": height,
                "fx": float(k_np[0, 0]),
                "fy": float(k_np[1, 1]),
                "cx": float(k_np[0, 2]),
                "cy": float(k_np[1, 2]),
            }
            for r in range(3):
                for c in range(3):
                    row[f"r{r}{c}"] = float(w2c_np[r, c])
                row[f"t{r}"] = float(w2c_np[r, 3])
            rows.append(row)
            print(
                f"[{index + 1:02d}/{len(dataset)}] "
                f"{runner.parser.image_names[index]} valid={valid_counts[-1]}"
            )
            del colors, alphas, depth, depth_u16
            torch.cuda.empty_cache()

    fieldnames = [
        "file",
        "image",
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
        "r00",
        "r01",
        "r02",
        "r10",
        "r11",
        "r12",
        "r20",
        "r21",
        "r22",
        "t0",
        "t1",
        "t2",
    ]
    with (args.output_dir / "frames.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "ply": str(args.ply.resolve()),
        "frame_count": len(rows),
        "max_depth": args.max_depth,
        "depth_scale": args.depth_scale,
        "valid_pixels_total": int(sum(valid_counts)),
        "valid_pixels_per_frame": valid_counts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if (args.voxel_size is None) != (args.sdf_trunc is None):
        raise ValueError("--voxel-size and --sdf-trunc must be provided together")
    if args.voxel_size is not None:
        (args.output_dir / "metadata.txt").write_text(
            "\n".join(
                [
                    f"voxel_size={args.voxel_size:.17g}",
                    f"sdf_trunc={args.sdf_trunc:.17g}",
                    f"depth_scale={args.depth_scale:.17g}",
                    f"depth_trunc={args.max_depth:.17g}",
                    f"frame_count={len(rows)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
