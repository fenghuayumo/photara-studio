"""Reconstruct an Open3D/pygsplat TSDF mesh from Photara-exported frames."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def load_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    return values


def keep_largest_clusters(
    mesh: o3d.geometry.TriangleMesh, count: int
) -> o3d.geometry.TriangleMesh:
    if count <= 0 or len(mesh.triangles) == 0:
        return mesh
    labels, sizes, _ = mesh.cluster_connected_triangles()
    labels_np = np.asarray(labels)
    sizes_np = np.asarray(sizes)
    threshold = max(int(np.sort(sizes_np)[-count]), 50)
    mesh.remove_triangles_by_mask(sizes_np[labels_np] < threshold)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--keep-clusters", type=int, default=1)
    args = parser.parse_args()

    metadata = load_metadata(args.frames_dir / "metadata.txt")
    voxel_size = float(metadata["voxel_size"])
    sdf_trunc = float(metadata["sdf_trunc"])
    depth_scale = float(metadata["depth_scale"])
    depth_trunc = float(metadata["depth_trunc"])
    with (args.frames_dir / "frames.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        frames = list(csv.DictReader(stream))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    started = time.perf_counter()
    valid_pixels = 0
    for index, frame in enumerate(frames):
        width = int(frame["width"])
        height = int(frame["height"])
        depth = np.fromfile(
            args.frames_dir / frame["file"], dtype="<u2"
        ).reshape(height, width)
        valid_pixels += int(np.count_nonzero(depth))
        color = np.zeros((height, width, 3), dtype=np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(frame["fx"]),
            float(frame["fy"]),
            float(frame["cx"]),
            float(frame["cy"]),
        )
        extrinsic = np.eye(4, dtype=np.float64)
        for row in range(3):
            for column in range(3):
                extrinsic[row, column] = float(
                    frame[f"r{row}{column}"]
                )
            extrinsic[row, 3] = float(frame[f"t{row}"])
        volume.integrate(
            rgbd, intrinsic, np.ascontiguousarray(extrinsic)
        )
        if (index + 1) % 10 == 0 or index + 1 == len(frames):
            print(f"integrated {index + 1}/{len(frames)} frames")

    raw = volume.extract_triangle_mesh()
    raw.compute_vertex_normals()
    raw_path = args.output.with_name(
        f"{args.output.stem}_raw{args.output.suffix}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(raw_path), raw):
        raise RuntimeError(f"failed to write {raw_path}")
    post = keep_largest_clusters(raw, args.keep_clusters)
    post.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(args.output), post):
        raise RuntimeError(f"failed to write {args.output}")
    print(
        "Open3D TSDF complete:",
        f"frames={len(frames)}",
        f"valid_pixels={valid_pixels}",
        f"raw_vertices={len(raw.vertices)}",
        f"raw_faces={len(raw.triangles)}",
        f"post_vertices={len(post.vertices)}",
        f"post_faces={len(post.triangles)}",
        f"elapsed_s={time.perf_counter() - started:.3f}",
    )


if __name__ == "__main__":
    main()
