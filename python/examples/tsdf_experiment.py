"""Run splat depth rendering and TSDF meshing without rebuilding C++."""

from __future__ import annotations

import argparse
from pathlib import Path

import aetherscan as aes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-scale", type=float, default=1.0)
    parser.add_argument("--truncation-voxels", type=float, default=4.0)
    parser.add_argument("--bounds-padding", type=float, default=2.0)
    parser.add_argument("--smooth-iters", type=int, default=2)
    args = parser.parse_args()

    request = aes.DatasetRequest()
    request.source = args.colmap
    request.image_directory = args.images
    request.initial_point_cloud = args.points
    request.format = aes.DatasetFormat.COLMAP
    scene = aes.load_dataset(request)
    model = aes.load_3dgs(args.model)

    training = aes.TrainingOptions()
    training.use_mask = False
    training.use_source_resolution = True

    tsdf = aes.TsdfOptions()
    tsdf.diagnostics_dir = args.output.parent
    tsdf.fusion.mesh_method = aes.MeshMethod.TSDF
    tsdf.fusion.mesh_tsdf_voxel_scale = args.voxel_scale
    tsdf.fusion.mesh_tsdf_truncation_voxels = args.truncation_voxels
    tsdf.fusion.mesh_tsdf_bounds_padding = args.bounds_padding
    tsdf.fusion.mesh_tsdf_smooth_iters = args.smooth_iters

    result = aes.extract_tsdf(model, scene, training, tsdf)
    result.save(args.output)
    print(
        f"mesh={args.output} vertices={result.vertex_count} "
        f"faces={result.face_count} valid_depth_pixels="
        f"{result.valid_depth_pixels}"
    )


if __name__ == "__main__":
    main()
