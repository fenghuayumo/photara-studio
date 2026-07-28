#pragma once

#include "mvs/options.hpp"
#include "mvs/types.hpp"
#include "sfm/scene.hpp"

namespace aetherscan::mvs {

// Build an MVS working scene from a registered SfM reconstruction.
// Images are loaded, undistorted, and downscaled to the working resolution.
MvsScene build_mvs_scene(const sfm::Scene& sfm_scene, const DensifyOptions& options);

// Score and attach neighbor lists on every view (covisibility + baseline angle).
void select_neighbors(MvsScene& scene, const DensifyOptions& options);

// Estimate a depth/normal/confidence map for every reference view. The default
// uses CUDA whenever a GPU is detected and CPU only when none is available.
void estimate_depth_maps(MvsScene& scene, const DensifyOptions& options);

// Fuse depth maps into a multi-view consistent DenseCloud.
void fuse_depth_maps(MvsScene& scene, const DensifyOptions& options);

// Extract a triangle mesh from the dense products. MVS uses CGAL Delaunay
// tetrahedralization + visibility graph-cut; GGGS depth uses TSDF.
void reconstruct_mesh(MvsScene& scene, const DensifyOptions& options);

// Full Stage-A Fast MVS: neighbors -> depth -> fuse -> optional mesh.
void densify(MvsScene& scene, const DensifyOptions& options);

// Convenience: SfM scene -> densified MvsScene.
MvsScene densify_from_sfm(const sfm::Scene& sfm_scene, const DensifyOptions& options);

}  // namespace aetherscan::mvs
