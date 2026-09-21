#include "mvs/internal.hpp"

#include "core/logging.hpp"

#include <stdexcept>

namespace photara::mvs {

void reconstruct_mesh(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.mesh");
    scene.mesh = {};
    if (options.mesh_method == MeshMethod::none) {
        stage.finish();
        return;
    }
    if (options.mesh_method == MeshMethod::delaunay_cut) {
        if (!detail::reconstruct_mesh_global_cgal(scene, options))
            throw std::runtime_error(
                "Global Delaunay meshing is unavailable or produced no "
                "surface. Configure a CGAL-enabled build.");
    } else if (options.mesh_method == MeshMethod::tsdf) {
        if (!detail::reconstruct_mesh_tsdf(scene, options))
            throw std::runtime_error(
                "TSDF meshing produced no surface. Check GGGS alpha/depth "
                "coverage or increase the TSDF truncation band.");
    } else {
        throw std::runtime_error("Unsupported MVS mesh method");
    }
    const OrientedBoundingBox* cleanup_bounds = nullptr;
    if (options.mesh_method == MeshMethod::tsdf &&
        scene.subject_bounds.valid)
        cleanup_bounds = &scene.subject_bounds;
    detail::clean_mesh(scene.mesh, options, cleanup_bounds);
    stage.finish();
}

}  // namespace photara::mvs
