#include "mvs/densify.hpp"

#include "core/logging.hpp"
#include "mvs/internal.hpp"

#include <algorithm>
#include <stdexcept>

namespace aetherscan::mvs {

void densify(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.densify");
#if !defined(AETHERSCAN_HAS_CGAL)
    if (options.build_mesh &&
        options.mesh_method == MeshMethod::delaunay_cut)
        throw std::runtime_error(
            "Global Delaunay meshing requires a CGAL-enabled build. "
            "Install CGAL or explicitly select TSDF/projective meshing.");
#endif
    if (!options.roi_path.empty()) {
        if (!detail::load_manual_roi(options.roi_path, scene.roi))
            throw std::runtime_error(
                "Invalid manual ROI file: " + options.roi_path.string());
        scene.roi_automatic = false;
        detail::build_projected_foreground_masks(scene, options);
    }
    select_neighbors(scene, options);

    if (options.auto_roi && !scene.roi.valid) {
        DensifyOptions coarse = options;
        coarse.estimation_iters = std::min(2U, options.estimation_iters);
        coarse.geometric_iters = 0;
        coarse.geometric_consistency = false;
        coarse.random_iters = std::min(4U, options.random_iters);
        coarse.min_views_fuse = std::min(2U, options.min_views_fuse);
        coarse.mesh_method = MeshMethod::depth_projective;
        coarse.mesh_pixel_step = std::max(3U, options.mesh_pixel_step);
        coarse.mesh_close_hole_edges = 0;

        estimate_depth_maps(scene, coarse);
        fuse_depth_maps(scene, coarse);
        if (detail::estimate_automatic_roi(scene, options)) {
            reconstruct_mesh(scene, coarse);
            detail::build_projected_foreground_masks(scene, options);
            for (auto& view : scene.views) view.depth_map = {};
            scene.dense_cloud.points.clear();
            scene.mesh = {};
        } else {
            core::Logger::instance().warning(
                "automatic ROI failed; retaining unconstrained coarse result");
            if (options.build_mesh && options.mesh_method != MeshMethod::none)
                reconstruct_mesh(scene, options);
            stage.finish();
            return;
        }
    }

    estimate_depth_maps(scene, options);
    fuse_depth_maps(scene, options);
    if (options.build_mesh && options.mesh_method != MeshMethod::none)
        reconstruct_mesh(scene, options);
    stage.finish();
}

MvsScene densify_from_sfm(
    const sfm::Scene& sfm_scene, const DensifyOptions& options) {
    MvsScene scene = build_mvs_scene(sfm_scene, options);
    densify(scene, options);
    return scene;
}

}  // namespace aetherscan::mvs
