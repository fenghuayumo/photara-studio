#include "mvs/densify.hpp"

#include "core/logging.hpp"
#include "mvs/export.hpp"
#include "mvs/internal.hpp"

#include <algorithm>
#include <stdexcept>

namespace aetherscan::mvs {

void densify(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.densify");
#if !defined(AETHERSCAN_HAS_CGAL)
    if (options.auto_roi && !scene.roi.valid)
        throw std::runtime_error(
            "Automatic ROI coarse meshing requires the CGAL global Delaunay "
            "backend. Configure a CGAL-enabled build.");
    if (options.build_mesh &&
        options.mesh_method == MeshMethod::delaunay_cut)
        throw std::runtime_error(
            "Global Delaunay meshing requires a CGAL-enabled build. "
            "Install CGAL or use GGGS TSDF meshing.");
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
        // Foreground masks require a globally coherent surface. Reuse the
        // production CGAL Delaunay/free-space graph-cut backend and cap only
        // its candidate count for this coarse pass.
        coarse.mesh_method = MeshMethod::delaunay_cut;
        constexpr std::uint64_t maximum_coarse_mesh_points = 500'000;
        coarse.mesh_max_points = options.mesh_max_points == 0
            ? maximum_coarse_mesh_points
            : std::min(
                  options.mesh_max_points, maximum_coarse_mesh_points);

        estimate_depth_maps(scene, coarse);
        fuse_depth_maps(scene, coarse);
        if (detail::estimate_automatic_roi(scene, options)) {
            reconstruct_mesh(scene, coarse);
            if (!options.coarse_mesh_output_path.empty()) {
                save_mesh_ply(scene.mesh, options.coarse_mesh_output_path);
                core::Logger::instance().info(
                    "coarse_mesh_ply=", options.coarse_mesh_output_path,
                    " vertices=", scene.mesh.vertices.size(),
                    " faces=", scene.mesh.faces.size());
            }
            detail::build_projected_foreground_masks(scene, options);
            if (options.coarse_preview_only) {
                core::Logger::instance().info(
                    "coarse preview complete; skipping final MVS pass");
                stage.finish();
                return;
            }
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
