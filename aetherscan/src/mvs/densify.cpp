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
    }
    select_neighbors(scene, options);

    estimate_depth_maps(scene, options);
    fuse_depth_maps(scene, options);
    // Automatic ROI is now a post-MVS operation. It may crop an explicitly
    // requested reconstruction volume, but it is never projected back into a
    // second PatchMatch pass. This keeps MVS -> mesh -> rendered mask -> GGGS
    // a strictly forward pipeline.
    if (options.auto_roi && !scene.roi.valid) {
        if (!detail::estimate_automatic_roi(scene, options))
            core::Logger::instance().warning(
                "automatic ROI failed; retaining the complete MVS cloud");
    }
    if (options.build_depth_roi_masks && scene.roi.valid)
        detail::build_depth_roi_foreground_masks(scene, options);
    if (options.build_mesh && options.mesh_method != MeshMethod::none)
        reconstruct_mesh(scene, options);
    if (options.coarse_preview_only)
        core::Logger::instance().warning(
            "--coarse-preview-only is deprecated: the pipeline now performs "
            "one MVS pass and exports the actual MVS mesh");
    stage.finish();
}

MvsScene densify_from_sfm(
    const sfm::Scene& sfm_scene, const DensifyOptions& options) {
    MvsScene scene = build_mvs_scene(sfm_scene, options);
    densify(scene, options);
    return scene;
}

}  // namespace aetherscan::mvs
