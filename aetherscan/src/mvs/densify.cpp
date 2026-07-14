#include "mvs/densify.hpp"

#include "core/logging.hpp"

namespace aetherscan::mvs {

void densify(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.densify");
    select_neighbors(scene, options);
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
