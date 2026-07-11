#include "aetherscan/sfm/reconstruct.hpp"

#include "aetherscan/sfm/triangulation.hpp"

namespace aetherscan::sfm {

ReconstructionSummary run_incremental_mapping(
    Scene& scene,
    const StarInitConfig& star,
    const ResectionConfig& resection) {
    ReconstructionSummary summary;
    if (!star_initialize(scene, star)) return summary;
    register_images(scene, resection);

    summary.registered_views = scene.registered_count();
    summary.failed_views =
        static_cast<unsigned>(scene.images.size()) - summary.registered_views;
    for (const Track& track : scene.tracks) {
        if (track.is_triangulated()) ++summary.landmarks;
    }
    summary.valid = summary.registered_views >= 2 && summary.landmarks > 0;
    return summary;
}

ReconstructionSummary reconstruct(
    Scene& scene_out,
    const std::vector<std::filesystem::path>& image_paths,
    const ReconstructionConfig& config) {
    FrontEndResult frontend = run_frontend(image_paths, config.frontend);
    scene_out = std::move(frontend.scene);
    return run_incremental_mapping(scene_out, config.star, config.resection);
}

}  // namespace aetherscan::sfm
