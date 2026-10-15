#pragma once
#include "splat/types.hpp"
#include <filesystem>
#include <vector>

namespace photara::splat {
// Pinhole camera-Z depth in scene units, and camera-space predicted unit normals.
// Zero depth/normal means unavailable. Prior validity never masks RGB fitting.
struct TrainingPrior {
    Camera camera;
    std::vector<float> depth;
    std::vector<float> normals; // CHW
    void validate() const;
};
[[nodiscard]] std::filesystem::path training_prior_path(
    const std::filesystem::path& directory, std::uint32_t view_id);
void save_training_prior(const std::filesystem::path& path,
    const std::filesystem::path& source_image, const TrainingPrior& prior);
[[nodiscard]] TrainingPrior load_training_prior(const std::filesystem::path& path,
    const std::filesystem::path& source_image, const Camera& full_camera);
[[nodiscard]] TrainingPrior resize_training_prior(const TrainingPrior& prior, const Camera& camera);
} // namespace photara::splat
