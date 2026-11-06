#pragma once
#include "splat/types.hpp"
#include <filesystem>
#include <vector>

namespace photara::splat {
// Scene-unit camera-Z depth for pinhole/fisheye, ray distance for panorama.
// Predicted unit normals are in the original camera coordinate system.
// Either map may be empty when not requested. Zero pixels mean unavailable.
// Prior validity never masks RGB fitting.
struct TrainingPrior {
    Camera camera;
    std::vector<float> depth;
    std::vector<float> normals; // CHW
    void validate() const;
};
[[nodiscard]] std::filesystem::path training_prior_path(
    const std::filesystem::path& directory,
    std::uint32_t view_id);
void save_training_prior(
    const std::filesystem::path& path,
    const std::filesystem::path& source_image,
    const TrainingPrior& prior);
[[nodiscard]] TrainingPrior load_training_prior(
    const std::filesystem::path& path,
    const std::filesystem::path& source_image,
    const Camera& full_camera,
    bool require_depth = true,
    bool require_normals = true);
[[nodiscard]] TrainingPrior resize_training_prior(const TrainingPrior& prior, const Camera& camera);

// CUDA runtime representation: octahedral signed 16-bit coordinates packed
// into one word per normal. 0x80008000 denotes an unavailable normal.
struct PackedTrainingPrior {
    std::vector<float> depth;
    std::vector<int> normals;
};
[[nodiscard]] PackedTrainingPrior load_packed_training_prior(
    const std::filesystem::path& path, const std::filesystem::path& source_image,
    const Camera& full_camera, const Camera& camera, bool require_depth, bool require_normals);
} // namespace photara::splat
