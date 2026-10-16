#pragma once
#include "io/image.hpp"
#include "splat/prior.hpp"
#include <filesystem>
#include <vector>

namespace editor {
struct PriorPreviewData {
    std::filesystem::path source;
    photara::splat::TrainingPrior prior;
    photara::io::RgbImage photo, depth, normal;
    float depth_near{}, depth_far{};
};
std::vector<std::filesystem::path> list_prior_previews(const std::filesystem::path& directory);
PriorPreviewData load_prior_preview(const std::filesystem::path& file);
std::filesystem::path find_prior_preview(
    const std::filesystem::path& directory, const std::filesystem::path& source);
photara::splat::Camera prior_preview_camera(const photara::splat::Camera& camera);
photara::io::RgbImage colorize_prior_depth(const photara::splat::TrainingPrior& prior,float near_depth,float far_depth,bool grayscale=false);
} // namespace editor
