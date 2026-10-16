#pragma once
#include "io/image.hpp"
#include "splat/prior.hpp"
#include <filesystem>
#include <vector>
#include <optional>
#include <array>

namespace editor {
struct PriorPreviewData {
    std::filesystem::path source;
    photara::splat::TrainingPrior prior;
    photara::io::RgbImage photo, depth, normal;
    float depth_near{}, depth_far{};
};
struct PriorPreviewPixel {
    std::uint32_t x{}, y{};
    std::optional<float> depth;
    std::optional<std::array<float,3>> normal;
    bool depth_valid{}, normal_valid{};
};
std::optional<PriorPreviewPixel> sample_prior_preview(const PriorPreviewData& data, float u, float v);
std::vector<std::filesystem::path> list_prior_previews(const std::filesystem::path& directory);
PriorPreviewData load_prior_preview(const std::filesystem::path& file);
std::filesystem::path find_prior_preview(
    const std::filesystem::path& directory, const std::filesystem::path& source);
photara::splat::Camera prior_preview_camera(const photara::splat::Camera& camera);
photara::io::RgbImage colorize_prior_depth(const photara::splat::TrainingPrior& prior,float near_depth,float far_depth,bool grayscale=false);
} // namespace editor
