#pragma once

#include "splat/types.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace aetherscan::splat {

// Forward-only preview shading. Training still uses Rasterizer::forward /
// backward; this path never builds a gradient tape.
enum class VisualizationMode : unsigned { splat = 0, points = 1, rings = 2 };

struct VisualizeOptions {
    VisualizationMode mode{VisualizationMode::splat};
    unsigned active_sh_degree{0};
    std::array<float, 3> background{0.F, 0.F, 0.F};
    float kernel_size{0.F};
    float scale_modifier{1.F};
    float point_size_px{2.5F};
    float ring_scale{2.5F};
};

[[nodiscard]] inline const char* visualization_mode_name(
    const VisualizationMode mode) noexcept {
    switch (mode) {
        case VisualizationMode::points: return "points";
        case VisualizationMode::rings: return "rings";
        case VisualizationMode::splat:
        default: return "splat";
    }
}

inline bool parse_visualization_mode(
    const std::string_view text, VisualizationMode& mode) noexcept {
    if (text == "points" || text == "1") {
        mode = VisualizationMode::points;
        return true;
    }
    if (text == "rings" || text == "2") {
        mode = VisualizationMode::rings;
        return true;
    }
    if (text == "splat" || text == "0") {
        mode = VisualizationMode::splat;
        return true;
    }
    return false;
}

inline bool write_visualization_sidecar(
    const std::filesystem::path& path, const VisualizeOptions& options,
    const std::uint64_t revision) {
    if (path.empty()) return false;
    std::ofstream output(path, std::ios::trunc);
    if (!output) return false;
    output << revision << '\n'
           << visualization_mode_name(options.mode) << '\n'
           << options.point_size_px << '\n'
           << options.ring_scale << '\n';
    return static_cast<bool>(output);
}

inline bool load_visualization_sidecar(
    const std::filesystem::path& path, VisualizeOptions& options,
    std::uint64_t& revision) {
    if (path.empty()) return false;
    std::ifstream input(path);
    if (!input) return false;
    std::uint64_t parsed_revision{};
    std::string mode_text;
    float point_size = options.point_size_px;
    float ring_scale = options.ring_scale;
    input >> parsed_revision >> mode_text >> point_size >> ring_scale;
    VisualizationMode mode = options.mode;
    if (!input || !parse_visualization_mode(mode_text, mode)) return false;
    revision = parsed_revision;
    options.mode = mode;
    options.point_size_px = std::max(0.5F, point_size);
    options.ring_scale = std::clamp(ring_scale, 0.5F, 8.F);
    return true;
}

// Planar float RGB [3,H,W] on CUDA. Splat mode reuses the training rasterizer
// forward pass; points and rings use a dedicated non-differentiable kernel.
[[nodiscard]] tinytensor::Tensor visualize(
    const GaussianModel& model, const Camera& camera,
    const VisualizeOptions& options = {});

}  // namespace aetherscan::splat
