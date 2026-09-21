#pragma once

#include "splat/types.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <string_view>

namespace photara::splat {

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

[[nodiscard]] inline const char* camera_model_token(
    const CameraModel model) noexcept {
    switch (model) {
        case CameraModel::opencv_fisheye:
            return "opencv_fisheye";
        case CameraModel::equirectangular:
            return "equirectangular";
        case CameraModel::pinhole:
        case CameraModel::automatic:
        default:
            return "pinhole";
    }
}

inline bool parse_camera_model_token(
    const std::string_view text, CameraModel& model) noexcept {
    if (text == "opencv_fisheye" || text == "fisheye") {
        model = CameraModel::opencv_fisheye;
        return true;
    }
    if (text == "equirectangular" || text == "equirect") {
        model = CameraModel::equirectangular;
        return true;
    }
    if (text == "pinhole") {
        model = CameraModel::pinhole;
        return true;
    }
    return false;
}

// Sidecar used by the editor orbit / 2D QA cameras. Layout:
//   revision
//   world_to_camera[16]
//   px py pz
//   fx fy cx cy width height [model k1 k2 k3 k4]
//   [vis_mode point_size ring_scale]
// The model token is optional so older pinhole-only files still load.
inline bool write_preview_camera_sidecar(
    const std::filesystem::path& path, const Camera& camera,
    const std::uint64_t revision, const char* vis_mode = nullptr,
    const float point_size_px = 2.5F, const float ring_scale = 2.5F) {
    if (path.empty()) return false;
    std::ofstream output(path, std::ios::trunc);
    if (!output) return false;
    output << std::setprecision(9);
    output << revision << '\n';
    for (std::size_t i = 0; i < camera.world_to_camera.size(); ++i) {
        if (i) output << ' ';
        output << camera.world_to_camera[i];
    }
    output << '\n'
           << camera.position[0] << ' ' << camera.position[1] << ' '
           << camera.position[2] << '\n'
           << camera.fx << ' ' << camera.fy << ' ' << camera.cx << ' '
           << camera.cy << ' ' << camera.width << ' ' << camera.height << ' '
           << camera_model_token(camera.model) << ' ' << camera.k1 << ' '
           << camera.k2 << ' ' << camera.k3 << ' ' << camera.k4 << '\n';
    if (vis_mode != nullptr && vis_mode[0] != '\0')
        output << vis_mode << ' ' << point_size_px << ' ' << ring_scale << '\n';
    return static_cast<bool>(output);
}

inline bool load_preview_camera_sidecar(
    const std::filesystem::path& path, Camera& camera, std::uint64_t& revision,
    VisualizeOptions* vis = nullptr) {
    if (path.empty()) return false;
    std::ifstream input(path);
    if (!input) return false;
    std::uint64_t parsed_revision{};
    input >> parsed_revision;
    for (float& value : camera.world_to_camera) input >> value;
    input >> camera.position[0] >> camera.position[1] >> camera.position[2];
    double fx{}, fy{}, cx{}, cy{};
    unsigned width{}, height{};
    input >> fx >> fy >> cx >> cy >> width >> height;
    if (!input || width == 0 || height == 0 || !(fx > 0.0) || !(fy > 0.0))
        return false;
    constexpr unsigned k_max_preview_extent = 4096;
    camera.fx = static_cast<float>(fx);
    camera.fy = static_cast<float>(fy);
    camera.cx = static_cast<float>(cx);
    camera.cy = static_cast<float>(cy);
    camera.width = std::min(width, k_max_preview_extent);
    camera.height = std::min(height, k_max_preview_extent);
    camera.model = CameraModel::pinhole;
    camera.k1 = camera.k2 = camera.k3 = camera.k4 = 0.F;

    const auto apply_vis = [&](const std::string& mode_text,
                               const float point_size,
                               const float ring_scale) {
        if (vis == nullptr) return;
        VisualizationMode mode = vis->mode;
        if (!parse_visualization_mode(mode_text, mode)) return;
        vis->mode = mode;
        vis->point_size_px = std::max(0.5F, point_size);
        vis->ring_scale = std::clamp(ring_scale, 0.5F, 8.F);
    };

    std::string token;
    if (input >> token) {
        CameraModel model = CameraModel::pinhole;
        if (parse_camera_model_token(token, model)) {
            float k1 = 0.F, k2 = 0.F, k3 = 0.F, k4 = 0.F;
            if (!(input >> k1 >> k2 >> k3 >> k4)) return false;
            camera.model = model;
            camera.k1 = k1;
            camera.k2 = k2;
            camera.k3 = k3;
            camera.k4 = k4;
            std::string mode_text;
            float point_size = vis != nullptr ? vis->point_size_px : 2.5F;
            float ring_scale = vis != nullptr ? vis->ring_scale : 2.5F;
            if (input >> mode_text >> point_size >> ring_scale)
                apply_vis(mode_text, point_size, ring_scale);
        } else {
            float point_size = vis != nullptr ? vis->point_size_px : 2.5F;
            float ring_scale = vis != nullptr ? vis->ring_scale : 2.5F;
            if (input >> point_size >> ring_scale)
                apply_vis(token, point_size, ring_scale);
        }
    }
    revision = parsed_revision;
    return true;
}

// Planar float RGB [3,H,W] on CUDA. Splat mode reuses the training rasterizer
// forward pass; points and rings use a dedicated non-differentiable kernel.
[[nodiscard]] tinytensor::Tensor visualize(
    const GaussianModel& model, const Camera& camera,
    const VisualizeOptions& options = {});

}  // namespace photara::splat
