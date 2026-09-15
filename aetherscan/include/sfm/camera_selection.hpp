#pragma once
#include "sfm/geometry.hpp"
#include <string>
#include <array>

namespace aetherscan::sfm {
// True when the image dimensions describe an equirectangular chart: a full turn
// horizontally and a half turn vertically, i.e. 2:1 within a small tolerance.
// This is the only size-based evidence that a still image covers the sphere, so
// panorama auto-selection is gated on it.
[[nodiscard]] bool is_equirectangular_image_size(
    std::uint32_t width, std::uint32_t height);

struct CameraModelProbe {
    std::vector<Vec2> first, second;
};
struct CameraModelSelection {
    CameraModel model{CameraModel::pinhole};
    double focal_pixels{};
    std::array<double,4> distortion{};
    double pinhole_score{}, fisheye_score{}, equirect_score{};
    unsigned informative_pairs{};
    bool confident{};
    std::string reason;
};
// Compare the same raw correspondences under calibrated projection hypotheses.
// Probes must share the supplied camera's dimensions. Ambiguity favors pinhole.
// A panorama (equirectangular) hypothesis is only considered when the caller
// reports a 2:1 equirectangular image size: the chart has no free intrinsics to
// calibrate, so the aspect ratio is the only admissible evidence that the image
// is a full sphere at all.
CameraModelSelection select_camera_model(
    const PinholeCamera& camera, const std::vector<CameraModelProbe>& probes,
    double supplied_focal = 0.0, bool fisheye_lens_hint = false,
    CameraModel requested = CameraModel::automatic,
    bool panorama_size_hint = false);
} // namespace aetherscan::sfm
