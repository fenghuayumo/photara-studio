#pragma once
#include "sfm/geometry.hpp"
#include <string>

namespace aetherscan::sfm {
struct CameraModelProbe {
    std::vector<Vec2> first, second;
};
struct CameraModelSelection {
    CameraModel model{CameraModel::pinhole};
    double focal_pixels{};
    double pinhole_score{}, fisheye_score{};
    unsigned informative_pairs{};
    bool confident{};
    std::string reason;
};
// Compare the same raw correspondences under calibrated projection hypotheses.
// Probes must share the supplied camera's dimensions. Ambiguity favors pinhole.
CameraModelSelection select_camera_model(
    const PinholeCamera& camera, const std::vector<CameraModelProbe>& probes,
    double supplied_focal = 0.0, bool fisheye_lens_hint = false);
} // namespace aetherscan::sfm
