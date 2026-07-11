#pragma once

#include "aetherscan/sfm/scene.hpp"

#include <array>

namespace aetherscan::sfm {

struct TwoViewOptions {
    double ransac_threshold_pixels{1.5};
    double confidence{0.999};
    std::size_t maximum_iterations{10000};
    std::size_t minimum_inliers{30};
};

struct TwoViewGeometry {
    bool valid{false};
    std::array<double, 9> essential{};
    std::array<double, 9> fundamental{};
    std::array<double, 9> homography{};
    std::array<double, 9> rotation{};
    std::array<double, 3> translation_direction{};
    features::MatchSet inliers;
    std::size_t homography_inliers{};
};

TwoViewGeometry verify_two_view(
    const Camera& first_camera,
    const Camera& second_camera,
    const features::FeatureSet& first_features,
    const features::FeatureSet& second_features,
    const features::MatchSet& matches,
    const TwoViewOptions& options = {});

}  // namespace aetherscan::sfm
