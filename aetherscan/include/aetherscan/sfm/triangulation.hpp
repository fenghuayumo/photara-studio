#pragma once

#include "aetherscan/sfm/scene.hpp"

namespace aetherscan::sfm {

struct TriangulationOptions {
    double minimum_depth{1e-6};
    double maximum_reprojection_error{3.0};
    double minimum_angle_degrees{1.0};
};

struct TriangulationResult {
    bool valid{false};
    std::array<double, 3> position{};
    double mean_reprojection_error{};
    double maximum_angle_degrees{};
    std::size_t supporting_views{};
};

TriangulationResult triangulate_track(
    const Scene& scene,
    const Track& track,
    const TriangulationOptions& options = {});

}  // namespace aetherscan::sfm
