#pragma once

#include "sfm/scene.hpp"

namespace photara::sfm::detail {

// Internal regression screen, not a pose-accuracy certificate. No optimized
// landmarks, robust weights or reference reconstruction enter these metrics.
struct WarmStartGeometry {
    bool finite_centers{true};
    bool noncollapsed_centers{true};
    unsigned pair_constraints{};
    unsigned collapsed_baselines{};
    unsigned heldout_tracks{};
    unsigned heldout_observations{};
    unsigned positive_observations{};
    double pair_p90_radians{};
    double worst_camera_pair_radians{};
    double heldout_p90_radians{};
    double pixel_noise_radians{};
    double layout_change_p90_baselines{};
    std::vector<double> camera_pair_radians;
    std::vector<double> camera_heldout_radians;
    std::vector<unsigned> camera_heldout_observations;
    std::vector<unsigned> camera_positive_observations;
};

WarmStartGeometry evaluate_warm_start_geometry(
    const Scene& scene, const std::vector<Index>& optimized_tracks);

// Gauge-invariant bulk movement, in typical local view-graph baseline units.
double warm_start_layout_change(const Scene& candidate, const std::vector<Vec3>& baseline_centers);

// Returns nullptr on acceptance, otherwise a stable diagnostic reason.
const char* warm_start_geometry_rejection(
    const WarmStartGeometry& baseline, const WarmStartGeometry& candidate);

unsigned warm_start_unverified_direction_changes(
    const WarmStartGeometry& baseline, const WarmStartGeometry& candidate);

// Local pre-triangulation errors nominate cameras for final geometry auditing;
// they cannot by themselves justify retrying the whole scene from random points.
std::vector<Index> warm_start_local_geometry_risks(
    const WarmStartGeometry& baseline, const WarmStartGeometry& candidate);

} // namespace photara::sfm::detail
