#pragma once

#include "sfm/hierarchical.hpp"
#include "sfm/checkpoint.hpp"
#include <functional>
#include <string>

namespace photara::sfm {

// Frontend-only structural screen: ignores all inherited poses and tracks.
// Marks cameras outside the largest bridge-free verified-pair block so dense
// internal connectivity cannot hide a branch attached through a single bridge.
std::vector<std::uint8_t> find_structural_pair_risks(const Scene& scene);
// Resection can constrain a singleton without independently triangulated local
// depth. Fit against stable-only depths, then validate withheld correspondences.
std::vector<Index> recover_stable_resections(Scene& scene);
// Explicit frozen anchors for offline quality recovery. Untrusted views never
// contribute depths used to estimate or validate their own replacement poses.
std::vector<Index> recover_stable_resections(
    Scene& scene, const std::vector<std::uint8_t>& stable);
// Local frozen-map proposals, never pair geometry. Two-view depths need >=2deg
// parallax, fit and held-out support from >=3 anchor cameras, >=12 held-out
// inliers and >=30 final observations. The ordinary overload keeps its stricter
// per-depth three-view requirement. Only unregistered targets may enter here.
std::vector<Index> recover_stable_resections(
    Scene& scene, const std::vector<std::uint8_t>& stable,
    const std::vector<RawPairMatches>& tentative_matches);
// Screen connections relative to each view's strongest pair, recover weak
// branches from independently validated anchors, and unregister unsupported
// poses. Strong-component poses and camera intrinsics stay fixed.
struct WeakBranchRecoveryOptions {
    // Disable for an offline comparison with full rebuild/triangulation,
    // including the frozen-boundary local BA neighborhood.
    bool reuse_verified_geometry{true};
};
unsigned recover_weakly_connected_views(
    Scene& scene, const WeakBranchRecoveryOptions& options = {});

struct SubmapRecoveryOptions {
    unsigned minimum_shared_points{30};
    unsigned minimum_camera_observations{30};
    double maximum_reprojection_error{2.0};
    double minimum_inlier_ratio{0.8};
    double relative_alignment_error{0.01};
    float minimum_angle_degrees{1.F};
    // Offline initializer receives only the cleared, internal-only submap.
    // Empty preserves the production star initializer and all acceptance gates.
    std::function<bool(Scene&)> independent_initializer;
    // Optional evidence export before alignment, including rejected submaps.
    std::function<void(const HierarchicalSubscene&)> independent_model_callback;
};

struct SubmapRecoveryReport {
    std::vector<Index> images;
    unsigned reconstructed_views{};
    unsigned local_landmarks{};
    unsigned boundary_matches{};
    unsigned candidate_tracks{};
    unsigned stable_multiview_tracks{};
    unsigned stable_triangulated_tracks{};
    unsigned shared_points{};
    unsigned fit_inliers{};
    unsigned validation_points{};
    unsigned validation_inliers{};
    bool accepted{};
    std::string reason;
    Similarity3 transform;
};

// Copy only internal verified pairs and keypoints. No inherited poses, points,
// tracks, descriptors or registration state; intrinsics are held fixed.
HierarchicalSubscene make_independent_submap(
    const Scene& parent, const std::vector<Index>& images);

// Experimental recovery, deliberately separate from the default mapper.
// The stable mask is frozen by the caller for the entire recovery pass.
// Rebuilds local geometry from images/pairs and stable depths from stable-only
// observations. Held-out landmarks must validate Sim(3) in 3D AND pixels.
// Failure leaves parent geometry unchanged; success never changes stable poses.
SubmapRecoveryReport recover_independent_submap(
    Scene& parent, const std::vector<Index>& images,
    const std::vector<std::uint8_t>& stable,
    const SubmapRecoveryOptions& options = {});

}  // namespace photara::sfm
