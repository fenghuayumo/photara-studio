#pragma once

#include "mvs/options.hpp"
#include "mvs/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace aetherscan::mvs::detail {

struct ViewImage {
    std::vector<float> gray;  // [0,1]
    std::vector<std::uint8_t> mask;  // empty = all foreground
};

std::vector<ViewImage> load_view_images(
    const MvsScene& scene, const DensifyOptions& options);

bool load_manual_roi(
    const std::filesystem::path& path, OrientedBoundingBox& roi);
bool estimate_automatic_roi(MvsScene& scene, const DensifyOptions& options);
void build_depth_roi_foreground_masks(
    MvsScene& scene, const DensifyOptions& options);
void build_projected_foreground_masks(
    MvsScene& scene, const DensifyOptions& options);

// Scalable global Delaunay/visibility graph-cut backend. Returns false when
// CGAL support is not built or a valid global surface cannot be extracted.
bool reconstruct_mesh_global_cgal(
    MvsScene& scene, const DensifyOptions& options);

// Sparse TSDF fusion over the scene depth maps. Populates scene.mesh and
// returns false when the maps contain insufficient signed-distance support.
bool reconstruct_mesh_tsdf(MvsScene& scene, const DensifyOptions& options);

// Backend-independent topology cleanup and normal recomputation.
void clean_mesh(
    Mesh& mesh, const DensifyOptions& options,
    const OrientedBoundingBox* roi = nullptr);

// CGAL orients a tetrahedron facet according to the index of its opposite
// vertex. Keeping that orientation is important for the signed
// plane/circumsphere angle used by the Delaunay graph-cut regularizer.
[[nodiscard]] constexpr std::array<int, 3>
oriented_tetrahedron_facet_vertices(const int opposite) noexcept {
    return (opposite & 1) == 0
        ? std::array<int, 3>{
              (opposite + 2) & 3, (opposite + 1) & 3,
              (opposite + 3) & 3}
        : std::array<int, 3>{
              (opposite + 1) & 3, (opposite + 2) & 3,
              (opposite + 3) & 3};
}

struct DepthEvidence {
    float positive_confidence{0.F};
    float negative_confidence{0.F};
    unsigned supporting_views{0};
    unsigned conflicting_views{0};
};

// OpenMVS AdjustConfidence accepts a depth only when it has enough agreeing
// views and their confidence outweighs occlusion/free-space conflicts.
[[nodiscard]] inline bool accepts_depth_evidence(
    const DepthEvidence& evidence, const unsigned minimum_support) noexcept {
    return evidence.supporting_views >= minimum_support &&
           evidence.positive_confidence > evidence.negative_confidence;
}

// Return the OpenMVS weak-surface sink multiplier, or zero when beta/gamma do
// not identify a sufficiently strong free-space discontinuity.
[[nodiscard]] inline float weak_surface_sink_multiplier(
    const float beta, const float gamma,
    const DensifyOptions& options) noexcept {
    if (!(beta > 0.F) || !std::isfinite(beta) || !std::isfinite(gamma) ||
        gamma < 0.F)
        return 0.F;
    const float difference = beta - gamma;
    const float ratio = gamma / beta;
    return ratio < options.mesh_k_free_space_rel &&
                   difference > options.mesh_k_free_space_abs &&
                   gamma < options.mesh_k_free_space_outlier
        ? difference
        : 0.F;
}

[[nodiscard]] inline float weak_surface_support_scale(
    const float calibration_difference,
    const DensifyOptions& options) noexcept {
    if (!(calibration_difference > 0.F) ||
        !(options.mesh_k_free_space_abs > 0.F) ||
        !std::isfinite(calibration_difference))
        return 1.F;
    return std::max(
        1.F, calibration_difference / options.mesh_k_free_space_abs);
}

[[nodiscard]] inline bool sample_gray(
    const std::vector<float>& image, const std::uint32_t width,
    const std::uint32_t height, const float x, const float y, float& value) {
    if (x < 0.F || y < 0.F || x > static_cast<float>(width - 1) ||
        y > static_cast<float>(height - 1))
        return false;
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, static_cast<int>(width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(height) - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const auto at = [&](const int px, const int py) {
        return image[static_cast<std::size_t>(py) * width +
                     static_cast<std::size_t>(px)];
    };
    value = (at(x0, y0) * (1.F - tx) + at(x1, y0) * tx) * (1.F - ty) +
            (at(x0, y1) * (1.F - tx) + at(x1, y1) * tx) * ty;
    return true;
}

[[nodiscard]] inline Mat3f relative_rotation(
    const sfm::Pose3D& ref, const sfm::Pose3D& src) {
    return (src.R * ref.R.transpose()).cast<float>();
}

[[nodiscard]] inline Vec3f relative_translation(
    const sfm::Pose3D& ref, const sfm::Pose3D& src) {
    return (src.R * (ref.C - src.C)).cast<float>();
}

[[nodiscard]] inline Mat3f plane_homography(
    const MvsView& ref, const MvsView& src, const float depth,
    const Vec3f& normal_cam, const Vec3f& x0_cam) {
    const Mat3f R = relative_rotation(ref.pose, src.pose);
    const Vec3f t = relative_translation(ref.pose, src.pose);
    const float denom = normal_cam.dot(x0_cam) * depth;
    Mat3f H_cam = R;
    if (std::abs(denom) > 1e-8F)
        H_cam += t * (normal_cam.transpose() / denom);
    return src.K() * H_cam * ref.K().inverse();
}

}  // namespace aetherscan::mvs::detail
