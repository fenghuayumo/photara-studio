#pragma once

#include "mvs/options.hpp"
#include "mvs/types.hpp"

#include <algorithm>
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
void build_projected_foreground_masks(
    MvsScene& scene, const DensifyOptions& options);

// Scalable global Delaunay/visibility graph-cut backend. Returns false when
// CGAL support is not built or a valid global surface cannot be extracted.
bool reconstruct_mesh_global_cgal(
    MvsScene& scene, const DensifyOptions& options);

// Backend-independent topology cleanup and normal recomputation.
void clean_mesh(
    Mesh& mesh, const DensifyOptions& options,
    const OrientedBoundingBox* roi = nullptr);

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
