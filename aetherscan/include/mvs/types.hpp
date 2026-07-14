#pragma once

#include "sfm/types.hpp"

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace aetherscan::mvs {

using Index = sfm::Index;
inline constexpr Index k_invalid = sfm::k_invalid;

using Vec2f = Eigen::Vector2f;
using Vec3f = Eigen::Vector3f;
using Mat3f = Eigen::Matrix3f;

// Per-reference depth estimation product.
struct DepthMap {
    Index view_id{k_invalid};
    std::uint32_t width{};
    std::uint32_t height{};
    float depth_min{0.F};
    float depth_max{0.F};
    std::vector<float> depth;       // 0 = invalid
    std::vector<Vec3f> normal;      // camera space; zero if invalid
    std::vector<float> confidence;  // photometric cost (1-NCC); lower = better

    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    }

    [[nodiscard]] std::size_t index(const int x, const int y) const noexcept {
        return static_cast<std::size_t>(y) * width + static_cast<std::size_t>(x);
    }

    void resize(const std::uint32_t w, const std::uint32_t h) {
        width = w;
        height = h;
        const std::size_t n = size();
        depth.assign(n, 0.F);
        normal.assign(n, Vec3f::Zero());
        confidence.assign(n, 2.F);
    }
};

struct NeighborScore {
    Index view_id{k_invalid};
    float score{0.F};
    float angle_deg{0.F};
    float scale{1.F};
    unsigned shared_points{0};
};

struct DensePoint {
    Vec3f position{Vec3f::Zero()};
    Vec3f normal{Vec3f::Zero()};
    Vec3f color{Vec3f::Zero()};  // RGB in [0, 1]
    float weight{0.F};
    std::vector<Index> views;
};

struct DenseCloud {
    std::vector<DensePoint> points;
};

struct Mesh {
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;   // optional per-vertex; may be empty
    std::vector<Vec3f> colors;    // optional per-vertex RGB [0,1]; may be empty
    std::vector<Eigen::Vector3i> faces;
};

// Working view for densify. Intrinsics are pinhole at working resolution
// (images are undistorted + optionally downscaled).
struct MvsView {
    Index id{k_invalid};
    Index sfm_image_id{k_invalid};
    std::filesystem::path path;
    sfm::Pose3D pose;
    // Working-resolution ideal pinhole intrinsics (after undistort + downscale).
    float fx{1.F};
    float fy{1.F};
    float cx{0.F};
    float cy{0.F};
    std::uint32_t width{};
    std::uint32_t height{};
    // Original camera (source image) for undistortion.
    float src_fx{1.F};
    float src_fy{1.F};
    float src_cx{0.F};
    float src_cy{0.F};
    float k1{0.F};
    float k2{0.F};
    float p1{0.F};
    float p2{0.F};
    std::uint32_t src_width{};
    std::uint32_t src_height{};
    std::vector<NeighborScore> neighbors;
    DepthMap depth_map;

    [[nodiscard]] Mat3f K() const {
        Mat3f k = Mat3f::Identity();
        k(0, 0) = fx;
        k(1, 1) = fy;
        k(0, 2) = cx;
        k(1, 2) = cy;
        return k;
    }

    [[nodiscard]] Vec3f unproject(const float u, const float v, const float depth) const {
        return {(u - cx) / fx * depth, (v - cy) / fy * depth, depth};
    }

    [[nodiscard]] bool project(
        const Vec3f& camera_point, float& u, float& v) const {
        if (camera_point.z() <= 1e-6F) return false;
        const float inv_z = 1.F / camera_point.z();
        u = fx * camera_point.x() * inv_z + cx;
        v = fy * camera_point.y() * inv_z + cy;
        return std::isfinite(u) && std::isfinite(v);
    }
};

struct SparsePoint {
    Vec3f position{Vec3f::Zero()};
    std::vector<Index> view_ids;  // MvsView indices that observe this point
};

struct MvsScene {
    std::vector<MvsView> views;
    std::vector<SparsePoint> sparse_points;
    DenseCloud dense_cloud;
    Mesh mesh;
    unsigned thread_count{0};
};

}  // namespace aetherscan::mvs
