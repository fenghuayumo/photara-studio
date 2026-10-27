#pragma once

#include <array>
#include <cstdint>

namespace splat_render {

// Forward-only placement for [previous end_index, end_index). Attributes stay
// in model space; positive uniform scale preserves the Gaussian model.
struct ModelTransform {
    std::uint32_t end_index{};
    std::array<float, 3> translation{};
    float scale{1.F};
    std::array<float, 4> rotation{1.F, 0.F, 0.F, 0.F};
    bool operator==(const ModelTransform&) const = default;
};

inline std::array<float, 3> rotate_vector(
    const std::array<float, 4>& q, const std::array<float, 3>& v) {
    const std::array<float, 3> t{
        2.F * (q[2] * v[2] - q[3] * v[1]),
        2.F * (q[3] * v[0] - q[1] * v[2]),
        2.F * (q[1] * v[1] - q[2] * v[0])};
    return {v[0] + q[0] * t[0] + q[2] * t[2] - q[3] * t[1],
            v[1] + q[0] * t[1] + q[3] * t[0] - q[1] * t[2],
            v[2] + q[0] * t[2] + q[1] * t[1] - q[2] * t[0]};
}

inline std::array<float, 3> transform_point(
    const ModelTransform& placement, const std::array<float, 3>& point) {
    auto result = rotate_vector(placement.rotation, point);
    for (int axis = 0; axis < 3; ++axis)
        result[axis] = placement.scale * result[axis] + placement.translation[axis];
    return result;
}

} // namespace splat_render
