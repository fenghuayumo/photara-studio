#pragma once

#include "mvs/types.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace photara::texture {

[[nodiscard]] inline bool has_effective_foreground_mask(
    const mvs::MvsView& view) noexcept {
    return view.foreground_mask.size() ==
        static_cast<std::size_t>(view.width) * view.height;
}

[[nodiscard]] inline float effective_foreground_coverage(
    const mvs::MvsView& view, const std::size_t pixel) {
    if (!has_effective_foreground_mask(view) ||
        pixel >= view.foreground_mask.size())
        throw std::out_of_range(
            "Texture foreground-mask pixel is out of range");
    return static_cast<float>(view.foreground_mask[pixel]) / 255.F;
}

// Converts Photara's OpenCV-style camera (positive Z, image Y down, integer
// pixel centers) to photara_drender clip coordinates. photara_drender maps NDC to
// pixel-corner coordinates and subtracts 0.5 before sampling the photograph,
// hence the explicit +0.5 principal-point offset below.
[[nodiscard]] inline std::array<float, 16> world_to_clip_row_major(
    const mvs::MvsView& view, const float near_z, const float far_z) {
    if (view.width == 0 || view.height == 0 ||
        !(near_z > 0.F) || !(far_z > near_z) ||
        !std::isfinite(near_z) || !std::isfinite(far_z))
        throw std::invalid_argument(
            "Invalid camera or depth range for texture projection");

    const Eigen::Matrix3f rotation = view.pose.R.cast<float>();
    const Eigen::Vector3f translation =
        view.pose.translation().cast<float>();
    const float width = static_cast<float>(view.width);
    const float height = static_cast<float>(view.height);
    const float x_offset = 2.F * (view.cx + 0.5F) / width - 1.F;
    const float y_offset = 2.F * (view.cy + 0.5F) / height - 1.F;
    const float depth_scale = (far_z + near_z) / (far_z - near_z);
    const float depth_offset =
        -2.F * far_z * near_z / (far_z - near_z);

    std::array<float, 16> matrix{};
    for (int column = 0; column < 3; ++column) {
        matrix[0 * 4 + column] =
            (2.F * view.fx / width) * rotation(0, column) +
            x_offset * rotation(2, column);
        matrix[1 * 4 + column] =
            (2.F * view.fy / height) * rotation(1, column) +
            y_offset * rotation(2, column);
        matrix[2 * 4 + column] =
            depth_scale * rotation(2, column);
        matrix[3 * 4 + column] = rotation(2, column);
    }
    matrix[0 * 4 + 3] =
        (2.F * view.fx / width) * translation.x() +
        x_offset * translation.z();
    matrix[1 * 4 + 3] =
        (2.F * view.fy / height) * translation.y() +
        y_offset * translation.z();
    matrix[2 * 4 + 3] =
        depth_scale * translation.z() + depth_offset;
    matrix[3 * 4 + 3] = translation.z();
    return matrix;
}

}  // namespace photara::texture
