#pragma once
#include <algorithm>
#include <array>

namespace splat_render {
// Shared with AI prior previews. Near is warm; far is cool.
inline std::array<float, 3> depth_color(float t, bool grayscale) {
    t = std::clamp(t, 0.F, 1.F);
    if (grayscale) return {1.F - t, 1.F - t, 1.F - t};
    constexpr std::array<float, 3> near_color{0.90F, 0.80F, 0.53F};
    constexpr std::array<float, 3> middle_color{0.51F, 0.73F, 0.65F};
    constexpr std::array<float, 3> far_color{0.16F, 0.24F, 0.40F};
    const auto& a = t < 0.5F ? near_color : middle_color;
    const auto& b = t < 0.5F ? middle_color : far_color;
    const float u = t < 0.5F ? t * 2.F : (t - 0.5F) * 2.F;
    return {a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u,
            a[2] + (b[2] - a[2]) * u};
}
} // namespace splat_render
