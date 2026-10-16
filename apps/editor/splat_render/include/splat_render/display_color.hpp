#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace splat_render {
// Legacy AI-prior rainbow: near red, middle green, far blue.
inline std::array<float, 3> depth_color(float t, bool grayscale) {
    t = std::clamp(t, 0.F, 1.F);
    if (grayscale) return {1.F - t, 1.F - t, 1.F - t};
    const float u = 1.F-t;
    return {std::clamp(1.5F-std::abs(4.F*u-3.F),0.F,1.F),
            std::clamp(1.5F-std::abs(4.F*u-2.F),0.F,1.F),
            std::clamp(1.5F-std::abs(4.F*u-1.F),0.F,1.F)};
}
inline std::array<float,3> normal_color(float x,float y,float z) {
    const float length=std::sqrt(x*x+y*y+z*z);
    if (!std::isfinite(length) || length<1e-6F) return {};
    return {std::clamp(.5F+.5F*x/length,0.F,1.F),
            std::clamp(.5F+.5F*y/length,0.F,1.F),
            std::clamp(.5F+.5F*z/length,0.F,1.F)};
}
} // namespace splat_render
