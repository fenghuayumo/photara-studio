#pragma once
#include "splat/visualize.hpp"
#include <limits>

namespace photara::splat::detail {
struct DiagnosticPush {
    unsigned pixels{}, stage{}, channel{}, grayscale{}, world{};
    float near_depth{}, far_depth{};
    float rotation[9]{};
};
inline DiagnosticPush diagnostic_push(const Camera &camera, const VisualizeOptions &options) {
    DiagnosticPush push;
    push.pixels = camera.width * camera.height;
    push.channel = static_cast<unsigned>(options.channel);
    push.grayscale = options.grayscale_depth;
    push.world = options.world_normals;
    push.near_depth = options.depth_near;
    push.far_depth = options.depth_far;
    const auto &m = camera.world_to_camera;
    const float rotation[] = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]};
    std::copy(std::begin(rotation), std::end(rotation), push.rotation);
    return push;
}
#if defined(TINYTENSOR_HAS_VULKAN)
tinytensor::Tensor visualize_diagnostics_vulkan(const RenderResult &rendered, const Camera &camera,
                                                const VisualizeOptions &options);
#endif
} // namespace photara::splat::detail
