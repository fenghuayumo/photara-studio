#pragma once

#include "splat/options.hpp"

#include <algorithm>

namespace photara::splat::detail {

struct MultiViewDepthQueryOptions {
    float bracket{};
    float tolerance{};
};

inline MultiViewDepthQueryOptions multi_view_depth_query_options(
    const TrainingOptions& options, const float scene_extent) {
    return {
        options.multi_view_depth_bracket > 0.F
            ? options.multi_view_depth_bracket
            : options.multi_view_depth_bracket == 0.F
                ? 4.F * std::max(scene_extent, 1.0e-3F)
                : 0.F,
        // Scene-wide depth error does not bound implicit median gradients
        // for thin Gaussians. Use all refinements unless explicitly requested.
        options.multi_view_depth_tolerance > 0.F
            ? options.multi_view_depth_tolerance : 0.F};
}

}  // namespace photara::splat::detail
