#pragma once

#include "sfm/scene.hpp"

#include <cstddef>

namespace aetherscan::sfm {

// True when every triangulated track already carries RGB from Align (or an
// upgraded working copy). Train/MVS can then skip re-decoding photos.
[[nodiscard]] bool triangulated_tracks_have_color(const Scene& scene);

// Sample source photos at inlier keypoints and store sRGB on triangulated
// tracks. Already-coloured tracks are left unchanged. Returns how many tracks
// were newly coloured. Align calls this once so Train inherits appearance.
std::size_t colour_triangulated_tracks(Scene& scene);

}  // namespace aetherscan::sfm
