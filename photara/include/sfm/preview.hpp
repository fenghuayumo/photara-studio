#pragma once
#include "sfm/scene.hpp"
namespace photara::sfm {
// Visualization only: bounded points, remapped observations, no descriptors.
Scene make_alignment_preview(const Scene& source, std::size_t maximum_points = 30000);
void save_alignment_preview(const Scene& source, const std::filesystem::path& path);
}
