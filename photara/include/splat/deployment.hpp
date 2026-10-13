#pragma once
#include "mvs/types.hpp"
#include "splat/types.hpp"
#include <filesystem>
#include <functional>
#include <vector>

namespace photara::splat {
struct ProcessingControl {
    bool vulkan{true};
    std::function<void(float)> progress;
    std::function<bool()> cancelled;
};
struct LodOptions : ProcessingControl {
    unsigned levels{4};
    float keep_ratio{0.5F};
    unsigned chunk_count{512000};
    unsigned chunk_min{8000};
    float chunk_extent{16.F};
};
// Writes PlayCanvas Streamed SOG v1 (unbundled SOG chunks).
// Destination must be new; publishing happens only after successful processing.
void export_streamed_sog(const GaussianModel &, const std::filesystem::path &,
                         const LodOptions & = {});

struct CollisionOptions : ProcessingControl {
    float voxel_size{0.05F};
    float opacity{0.1F};
    unsigned max_voxels{16000000};
    enum class Fill { none, interior, ground } fill{Fill::none};
    // Collision data uses the model coordinate frame unless this is enabled.
    bool playcanvas_coordinates{true};
};
struct CollisionGrid {
    std::array<float, 3> origin{};
    std::array<float, 3> scene_min{}, scene_max{};
    std::array<unsigned, 3> dimensions{};
    float voxel_size{};
    bool playcanvas_coordinates{};
    std::vector<std::uint8_t> solid; // x + nx * (y + ny * z)
};
CollisionGrid voxelize_gaussians(const GaussianModel &, const CollisionOptions & = {});
// Exposed faces are shared at integer-grid corners and oriented outwards.
mvs::Mesh collision_mesh(const CollisionGrid &);
void export_collision(const GaussianModel &, const std::filesystem::path &,
                      const CollisionOptions & = {});
} // namespace photara::splat
