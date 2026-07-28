#pragma once

#include "mvs/types.hpp"

#include <cstdint>
#include <filesystem>

namespace aetherscan::texture {

struct MeshMaskOptions {
    // Render each axis at this multiple, then area-downsample. asdiff_render's
    // current raster output is single-sample; 2x produces stable soft edges
    // without teaching the MVS working resolution to the training mask.
    std::uint32_t supersample{2};
    // Reject bridge triangles before rasterization. The limit is the median
    // MVS-mesh edge length times this factor; zero disables the filter.
    float maximum_edge_factor{5.F};
    // Source-resolution topology cleanup only. These deliberately small
    // limits remove rasterized mesh specks and pinholes without closing real
    // gaps between subject parts.
    std::uint32_t minimum_island_pixels{64};
    std::uint32_t maximum_hole_pixels{64};
    std::uint32_t vulkan_device_index{0};
    // When non-empty, write a flat normal-shaded render from the exact same
    // asdiff raster pass as each mask. This makes camera/mesh alignment
    // inspectable without introducing a second renderer.
    std::filesystem::path preview_directory;
};

struct MeshMaskSummary {
    std::size_t image_count{};
    std::uint64_t pixel_count{};
    std::uint64_t foreground_pixels{};
    std::uint64_t soft_edge_pixels{};
    std::uint64_t rendered_faces{};
    std::uint64_t rejected_bridge_faces{};
    std::uint64_t removed_island_pixels{};
    std::uint64_t filled_hole_pixels{};
    std::size_t preview_count{};
};

// Rasterize the completed MVS mesh with the source-resolution ideal pinhole
// cameras used by GGGS. One grayscale PNG (stored as RGB for portable image
// IO) is written per source image and can be consumed as TrainingOptions.mask_dir.
[[nodiscard]] MeshMaskSummary render_mesh_foreground_masks(
    const mvs::MvsScene& scene,
    const std::filesystem::path& output_directory,
    const MeshMaskOptions& options = {});

// Export working-camera foreground coverage at the source ideal-pinhole
// resolution. Intended for inspecting depth/ROI masks; GGGS consumes the
// in-memory working masks directly.
[[nodiscard]] MeshMaskSummary export_view_foreground_masks(
    const mvs::MvsScene& scene,
    const std::filesystem::path& output_directory);

}  // namespace aetherscan::texture
