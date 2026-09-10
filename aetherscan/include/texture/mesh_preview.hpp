#pragma once

#include "io/image.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace aetherscan::texture {

// Orbit / capture camera in the same world-to-camera convention as the
// editor splat preview (column-major 4x4, OpenCV X-right Y-down Z-forward).
struct MeshPreviewCamera {
    std::array<float, 16> world_to_camera{};
    std::array<float, 3> position{};
    float fx{1.F};
    float fy{1.F};
    float cx{};
    float cy{};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct MeshPreviewOptions {
    bool wireframe = false;
    bool vertex_colour = false;
    std::array<float, 3> clay{0.77F, 0.73F, 0.68F};
    std::array<float, 3> background{0.027F, 0.031F, 0.043F};
};

// GPU triangle rasterizer (aether_drender) with a real Z-buffer. Keeps the
// Vulkan context alive across frames; construct once per editor session.
class MeshPreviewRasterizer {
public:
    MeshPreviewRasterizer();
    ~MeshPreviewRasterizer();
    MeshPreviewRasterizer(MeshPreviewRasterizer&&) noexcept;
    MeshPreviewRasterizer& operator=(MeshPreviewRasterizer&&) noexcept;
    MeshPreviewRasterizer(const MeshPreviewRasterizer&) = delete;
    MeshPreviewRasterizer& operator=(const MeshPreviewRasterizer&) = delete;

    // `positions` is packed xyz, `normals` packed xyz or empty, `colours`
    // packed rgb in [0,1] or empty, `indices` a triangle list.
    [[nodiscard]] io::RgbImage render(
        const std::vector<float>& positions,
        const std::vector<float>& normals,
        const std::vector<float>& colours,
        const std::vector<std::uint32_t>& indices,
        const MeshPreviewCamera& camera,
        const MeshPreviewOptions& options);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aetherscan::texture
