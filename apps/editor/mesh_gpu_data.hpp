#pragma once

#include <cstdint>
#include <vector>

namespace editor::gpu {

// Matches the mesh shader's four 16-byte vertex attributes. Packed on the
// loader thread, then shared with the renderer without another CPU copy.
struct alignas(16) MeshGpuVertex {
    float px{}, py{}, pz{}, pad0{};
    float nx{}, ny{}, nz{-1.F}, pad1{};
    float r{1.F}, g{1.F}, b{1.F}, pad2{};
    float u{}, v{}, pad3{}, pad4{};
};
static_assert(sizeof(MeshGpuVertex) == 64);

struct MeshGpuData {
    std::vector<MeshGpuVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct MeshGpuTexture {
    std::uint32_t width{}, height{};
    std::vector<std::uint8_t> pixels; // GPU-ready RGBA, four bytes per pixel.
};

} // namespace editor::gpu
