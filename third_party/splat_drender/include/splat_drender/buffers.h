// Workspace layouts shared between the host pipeline and device kernels.
//
// Every pool is a flat byte buffer carved into fixed-order arrays. The
// from_pool() helpers mirror the bytes() calculators; both must be updated
// together.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace splat_drender::ws {

constexpr std::size_t kAlign = 128;

inline std::size_t align_up(std::size_t v) {
    return (v + kAlign - 1) & ~(kAlign - 1);
}

inline char* take(char*& cursor, std::size_t bytes) {
    char* p = cursor;
    cursor += align_up(bytes);
    return p;
}

// --------------------------------------------------------------- gaussian --
// Forward per-Gaussian state persisted until backward(). Scratch used only
// while building draw lists (tiles_touched, depth keys, scan) lives here
// too so a single pool covers preprocess + the matching backward.
struct GaussianState {
    float2* mean2d = nullptr;
    float4* ray_plane = nullptr;      // plane a,b,c=peak-depth fn, rsigma
    float3* normal = nullptr;         // camera-space footprint normal
    float4* conic_opacity = nullptr;  // conic (xx,xy,yy), opacity*coef
    float3* rgb = nullptr;            // SH color (unused with precomp colors)
    bool* clamped = nullptr;          // per-channel SH color clamp flags
    unsigned* tiles_touched = nullptr;
    unsigned* visible_flag = nullptr;     // 0/1, InclusiveSum input
    unsigned* visible_offset = nullptr;   // InclusiveSum of visible_flag
    unsigned* depth_key = nullptr;        // ordered float bits of depth
    ushort4* screen_bounds = nullptr;     // [x0,x1) [y0,y1) pixel bounds
    int* radius = nullptr;                // 0 = culled
    unsigned* n_visible = nullptr;        // device scalar
    unsigned* n_instances = nullptr;      // device scalar
    char* scan_scratch = nullptr;
    std::size_t scan_bytes = 0;

    static std::size_t bytes(std::size_t n, std::size_t cub_scan_bytes) {
        // from_pool aligns every array independently; mirror that exactly.
        return align_up(n * sizeof(float2)) +
               align_up(n * sizeof(float4)) * 2 +
               align_up(n * sizeof(float3)) * 2 +
               align_up(n * 3 * sizeof(bool)) +
               align_up(n * sizeof(unsigned)) * 4 +
               align_up(n * sizeof(ushort4)) +
               align_up(n * sizeof(int)) +
               align_up(2 * sizeof(unsigned)) +
               align_up(cub_scan_bytes) + kAlign;
    }

    static GaussianState from_pool(char* base, std::size_t n,
                                   std::size_t cub_scan_bytes) {
        GaussianState s;
        char* c = base;
        s.mean2d = reinterpret_cast<float2*>(take(c, n * sizeof(float2)));
        s.ray_plane = reinterpret_cast<float4*>(take(c, n * sizeof(float4)));
        s.normal = reinterpret_cast<float3*>(take(c, n * sizeof(float3)));
        s.conic_opacity = reinterpret_cast<float4*>(take(c, n * sizeof(float4)));
        s.rgb = reinterpret_cast<float3*>(take(c, n * sizeof(float3)));
        s.clamped = reinterpret_cast<bool*>(take(c, n * 3 * sizeof(bool)));
        s.tiles_touched = reinterpret_cast<unsigned*>(take(c, n * sizeof(unsigned)));
        s.visible_flag = reinterpret_cast<unsigned*>(take(c, n * sizeof(unsigned)));
        s.visible_offset = reinterpret_cast<unsigned*>(take(c, n * sizeof(unsigned)));
        s.depth_key = reinterpret_cast<unsigned*>(take(c, n * sizeof(unsigned)));
        s.screen_bounds = reinterpret_cast<ushort4*>(take(c, n * sizeof(ushort4)));
        s.radius = reinterpret_cast<int*>(take(c, n * sizeof(int)));
        s.n_visible = reinterpret_cast<unsigned*>(take(c, 2 * sizeof(unsigned)));
        s.n_instances = s.n_visible + 1;
        s.scan_scratch = take(c, cub_scan_bytes);
        s.scan_bytes = cub_scan_bytes;
        return s;
    }
};

// Backward-only per-Gaussian intermediates (fresh scratch each call).
struct GradState {
    float4* d_conic = nullptr;     // xyz: conic grads, w: opacity grad
    float4* d_ray_plane = nullptr;
    float3* d_normal = nullptr;
    float3* d_mean2d = nullptr;    // xy grads, z: refine magnitude
    float3* d_color = nullptr;     // blended RGB grads, separate from SH

    static std::size_t bytes(std::size_t n) {
        // from_pool aligns each array independently; mirror that exactly.
        return align_up(n * sizeof(float4)) * 2 +
               align_up(n * sizeof(float3)) * 3 + kAlign;
    }

    static GradState from_pool(char* base, std::size_t n) {
        GradState s;
        char* c = base;
        s.d_conic = reinterpret_cast<float4*>(take(c, n * sizeof(float4)));
        s.d_ray_plane = reinterpret_cast<float4*>(take(c, n * sizeof(float4)));
        s.d_normal = reinterpret_cast<float3*>(take(c, n * sizeof(float3)));
        s.d_mean2d = reinterpret_cast<float3*>(take(c, n * sizeof(float3)));
        s.d_color = reinterpret_cast<float3*>(take(c, n * sizeof(float3)));
        return s;
    }
};

// --------------------------------------------------------------- instance --
// FasterGS double radix sort (order preserving):
//   1. visible Gaussians stably sorted by ordered 32-bit depth key,
//   2. (gaussian,tile) instances emitted in that depth order, then stably
//      sorted by tile id. CUB DoubleBuffer ping-pongs; selector is stored
//      in ForwardResult so backward() rebuilds the same current pointer.
struct InstanceState {
    unsigned* depth_key[2] = {nullptr, nullptr};
    unsigned* depth_value[2] = {nullptr, nullptr};
    unsigned* compact_offset = nullptr;  // InclusiveSum of tiles along depth order
    unsigned* tile_key[2] = {nullptr, nullptr};
    unsigned* instance_value[2] = {nullptr, nullptr};
    char* sort_scratch = nullptr;
    std::size_t sort_bytes = 0;

    static std::size_t bytes(std::size_t visible, std::size_t instances,
                             std::size_t cub_sort_bytes) {
        // from_pool aligns every array independently; mirror that exactly.
        return align_up(visible * sizeof(unsigned)) * 5 +
               align_up(instances * sizeof(unsigned)) * 4 +
               align_up(cub_sort_bytes) + kAlign;
    }

    static InstanceState from_pool(char* base, std::size_t visible,
                                   std::size_t instances,
                                   std::size_t cub_sort_bytes) {
        InstanceState s;
        char* c = base;
        s.depth_key[0] = reinterpret_cast<unsigned*>(take(c, visible * sizeof(unsigned)));
        s.depth_key[1] = reinterpret_cast<unsigned*>(take(c, visible * sizeof(unsigned)));
        s.depth_value[0] = reinterpret_cast<unsigned*>(take(c, visible * sizeof(unsigned)));
        s.depth_value[1] = reinterpret_cast<unsigned*>(take(c, visible * sizeof(unsigned)));
        s.compact_offset = reinterpret_cast<unsigned*>(take(c, visible * sizeof(unsigned)));
        s.tile_key[0] = reinterpret_cast<unsigned*>(take(c, instances * sizeof(unsigned)));
        s.tile_key[1] = reinterpret_cast<unsigned*>(take(c, instances * sizeof(unsigned)));
        s.instance_value[0] = reinterpret_cast<unsigned*>(take(c, instances * sizeof(unsigned)));
        s.instance_value[1] = reinterpret_cast<unsigned*>(take(c, instances * sizeof(unsigned)));
        s.sort_scratch = take(c, cub_sort_bytes);
        s.sort_bytes = cub_sort_bytes;
        return s;
    }
};

// ------------------------------------------------------------------ pixel --
struct PixelState {
    unsigned* n_contrib = nullptr;   // per-pixel last contributor + 1

    static std::size_t bytes(std::size_t pixels) {
        return align_up(pixels * sizeof(unsigned)) + kAlign;
    }

    static PixelState from_pool(char* base, std::size_t pixels) {
        PixelState s;
        char* c = base;
        s.n_contrib = reinterpret_cast<unsigned*>(take(c, pixels * sizeof(unsigned)));
        return s;
    }
};

// ------------------------------------------------------------------- tile --
struct TileState {
    uint2* range = nullptr;            // sorted instance range per tile
    unsigned* max_contributor = nullptr;

    static std::size_t bytes(std::size_t tiles) {
        return align_up(tiles * sizeof(uint2)) +
               align_up(tiles * sizeof(unsigned)) + kAlign;
    }

    static TileState from_pool(char* base, std::size_t tiles) {
        TileState s;
        char* c = base;
        s.range = reinterpret_cast<uint2*>(take(c, tiles * sizeof(uint2)));
        s.max_contributor = reinterpret_cast<unsigned*>(take(c, tiles * sizeof(unsigned)));
        return s;
    }
};

// ------------------------------------------------------------------ point --
// Each visible query point lands in exactly one tile, so the instance
// arrays are sized to `points` (not a runtime instance count). That keeps
// forward/backward from_pool() layouts identical without storing the
// Gaussian instance list a second time.
struct PointState {
    float2* point2d = nullptr;
    float2* grad_point2d = nullptr;
    float* point_t = nullptr;
    unsigned* touched = nullptr;
    unsigned* tile_offset = nullptr;  // InclusiveSum of touched
    char* scan_scratch = nullptr;
    std::size_t scan_bytes = 0;

    unsigned* point_key[2] = {nullptr, nullptr};
    unsigned* point_value[2] = {nullptr, nullptr};
    char* sort_scratch = nullptr;
    std::size_t sort_bytes = 0;

    uint2* point_range = nullptr;

    static std::size_t bytes(std::size_t points, std::size_t tiles,
                             std::size_t cub_bytes) {
        // from_pool aligns every array independently; mirror that exactly.
        return align_up(points * sizeof(float2)) * 2 +
               align_up(points * sizeof(float)) +
               align_up(points * sizeof(unsigned)) * 6 +
               align_up(cub_bytes) * 2 +
               align_up(tiles * sizeof(uint2)) + kAlign;
    }

    static PointState from_pool(char* base, std::size_t points, std::size_t tiles,
                                std::size_t cub_bytes) {
        PointState s;
        char* c = base;
        s.point2d = reinterpret_cast<float2*>(take(c, points * sizeof(float2)));
        s.grad_point2d = reinterpret_cast<float2*>(take(c, points * sizeof(float2)));
        s.point_t = reinterpret_cast<float*>(take(c, points * sizeof(float)));
        s.touched = reinterpret_cast<unsigned*>(take(c, points * sizeof(unsigned)));
        s.tile_offset = reinterpret_cast<unsigned*>(take(c, points * sizeof(unsigned)));
        s.scan_scratch = take(c, cub_bytes);
        s.point_key[0] = reinterpret_cast<unsigned*>(take(c, points * sizeof(unsigned)));
        s.point_key[1] = reinterpret_cast<unsigned*>(take(c, points * sizeof(unsigned)));
        s.point_value[0] = reinterpret_cast<unsigned*>(take(c, points * sizeof(unsigned)));
        s.point_value[1] = reinterpret_cast<unsigned*>(take(c, points * sizeof(unsigned)));
        s.sort_scratch = take(c, cub_bytes);
        s.point_range = reinterpret_cast<uint2*>(take(c, tiles * sizeof(uint2)));
        s.scan_bytes = cub_bytes;
        s.sort_bytes = cub_bytes;
        return s;
    }
};

}  // namespace splat_drender::ws
