// Public host API of splat_drender.
//
// The rasterizer keeps its forward intermediates in caller-owned device
// workspaces ("pools") described by buffers.h. forward() fills them,
// backward() consumes them; both must observe the same Gaussian/camera
// inputs and pool contents between the two calls.
#pragma once

#include "splat_drender/camera.h"

#include <cstddef>
#include <functional>

namespace splat_drender {

struct CameraView {
    int width = 0, height = 0;
    float fx = 1.f, fy = 1.f, cx = 0.f, cy = 0.f;
    CameraMode mode = CameraMode::pinhole;
    float k1 = 0.f, k2 = 0.f, k3 = 0.f, k4 = 0.f;
    // Column-major 4x4 world-to-camera transform (16 floats, device).
    const float* world_to_camera = nullptr;
    // Camera center in world space (3 floats, device).
    const float* center = nullptr;
};

struct Gaussians {
    int count = 0;
    const float* means = nullptr;        // [N,3]
    // Exactly one of sh / colors must be non-null.
    const float* sh = nullptr;           // [N,sh_bases,3]
    const float* colors = nullptr;       // [N,3] precomputed view color
    const float* opacities = nullptr;    // [N], already activated
    // Exactly one of (scales, rotations) / covariances must be provided.
    const float* scales = nullptr;       // [N,3], already activated
    const float* rotations = nullptr;    // [N,4], scalar-first
    const float* covariances = nullptr;  // [N,6], upper triangular
    int sh_degree = 0;
    int sh_bases = 0;
};

struct RenderSettings {
    float background[3] = {0.f, 0.f, 0.f};
    float scale_modifier = 1.f;
    float kernel_size = 0.f;
    bool need_depth = true;
    bool debug = false;
    // Point queries: sort a fixed-size list with a sentinel for invalid points,
    // eliminating the point-count host readback. SampleCounts::point_instances
    // is -1 (not collected) on this path; per-tile ranges remain exact.
    bool device_point_lists = false;
};

struct RenderOutputs {
    float* color = nullptr;         // [3,H,W]
    float* alpha = nullptr;         // [H,W]   1 - final transmittance
    float* median_depth = nullptr;  // [H,W]   camera-space Z
    float* normal = nullptr;        // [3,H,W] camera space, alpha-normalized
    float* visibility = nullptr;    // [N]     contributed >=1 accepted splat
    int* radii = nullptr;           // [N]     pixel radius, 0 = invisible
};

// Read-only view of the forward outputs consumed by backward().
struct ForwardOutputsView {
    const float* alpha = nullptr;         // [H,W]
    const float* median_depth = nullptr;  // [H,W]
    const float* normal = nullptr;        // [3,H,W]
    const int* radii = nullptr;           // [N]
};

struct LossGradients {
    const float* color = nullptr;         // [3,H,W]
    const float* alpha = nullptr;         // [H,W]
    const float* median_depth = nullptr;  // [H,W]
    const float* normal = nullptr;        // [3,H,W]
};

struct ModelGradients {
    float* means = nullptr;        // [N,3]
    float* sh = nullptr;           // [N,sh_bases,3]
    float* colors = nullptr;       // [N,3]
    float* opacities = nullptr;    // [N]
    float* scales = nullptr;       // [N,3]
    float* rotations = nullptr;    // [N,4]
    float* covariances = nullptr;  // [N,6]
    // Brush-style densification signal: accumulated per-Gaussian
    // ||dL/d(mean2d)|| * (W,H) / final alpha. Null disables it.
    float* refine_weight = nullptr;  // [N]
};

// Forward outputs that backward() needs again.
struct ForwardResult {
    int instance_count = 0;
    int visible_count = 0;
    // CUB DoubleBuffer selector after the tile-id sort (0 or 1).
    int instance_selector = 0;
};

struct SampleOutputs {
    float3* ray_points = nullptr;   // [P] median-depth point along pixel ray
    float* median_depth = nullptr;  // [P]
    unsigned* n_contrib = nullptr;  // [P] last contributing instance + 1
    bool* inside = nullptr;         // [P]
};

// Read-only view of the sample outputs consumed by sample_depth_backward().
struct SampleOutputsView {
    const float* median_depth = nullptr;  // [P]
    const unsigned* n_contrib = nullptr;  // [P]
    const bool* inside = nullptr;         // [P]
};

struct OccupancyOutputs {
    float* occupancy = nullptr;  // [P] 1 - transmittance along the ray
    bool* inside = nullptr;      // [P]
};

struct SampleGradients {
    float3* points = nullptr;  // [P] dL/d(world query point)
};

// Workspace pools. Each callback grows its pool to the requested number of
// bytes and returns the current base pointer (like a cached cudaMalloc).
struct WorkspacePools {
    std::function<char*(std::size_t)> gaussian;  // per-Gaussian forward state
    std::function<char*(std::size_t)> grad;      // per-Gaussian backward scratch
    std::function<char*(std::size_t)> instance;  // sorted draw/sort scratch
    std::function<char*(std::size_t)> pixel;     // per-pixel forward state
    std::function<char*(std::size_t)> tile;      // per-tile state
    std::function<char*(std::size_t)> point;     // point-query state
};

class Rasterizer {
public:
    // Renders one view. Returns the number of (Gaussian, tile) instances
    // drawn; keep it for the matching backward() call.
    static ForwardResult forward(
        const WorkspacePools& pools, const Gaussians& g, const CameraView& cam,
        const RenderSettings& s, const RenderOutputs& out);

    // Consumes the pools filled by forward() plus per-pixel loss gradients
    // and accumulates model gradients (buffers are NOT zeroed here).
    static void backward(
        const WorkspacePools& pools, const Gaussians& g, const CameraView& cam,
        const RenderSettings& s, const ForwardResult& fwd,
        const ForwardOutputsView& fwd_out, const LossGradients& dL,
        const ModelGradients& grads);

    // Median-depth sampling of world-space query points (multi-view
    // photometric / geometric losses).
    struct SampleCounts {
        int point_instances = 0; // -1 when device_point_lists skips count readback
        int gaussian_instances = 0;
        int visible_count = 0;
        int tile_blocks = 0;
        int instance_selector = 0;
        int point_selector = 0;
    };
    static SampleCounts sample_depth(
        const WorkspacePools& pools, const Gaussians& g, const CameraView& cam,
        const RenderSettings& s, const float* world_points, int point_count,
        const SampleOutputs& out);

    static void sample_depth_backward(
        const WorkspacePools& pools, const Gaussians& g, const CameraView& cam,
        const RenderSettings& s, const float* world_points, int point_count,
        const SampleCounts& counts, const SampleOutputsView& fwd,
        const float3* dL_dray_points, const SampleGradients& point_grads,
        const ModelGradients& grads);

    // Occupancy = 1 - transmittance at each query point's ray position.
    static void evaluate_occupancy(
        const WorkspacePools& pools, const Gaussians& g, const CameraView& cam,
        const RenderSettings& s, const float* world_points, int point_count,
        const OccupancyOutputs& out);
};

}  // namespace splat_drender
