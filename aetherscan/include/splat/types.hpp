#pragma once

#include "internal/tensor_impl.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aetherscan::splat {

// Camera matrices follow AetherScan's world-to-camera convention. The
// rasterizer-facing matrix is stored transposed/contiguous, matching the
// column-major layout used by the CUDA GGGS kernels.
struct Camera {
    std::array<float, 16> world_to_camera{};
    std::array<float, 3> position{};
    float fx{1.F};
    float fy{1.F};
    float cx{0.F};
    float cy{0.F};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct GaussianModel {
    tinytensor::Tensor means;           // [N,3], world space
    tinytensor::Tensor log_scales;      // [N,3]
    tinytensor::Tensor quaternions;     // [N,4], scalar-first
    tinytensor::Tensor opacity_logits;  // [N,1]
    tinytensor::Tensor sh;              // [N,K,3]
    // GaussianWrapping normal-field features. xyz is a learnable direction;
    // w is an orientation logit. The surface normal used by training/PAM is
    // normalize(xyz) * tanh(w). Kept optional for legacy 3DGS PLY files.
    tinytensor::Tensor normal_features; // [N,4], optional
    // Mip-Splatting screen-space footprint converted to a world-space radius.
    // This is derived state (no gradient) and is recomputed as means change.
    tinytensor::Tensor filter_3d;        // [N,1], optional
    unsigned sh_degree{3};

    [[nodiscard]] std::size_t size() const noexcept {
        return means.is_valid() && means.shape().rank() > 0 ? means.shape()[0] : 0;
    }
};

struct TrainingView {
    Camera camera;
    tinytensor::Tensor rgb;     // [3,H,W], linear float RGB
    tinytensor::Tensor gray;    // [H,W], optional NCC luminance
    tinytensor::Tensor depth;   // [H,W], zero when unavailable
    tinytensor::Tensor normal;  // [3,H,W], camera space
    tinytensor::Tensor mask;    // [H,W], 0 or 1
    bool has_mask{false};
};

struct ModelGradients {
    tinytensor::Tensor means;
    tinytensor::Tensor log_scales;
    tinytensor::Tensor quaternions;
    tinytensor::Tensor opacity_logits;
    tinytensor::Tensor sh;
    tinytensor::Tensor normal_features;
    // Gradient of RasterizeOptions::colors_precomp. This is populated only
    // for an explicit precomputed-color render (normal-field training).
    tinytensor::Tensor colors_precomp;
    // Per-Gaussian image-plane refine weight emitted by the GGGS backward
    // kernel. This drives default/ADC+/ADC-IGS densification.
    tinytensor::Tensor refine_weight;
};

struct RasterContextImpl;

struct RasterContext {
    std::shared_ptr<RasterContextImpl> impl;
};

struct RenderResult {
    tinytensor::Tensor color;         // [3,H,W]
    tinytensor::Tensor alpha;         // [H,W]
    tinytensor::Tensor median_depth;  // [H,W]
    tinytensor::Tensor normal;        // [3,H,W]
    tinytensor::Tensor radii;         // [N], int32
    // [N], 1 only when the Gaussian survives alpha/transmittance tests and
    // contributes to at least one pixel. This is stricter than radii > 0,
    // which only means that the Gaussian projected into the camera frustum.
    tinytensor::Tensor visibility;
    RasterContext context;
    int rendered_instances{};
};

struct DepthSampleContextImpl;

struct DepthSampleResult {
    tinytensor::Tensor camera_points;  // [P,3], neighbour camera space
    tinytensor::Tensor inside;         // [P], bool
    std::shared_ptr<DepthSampleContextImpl> context;
};

struct DepthSampleGradients {
    ModelGradients model;
    tinytensor::Tensor points;         // [P,3], world space
};

struct OccupancyResult {
    // GaussianWrapping calls this alpha_integrated: 1 - transmittance at each
    // world-space query point along the selected camera ray.
    tinytensor::Tensor occupancy;  // [P]
    tinytensor::Tensor inside;     // [P], bool camera-frustum membership
};

}  // namespace aetherscan::splat
