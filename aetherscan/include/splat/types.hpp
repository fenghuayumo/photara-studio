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
    unsigned sh_degree{3};

    [[nodiscard]] std::size_t size() const noexcept {
        return means.is_valid() && means.shape().rank() > 0 ? means.shape()[0] : 0;
    }
};

struct TrainingView {
    Camera camera;
    tinytensor::Tensor rgb;     // [3,H,W], linear float RGB
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
    RasterContext context;
    int rendered_instances{};
};

}  // namespace aetherscan::splat
