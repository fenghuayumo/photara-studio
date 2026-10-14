#pragma once
#include "inference/geometry_prior.hpp"
#include "inference/options.hpp"
#include "io/image.hpp"

namespace photara::inference {
struct GeometryInput {
    const io::RgbImage& image; // Pinhole RGB; caller handles camera warping.
    std::optional<PinholeIntrinsics> intrinsics;
};
// Caller-owned model sessions are not concurrently callable.
class GeometryPredictor {
public:
    virtual ~GeometryPredictor() = default;
    [[nodiscard]] virtual const ModelInfo& model_info() const noexcept = 0;
    [[nodiscard]] virtual const DeviceInfo& device_info() const noexcept = 0;
    virtual GeometryPrior predict(
        const GeometryInput& input, const PredictOptions& options = {}) = 0;
};
} // namespace photara::inference
