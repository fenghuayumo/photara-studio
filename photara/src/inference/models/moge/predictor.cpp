#include "inference/models/moge.hpp"
#include "decoder.hpp"
#include "preprocess.hpp"
#include <cmath>
#include <stdexcept>

namespace photara::inference {
namespace {
MogeOptions checked(MogeOptions options) {
    if (options.runtime.backend != Backend::vulkan || options.runtime.allow_cpu_fallback)
        throw std::invalid_argument("MoGe prediction requires Vulkan with CPU fallback disabled");
    if (!options.num_tokens || options.num_tokens > 4096)
        throw std::invalid_argument("MoGe token budget must lie in 1..4096");
    if (!std::isfinite(options.mask_threshold) || options.mask_threshold <= 0 || options.mask_threshold >= 1)
        throw std::invalid_argument("MoGe mask threshold must lie strictly between zero and one");
    return options;
}
class MogePredictor final : public GeometryPredictor {
public:
    MogePredictor(const std::filesystem::path& path, const MogeOptions& options)
        : options_(checked(options)), runtime_(options_.runtime),
          weights_(path, runtime_.backend(), options_.maximum_tensor_bytes) {}
    const ModelInfo& model_info() const noexcept override { return weights_.info().model; }
    const DeviceInfo& device_info() const noexcept override { return runtime_.device(); }
    GeometryPrior predict(const GeometryInput& input, const PredictOptions& options) override {
        const auto boundary = [&](float progress) {
            if (options.stop.stop_requested()) throw std::runtime_error("MoGe prediction cancelled");
            if (options.progress) options.progress(progress);
            if (options.stop.stop_requested()) throw std::runtime_error("MoGe prediction cancelled");
        };
        boundary(0.F);
        if (input.intrinsics) {
            const auto& k = *input.intrinsics;
            if (!std::isfinite(k.fx) || k.fx <= 0 || !std::isfinite(k.fy) || k.fy <= 0 ||
                !std::isfinite(k.cx) || !std::isfinite(k.cy))
                throw std::invalid_argument("Invalid MoGe input intrinsics");
        }
        auto image = detail::preprocess_moge(input.image, options_.num_tokens);
        boundary(0.1F);
        auto encoded = detail::encode_moge_vitb(weights_, runtime_, image.rows, image.columns, image.normalized_rgb);
        boundary(0.5F);
        auto raw = detail::decode_moge_vitb(weights_, runtime_, encoded,
            static_cast<float>(input.image.width) / input.image.height);
        boundary(0.9F);
        MogeGeometryOptions geometry;
        geometry.width = input.image.width;
        geometry.height = input.image.height;
        geometry.intrinsics = input.intrinsics;
        geometry.mask_threshold = options_.mask_threshold;
        auto result = recover_moge_geometry(raw, geometry, model_info(), device_info());
        if (result.recovery.status != MogeRecoveryStatus::recovered)
            throw std::runtime_error("MoGe geometry recovery failed: insufficient or degenerate support");
        result.prior.preprocessing_revision = "moge2-rgb-aa-imagenet-geometry-v1";
        result.prior.validate();
        boundary(1.F);
        return std::move(result.prior);
    }
private:
    MogeOptions options_;
    detail::GgmlRuntime runtime_; // Must outlive the weight buffer.
    detail::MogeWeights weights_;
};
}
std::unique_ptr<GeometryPredictor> create_moge_predictor(
    const std::filesystem::path& path, const MogeOptions& options) {
    return std::make_unique<MogePredictor>(path, options);
}
} // namespace photara::inference
