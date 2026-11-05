#include "inference/models/moge.hpp"
#include "decoder.hpp"
#include "preprocess.hpp"
#include "postprocess_gpu.hpp"
#include <cmath>
#include <chrono>
#include "core/logging.hpp"
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
          weights_(path, runtime_.backend(), options_.maximum_tensor_bytes) {
        if (options_.gpu_geometry) geometry_gpu_ = detail::create_moge_geometry_gpu(device_info());
    }
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
        const auto start = std::chrono::steady_clock::now();
        auto image = detail::preprocess_moge(input.image, options_.num_tokens);
        boundary(0.1F);
        const auto preprocessed = std::chrono::steady_clock::now();
        auto encoded = detail::encode_moge_vitb(weights_, runtime_, image.rows, image.columns, image.normalized_rgb, false, &encoder_cache_);
        const auto encoded_at = std::chrono::steady_clock::now();
        boundary(0.5F);
        auto raw = detail::decode_moge_vitb(weights_, runtime_, encoded,
            static_cast<float>(input.image.width) / input.image.height, false, &decoder_cache_);
        const auto decoded_at = std::chrono::steady_clock::now();
        boundary(0.9F);
        MogeGeometryOptions geometry;
        geometry.width = input.image.width;
        geometry.height = input.image.height;
        geometry.intrinsics = input.intrinsics;
        geometry.mask_threshold = options_.mask_threshold;
        auto result = geometry_gpu_
            ? detail::recover_moge_geometry_gpu(raw, geometry, model_info(), device_info(), *geometry_gpu_)
            : recover_moge_geometry(raw, geometry, model_info(), device_info());
        if (result.recovery.status != MogeRecoveryStatus::recovered)
            throw std::runtime_error("MoGe geometry recovery failed: insufficient or degenerate support");
        result.prior.preprocessing_revision = "moge2-rgb-aa-imagenet-geometry-v1";
        result.prior.validate();
        const auto finished = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        core::Logger::instance().info("MoGe timing grid=", image.columns, "x", image.rows,
            " preprocess_ms=", ms(start, preprocessed), " encoder_ms=", ms(preprocessed, encoded_at),
            " decoder_ms=", ms(encoded_at, decoded_at), " geometry_ms=", ms(decoded_at, finished),
            " total_ms=", ms(start, finished));
        boundary(1.F);
        return std::move(result.prior);
    }
private:
    MogeOptions options_;
    detail::GgmlRuntime runtime_; // Must outlive the weight buffer.
    detail::MogeWeights weights_;
    detail::MogeGraphCache encoder_cache_, decoder_cache_;
    std::unique_ptr<detail::MogeGeometryGpu> geometry_gpu_;
};
}
std::unique_ptr<GeometryPredictor> create_moge_predictor(
    const std::filesystem::path& path, const MogeOptions& options) {
    return std::make_unique<MogePredictor>(path, options);
}
} // namespace photara::inference
