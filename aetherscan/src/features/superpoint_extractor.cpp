#include "aetherscan/features/features.hpp"

#include <stdexcept>
#include <utility>

namespace aetherscan::features {

SuperPointExtractor::SuperPointExtractor(SuperPointOptions options)
    : options_(std::move(options)) {}
SuperPointExtractor::~SuperPointExtractor() = default;
SuperPointExtractor::SuperPointExtractor(SuperPointExtractor&&) noexcept = default;
SuperPointExtractor& SuperPointExtractor::operator=(SuperPointExtractor&&) noexcept = default;

bool SuperPointExtractor::is_implemented() noexcept { return false; }

ExtractorInfo SuperPointExtractor::info() const {
    ExtractorInfo value;
    value.thread_safe = true;
    value.accepts_gray = true;
    value.accepts_rgb = false;
    value.metric = DescriptorMetric::inner_product;
    value.typical_descriptor_dimension = 256;
    return value;
}

std::unique_ptr<FeatureExtractor> SuperPointExtractor::clone() const {
    return std::make_unique<SuperPointExtractor>(options_);
}

FeatureSet SuperPointExtractor::extract_gray(
    std::span<const std::uint8_t> /*pixels*/, std::uint32_t /*width*/,
    std::uint32_t /*height*/, std::size_t /*row_stride*/) const {
    throw std::runtime_error(
        "SuperPointExtractor is a registered extension point but not implemented yet. "
        "Add ONNX/TensorRT inference in src/features/superpoint_extractor.cpp and set "
        "is_implemented() to true. Prefer create_extractor(\"superpoint\") once ready.");
}

void register_superpoint_feature_backends() {
    register_extractor("superpoint", [] {
        return std::unique_ptr<FeatureExtractor>(std::make_unique<SuperPointExtractor>());
    });
}

}  // namespace aetherscan::features
