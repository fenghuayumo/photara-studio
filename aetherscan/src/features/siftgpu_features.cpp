#include "features/features.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(AETHERSCAN_HAS_SIFTGPU)
#include <glad/glad.h>
#include <siftgpu/SiftGPU.h>
#endif

namespace aetherscan::features {

class SiftGpuExtractor::Impl {
public:
    explicit Impl(SiftGpuOptions value) : options(value) {
        if (options.maximum_features == 0 || options.maximum_image_dimension == 0 ||
            options.maximum_orientations == 0)
            throw std::invalid_argument("Invalid SiftGPU limits");
#if defined(AETHERSCAN_HAS_SIFTGPU)
        std::vector<std::string> arguments{
            "-cuda", std::to_string(options.device_index), "-maxd",
            std::to_string(options.maximum_image_dimension), "-t",
            std::to_string(options.peak_threshold), "-e", std::to_string(options.edge_threshold),
            "-mo", std::to_string(options.maximum_orientations), "-tc2",
            std::to_string(options.maximum_features), "-v", "0"};
        std::vector<const char*> argv;
        argv.reserve(arguments.size());
        for (const auto& argument : arguments) argv.push_back(argument.c_str());
        gpu.ParseParam(static_cast<int>(argv.size()), argv.data());
        available = gpu.CreateContextGL() == SiftGPU::SIFTGPU_FULL_SUPPORTED;
#endif
    }
    SiftGpuOptions options;
    bool available{false};
#if defined(AETHERSCAN_HAS_SIFTGPU)
    SiftGPU gpu;
#endif
};

SiftGpuExtractor::SiftGpuExtractor(SiftGpuOptions options)
    : impl_(std::make_unique<Impl>(options)) {}
SiftGpuExtractor::~SiftGpuExtractor() = default;
SiftGpuExtractor::SiftGpuExtractor(SiftGpuExtractor&&) noexcept = default;
SiftGpuExtractor& SiftGpuExtractor::operator=(SiftGpuExtractor&&) noexcept = default;

bool SiftGpuExtractor::is_built() noexcept {
#if defined(AETHERSCAN_HAS_SIFTGPU)
    return true;
#else
    return false;
#endif
}
bool SiftGpuExtractor::is_available() const noexcept { return impl_->available; }

ExtractorInfo SiftGpuExtractor::info() const {
    ExtractorInfo value;
    value.thread_safe = false;
    value.accepts_gray = true;
    value.accepts_rgb = false;
    value.metric =
        impl_->options.root_sift ? DescriptorMetric::l2_root : DescriptorMetric::l2;
    value.typical_descriptor_dimension = 128;
    return value;
}

std::unique_ptr<FeatureExtractor> SiftGpuExtractor::clone() const {
    return std::make_unique<SiftGpuExtractor>(impl_->options);
}

FeatureSet SiftGpuExtractor::extract_gray(
    const std::span<const std::uint8_t> pixels, const std::uint32_t width,
    const std::uint32_t height, std::size_t row_stride) const {
    if (!impl_->available)
        throw std::runtime_error("SiftGPU is not built or its CUDA/OpenGL context is unavailable");
    if (row_stride == 0) row_stride = width;
    if (width == 0 || height == 0 || row_stride < width ||
        pixels.size() < row_stride * static_cast<std::size_t>(height))
        throw std::invalid_argument("Invalid grayscale image view");
#if defined(AETHERSCAN_HAS_SIFTGPU)
    std::vector<std::uint8_t> contiguous;
    const std::uint8_t* data = pixels.data();
    if (row_stride != width) {
        contiguous.resize(static_cast<std::size_t>(width) * height);
        for (std::uint32_t row = 0; row < height; ++row)
            std::copy_n(
                pixels.data() + static_cast<std::size_t>(row) * row_stride, width,
                contiguous.data() + static_cast<std::size_t>(row) * width);
        data = contiguous.data();
    }
    // RunSIFT mutates GPU state; API is non-const on the underlying object.
    auto* mutable_impl = const_cast<Impl*>(impl_.get());
    if (!mutable_impl->gpu.RunSIFT(
            static_cast<int>(width), static_cast<int>(height), data, GL_LUMINANCE,
            GL_UNSIGNED_BYTE))
        throw std::runtime_error("SiftGPU extraction failed");
    const int count = mutable_impl->gpu.GetFeatureNum();
    std::vector<SiftGPU::SiftKeypoint> keys(static_cast<std::size_t>(count));
    std::vector<float> descriptors(static_cast<std::size_t>(count) * 128);
    mutable_impl->gpu.GetFeatureVector(keys.data(), descriptors.data());
    FeatureSet result;
    result.image_width = width;
    result.image_height = height;
    result.descriptor_dimension = 128;
    result.metric = info().metric;
    result.extractor_name = std::string(name());
    result.keypoints.reserve(keys.size());
    for (const auto& key : keys)
        result.keypoints.push_back({key.x, key.y, key.s, key.o, 0.0F});
    result.descriptors = std::move(descriptors);
    if (impl_->options.root_sift) {
        for (std::size_t row = 0; row < result.keypoints.size(); ++row) {
            float* descriptor = result.descriptors.data() + row * 128;
            double sum = 0.0;
            for (int column = 0; column < 128; ++column)
                sum += (std::max)(0.0F, descriptor[column]);
            const float inverse = static_cast<float>(1.0 / (std::max)(sum, 1e-12));
            for (int column = 0; column < 128; ++column)
                descriptor[column] =
                    std::sqrt((std::max)(0.0F, descriptor[column]) * inverse);
        }
    }
    return result;
#else
    (void)pixels;
    (void)width;
    (void)height;
    (void)row_stride;
    return {};
#endif
}

void register_siftgpu_feature_backends() {
    register_extractor("siftgpu", [] {
        auto extractor = std::make_unique<SiftGpuExtractor>();
        if (!extractor->is_available())
            throw std::runtime_error("SiftGPU backend is not available on this machine");
        return std::unique_ptr<FeatureExtractor>(std::move(extractor));
    });
}

}  // namespace aetherscan::features
