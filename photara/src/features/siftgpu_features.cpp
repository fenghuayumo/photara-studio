#include "features/features.hpp"
#if defined(PHOTARA_MATCH_HAS_CUDA)
#include "matcher_cuda.hpp"
#include "sift_cuda.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace photara::features {
namespace {

FeatureSet quantized_copy(const FeatureSet& source) {
    FeatureSet copy = source;
    copy.compress_descriptors_u8();
    return copy;
}

}  // namespace

class SiftGpuExtractor::Impl {
public:
    explicit Impl(SiftGpuOptions value)
        : options(std::move(value)), owner_thread(std::this_thread::get_id()) {
#if defined(PHOTARA_MATCH_HAS_CUDA)
        engine = std::make_unique<SiftCudaEngine>(options);
        available = engine->available();
#else
        (void)options;
#endif
    }

    SiftGpuOptions options;
    std::thread::id owner_thread;
    bool available{false};
#if defined(PHOTARA_MATCH_HAS_CUDA)
    std::unique_ptr<SiftCudaEngine> engine;
#endif
};

SiftGpuExtractor::SiftGpuExtractor(SiftGpuOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
SiftGpuExtractor::~SiftGpuExtractor() = default;
SiftGpuExtractor::SiftGpuExtractor(SiftGpuExtractor&&) noexcept = default;
SiftGpuExtractor& SiftGpuExtractor::operator=(SiftGpuExtractor&&) noexcept = default;

bool SiftGpuExtractor::is_built() noexcept {
#if defined(PHOTARA_MATCH_HAS_CUDA)
    return true;
#else
    return false;
#endif
}

bool SiftGpuExtractor::is_available() const noexcept { return impl_->available; }

ExtractorInfo SiftGpuExtractor::info() const {
    ExtractorInfo value;
    value.thread_safe = false;
    value.thread_affine = true;
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
    auto result = extract_gray_deferred(pixels, width, height, row_stride);
    finalize_descriptors(result);
    return result;
}

FeatureSet SiftGpuExtractor::extract_gray_deferred(
    const std::span<const std::uint8_t> pixels, const std::uint32_t width,
    const std::uint32_t height, std::size_t row_stride) const {
    if (!impl_->available)
        throw std::runtime_error("CUDA SIFT extractor is not available");
    if (std::this_thread::get_id() != impl_->owner_thread)
        throw std::runtime_error(
            "SiftGPU extractor must run on its CUDA context owner thread");
#if defined(PHOTARA_MATCH_HAS_CUDA)
    return impl_->engine->extract(pixels, width, height, row_stride);
#else
    (void)pixels;
    (void)width;
    (void)height;
    (void)row_stride;
    return {};
#endif
}

void SiftGpuExtractor::finalize_descriptors(FeatureSet& result) const {
    if (result.metric == DescriptorMetric::l2_root) return;
    if (result.metric != DescriptorMetric::l2 || result.descriptor_dimension != 128 ||
        result.descriptors.size() != result.keypoints.size() * 128)
        throw std::invalid_argument("Invalid deferred SiftGPU descriptors");
    if (!impl_->options.root_sift) return;
    for (std::size_t row = 0; row < result.keypoints.size(); ++row) {
        float* descriptor = result.descriptors.data() + row * 128;
        double sum = 0.0;
        for (int column = 0; column < 128; ++column)
            sum += (std::max)(0.0F, descriptor[column]);
        const float inverse = static_cast<float>(1.0 / (std::max)(sum, 1e-12));
        for (int column = 0; column < 128; ++column)
            descriptor[column] = std::sqrt((std::max)(0.0F, descriptor[column]) * inverse);
    }
    result.metric = DescriptorMetric::l2_root;
}

class SiftGpuMatcher::Impl {
public:
    explicit Impl(SiftGpuMatcherOptions value)
        : options(value), owner_thread(std::this_thread::get_id()) {
        if (!(options.ratio_threshold > 0.F && options.ratio_threshold <= 1.F) ||
            options.maximum_features < 2)
            throw std::invalid_argument("Invalid SiftGPU matcher options");
#if defined(PHOTARA_MATCH_HAS_CUDA)
        available = sift_cuda_device_available(options.device_index);
#else
        (void)options;
#endif
    }

    SiftGpuMatcherOptions options;
    bool available{false};
    std::thread::id owner_thread;
    mutable std::unique_ptr<FeatureMatcher> native;
};

SiftGpuMatcher::SiftGpuMatcher(SiftGpuMatcherOptions options)
    : impl_(std::make_shared<Impl>(options)) {}
SiftGpuMatcher::SiftGpuMatcher(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
SiftGpuMatcher::~SiftGpuMatcher() = default;
SiftGpuMatcher::SiftGpuMatcher(SiftGpuMatcher&&) noexcept = default;
SiftGpuMatcher& SiftGpuMatcher::operator=(SiftGpuMatcher&&) noexcept = default;

bool SiftGpuMatcher::is_built() noexcept {
#if defined(PHOTARA_MATCH_HAS_CUDA)
    return true;
#else
    return false;
#endif
}

bool SiftGpuMatcher::is_available() const noexcept { return impl_->available; }

std::unique_ptr<FeatureMatcher> SiftGpuMatcher::clone() const {
    return std::unique_ptr<FeatureMatcher>(new SiftGpuMatcher(impl_));
}

void SiftGpuMatcher::clear_prepared() {
    if (impl_->native) impl_->native->clear_prepared();
}

std::vector<MatchSet> SiftGpuMatcher::match_batch(std::span<const Pair> pairs) const {
    if (!impl_->available)
        throw std::runtime_error("CUDA matcher is not available");
    if (std::this_thread::get_id() != impl_->owner_thread)
        throw std::runtime_error("CUDA matcher must run on its owner thread");
#if defined(PHOTARA_MATCH_HAS_CUDA)
    if (!impl_->native) impl_->native = make_native_cuda_matcher(impl_->options);
    const bool uint8 = std::all_of(pairs.begin(), pairs.end(), [](const Pair& pair) {
        return pair.first->storage == DescriptorStorage::uint8 &&
               pair.second->storage == DescriptorStorage::uint8;
    });
    if (uint8) return impl_->native->match_batch(pairs);

    std::vector<FeatureSet> owned(pairs.size() * 2);
    std::vector<Pair> converted(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const FeatureSet* first = pairs[i].first;
        const FeatureSet* second = pairs[i].second;
        if (first->storage != DescriptorStorage::uint8) {
            owned[i * 2] = quantized_copy(*first);
            first = &owned[i * 2];
        }
        if (second->storage != DescriptorStorage::uint8) {
            owned[i * 2 + 1] = quantized_copy(*second);
            second = &owned[i * 2 + 1];
        }
        converted[i] = Pair{first, second};
    }
    return impl_->native->match_batch(converted);
#else
    (void)pairs;
    return {};
#endif
}

MatchSet SiftGpuMatcher::match(const FeatureSet& query, const FeatureSet& train) const {
    const Pair pair{&query, &train};
    return std::move(match_batch(std::span(&pair, 1))[0]);
}

void register_siftgpu_feature_backends() {
    register_extractor("siftgpu", [] {
        auto extractor = std::make_unique<SiftGpuExtractor>();
        if (!extractor->is_available())
            throw std::runtime_error("CUDA SIFT extractor is not available");
        return std::unique_ptr<FeatureExtractor>(std::move(extractor));
    });
    auto make_gpu_mutual_ratio = [] {
        auto matcher = std::make_unique<SiftGpuMatcher>();
        if (!matcher->is_available())
            throw std::runtime_error(
                "gpu_mutual_ratio matcher is not available on this machine");
        return std::unique_ptr<FeatureMatcher>(std::move(matcher));
    };
    register_matcher("gpu_mutual_ratio", make_gpu_mutual_ratio);
    register_matcher("siftgpu", make_gpu_mutual_ratio);
}

}  // namespace photara::features
