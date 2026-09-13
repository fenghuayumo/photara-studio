#include "features/features.hpp"
#if defined(AETHERSCAN_MATCH_HAS_CUDA)
#include "matcher_cuda.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(AETHERSCAN_HAS_SIFTGPU)
#include <glad/glad.h>
#include <siftgpu/SiftGPU.h>
#endif

namespace aetherscan::features {

class SiftGpuExtractor::Impl {
public:
    explicit Impl(SiftGpuOptions value)
        : options(value), owner_thread(std::this_thread::get_id()) {
        if (options.maximum_features == 0 || options.maximum_image_dimension == 0 ||
            options.maximum_orientations == 0 || options.octave_layers == 0)
            throw std::invalid_argument("Invalid SiftGPU limits");
#if defined(AETHERSCAN_HAS_SIFTGPU)
        const unsigned octave_factor =
            1U << static_cast<unsigned>(std::max(0, -options.first_octave));
        std::vector<std::string> arguments{
            "-cuda", std::to_string(options.device_index), "-maxd",
            std::to_string(options.maximum_image_dimension * octave_factor), "-t",
            std::to_string(options.peak_threshold), "-e", std::to_string(options.edge_threshold),
            "-mo", std::to_string(options.maximum_orientations), "-tc2",
            std::to_string(options.maximum_features), "-fo",
            std::to_string(options.first_octave), "-d",
            std::to_string(options.octave_layers), "-v", "0"};
        std::vector<const char*> argv;
        argv.reserve(arguments.size());
        for (const auto& argument : arguments) argv.push_back(argument.c_str());
        gpu.ParseParam(static_cast<int>(argv.size()), argv.data());
        available = gpu.CreateContextGL() == SiftGPU::SIFTGPU_FULL_SUPPORTED;
#endif
    }
    SiftGpuOptions options;
    bool available{false};
    std::thread::id owner_thread;
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
    if (!impl_->available)
        throw std::runtime_error("SiftGPU is not built or its CUDA/OpenGL context is unavailable");
    if (std::this_thread::get_id() != impl_->owner_thread)
        throw std::runtime_error(
            "SiftGPU extractor must run on its CUDA context owner thread");
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

class SiftGpuMatcher::Impl {
public:
    explicit Impl(SiftGpuMatcherOptions value)
        : options(value), owner_thread(std::this_thread::get_id()) {
        if (!(options.ratio_threshold > 0.F &&
              options.ratio_threshold <= 1.F) ||
            options.maximum_features < 2)
            throw std::invalid_argument("Invalid SiftGPU matcher options");
#if defined(AETHERSCAN_HAS_SIFTGPU)
        (void)options.device_index;
        gpu.reset(CreateNewSiftMatchGPU(
            static_cast<int>(options.maximum_features)));
        if (!gpu) return;
        gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
        std::array<std::string, 2> device_arguments{
            "-cuda", std::to_string(options.device_index)};
        std::array<char*, 2> device_argv{
            device_arguments[0].data(), device_arguments[1].data()};
        gpu->SetDeviceParam(
            static_cast<int>(device_argv.size()), device_argv.data());
        if (!gpu->CreateContextGL()) {
            gpu.reset();
            return;
        }
        // SiftMatchCU grows its textures in SetDescriptors/GetSiftMatch.
        // Allocate(maximum_features) eagerly reserves the worst-case square
        // distance matrix even when this instance only uses the native path.
        available = true;
#endif
    }

    SiftGpuMatcherOptions options;
    bool available{false};
    std::thread::id owner_thread;
    mutable std::mutex mutex;
    mutable std::unique_ptr<FeatureMatcher> native;
    // Skip SetDescriptors when the same FeatureSet remains bound to a slot
    // (openMVS prevImageID1/prevImageID2 cache).
    mutable std::uint64_t slot0_identity{0};
    mutable std::uint64_t slot0_generation{
        std::numeric_limits<std::uint64_t>::max()};
    mutable std::uint64_t slot1_identity{0};
    mutable std::uint64_t slot1_generation{
        std::numeric_limits<std::uint64_t>::max()};
#if defined(AETHERSCAN_HAS_SIFTGPU)
    std::unique_ptr<SiftMatchGPU> gpu;
#endif
};

SiftGpuMatcher::SiftGpuMatcher(SiftGpuMatcherOptions options)
    : impl_(std::make_shared<Impl>(options)) {}
SiftGpuMatcher::SiftGpuMatcher(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SiftGpuMatcher::~SiftGpuMatcher() = default;
SiftGpuMatcher::SiftGpuMatcher(SiftGpuMatcher&&) noexcept = default;
SiftGpuMatcher& SiftGpuMatcher::operator=(SiftGpuMatcher&&) noexcept = default;

bool SiftGpuMatcher::is_built() noexcept {
#if defined(AETHERSCAN_HAS_SIFTGPU)
    return true;
#else
    return false;
#endif
}

bool SiftGpuMatcher::is_available() const noexcept {
    return impl_->available;
}

std::unique_ptr<FeatureMatcher> SiftGpuMatcher::clone() const {
    return std::unique_ptr<FeatureMatcher>(new SiftGpuMatcher(impl_));
}

void SiftGpuMatcher::clear_prepared() { impl_->native.reset(); }

std::vector<MatchSet> SiftGpuMatcher::match_batch(std::span<const Pair> pairs) const {
    if (std::this_thread::get_id() != impl_->owner_thread)
        throw std::runtime_error("SiftGPU matcher must run on its CUDA context owner thread");
#if defined(AETHERSCAN_MATCH_HAS_CUDA)
    if (impl_->available && impl_->options.native_cuda && !pairs.empty() &&
        std::all_of(pairs.begin(), pairs.end(), [](const Pair& pair) {
            return pair.first->storage == DescriptorStorage::uint8 &&
                   pair.second->storage == DescriptorStorage::uint8;
        })) {
        if (!impl_->native) impl_->native = make_native_cuda_matcher(impl_->options);
        return impl_->native->match_batch(pairs);
    }
#endif
    return FeatureMatcher::match_batch(pairs);
}

MatchSet SiftGpuMatcher::match(
    const FeatureSet& query, const FeatureSet& train) const {
#if defined(AETHERSCAN_MATCH_HAS_CUDA)
    if (impl_->available && impl_->options.native_cuda &&
        query.storage == DescriptorStorage::uint8 && train.storage == DescriptorStorage::uint8) {
        const Pair pair{&query, &train};
        return std::move(match_batch(std::span(&pair, 1))[0]);
    }
#endif
    if (!impl_->available)
        throw std::runtime_error(
            "SiftGPU matcher is not built or its CUDA context is unavailable");
    if (std::this_thread::get_id() != impl_->owner_thread)
        throw std::runtime_error(
            "SiftGPU matcher must run on its CUDA context owner thread");
    query.validate();
    train.validate();
    if (query.descriptor_dimension != 128 ||
        train.descriptor_dimension != 128)
        throw std::invalid_argument(
            "SiftGPU matcher requires 128-dimensional descriptors");
    if (query.keypoints.empty() || train.keypoints.empty()) return {};
    if (query.keypoints.size() > impl_->options.maximum_features ||
        train.keypoints.size() > impl_->options.maximum_features)
        throw std::invalid_argument(
            "SiftGPU matcher feature count exceeds configured maximum");
#if defined(AETHERSCAN_HAS_SIFTGPU)
    std::lock_guard lock(impl_->mutex);
    if (impl_->slot0_identity != query.descriptor_identity ||
        impl_->slot0_generation != query.descriptor_generation) {
        if (query.storage == DescriptorStorage::uint8) {
            if (query.descriptors_u8.empty()) return {};
            impl_->gpu->SetDescriptors(
                0, static_cast<int>(query.keypoints.size()),
                query.descriptors_u8.data());
        } else {
            if (query.descriptors.empty()) return {};
            impl_->gpu->SetDescriptors(
                0, static_cast<int>(query.keypoints.size()),
                query.descriptors.data());
        }
        impl_->slot0_identity = query.descriptor_identity;
        impl_->slot0_generation = query.descriptor_generation;
    }
    if (impl_->slot1_identity != train.descriptor_identity ||
        impl_->slot1_generation != train.descriptor_generation) {
        if (train.storage == DescriptorStorage::uint8) {
            if (train.descriptors_u8.empty()) return {};
            impl_->gpu->SetDescriptors(
                1, static_cast<int>(train.keypoints.size()),
                train.descriptors_u8.data());
        } else {
            if (train.descriptors.empty()) return {};
            impl_->gpu->SetDescriptors(
                1, static_cast<int>(train.keypoints.size()),
                train.descriptors.data());
        }
        impl_->slot1_identity = train.descriptor_identity;
        impl_->slot1_generation = train.descriptor_generation;
    }
    const int maximum_matches = static_cast<int>(
        impl_->options.mutual_check
            ? std::min(query.keypoints.size(), train.keypoints.size())
            : query.keypoints.size());
    std::vector<std::array<std::uint32_t, 2>> pairs(
        static_cast<std::size_t>(maximum_matches));
    const int match_count = impl_->gpu->GetSiftMatch(
        maximum_matches,
        reinterpret_cast<std::uint32_t (*)[2]>(pairs.data()),
        0.7F,
        impl_->options.ratio_threshold,
        impl_->options.mutual_check ? 1 : 0);
    MatchSet result;
    if (match_count < 0)
        throw std::runtime_error("SiftGPU matching failed");
    if (match_count == 0) return result;
    result.matches.reserve(static_cast<std::size_t>(match_count));
    for (int i = 0; i < match_count; ++i) {
        const auto [query_index, train_index] =
            pairs[static_cast<std::size_t>(i)];
        if (query_index >= query.keypoints.size() ||
            train_index >= train.keypoints.size())
            continue;
        result.matches.push_back(
            {query_index, train_index, 1.F});
    }
    return result;
#else
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
    auto make_gpu_mutual_ratio = [] {
        auto matcher = std::make_unique<SiftGpuMatcher>();
        if (!matcher->is_available())
            throw std::runtime_error(
                "gpu_mutual_ratio matcher is not available on this machine");
        return std::unique_ptr<FeatureMatcher>(std::move(matcher));
    };
    register_matcher("gpu_mutual_ratio", make_gpu_mutual_ratio);
    // Legacy alias: SIFT is an extractor; matching is descriptor mutual-ratio.
    register_matcher("siftgpu", make_gpu_mutual_ratio);
}

}  // namespace aetherscan::features
