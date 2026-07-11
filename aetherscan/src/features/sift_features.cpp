#include "aetherscan/features/features.hpp"

#include "aetherscan/parallel/thread_pool.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif

extern "C" {
#include "vl/sift.h"
}

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace aetherscan::features {
namespace {

inline bool use_openmp(const bool requested, const std::size_t work_items) {
#if defined(AETHERSCAN_HAS_OPENMP)
    return requested && parallel::allow_inner_parallelism() && work_items > 64;
#else
    (void)requested;
    (void)work_items;
    return false;
#endif
}

inline float squared_l2(const float* left, const float* right, const std::size_t dimension) {
#if defined(__AVX2__)
    if (dimension == 128) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (std::size_t index = 0; index < 128; index += 16) {
            const __m256 a0 = _mm256_loadu_ps(left + index);
            const __m256 b0 = _mm256_loadu_ps(right + index);
            const __m256 d0 = _mm256_sub_ps(a0, b0);
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
            const __m256 a1 = _mm256_loadu_ps(left + index + 8);
            const __m256 b1 = _mm256_loadu_ps(right + index + 8);
            const __m256 d1 = _mm256_sub_ps(a1, b1);
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
        }
        const __m256 sum = _mm256_add_ps(acc0, acc1);
        __m128 low = _mm256_castps256_ps128(sum);
        __m128 high = _mm256_extractf128_ps(sum, 1);
        __m128 folded = _mm_add_ps(low, high);
        folded = _mm_hadd_ps(folded, folded);
        folded = _mm_hadd_ps(folded, folded);
        return _mm_cvtss_f32(folded);
    }
#endif
    float sum = 0.0F;
    std::size_t index = 0;
    for (; index + 4 <= dimension; index += 4) {
        const float d0 = left[index] - right[index];
        const float d1 = left[index + 1] - right[index + 1];
        const float d2 = left[index + 2] - right[index + 2];
        const float d3 = left[index + 3] - right[index + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (; index < dimension; ++index) {
        const float delta = left[index] - right[index];
        sum += delta * delta;
    }
    return sum;
}

void knn2_squared_l2(
    const float* query, const std::size_t query_count, const float* train,
    const std::size_t train_count, const std::size_t dimension, const bool parallel,
    std::vector<int>& best0, std::vector<float>& dist0, std::vector<int>& best1,
    std::vector<float>& dist1) {
    best0.assign(query_count, -1);
    best1.assign(query_count, -1);
    dist0.assign(query_count, std::numeric_limits<float>::infinity());
    dist1.assign(query_count, std::numeric_limits<float>::infinity());
    const bool openmp = use_openmp(parallel, query_count);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 16) if (openmp)
#else
    (void)openmp;
#endif
    for (int query_index = 0; query_index < static_cast<int>(query_count); ++query_index) {
        const float* query_row = query + static_cast<std::size_t>(query_index) * dimension;
        float first_distance = std::numeric_limits<float>::infinity();
        float second_distance = std::numeric_limits<float>::infinity();
        int first_index = -1;
        int second_index = -1;
        for (std::size_t train_index = 0; train_index < train_count; ++train_index) {
            const float distance =
                squared_l2(query_row, train + train_index * dimension, dimension);
            if (distance < first_distance) {
                second_distance = first_distance;
                second_index = first_index;
                first_distance = distance;
                first_index = static_cast<int>(train_index);
            } else if (distance < second_distance) {
                second_distance = distance;
                second_index = static_cast<int>(train_index);
            }
        }
        best0[static_cast<std::size_t>(query_index)] = first_index;
        best1[static_cast<std::size_t>(query_index)] = second_index;
        dist0[static_cast<std::size_t>(query_index)] = first_distance;
        dist1[static_cast<std::size_t>(query_index)] = second_distance;
    }
}

void knn1_squared_l2_subset(
    const float* query, const std::size_t query_count, const float* train,
    const std::size_t dimension, const std::vector<int>& train_indices, const bool parallel,
    std::vector<int>& best_query, std::vector<float>& best_distance) {
    best_query.assign(train_indices.size(), -1);
    best_distance.assign(train_indices.size(), std::numeric_limits<float>::infinity());
    const bool openmp = use_openmp(parallel, train_indices.size());
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 16) if (openmp)
#else
    (void)openmp;
#endif
    for (int slot = 0; slot < static_cast<int>(train_indices.size()); ++slot) {
        const int train_index = train_indices[static_cast<std::size_t>(slot)];
        const float* train_row = train + static_cast<std::size_t>(train_index) * dimension;
        float best = std::numeric_limits<float>::infinity();
        int best_index = -1;
        for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
            const float distance =
                squared_l2(train_row, query + query_index * dimension, dimension);
            if (distance < best) {
                best = distance;
                best_index = static_cast<int>(query_index);
            }
        }
        best_query[static_cast<std::size_t>(slot)] = best_index;
        best_distance[static_cast<std::size_t>(slot)] = best;
    }
}

void apply_root_sift(FeatureSet& features) {
    if (features.descriptor_dimension == 0 || features.descriptors.empty()) return;
    const bool openmp = use_openmp(true, features.keypoints.size());
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (openmp)
#else
    (void)openmp;
#endif
    for (int row = 0; row < static_cast<int>(features.keypoints.size()); ++row) {
        float* descriptor = features.descriptors.data() +
                            static_cast<std::size_t>(row) * features.descriptor_dimension;
        double sum = 0.0;
        for (std::size_t column = 0; column < features.descriptor_dimension; ++column)
            sum += (std::max)(0.0F, descriptor[column]);
        const float inverse = static_cast<float>(1.0 / (std::max)(sum, 1e-12));
        for (std::size_t column = 0; column < features.descriptor_dimension; ++column)
            descriptor[column] =
                std::sqrt((std::max)(0.0F, descriptor[column]) * inverse);
    }
}

struct VlRuntime {
    VlRuntime() { vl_constructor(); }
    ~VlRuntime() { vl_destructor(); }
};

void ensure_vl() {
    static VlRuntime runtime;
    (void)runtime;
}

}  // namespace

void FeatureSet::validate() const {
    if (image_width == 0 || image_height == 0)
        throw std::invalid_argument("Feature image dimensions must be non-zero");
    if (descriptor_dimension == 0) {
        if (!descriptors.empty()) throw std::invalid_argument("Descriptor data has zero dimension");
    } else if (descriptors.size() != keypoints.size() * descriptor_dimension) {
        throw std::invalid_argument("Descriptor storage does not match keypoint count");
    }
}

class SiftExtractor::Impl {
public:
    explicit Impl(SiftOptions value) : options(value) {
        if (value.maximum_features == 0 || value.octave_layers == 0)
            throw std::invalid_argument("SIFT feature limits must be positive");
        ensure_vl();
    }
    SiftOptions options;
};

SiftExtractor::SiftExtractor(SiftOptions options) : impl_(std::make_unique<Impl>(options)) {}
SiftExtractor::~SiftExtractor() = default;
SiftExtractor::SiftExtractor(SiftExtractor&&) noexcept = default;
SiftExtractor& SiftExtractor::operator=(SiftExtractor&&) noexcept = default;

ExtractorInfo SiftExtractor::info() const {
    ExtractorInfo value;
    value.thread_safe = false;
    value.accepts_gray = true;
    value.accepts_rgb = false;
    value.metric =
        impl_->options.root_sift ? DescriptorMetric::l2_root : DescriptorMetric::l2;
    value.typical_descriptor_dimension = 128;
    return value;
}

std::unique_ptr<FeatureExtractor> SiftExtractor::clone() const {
    return std::make_unique<SiftExtractor>(impl_->options);
}

FeatureSet SiftExtractor::extract_gray(
    const std::span<const std::uint8_t> pixels, const std::uint32_t width,
    const std::uint32_t height, std::size_t row_stride) const {
    if (row_stride == 0) row_stride = width;
    if (width == 0 || height == 0 || row_stride < width ||
        pixels.size() < row_stride * static_cast<std::size_t>(height))
        throw std::invalid_argument("Invalid grayscale image view");

    std::vector<float> float_image(static_cast<std::size_t>(width) * height);
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint8_t* row = pixels.data() + static_cast<std::size_t>(y) * row_stride;
        float* destination = float_image.data() + static_cast<std::size_t>(y) * width;
        for (std::uint32_t x = 0; x < width; ++x) destination[x] = static_cast<float>(row[x]);
    }

    const int octaves = (std::max)(
        1, static_cast<int>(std::floor(std::log2((std::min)(width, height)))) - 3);
    VlSiftFilt* filter = vl_sift_new(
        static_cast<int>(width), static_cast<int>(height), octaves,
        static_cast<int>(impl_->options.octave_layers), 0);
    if (!filter) throw std::runtime_error("Failed to create VLFeat SIFT filter");
    vl_sift_set_edge_thresh(filter, impl_->options.edge_threshold);
    vl_sift_set_peak_thresh(
        filter,
        255.0 * impl_->options.contrast_threshold /
            static_cast<double>(impl_->options.octave_layers));

    FeatureSet result;
    result.image_width = width;
    result.image_height = height;
    result.descriptor_dimension = 128;
    result.metric = info().metric;
    result.extractor_name = std::string(name());
    result.keypoints.reserve(impl_->options.maximum_features);
    result.descriptors.reserve(impl_->options.maximum_features * 128);

    vl_sift_process_first_octave(filter, float_image.data());
    vl_sift_pix descriptor[128];
    while (true) {
        vl_sift_detect(filter);
        const VlSiftKeypoint* keys = vl_sift_get_keypoints(filter);
        const int key_count = vl_sift_get_nkeypoints(filter);
        vl_sift_update_gradient(filter);
        for (int index = 0; index < key_count; ++index) {
            if (result.keypoints.size() >= impl_->options.maximum_features) break;
            double angles[4] = {};
            const int angle_count =
                vl_sift_calc_keypoint_orientations(filter, angles, keys + index);
            for (int angle = 0; angle < angle_count; ++angle) {
                if (result.keypoints.size() >= impl_->options.maximum_features) break;
                vl_sift_calc_keypoint_descriptor(
                    filter, descriptor, keys + index, angles[angle]);
                result.keypoints.push_back(
                    {keys[index].x, keys[index].y, keys[index].sigma,
                     static_cast<float>(angles[angle]), 0.0F});
                result.descriptors.insert(
                    result.descriptors.end(), descriptor, descriptor + 128);
            }
        }
        if (result.keypoints.size() >= impl_->options.maximum_features) break;
        if (vl_sift_process_next_octave(filter)) break;
    }
    vl_sift_delete(filter);
    if (impl_->options.root_sift) apply_root_sift(result);
    return result;
}

MutualRatioMatcher::MutualRatioMatcher(DescriptorMatcherOptions options)
    : options_(options) {
    if (!(options_.ratio_threshold > 0.0F && options_.ratio_threshold <= 1.0F))
        throw std::invalid_argument("Descriptor ratio threshold must be in (0, 1]");
}

std::unique_ptr<FeatureMatcher> MutualRatioMatcher::clone() const {
    return std::make_unique<MutualRatioMatcher>(options_);
}

MatchSet MutualRatioMatcher::match(const FeatureSet& query, const FeatureSet& train) const {
    query.validate();
    train.validate();
    if (query.descriptor_dimension == 0 || train.descriptor_dimension == 0) return {};
    if (query.descriptor_dimension != train.descriptor_dimension)
        throw std::invalid_argument("Feature descriptor dimensions differ");
    if (query.keypoints.empty() || train.keypoints.empty()) return {};

    std::vector<int> best0, best1;
    std::vector<float> dist0, dist1;
    knn2_squared_l2(
        query.descriptors.data(), query.keypoints.size(), train.descriptors.data(),
        train.keypoints.size(), query.descriptor_dimension, options_.parallel, best0, dist0,
        best1, dist1);

    const float ratio_squared = options_.ratio_threshold * options_.ratio_threshold;
    struct Candidate {
        FeatureIndex query{};
        FeatureIndex train{};
        float squared_distance{};
    };
    std::vector<Candidate> candidates;
    candidates.reserve(query.keypoints.size() / 4);
    for (std::size_t index = 0; index < query.keypoints.size(); ++index) {
        if (best0[index] < 0 || best1[index] < 0) continue;
        if (dist0[index] >= ratio_squared * dist1[index]) continue;
        candidates.push_back(
            {static_cast<FeatureIndex>(index), static_cast<FeatureIndex>(best0[index]),
             dist0[index]});
    }

    MatchSet result;
    result.matches.reserve(candidates.size());
    if (!options_.mutual_check) {
        for (const auto& candidate : candidates)
            result.matches.push_back(
                {candidate.query, candidate.train,
                 1.0F / (1.0F + std::sqrt(candidate.squared_distance))});
        return result;
    }

    std::vector<int> unique_trains;
    unique_trains.reserve(candidates.size());
    std::vector<int> train_slot(train.keypoints.size(), -1);
    for (const auto& candidate : candidates) {
        if (train_slot[candidate.train] >= 0) continue;
        train_slot[candidate.train] = static_cast<int>(unique_trains.size());
        unique_trains.push_back(static_cast<int>(candidate.train));
    }

    std::vector<int> reverse_best;
    std::vector<float> reverse_distance;
    knn1_squared_l2_subset(
        query.descriptors.data(), query.keypoints.size(), train.descriptors.data(),
        query.descriptor_dimension, unique_trains, options_.parallel, reverse_best,
        reverse_distance);

    for (const auto& candidate : candidates) {
        const int slot = train_slot[candidate.train];
        if (slot < 0 ||
            reverse_best[static_cast<std::size_t>(slot)] != static_cast<int>(candidate.query))
            continue;
        result.matches.push_back(
            {candidate.query, candidate.train,
             1.0F / (1.0F + std::sqrt(candidate.squared_distance))});
    }
    return result;
}

MatchSet match_descriptors(
    const FeatureSet& query, const FeatureSet& train, const DescriptorMatcherOptions& options) {
    return MutualRatioMatcher(options).match(query, train);
}

void register_sift_feature_backends() {
    register_extractor("sift", [] { return std::make_unique<SiftExtractor>(); });
    register_matcher("mutual_ratio", [] { return std::make_unique<MutualRatioMatcher>(); });
}

}  // namespace aetherscan::features
