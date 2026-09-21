#include "features/features.hpp"

#include "parallel/thread_pool.hpp"

#include <hnswlib/hnswlib.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif

extern "C" {
#include "vl/sift.h"
}

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace photara::features {
namespace {

inline bool use_openmp(const bool requested, const std::size_t work_items) {
#if defined(PHOTARA_HAS_OPENMP)
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
#if defined(PHOTARA_HAS_OPENMP)
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
#if defined(PHOTARA_HAS_OPENMP)
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
#if defined(PHOTARA_HAS_OPENMP)
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

class MutualRatioMatcher::SharedState {
public:
    class Index {
    public:
        Index(
            std::span<const float> rows,
            const std::size_t keypoint_count,
            const std::size_t dimension,
            const DescriptorMatcherOptions& options)
            : space(dimension),
              graph(&space, keypoint_count, options.ann_m,
                    options.ann_ef_construction, 42) {
            for (std::size_t row = 0; row < keypoint_count; ++row) {
                graph.addPoint(rows.data() + row * dimension, row);
            }
            graph.setEf(options.ann_ef_search);
        }

        hnswlib::L2Space space;
        hnswlib::HierarchicalNSW<float> graph;
    };

    struct Entry {
        const float* descriptors{};
        std::size_t descriptor_count{};
        std::size_t keypoint_count{};
        std::size_t dimension{};
        std::uint64_t generation{};
        std::uint64_t identity{};
        std::shared_ptr<Index> index;
        std::uint64_t stamp{};
        unsigned pins{0};
    };

    static bool matches(
        const Entry& entry,
        const FeatureSet& features,
        const float* rows,
        const std::size_t row_count) {
        return entry.descriptors == rows &&
               entry.descriptor_count == row_count &&
               entry.keypoint_count == features.keypoints.size() &&
               entry.dimension == features.descriptor_dimension &&
               entry.generation == features.descriptor_generation &&
               entry.identity == features.descriptor_identity;
    }

    void prepare(
        FeatureSet& features,
        const DescriptorMatcherOptions& options) {
        if (!options.approximate ||
            features.keypoints.size() < options.ann_min_features ||
            features.descriptor_dimension == 0)
            return;
        const auto rows = features.descriptor_rows_float();
        if (rows.empty()) return;
        {
            std::lock_guard lock(mutex_);
            const auto found = indices_.find(&features);
            if (found != indices_.end() &&
                matches(
                    found->second, features, rows.data(), rows.size())) {
                found->second.stamp = ++clock_;
                return;
            }
        }
        auto index = std::make_shared<Index>(
            rows, features.keypoints.size(), features.descriptor_dimension,
            options);
        std::lock_guard lock(mutex_);
        Entry& entry = indices_[&features];
        const unsigned pins = entry.pins;
        entry = {
            rows.data(), rows.size(), features.keypoints.size(),
            features.descriptor_dimension, features.descriptor_generation,
            features.descriptor_identity, std::move(index), ++clock_, pins};
        evict_unlocked(options.max_cached_indices);
    }

    void pin(const FeatureSet& features) {
        std::lock_guard lock(mutex_);
        ++indices_[&features].pins;
    }

    void unpin(const FeatureSet& features) {
        std::lock_guard lock(mutex_);
        const auto found = indices_.find(&features);
        if (found == indices_.end() || found->second.pins == 0) return;
        --found->second.pins;
    }

    [[nodiscard]] std::shared_ptr<Index> find(FeatureSet& features) {
        const auto rows = features.descriptor_rows_float();
        std::lock_guard lock(mutex_);
        const auto found = indices_.find(&features);
        if (found == indices_.end() ||
            !matches(found->second, features, rows.data(), rows.size()))
            return nullptr;
        found->second.stamp = ++clock_;
        return found->second.index;
    }

    void clear() {
        std::lock_guard lock(mutex_);
        indices_.clear();
    }

private:
    void evict_unlocked(const std::size_t budget) {
        if (budget == 0 || indices_.size() <= budget) return;
        while (indices_.size() > budget) {
            auto victim = indices_.end();
            for (auto it = indices_.begin(); it != indices_.end(); ++it) {
                if (it->second.pins > 0 || !it->second.index) continue;
                if (victim == indices_.end() ||
                    it->second.stamp < victim->second.stamp)
                    victim = it;
            }
            if (victim == indices_.end()) break;
            indices_.erase(victim);
        }
    }

    mutable std::mutex mutex_;
    std::unordered_map<const FeatureSet*, Entry> indices_;
    std::uint64_t clock_{0};
};

void FeatureSet::validate() const {
    if (image_width == 0 || image_height == 0)
        throw std::invalid_argument("Feature image dimensions must be non-zero");
    if (descriptor_dimension == 0) {
        if (!descriptors.empty() || !descriptors_u8.empty())
            throw std::invalid_argument("Descriptor data has zero dimension");
        return;
    }
    const std::size_t expected = keypoints.size() * descriptor_dimension;
    if (storage == DescriptorStorage::float32) {
        if (descriptors.size() != expected)
            throw std::invalid_argument(
                "Descriptor storage does not match keypoint count");
    } else if (descriptors_u8.size() != expected) {
        throw std::invalid_argument(
            "Uint8 descriptor storage does not match keypoint count");
    }
}

std::span<const float> FeatureSet::descriptor_rows_float() {
    if (storage == DescriptorStorage::float32) return descriptors;
    if (descriptors_u8.empty() || descriptor_dimension == 0) return {};
    const std::size_t expected = keypoints.size() * descriptor_dimension;
    if (descriptors.size() != expected) {
        descriptors.resize(expected);
        for (std::size_t row = 0; row < keypoints.size(); ++row) {
            float* descriptor =
                descriptors.data() + row * descriptor_dimension;
            double squared_norm = 0.0;
            for (std::size_t column = 0; column < descriptor_dimension;
                 ++column) {
                const float value = static_cast<float>(
                    descriptors_u8[row * descriptor_dimension + column]);
                descriptor[column] = value;
                squared_norm += static_cast<double>(value) * value;
            }
            const float inverse_norm = static_cast<float>(
                1.0 / std::sqrt(std::max(squared_norm, 1e-24)));
            for (std::size_t column = 0; column < descriptor_dimension;
                 ++column)
                descriptor[column] *= inverse_norm;
        }
        mark_descriptors_modified();
    }
    return descriptors;
}

void FeatureSet::compress_descriptors_u8() {
    if (storage == DescriptorStorage::uint8) return;
    if (descriptors.empty() || descriptor_dimension == 0) {
        descriptors.clear();
        descriptors.shrink_to_fit();
        storage = DescriptorStorage::uint8;
        mark_descriptors_modified();
        return;
    }
    descriptors_u8.resize(descriptors.size());
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        const float value = std::clamp(descriptors[i], 0.F, 1.F);
        // SiftMatchGPU's byte-descriptor overload expects descriptors scaled
        // to a norm of 512 (the de-facto SIFT/RootSIFT byte convention used
        // by COLMAP and openMVS), with values above 255 saturated.
        descriptors_u8[i] = static_cast<std::uint8_t>(std::min(
            255L, std::lround(value * 512.F)));
    }
    descriptors.clear();
    descriptors.shrink_to_fit();
    storage = DescriptorStorage::uint8;
    mark_descriptors_modified();
}

void FeatureSet::release_descriptors() noexcept {
    descriptors.clear();
    descriptors.shrink_to_fit();
    descriptors_u8.clear();
    descriptors_u8.shrink_to_fit();
    storage = DescriptorStorage::float32;
    mark_descriptors_modified();
}

class SiftExtractor::Impl {
public:
    explicit Impl(SiftOptions value) : options(value) {
        if (value.maximum_features == 0 || value.octave_layers == 0 ||
            value.grid_size == 0 || value.max_features_per_cell == 0 ||
            value.min_features_per_cell > value.max_features_per_cell)
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

    FeatureSet result;
    result.image_width = width;
    result.image_height = height;
    result.descriptor_dimension = 128;
    result.metric = info().metric;
    result.extractor_name = std::string(name());
    result.keypoints.reserve(impl_->options.maximum_features);
    result.descriptors.reserve(impl_->options.maximum_features * 128);

    const auto extract_cell = [&](const std::uint32_t roi_x, const std::uint32_t roi_y,
                                  const std::uint32_t roi_width,
                                  const std::uint32_t roi_height,
                                  const std::uint32_t core_x,
                                  const std::uint32_t core_y,
                                  const std::uint32_t core_width,
                                  const std::uint32_t core_height,
                                  const double contrast) {
        FeatureSet cell;
        cell.image_width = width;
        cell.image_height = height;
        cell.descriptor_dimension = 128;
        cell.metric = result.metric;
        cell.extractor_name = result.extractor_name;
        cell.keypoints.reserve(impl_->options.max_features_per_cell);
        cell.descriptors.reserve(impl_->options.max_features_per_cell * 128);

        std::vector<float> float_image(static_cast<std::size_t>(roi_width) * roi_height);
        for (std::uint32_t y = 0; y < roi_height; ++y) {
            const std::uint8_t* source =
                pixels.data() + static_cast<std::size_t>(roi_y + y) * row_stride + roi_x;
            float* destination =
                float_image.data() + static_cast<std::size_t>(y) * roi_width;
            for (std::uint32_t x = 0; x < roi_width; ++x)
                destination[x] = static_cast<float>(source[x]);
        }

        const int octaves = (std::max)(
            1, static_cast<int>(std::floor(
                   std::log2((std::min)(roi_width, roi_height)))) -
                   3);
        VlSiftFilt* filter = vl_sift_new(
            static_cast<int>(roi_width), static_cast<int>(roi_height), octaves,
            static_cast<int>(impl_->options.octave_layers),
            impl_->options.first_octave);
        if (!filter) throw std::runtime_error("Failed to create VLFeat SIFT filter");
        vl_sift_set_edge_thresh(filter, impl_->options.edge_threshold);
        vl_sift_set_peak_thresh(
            filter, 255.0 * contrast /
                        static_cast<double>(impl_->options.octave_layers));

        const float local_core_x0 = static_cast<float>(core_x - roi_x);
        const float local_core_y0 = static_cast<float>(core_y - roi_y);
        const float local_core_x1 = local_core_x0 + static_cast<float>(core_width);
        const float local_core_y1 = local_core_y0 + static_cast<float>(core_height);
        vl_sift_pix descriptor[128];
        if (vl_sift_process_first_octave(filter, float_image.data()) == 0) {
            while (true) {
                vl_sift_detect(filter);
                const VlSiftKeypoint* keys = vl_sift_get_keypoints(filter);
                const int key_count = vl_sift_get_nkeypoints(filter);
                vl_sift_update_gradient(filter);
                for (int index = 0; index < key_count; ++index) {
                    if (cell.keypoints.size() >= impl_->options.max_features_per_cell)
                        break;
                    const VlSiftKeypoint& key = keys[index];
                    if (key.x < local_core_x0 || key.x >= local_core_x1 ||
                        key.y < local_core_y0 || key.y >= local_core_y1)
                        continue;
                    double angles[4] = {};
                    const int angle_count =
                        vl_sift_calc_keypoint_orientations(filter, angles, &key);
                    for (int angle = 0; angle < angle_count; ++angle) {
                        if (cell.keypoints.size() >=
                            impl_->options.max_features_per_cell)
                            break;
                        vl_sift_calc_keypoint_descriptor(
                            filter, descriptor, &key, angles[angle]);
                        cell.keypoints.push_back(
                            {key.x + static_cast<float>(roi_x),
                             key.y + static_cast<float>(roi_y), key.sigma,
                             static_cast<float>(angles[angle]), 1.0F});
                        cell.descriptors.insert(
                            cell.descriptors.end(), descriptor, descriptor + 128);
                    }
                }
                if (cell.keypoints.size() >= impl_->options.max_features_per_cell ||
                    vl_sift_process_next_octave(filter))
                    break;
            }
        }
        vl_sift_delete(filter);
        return cell;
    };

    const std::size_t grid = impl_->options.grid_size;
    const std::uint32_t cell_width = width / static_cast<std::uint32_t>(grid);
    const std::uint32_t cell_height = height / static_cast<std::uint32_t>(grid);
    const std::uint32_t border = static_cast<std::uint32_t>((std::min)(
        impl_->options.cell_border,
        static_cast<std::size_t>((std::min)(cell_width, cell_height) / 2)));
    for (std::size_t row = 0; row < grid; ++row) {
        for (std::size_t col = 0; col < grid; ++col) {
            const std::uint32_t core_x =
                static_cast<std::uint32_t>(col) * cell_width;
            const std::uint32_t core_y =
                static_cast<std::uint32_t>(row) * cell_height;
            const std::uint32_t core_width =
                col + 1 == grid ? width - core_x : cell_width;
            const std::uint32_t core_height =
                row + 1 == grid ? height - core_y : cell_height;
            const std::uint32_t roi_x = core_x > border ? core_x - border : 0;
            const std::uint32_t roi_y = core_y > border ? core_y - border : 0;
            const std::uint32_t roi_x1 =
                (std::min)(width, core_x + core_width + border);
            const std::uint32_t roi_y1 =
                (std::min)(height, core_y + core_height + border);

            FeatureSet selected;
            for (unsigned retry = 0; retry <= impl_->options.adaptive_retries; ++retry) {
                const double contrast =
                    impl_->options.contrast_threshold * std::pow(0.5, retry);
                selected = extract_cell(
                    roi_x, roi_y, roi_x1 - roi_x, roi_y1 - roi_y,
                    core_x, core_y, core_width, core_height, contrast);
                if (selected.keypoints.size() >=
                        impl_->options.min_features_per_cell ||
                    retry == impl_->options.adaptive_retries)
                    break;
            }
            const std::size_t available =
                impl_->options.maximum_features - result.keypoints.size();
            const std::size_t count =
                (std::min)(available, selected.keypoints.size());
            result.keypoints.insert(
                result.keypoints.end(), selected.keypoints.begin(),
                selected.keypoints.begin() + static_cast<std::ptrdiff_t>(count));
            result.descriptors.insert(
                result.descriptors.end(), selected.descriptors.begin(),
                selected.descriptors.begin() +
                    static_cast<std::ptrdiff_t>(count * result.descriptor_dimension));
            if (result.keypoints.size() >= impl_->options.maximum_features) break;
        }
        if (result.keypoints.size() >= impl_->options.maximum_features) break;
    }
    if (impl_->options.root_sift) apply_root_sift(result);
    return result;
}

MutualRatioMatcher::MutualRatioMatcher(DescriptorMatcherOptions options)
    : MutualRatioMatcher(std::move(options), std::make_shared<SharedState>()) {}

MutualRatioMatcher::MutualRatioMatcher(
    DescriptorMatcherOptions options,
    std::shared_ptr<SharedState> shared)
    : options_(std::move(options)), shared_(std::move(shared)) {
    if (!(options_.ratio_threshold > 0.0F && options_.ratio_threshold <= 1.0F))
        throw std::invalid_argument("Descriptor ratio threshold must be in (0, 1]");
    if (options_.ann_m < 2 || options_.ann_ef_construction < 2 ||
        options_.ann_ef_search < 2)
        throw std::invalid_argument("ANN search parameters must be at least 2");
}

std::unique_ptr<FeatureMatcher> MutualRatioMatcher::clone() const {
    return std::unique_ptr<FeatureMatcher>(
        new MutualRatioMatcher(options_, shared_));
}

void MutualRatioMatcher::prepare(const FeatureSet& features) {
    auto& mutable_features = const_cast<FeatureSet&>(features);
    mutable_features.validate();
    shared_->prepare(mutable_features, options_);
}

void MutualRatioMatcher::clear_prepared() {
    shared_->clear();
}

void MutualRatioMatcher::pin(const FeatureSet& features) {
    shared_->pin(features);
}

void MutualRatioMatcher::unpin(const FeatureSet& features) {
    shared_->unpin(features);
}

MatchSet MutualRatioMatcher::match(const FeatureSet& query, const FeatureSet& train) const {
    auto& mutable_query = const_cast<FeatureSet&>(query);
    auto& mutable_train = const_cast<FeatureSet&>(train);
    mutable_query.validate();
    mutable_train.validate();
    if (query.descriptor_dimension == 0 || train.descriptor_dimension == 0) return {};
    if (query.descriptor_dimension != train.descriptor_dimension)
        throw std::invalid_argument("Feature descriptor dimensions differ");
    if (query.keypoints.empty() || train.keypoints.empty()) return {};

    const auto query_rows = mutable_query.descriptor_rows_float();
    const auto train_rows = mutable_train.descriptor_rows_float();
    if (query_rows.empty() || train_rows.empty()) return {};

    std::vector<int> best0, best1;
    std::vector<float> dist0, dist1;
    const auto query_index = shared_->find(mutable_query);
    const auto train_index = shared_->find(mutable_train);
    const bool use_ann =
        query_index && train_index && train.keypoints.size() >= 2;
    if (use_ann) {
        best0.assign(query.keypoints.size(), -1);
        best1.assign(query.keypoints.size(), -1);
        dist0.assign(query.keypoints.size(), std::numeric_limits<float>::infinity());
        dist1.assign(query.keypoints.size(), std::numeric_limits<float>::infinity());
        for (std::size_t row = 0; row < query.keypoints.size(); ++row) {
            auto nearest = train_index->graph.searchKnn(
                query_rows.data() + row * query.descriptor_dimension, 2);
            if (nearest.size() < 2) continue;
            const auto second = nearest.top();
            nearest.pop();
            const auto first = nearest.top();
            if (first.second >= train.keypoints.size() ||
                second.second >= train.keypoints.size())
                continue;
            best0[row] = static_cast<int>(first.second);
            dist0[row] = first.first;
            best1[row] = static_cast<int>(second.second);
            dist1[row] = second.first;
        }
    } else {
        knn2_squared_l2(
            query_rows.data(), query.keypoints.size(), train_rows.data(),
            train.keypoints.size(), query.descriptor_dimension, options_.parallel,
            best0, dist0, best1, dist1);
    }

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
    if (use_ann) {
        reverse_best.assign(unique_trains.size(), -1);
        reverse_distance.assign(
            unique_trains.size(), std::numeric_limits<float>::infinity());
        for (std::size_t slot = 0; slot < unique_trains.size(); ++slot) {
            const int train_id = unique_trains[slot];
            auto nearest = query_index->graph.searchKnn(
                train_rows.data() +
                    static_cast<std::size_t>(train_id) *
                        train.descriptor_dimension,
                1);
            if (nearest.empty()) continue;
            if (nearest.top().second >= query.keypoints.size()) continue;
            reverse_best[slot] = static_cast<int>(nearest.top().second);
            reverse_distance[slot] = nearest.top().first;
        }
    } else {
        knn1_squared_l2_subset(
            query_rows.data(), query.keypoints.size(), train_rows.data(),
            query.descriptor_dimension, unique_trains, options_.parallel, reverse_best,
            reverse_distance);
    }

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
    MutualRatioMatcher matcher(options);
    matcher.prepare(query);
    matcher.prepare(train);
    return matcher.match(query, train);
}

void register_sift_feature_backends() {
    register_extractor("sift", [] { return std::make_unique<SiftExtractor>(); });
    register_matcher("mutual_ratio", [] { return std::make_unique<MutualRatioMatcher>(); });
}

}  // namespace photara::features
