#include "sfm/vocabulary.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace aetherscan::sfm {
namespace {

constexpr std::uint32_t k_vocab_magic = 0x41535654u;  // ASVT
constexpr std::uint32_t k_vocab_version = 1;

float squared_l2(
    const float* left, const float* right, const std::size_t dimension) {
    float sum = 0.F;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float delta = left[i] - right[i];
        sum += delta * delta;
    }
    return sum;
}

template <class T>
void write_pod(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <class T>
void write_vector(std::ostream& output, const std::vector<T>& values) {
    const std::uint64_t count = values.size();
    write_pod(output, count);
    if (!values.empty()) {
        output.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
}

template <class T>
T read_pod(std::istream& input) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("Failed reading vocabulary file");
    return value;
}

template <class T>
std::vector<T> read_vector(std::istream& input, const std::size_t maximum) {
    const std::uint64_t count = read_pod<std::uint64_t>(input);
    if (count > maximum)
        throw std::runtime_error("Vocabulary vector exceeds size limit");
    std::vector<T> values(static_cast<std::size_t>(count));
    if (!values.empty()) {
        input.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!input) throw std::runtime_error("Failed reading vocabulary vector");
    }
    return values;
}

}  // namespace

std::vector<features::FeatureIndex> sample_descriptors_spatially(
    const features::FeatureSet& features,
    const std::size_t maximum,
    const unsigned grid_size) {
    if (maximum == 0 || features.keypoints.empty() || grid_size == 0) return {};
    const std::size_t keep = std::min(maximum, features.keypoints.size());
    if (keep == features.keypoints.size()) {
        std::vector<features::FeatureIndex> all(features.keypoints.size());
        std::iota(all.begin(), all.end(), features::FeatureIndex{0});
        return all;
    }

    const float width = static_cast<float>(
        std::max<std::uint32_t>(1, features.image_width));
    const float height = static_cast<float>(
        std::max<std::uint32_t>(1, features.image_height));
    const unsigned cells = grid_size * grid_size;
    std::vector<std::vector<features::FeatureIndex>> buckets(cells);
    for (features::FeatureIndex index = 0; index < features.keypoints.size();
         ++index) {
        const auto& keypoint = features.keypoints[index];
        const unsigned cell_x = std::min(
            grid_size - 1U,
            static_cast<unsigned>(
                std::max(0.F, std::min(keypoint.x, width - 1.F)) / width *
                grid_size));
        const unsigned cell_y = std::min(
            grid_size - 1U,
            static_cast<unsigned>(
                std::max(0.F, std::min(keypoint.y, height - 1.F)) / height *
                grid_size));
        buckets[cell_y * grid_size + cell_x].push_back(index);
    }

    const auto quality = [&](const features::FeatureIndex index) {
        const auto& keypoint = features.keypoints[index];
        const float response =
            keypoint.response / (keypoint.response + 0.03F);
        const float scale = std::clamp(keypoint.scale, 2.F, 20.F);
        const float size_weight = 0.5F + (scale - 2.F) / 18.F;
        return response * size_weight;
    };
    for (auto& bucket : buckets) {
        std::sort(
            bucket.begin(), bucket.end(),
            [&](const features::FeatureIndex left,
                const features::FeatureIndex right) {
                const float quality_left = quality(left);
                const float quality_right = quality(right);
                if (quality_left != quality_right)
                    return quality_left > quality_right;
                return left < right;
            });
    }

    std::vector<features::FeatureIndex> selected;
    selected.reserve(keep);
    std::vector<std::size_t> cursor(cells, 0);
    while (selected.size() < keep) {
        bool progressed = false;
        for (unsigned cell = 0; cell < cells && selected.size() < keep; ++cell) {
            if (cursor[cell] >= buckets[cell].size()) continue;
            selected.push_back(buckets[cell][cursor[cell]++]);
            progressed = true;
        }
        if (!progressed) break;
    }
    return selected;
}

void VocabularyTree::build_into(
    const std::size_t node_index,
    std::vector<std::size_t>& subset,
    const unsigned depth,
    std::span<const float> descriptors) {
    if (node_index >= nodes_.size())
        throw std::runtime_error("Vocabulary node index out of range");
    if (centroids_.size() < nodes_.size() * dimension_)
        centroids_.resize(nodes_.size() * dimension_, 0.F);

    const bool leaf =
        depth >= config_.depth ||
        subset.size() <= config_.branching ||
        subset.size() <= 1;
    if (leaf) {
        nodes_[node_index] = {};
        nodes_[node_index].word_id = static_cast<std::int32_t>(word_count_++);
        float* centroid = centroids_.data() + node_index * dimension_;
        std::fill(centroid, centroid + dimension_, 0.F);
        for (const std::size_t id : subset) {
            const float* row = descriptors.data() + id * dimension_;
            for (std::size_t d = 0; d < dimension_; ++d) centroid[d] += row[d];
        }
        const float inv =
            subset.empty() ? 0.F : 1.F / static_cast<float>(subset.size());
        for (std::size_t d = 0; d < dimension_; ++d) centroid[d] *= inv;
        return;
    }

    const std::uint32_t branching = static_cast<std::uint32_t>(
        std::min<std::size_t>(config_.branching, subset.size()));
    std::mt19937 random(
        config_.seed + static_cast<std::uint32_t>(node_index * 9973u));
    std::vector<std::size_t> centers;
    centers.reserve(branching);
    {
        std::vector<std::size_t> order = subset;
        std::shuffle(order.begin(), order.end(), random);
        for (std::size_t i = 0; i < branching; ++i) centers.push_back(order[i]);
    }

    std::vector<std::vector<float>> center_values(
        branching, std::vector<float>(dimension_));
    for (std::uint32_t c = 0; c < branching; ++c) {
        const float* row = descriptors.data() + centers[c] * dimension_;
        std::copy(row, row + dimension_, center_values[c].begin());
    }

    std::vector<std::uint32_t> assignment(subset.size(), 0);
    for (unsigned iteration = 0; iteration < config_.max_iterations;
         ++iteration) {
        // Nearest-center search is the dominant k-means cost and writes only
        // assignment[i], so a parallel_for reproduces the serial result
        // bit-for-bit (the reduction steps below stay serial and ordered).
        parallel::parallel_for(
            subset.size(), parallel::resolve_thread_count(0),
            [&](const std::size_t i) {
                const float* row = descriptors.data() + subset[i] * dimension_;
                float best = std::numeric_limits<float>::infinity();
                std::uint32_t best_center = 0;
                for (std::uint32_t c = 0; c < branching; ++c) {
                    const float distance =
                        squared_l2(row, center_values[c].data(), dimension_);
                    if (distance < best) {
                        best = distance;
                        best_center = c;
                    }
                }
                assignment[i] = best_center;
            });

        std::vector<std::vector<float>> sums(
            branching, std::vector<float>(dimension_, 0.F));
        std::vector<std::size_t> counts(branching, 0);
        for (std::size_t i = 0; i < subset.size(); ++i) {
            const std::uint32_t center = assignment[i];
            const float* row = descriptors.data() + subset[i] * dimension_;
            for (std::size_t d = 0; d < dimension_; ++d)
                sums[center][d] += row[d];
            ++counts[center];
        }
        for (std::uint32_t c = 0; c < branching; ++c) {
            if (counts[c] == 0) {
                std::uint32_t donor = 0;
                for (std::uint32_t other = 1; other < branching; ++other)
                    if (counts[other] > counts[donor]) donor = other;
                for (std::size_t i = 0; i < subset.size(); ++i) {
                    if (assignment[i] != donor) continue;
                    assignment[i] = c;
                    --counts[donor];
                    ++counts[c];
                    break;
                }
                sums.assign(branching, std::vector<float>(dimension_, 0.F));
                counts.assign(branching, 0);
                for (std::size_t i = 0; i < subset.size(); ++i) {
                    const std::uint32_t center = assignment[i];
                    const float* row =
                        descriptors.data() + subset[i] * dimension_;
                    for (std::size_t d = 0; d < dimension_; ++d)
                        sums[center][d] += row[d];
                    ++counts[center];
                }
            }
            const float inv =
                1.F / static_cast<float>(std::max<std::size_t>(1, counts[c]));
            for (std::size_t d = 0; d < dimension_; ++d)
                center_values[c][d] = sums[c][d] * inv;
        }
    }

    std::vector<std::vector<std::size_t>> children(branching);
    for (std::size_t i = 0; i < subset.size(); ++i)
        children[assignment[i]].push_back(subset[i]);

    const std::uint32_t first_child =
        static_cast<std::uint32_t>(nodes_.size());
    nodes_[node_index] = {};
    nodes_[node_index].first_child = first_child;
    nodes_[node_index].child_count = static_cast<std::uint16_t>(branching);
    nodes_.resize(first_child + branching);
    centroids_.resize(nodes_.size() * dimension_, 0.F);
    for (std::uint32_t c = 0; c < branching; ++c)
        build_into(
            first_child + c, children[c], depth + 1, descriptors);

    float* centroid = centroids_.data() + node_index * dimension_;
    std::fill(centroid, centroid + dimension_, 0.F);
    for (std::uint16_t child = 0; child < branching; ++child) {
        const float* child_centroid =
            centroids_.data() + (first_child + child) * dimension_;
        for (std::size_t d = 0; d < dimension_; ++d)
            centroid[d] += child_centroid[d];
    }
    const float inv = 1.F / static_cast<float>(branching);
    for (std::size_t d = 0; d < dimension_; ++d) centroid[d] *= inv;
}

void VocabularyTree::train(
    const std::span<const features::FeatureSet> images,
    const VocabularyConfig& config,
    const features::DescriptorMetric metric) {
    std::vector<const features::FeatureSet*> image_refs;
    image_refs.reserve(images.size());
    for (const features::FeatureSet& image : images)
        image_refs.push_back(&image);
    train(image_refs, config, metric);
}

void VocabularyTree::train(
    const std::span<const features::FeatureSet* const> images,
    const VocabularyConfig& config,
    const features::DescriptorMetric metric) {
    if (config.branching < 2 || config.depth == 0)
        throw std::invalid_argument("Invalid vocabulary configuration");
    config_ = config;
    metric_ = metric;
    nodes_.clear();
    centroids_.clear();
    word_count_ = 0;
    dimension_ = 0;
    for (const features::FeatureSet* image : images) {
        if (image == nullptr) continue;
        const features::FeatureSet& features = *image;
        if (features.descriptor_dimension > 0) {
            dimension_ = features.descriptor_dimension;
            break;
        }
    }
    if (dimension_ == 0) return;

    // Sampling and L2-normalizing each image's descriptors is independent per
    // image; rows are concatenated in image order afterwards so the training
    // buffer matches the serial collection exactly (including the cap).
    std::vector<std::vector<float>> image_rows(images.size());
    parallel::parallel_for(
        images.size(), parallel::resolve_thread_count(0),
        [&](const std::size_t image_index) {
            const features::FeatureSet* image = images[image_index];
            if (image == nullptr) return;
            const features::FeatureSet& features = *image;
            if (features.descriptor_dimension != dimension_ ||
                (features.descriptors.empty() &&
                 features.descriptors_u8.empty()) ||
                features.keypoints.empty())
                return;
            const auto samples = sample_descriptors_spatially(
                features, config.max_descriptors_per_image, config.sample_grid);
            std::vector<float>& rows = image_rows[image_index];
            rows.reserve(samples.size() * dimension_);
            for (const features::FeatureIndex index : samples) {
                if (features.storage ==
                    aetherscan::features::DescriptorStorage::float32) {
                    const float* row = features.descriptors.data() +
                        static_cast<std::size_t>(index) * dimension_;
                    rows.insert(rows.end(), row, row + dimension_);
                } else {
                    const std::uint8_t* row = features.descriptors_u8.data() +
                        static_cast<std::size_t>(index) * dimension_;
                    double squared_norm = 0.0;
                    for (std::size_t column = 0; column < dimension_; ++column)
                        squared_norm += static_cast<double>(row[column]) *
                                        static_cast<double>(row[column]);
                    const float inverse_norm = static_cast<float>(
                        1.0 / std::sqrt(std::max(squared_norm, 1e-24)));
                    for (std::size_t column = 0; column < dimension_; ++column)
                        rows.push_back(row[column] * inverse_norm);
                }
            }
        });
    std::vector<float> training;
    training.reserve(
        std::min(config.max_training_descriptors,
                 images.size() * config.max_descriptors_per_image) *
        dimension_);
    for (const std::vector<float>& rows : image_rows) {
        const std::size_t remaining =
            config.max_training_descriptors - training.size() / dimension_;
        const std::size_t copy =
            std::min(remaining, rows.size() / dimension_);
        training.insert(
            training.end(), rows.begin(),
            rows.begin() + static_cast<std::ptrdiff_t>(copy * dimension_));
        if (training.size() / dimension_ >= config.max_training_descriptors)
            break;
    }
    const std::size_t descriptor_count = training.size() / dimension_;
    if (descriptor_count < config.branching) {
        core::Logger::instance().warning(
            "vocabulary training skipped: descriptors=", descriptor_count);
        return;
    }

    std::vector<std::size_t> subset(descriptor_count);
    std::iota(subset.begin(), subset.end(), std::size_t{0});
    nodes_.assign(1, {});
    centroids_.assign(dimension_, 0.F);
    build_into(0, subset, 0, training);
    core::Logger::instance().info(
        "vocabulary trained: words=", word_count_, " nodes=", nodes_.size(),
        " descriptors=", descriptor_count, " dim=", dimension_);
}

std::uint32_t VocabularyTree::quantize(const std::span<const float> descriptor) const {
    if (nodes_.empty() || descriptor.size() != dimension_)
        throw std::invalid_argument("Vocabulary quantize input mismatch");
    std::size_t node = 0;
    while (nodes_[node].word_id < 0) {
        const Node& current = nodes_[node];
        if (current.child_count == 0)
            throw std::runtime_error("Corrupt vocabulary node");
        float best = std::numeric_limits<float>::infinity();
        std::size_t best_child = current.first_child;
        for (std::uint16_t child = 0; child < current.child_count; ++child) {
            const std::size_t child_index = current.first_child + child;
            const float distance = squared_l2(
                descriptor.data(),
                centroids_.data() + child_index * dimension_,
                dimension_);
            if (distance < best) {
                best = distance;
                best_child = child_index;
            }
        }
        node = best_child;
    }
    return static_cast<std::uint32_t>(nodes_[node].word_id);
}

std::uint32_t VocabularyTree::quantize(
    const std::span<const std::uint8_t> descriptor) const {
    if (nodes_.empty() || descriptor.size() != dimension_)
        throw std::invalid_argument("Vocabulary quantize input mismatch");
    double squared_norm = 0.0;
    for (const std::uint8_t value : descriptor)
        squared_norm += static_cast<double>(value) * value;
    const float inverse_norm = static_cast<float>(
        1.0 / std::sqrt(std::max(squared_norm, 1e-24)));
    std::size_t node = 0;
    while (nodes_[node].word_id < 0) {
        const Node& current = nodes_[node];
        if (current.child_count == 0)
            throw std::runtime_error("Corrupt vocabulary node");
        float best = std::numeric_limits<float>::infinity();
        std::size_t best_child = current.first_child;
        for (std::uint16_t child = 0; child < current.child_count; ++child) {
            const std::size_t child_index = current.first_child + child;
            const float* centroid =
                centroids_.data() + child_index * dimension_;
            float distance = 0.F;
            for (std::size_t column = 0; column < dimension_; ++column) {
                const float delta =
                    descriptor[column] * inverse_norm - centroid[column];
                distance += delta * delta;
            }
            if (distance < best) {
                best = distance;
                best_child = child_index;
            }
        }
        node = best_child;
    }
    return static_cast<std::uint32_t>(nodes_[node].word_id);
}

void VocabularyTree::save(const std::filesystem::path& path) const {
    if (nodes_.empty()) throw std::runtime_error("Cannot save empty vocabulary");
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Failed to create vocabulary file");
    write_pod(output, k_vocab_magic);
    write_pod(output, k_vocab_version);
    write_pod(output, config_.branching);
    write_pod(output, config_.depth);
    write_pod(output, config_.max_iterations);
    write_pod(output, config_.seed);
    write_pod(output, static_cast<std::uint64_t>(config_.max_descriptors_per_image));
    write_pod(output, static_cast<std::uint64_t>(config_.max_training_descriptors));
    write_pod(output, config_.sample_grid);
    write_pod(output, static_cast<std::uint32_t>(metric_));
    write_pod(output, static_cast<std::uint64_t>(dimension_));
    write_pod(output, static_cast<std::uint64_t>(word_count_));
    write_vector(output, nodes_);
    write_vector(output, centroids_);
    if (!output) throw std::runtime_error("Failed while writing vocabulary");
}

void VocabularyTree::load(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open vocabulary file");
    const auto magic = read_pod<std::uint32_t>(input);
    const auto version = read_pod<std::uint32_t>(input);
    if (magic != k_vocab_magic || version != k_vocab_version)
        throw std::runtime_error("Unsupported vocabulary file");
    config_.branching = read_pod<std::uint32_t>(input);
    config_.depth = read_pod<std::uint32_t>(input);
    config_.max_iterations = read_pod<std::uint32_t>(input);
    config_.seed = read_pod<std::uint32_t>(input);
    config_.max_descriptors_per_image =
        static_cast<std::size_t>(read_pod<std::uint64_t>(input));
    config_.max_training_descriptors =
        static_cast<std::size_t>(read_pod<std::uint64_t>(input));
    config_.sample_grid = read_pod<unsigned>(input);
    metric_ = static_cast<features::DescriptorMetric>(
        read_pod<std::uint32_t>(input));
    dimension_ = static_cast<std::size_t>(read_pod<std::uint64_t>(input));
    word_count_ = static_cast<std::size_t>(read_pod<std::uint64_t>(input));
    nodes_ = read_vector<Node>(input, 1'000'000);
    centroids_ = read_vector<float>(input, 1'000'000ull * 512ull);
    if (centroids_.size() != nodes_.size() * dimension_)
        throw std::runtime_error("Vocabulary centroid size mismatch");
}

}  // namespace aetherscan::sfm
