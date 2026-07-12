#include "sfm/retrieval.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace aetherscan::sfm {
namespace {

using Word = std::uint32_t;
using PairKey = std::uint64_t;

PairKey pair_key(Index first, Index second) {
    if (first > second) std::swap(first, second);
    return (static_cast<PairKey>(first) << 32U) | second;
}

std::pair<Index, Index> decode_pair(const PairKey key) {
    return {
        static_cast<Index>(key >> 32U),
        static_cast<Index>(key & 0xffffffffULL)};
}

}  // namespace

std::vector<RetrievedPair> retrieve_image_pairs(
    const std::vector<Image>& images,
    const RetrievalOptions& options) {
    if (images.size() < 2 || options.top_k == 0) return {};
    if (options.hash_tables == 0 || options.bits_per_word == 0 ||
        options.bits_per_word > 16 ||
        options.max_posting_images < 2 ||
        !(options.stop_word_ratio > 0.F && options.stop_word_ratio <= 1.F))
        throw std::invalid_argument("Invalid retrieval options");

    std::size_t dimension = 0;
    for (const Image& image : images) {
        if (image.features.descriptor_dimension > 0) {
            dimension = image.features.descriptor_dimension;
            break;
        }
    }
    if (dimension == 0) return {};

    const std::size_t bit_count =
        static_cast<std::size_t>(options.hash_tables) * options.bits_per_word;
    std::vector<std::pair<std::size_t, std::size_t>> comparisons(bit_count);
    std::mt19937 random(0xA37E5u);
    std::uniform_int_distribution<std::size_t> coordinate(0, dimension - 1);
    for (auto& comparison : comparisons) {
        comparison = {coordinate(random), coordinate(random)};
        if (comparison.first == comparison.second)
            comparison.second = (comparison.second + 1) % dimension;
    }

    using Histogram = std::unordered_map<Word, std::uint16_t>;
    std::vector<Histogram> histograms(images.size());
    std::unordered_map<Word, unsigned> document_frequency;
    for (std::size_t image_id = 0; image_id < images.size(); ++image_id) {
        const auto& features = images[image_id].features;
        if (features.descriptor_dimension != dimension ||
            features.keypoints.empty())
            continue;
        const std::size_t sample_count = std::min(
            features.keypoints.size(), options.max_descriptors_per_image);
        Histogram& histogram = histograms[image_id];
        histogram.reserve(sample_count);
        for (std::size_t sample = 0; sample < sample_count; ++sample) {
            const std::size_t row =
                sample * features.keypoints.size() / sample_count;
            const float* descriptor =
                features.descriptors.data() + row * dimension;
            for (unsigned table = 0; table < options.hash_tables; ++table) {
                Word code = 0;
                for (unsigned bit = 0; bit < options.bits_per_word; ++bit) {
                    const auto [left, right] =
                        comparisons[static_cast<std::size_t>(table) *
                                        options.bits_per_word +
                                    bit];
                    if (descriptor[left] > descriptor[right])
                        code |= Word{1} << bit;
                }
                const Word word =
                    (static_cast<Word>(table) << options.bits_per_word) | code;
                auto& count = histogram[word];
                if (count != std::numeric_limits<std::uint16_t>::max())
                    ++count;
            }
        }
        for (const auto& [word, count] : histogram) {
            (void)count;
            ++document_frequency[word];
        }
    }

    struct Posting {
        Index image{};
        double weight{};
    };
    std::unordered_map<Word, std::vector<Posting>> inverted;
    std::vector<double> norms(images.size(), 0.0);
    const std::size_t max_documents = std::max<std::size_t>(
        2, std::min(
               options.max_posting_images,
               static_cast<std::size_t>(
                   std::ceil(options.stop_word_ratio * images.size()))));
    for (std::size_t image_id = 0; image_id < histograms.size(); ++image_id) {
        for (const auto& [word, count] : histograms[image_id]) {
            const unsigned df = document_frequency[word];
            if (df > max_documents) continue;
            const double idf =
                std::log((images.size() + 1.0) / (df + 1.0)) + 1.0;
            const double tf = 1.0 + std::log(static_cast<double>(count));
            const double weight = tf * idf;
            inverted[word].push_back(
                {static_cast<Index>(image_id), weight});
            norms[image_id] += weight * weight;
        }
        norms[image_id] = std::sqrt(norms[image_id]);
    }

    std::unordered_map<PairKey, float> selected;
    std::vector<double> accumulator(images.size(), 0.0);
    std::vector<Index> touched;
    for (Index image_id = 0; image_id < histograms.size(); ++image_id) {
        if (!(norms[image_id] > 0.0)) continue;
        touched.clear();
        for (const auto& [word, count] : histograms[image_id]) {
            const auto posting_it = inverted.find(word);
            if (posting_it == inverted.end()) continue;
            const unsigned df = document_frequency[word];
            const double idf =
                std::log((images.size() + 1.0) / (df + 1.0)) + 1.0;
            const double query_weight =
                (1.0 + std::log(static_cast<double>(count))) * idf;
            for (const Posting& posting : posting_it->second) {
                if (posting.image == image_id) continue;
                if (accumulator[posting.image] == 0.0)
                    touched.push_back(posting.image);
                accumulator[posting.image] +=
                    query_weight * posting.weight;
            }
        }
        std::vector<std::pair<Index, float>> candidates;
        candidates.reserve(touched.size());
        for (const Index candidate : touched) {
            const double denominator = norms[image_id] * norms[candidate];
            if (denominator > 0.0) {
                const float score = static_cast<float>(
                    accumulator[candidate] / denominator);
                if (score > 0.F) candidates.push_back({candidate, score});
            }
            accumulator[candidate] = 0.0;
        }
        const std::size_t keep = std::min(options.top_k, candidates.size());
        std::partial_sort(
            candidates.begin(), candidates.begin() + keep, candidates.end(),
            [](const auto& left, const auto& right) {
                return left.second > right.second;
            });
        for (std::size_t i = 0; i < keep; ++i) {
            const PairKey key = pair_key(image_id, candidates[i].first);
            auto [position, inserted] = selected.try_emplace(key, candidates[i].second);
            if (!inserted)
                position->second = std::max(position->second, candidates[i].second);
        }
    }

    std::vector<RetrievedPair> result;
    result.reserve(selected.size());
    for (const auto& [key, score] : selected) {
        const auto [first, second] = decode_pair(key);
        result.push_back({first, second, score});
    }
    std::sort(
        result.begin(), result.end(),
        [](const RetrievedPair& left, const RetrievedPair& right) {
            return left.first < right.first ||
                   (left.first == right.first && left.second < right.second);
        });
    return result;
}

}  // namespace aetherscan::sfm
