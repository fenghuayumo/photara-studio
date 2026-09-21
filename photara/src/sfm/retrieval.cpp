#include "sfm/retrieval.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace photara::sfm {
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

struct Posting {
    Index image{};
    float weight{};
};

}  // namespace

VocabularyTree load_or_train_vocabulary(
    const std::vector<Image>& images,
    const RetrievalOptions& options) {
    VocabularyTree vocabulary;
    VocabularyConfig config = options.vocabulary;
    config.max_descriptors_per_image = options.max_descriptors_per_image;
    config.sample_grid = options.sample_grid;
    if (!options.vocabulary_path.empty() &&
        std::filesystem::exists(options.vocabulary_path)) {
        vocabulary.load(options.vocabulary_path);
        core::Logger::instance().info(
            "vocabulary loaded: path=", options.vocabulary_path,
            " words=", vocabulary.word_count());
        return vocabulary;
    }

    std::vector<const features::FeatureSet*> feature_sets;
    feature_sets.reserve(images.size());
    for (const Image& image : images) feature_sets.push_back(&image.features);
    features::DescriptorMetric metric = features::DescriptorMetric::l2_root;
    for (const Image& image : images) {
        if (!image.features.descriptors.empty()) {
            metric = image.features.metric;
            break;
        }
    }
    vocabulary.train(feature_sets, config, metric);
    if (!vocabulary.empty() && !options.vocabulary_path.empty()) {
        vocabulary.save(options.vocabulary_path);
        core::Logger::instance().info(
            "vocabulary saved: path=", options.vocabulary_path,
            " words=", vocabulary.word_count());
    }
    return vocabulary;
}

std::vector<RetrievedPair> retrieve_image_pairs(
    const std::vector<Image>& images,
    const RetrievalOptions& options) {
    if (images.size() < 2 || options.top_k == 0) return {};
    if (options.max_posting_images < 2 ||
        !(options.stop_word_ratio > 0.F && options.stop_word_ratio <= 1.F))
        throw std::invalid_argument("Invalid retrieval options");

    VocabularyTree vocabulary = load_or_train_vocabulary(images, options);
    if (vocabulary.empty() || vocabulary.word_count() == 0) return {};
    const std::size_t dimension = vocabulary.dimension();

    using Histogram = std::unordered_map<Word, std::uint16_t>;
    std::vector<Histogram> histograms(images.size());
    const unsigned threads = parallel::resolve_thread_count(0);
    parallel::parallel_for(
        images.size(), threads, [&](const std::size_t image_id) {
            const auto& features = images[image_id].features;
            if (features.descriptor_dimension != dimension ||
                features.keypoints.empty())
                return;
            const auto samples = sample_descriptors_spatially(
                features, options.max_descriptors_per_image, options.sample_grid);
            Histogram& histogram = histograms[image_id];
            histogram.reserve(samples.size());
            for (const features::FeatureIndex index : samples) {
                Word word{};
                if (features.storage ==
                    features::DescriptorStorage::float32) {
                    const float* row = features.descriptors.data() +
                        static_cast<std::size_t>(index) * dimension;
                    word = vocabulary.quantize(
                        std::span<const float>(row, dimension));
                } else {
                    const std::uint8_t* row =
                        features.descriptors_u8.data() +
                        static_cast<std::size_t>(index) * dimension;
                    word = vocabulary.quantize(
                        std::span<const std::uint8_t>(row, dimension));
                }
                auto& count = histogram[word];
                if (count != std::numeric_limits<std::uint16_t>::max()) ++count;
            }
        });

    const unsigned shard_count = std::max(1U, threads);
    std::vector<std::unordered_map<Word, unsigned>> df_shards(shard_count);
    parallel::parallel_for(
        images.size(), threads,
        [&](const std::size_t image_id, const unsigned tid) {
            auto& local = df_shards[tid % shard_count];
            for (const auto& [word, count] : histograms[image_id]) {
                (void)count;
                ++local[word];
            }
        });
    std::unordered_map<Word, unsigned> document_frequency;
    for (auto& shard : df_shards) {
        if (document_frequency.empty()) {
            document_frequency = std::move(shard);
            continue;
        }
        for (const auto& [word, count] : shard) document_frequency[word] += count;
    }

    const std::size_t max_documents = std::max<std::size_t>(
        2, std::min(
               options.max_posting_images,
               static_cast<std::size_t>(
                   std::ceil(options.stop_word_ratio * images.size()))));
    std::unordered_map<Word, float> idf_by_word;
    idf_by_word.reserve(document_frequency.size());
    const auto fill_idf = [&](const bool apply_stop_words) {
        idf_by_word.clear();
        for (const auto& [word, df] : document_frequency) {
            if (apply_stop_words && df > max_documents) continue;
            idf_by_word.emplace(
                word,
                static_cast<float>(
                    std::log((images.size() + 1.0) / (df + 1.0)) + 1.0));
        }
    };
    fill_idf(true);
    // Tiny scenes or collapsed vocabularies can mark every word as a stop
    // word; fall back to unfiltered IDF so retrieval still returns pairs.
    if (idf_by_word.empty()) fill_idf(false);

    struct WeightedWord {
        Word word{};
        float weight{};
    };
    std::vector<std::vector<WeightedWord>> image_words(images.size());
    std::vector<double> norms(images.size(), 0.0);
    parallel::parallel_for(
        images.size(), threads, [&](const std::size_t image_id) {
            auto& words = image_words[image_id];
            words.reserve(histograms[image_id].size());
            double norm_sq = 0.0;
            for (const auto& [word, count] : histograms[image_id]) {
                const auto idf_it = idf_by_word.find(word);
                if (idf_it == idf_by_word.end()) continue;
                const float weight = static_cast<float>(
                    (1.0 + std::log(static_cast<double>(count))) *
                    idf_it->second);
                words.push_back({word, weight});
                norm_sq += static_cast<double>(weight) * weight;
            }
            std::sort(
                words.begin(), words.end(),
                [](const WeightedWord& left, const WeightedWord& right) {
                    return left.word < right.word;
                });
            norms[image_id] = std::sqrt(norm_sq);
        });

    std::vector<std::pair<Word, Posting>> flat_postings;
    std::size_t posting_count = 0;
    for (const auto& words : image_words) posting_count += words.size();
    flat_postings.reserve(posting_count);
    for (Index image_id = 0; image_id < image_words.size(); ++image_id) {
        for (const WeightedWord& entry : image_words[image_id])
            flat_postings.push_back({entry.word, {image_id, entry.weight}});
    }
    std::sort(
        flat_postings.begin(), flat_postings.end(),
        [](const auto& left, const auto& right) {
            if (left.first != right.first) return left.first < right.first;
            return left.second.image < right.second.image;
        });

    std::vector<Word> words;
    std::vector<std::uint32_t> offsets;
    std::vector<Posting> postings;
    words.reserve(idf_by_word.size());
    offsets.reserve(idf_by_word.size() + 1);
    postings.reserve(flat_postings.size());
    offsets.push_back(0);
    for (std::size_t begin = 0; begin < flat_postings.size();) {
        const Word word = flat_postings[begin].first;
        std::size_t end = begin + 1;
        while (end < flat_postings.size() && flat_postings[end].first == word)
            ++end;
        words.push_back(word);
        for (std::size_t i = begin; i < end; ++i)
            postings.push_back(flat_postings[i].second);
        offsets.push_back(static_cast<std::uint32_t>(postings.size()));
        begin = end;
    }

    const auto find_word = [&](const Word word) -> std::ptrdiff_t {
        const auto it = std::lower_bound(words.begin(), words.end(), word);
        if (it == words.end() || *it != word) return -1;
        return it - words.begin();
    };

    // Reuse one dense score buffer per worker. Constructing and zero-filling an
    // image_count-sized vector for every query made retrieval O(N^2) even when
    // the inverted index touched only a small candidate set.
    std::vector<std::vector<double>> accumulators(
        threads, std::vector<double>(images.size(), 0.0));
    std::vector<std::unordered_map<PairKey, float>> selected_shards(threads);
    parallel::parallel_for(
        images.size(), threads,
        [&](const std::size_t image_id, const unsigned tid) {
            if (!(norms[image_id] > 0.0)) return;
            std::vector<double>& accumulator = accumulators[tid];
            std::vector<Index> touched;
            touched.reserve(64);
            for (const WeightedWord& entry : image_words[image_id]) {
                const std::ptrdiff_t word_index = find_word(entry.word);
                if (word_index < 0) continue;
                const std::uint32_t begin = offsets[word_index];
                const std::uint32_t end = offsets[word_index + 1];
                for (std::uint32_t i = begin; i < end; ++i) {
                    const Posting& posting = postings[i];
                    if (posting.image == static_cast<Index>(image_id)) continue;
                    if (accumulator[posting.image] == 0.0)
                        touched.push_back(posting.image);
                    accumulator[posting.image] +=
                        static_cast<double>(entry.weight) * posting.weight;
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
            }
            for (const Index candidate : touched) accumulator[candidate] = 0.0;
            const std::size_t keep = std::min(options.top_k, candidates.size());
            if (keep == 0) return;
            std::partial_sort(
                candidates.begin(), candidates.begin() + keep, candidates.end(),
                [](const auto& left, const auto& right) {
                    return left.second > right.second;
                });
            auto& selected = selected_shards[tid];
            for (std::size_t i = 0; i < keep; ++i) {
                const PairKey key =
                    pair_key(static_cast<Index>(image_id), candidates[i].first);
                auto [position, inserted] =
                    selected.try_emplace(key, candidates[i].second);
                if (!inserted)
                    position->second =
                        std::max(position->second, candidates[i].second);
            }
        });

    std::unordered_map<PairKey, float> selected;
    for (auto& shard : selected_shards) {
        if (selected.empty()) {
            selected = std::move(shard);
            continue;
        }
        for (const auto& [key, score] : shard) {
            auto [position, inserted] = selected.try_emplace(key, score);
            if (!inserted) position->second = std::max(position->second, score);
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

}  // namespace photara::sfm
