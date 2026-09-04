#pragma once

#include "sfm/scene.hpp"
#include "sfm/vocabulary.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace aetherscan::sfm {

struct RetrievalOptions {
    // Match openMVS' large-scene retrieval budget.  Twenty neighbors leaves
    // multi-room/loop datasets fragmented even when every retained pair is
    // geometrically sound; fifty remains linear per image while providing
    // enough cross-cluster links for robust global/hierarchical alignment.
    std::size_t top_k{50};
    std::size_t max_descriptors_per_image{2000};
    unsigned sample_grid{3};
    float stop_word_ratio{0.5F};
    std::size_t max_posting_images{64};
    VocabularyConfig vocabulary{};
    // Empty path trains an ephemeral vocabulary for this scene only.
    std::filesystem::path vocabulary_path;
};

struct RetrievedPair {
    Index first{k_invalid};
    Index second{k_invalid};
    float score{0.F};
};

// Learned hierarchical vocabulary + TF-IDF inverted index with stop-word /
// posting-length caps. Only proposes image pairs for descriptor matching.
std::vector<RetrievedPair> retrieve_image_pairs(
    const std::vector<Image>& images,
    const RetrievalOptions& options = {});

// Load vocabulary from options.path or train+optionally persist.
VocabularyTree load_or_train_vocabulary(
    const std::vector<Image>& images,
    const RetrievalOptions& options);

}  // namespace aetherscan::sfm
