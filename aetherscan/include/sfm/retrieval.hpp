#pragma once

#include "sfm/scene.hpp"

#include <cstddef>
#include <vector>

namespace aetherscan::sfm {

struct RetrievalOptions {
    std::size_t top_k{20};
    std::size_t max_descriptors_per_image{2000};
    unsigned hash_tables{4};
    unsigned bits_per_word{12};
    float stop_word_ratio{0.5F};
    std::size_t max_posting_images{64};
};

struct RetrievedPair {
    Index first{k_invalid};
    Index second{k_invalid};
    float score{0.F};
};

// Lightweight multi-table descriptor LSH + TF-IDF inverted index. This stage
// only proposes image pairs; descriptor ANN and geometry still verify them.
std::vector<RetrievedPair> retrieve_image_pairs(
    const std::vector<Image>& images,
    const RetrievalOptions& options = {});

}  // namespace aetherscan::sfm
