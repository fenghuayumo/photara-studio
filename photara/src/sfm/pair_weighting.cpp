#include "sfm/pair_weighting.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace photara::sfm {
namespace {

constexpr double k_pi = 3.14159265358979323846;

struct NeighborEdge {
    Index neighbor{k_invalid};
    Index pair_index{k_invalid};
};

bool valid_pair(const Scene& scene, const ImagePair& pair, unsigned min_inliers) {
    return pair.active && pair.relative_pose.has_value() &&
           pair.id1 < scene.images.size() && pair.id2 < scene.images.size() &&
           pair.id1 != pair.id2 && pair.num_inliers() >= min_inliers &&
           pair.weight_spatial > 0.F && pair.weight_geometry > 0.F;
}

Mat3 relative_rotation(const ImagePair& pair, Index from, Index to) {
    if (pair.id1 == from && pair.id2 == to)
        return pair.relative_pose->R;
    return pair.relative_pose->R.transpose();
}

double cycle_error(
    const ImagePair& uv, const ImagePair& vk, const ImagePair& ku,
    Index u, Index v, Index k) {
    const Mat3 closure = relative_rotation(ku, k, u) *
                         relative_rotation(vk, v, k) *
                         relative_rotation(uv, u, v);
    const Eigen::AngleAxisd angle_axis(closure);
    return std::isfinite(angle_axis.angle())
        ? std::abs(angle_axis.angle())
        : k_pi;
}

}  // namespace

PairWeightingSummary compute_pair_weights(
    Scene& scene, const PairWeightingOptions& options) {
    PairWeightingSummary summary;
    if (scene.images.empty() || scene.pairs.empty()) return summary;

    const unsigned min_inliers = std::max(1U, options.min_inliers);
    std::vector<unsigned> strongest(scene.images.size(), 0);
    std::vector<std::vector<NeighborEdge>> adjacency(scene.images.size());

    for (Index pair_index = 0; pair_index < scene.pairs.size(); ++pair_index) {
        ImagePair& pair = scene.pairs[pair_index];
        pair.weight_connectivity = 0.F;
        pair.weight_triplet = 0.F;
        pair.weight_cycle = 1.F;
        if (!valid_pair(scene, pair, min_inliers)) continue;
        strongest[pair.id1] = std::max(strongest[pair.id1], pair.num_inliers());
        strongest[pair.id2] = std::max(strongest[pair.id2], pair.num_inliers());
        adjacency[pair.id1].push_back({pair.id2, pair_index});
        adjacency[pair.id2].push_back({pair.id1, pair_index});
    }

    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end(),
            [](const NeighborEdge& left, const NeighborEdge& right) {
                return left.neighbor < right.neighbor;
            });
    }

    const double threshold = std::max(0.0, options.max_triplet_rotation_error_deg) *
                             k_pi / 180.0;
    for (Index pair_index = 0; pair_index < scene.pairs.size(); ++pair_index) {
        ImagePair& pair = scene.pairs[pair_index];
        if (!valid_pair(scene, pair, min_inliers)) continue;

        const double count = static_cast<double>(pair.num_inliers());
        const double max_first = std::max(1U, strongest[pair.id1]);
        const double max_second = std::max(1U, strongest[pair.id2]);
        pair.weight_connectivity = static_cast<float>(std::clamp(
            std::sqrt((count / max_first) * (count / max_second)), 0.0, 1.0));

        unsigned consistent = 0;
        unsigned inconsistent = 0;
        const auto& first = adjacency[pair.id1];
        const auto& second = adjacency[pair.id2];
        std::size_t i = 0;
        std::size_t j = 0;
        while (i < first.size() && j < second.size()) {
            if (first[i].neighbor < second[j].neighbor) {
                ++i;
                continue;
            }
            if (second[j].neighbor < first[i].neighbor) {
                ++j;
                continue;
            }
            const Index common = first[i].neighbor;
            if (common != pair.id1 && common != pair.id2) {
                const ImagePair& first_common = scene.pairs[first[i].pair_index];
                const ImagePair& second_common = scene.pairs[second[j].pair_index];
                const double error = cycle_error(
                    pair, second_common, first_common,
                    pair.id1, pair.id2, common);
                if (error <= threshold)
                    ++consistent;
                else
                    ++inconsistent;
            }
            ++i;
            ++j;
        }

        const unsigned total = consistent + inconsistent;
        summary.tested_triplets += total;
        if (total > 0) {
            pair.weight_triplet = static_cast<float>(consistent) /
                (static_cast<float>(total) +
                 std::max(0.F, options.triplet_saturation));
            if (consistent > 0) ++summary.supported_pairs;
            const float inconsistent_ratio =
                static_cast<float>(inconsistent) / static_cast<float>(total);
            if (total >= options.min_triplets_for_penalty &&
                inconsistent_ratio > options.max_inconsistent_triplet_ratio) {
                pair.weight_cycle = std::clamp(
                    options.inconsistent_triplet_scale, 0.F, 1.F);
                ++summary.inconsistent_pairs;
            }
        }
        ++summary.weighted_pairs;
    }

    // Each triangle is observed from all three incident edges.
    summary.tested_triplets /= 3;
    core::Logger::instance().info(
        "pair weighting: weighted=", summary.weighted_pairs,
        " cycle_supported=", summary.supported_pairs,
        " cycle_inconsistent=", summary.inconsistent_pairs,
        " triplets=", summary.tested_triplets);
    return summary;
}

}  // namespace photara::sfm
