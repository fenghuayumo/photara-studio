#pragma once

#include "sfm/types.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <vector>

namespace photara::sfm::detail {

// Stratify by the cameras that independently supplied each depth. Selection
// uses ancestry alone, before pose fitting or residual evaluation. A rare
// anchor group should not disappear from validation by an unlucky shuffle.
inline std::vector<bool> stratified_resection_holdout(
    const std::vector<std::vector<Index>>& ancestry, std::size_t count) {
    count = std::min(count, ancestry.size());
    std::map<std::vector<Index>, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < ancestry.size(); ++i) {
        auto anchors = ancestry[i];
        std::sort(anchors.begin(), anchors.end());
        anchors.erase(std::unique(anchors.begin(), anchors.end()), anchors.end());
        groups[anchors].push_back(i);
    }
    std::mt19937 random(0x53464d);
    std::vector<std::vector<std::size_t>> members;
    for (auto& [anchors, indices] : groups) {
        std::shuffle(indices.begin(), indices.end(), random);
        members.push_back(std::move(indices));
    }
    std::vector<std::size_t> selected(members.size());
    std::vector<bool> heldout(ancestry.size(), false);
    for (std::size_t n = 0; n < count; ++n) {
        std::size_t best = members.size();
        double deficit = -1e100;
        for (std::size_t g = 0; g < members.size(); ++g) {
            if (selected[g] >= members[g].size()) continue;
            const double value = double(count) * members[g].size() / ancestry.size() - selected[g];
            if (value > deficit) { deficit = value; best = g; }
        }
        if (best == members.size()) break;
        heldout[members[best][selected[best]++]] = true;
    }
    return heldout;
}

}  // namespace photara::sfm::detail
