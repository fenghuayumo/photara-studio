#include "aetherscan/sfm/scene.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace aetherscan::sfm {
namespace {

class DisjointSet {
public:
    explicit DisjointSet(const std::size_t count) : parent_(count), rank_(count, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    std::size_t find(std::size_t value) {
        while (parent_[value] != value) {
            parent_[value] = parent_[parent_[value]];
            value = parent_[value];
        }
        return value;
    }
    void unite(std::size_t first, std::size_t second) {
        first = find(first); second = find(second);
        if (first == second) return;
        if (rank_[first] < rank_[second]) std::swap(first, second);
        parent_[second] = first;
        if (rank_[first] == rank_[second]) ++rank_[first];
    }
private:
    std::vector<std::size_t> parent_;
    std::vector<std::uint8_t> rank_;
};

}  // namespace

void Scene::validate() const {
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        if (cameras[i].id != i || cameras[i].width == 0 || cameras[i].height == 0 ||
            cameras[i].fx <= 0.0 || cameras[i].fy <= 0.0)
            throw std::invalid_argument("Invalid or non-contiguous camera record");
    }
    for (std::size_t i = 0; i < views.size(); ++i) {
        if (views[i].id != i || views[i].camera_id >= cameras.size())
            throw std::invalid_argument("Invalid or non-contiguous view record");
        views[i].features.validate();
    }
    for (const auto& pair : verified_pairs) {
        if (pair.first_view >= views.size() || pair.second_view >= views.size() ||
            pair.first_view >= pair.second_view)
            throw std::invalid_argument("Invalid verified image pair");
    }
}

std::vector<Track> build_tracks(
    const std::vector<View>& views, const std::vector<VerifiedPair>& pairs,
    const std::size_t minimum_length) {
    if (minimum_length < 2) throw std::invalid_argument("Track length must be at least two");
    std::unordered_map<Id, std::size_t> positions;
    std::vector<std::size_t> offsets(views.size() + 1, 0);
    for (std::size_t i = 0; i < views.size(); ++i) {
        if (!positions.emplace(views[i].id, i).second)
            throw std::invalid_argument("Duplicate view ID");
        offsets[i + 1] = offsets[i] + views[i].features.keypoints.size();
    }
    DisjointSet components(offsets.back());
    for (const auto& pair : pairs) {
        const auto first_position = positions.find(pair.first_view);
        const auto second_position = positions.find(pair.second_view);
        if (first_position == positions.end() || second_position == positions.end())
            throw std::invalid_argument("Verified pair references an unknown view");
        for (const auto& match : pair.inliers.matches) {
            if (match.query >= views[first_position->second].features.keypoints.size() ||
                match.train >= views[second_position->second].features.keypoints.size())
                throw std::out_of_range("Verified match references an unknown feature");
            components.unite(offsets[first_position->second] + match.query,
                             offsets[second_position->second] + match.train);
        }
    }
    std::unordered_map<std::size_t, std::vector<TrackObservation>> grouped;
    for (std::size_t view = 0; view < views.size(); ++view)
        for (std::size_t feature = 0; feature < views[view].features.keypoints.size(); ++feature) {
            const std::size_t global = offsets[view] + feature;
            grouped[components.find(global)].push_back(
                {views[view].id, static_cast<features::FeatureIndex>(feature)});
        }
    std::vector<Track> tracks;
    for (auto& [root, observations] : grouped) {
        (void)root;
        if (observations.size() < minimum_length) continue;
        std::sort(observations.begin(), observations.end(),
                  [](const auto& a, const auto& b) { return a.view_id < b.view_id; });
        bool ambiguous = false;
        for (std::size_t i = 1; i < observations.size(); ++i)
            ambiguous |= observations[i - 1].view_id == observations[i].view_id;
        if (!ambiguous) tracks.push_back({static_cast<Id>(tracks.size()), std::move(observations)});
    }
    return tracks;
}

}  // namespace aetherscan::sfm
