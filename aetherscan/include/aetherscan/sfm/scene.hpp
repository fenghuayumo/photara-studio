#pragma once

#include "aetherscan/features/features.hpp"
#include "aetherscan/ba/problem.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <vector>

namespace aetherscan::sfm {

using Id = std::uint32_t;
inline constexpr Id invalid_id = std::numeric_limits<Id>::max();

struct Camera {
    Id id{invalid_id};
    std::uint32_t width{};
    std::uint32_t height{};
    double fx{1.0}, fy{1.0}, cx{}, cy{};
    double k1{}, k2{}, p1{}, p2{};
};

struct View {
    Id id{invalid_id};
    Id camera_id{invalid_id};
    std::filesystem::path image_path;
    features::FeatureSet features;
    ba::Pose pose;
    bool registered{false};
};

struct TrackObservation {
    Id view_id{invalid_id};
    features::FeatureIndex feature_index{};
};

struct Track {
    Id id{invalid_id};
    std::vector<TrackObservation> observations;
};

struct Landmark {
    Id id{invalid_id};
    std::array<double, 3> position{};
    Id track_id{invalid_id};
    double mean_reprojection_error{};
};

struct VerifiedPair {
    Id first_view{invalid_id};
    Id second_view{invalid_id};
    features::MatchSet inliers;
};

struct Scene {
    std::vector<Camera> cameras;
    std::vector<View> views;
    std::vector<VerifiedPair> verified_pairs;
    std::vector<Track> tracks;
    std::vector<Landmark> landmarks;

    void validate() const;
};

// Merges geometrically verified pair matches with union-find. Components that
// contain two features from the same view are rejected as ambiguous.
std::vector<Track> build_tracks(
    const std::vector<View>& views,
    const std::vector<VerifiedPair>& pairs,
    std::size_t minimum_length = 2);

}  // namespace aetherscan::sfm
