#pragma once

#include "aetherscan/features/types.hpp"
#include "aetherscan/sfm/types.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

namespace aetherscan::sfm {

struct Image {
    Index id{k_invalid};
    Index camera_id{k_invalid};
    std::filesystem::path path;
    features::FeatureSet features;
    Pose3D pose;
    bool registered{false};

    [[nodiscard]] bool has_features() const { return !features.keypoints.empty(); }
};

struct ImagePair {
    Index id1{k_invalid};
    Index id2{k_invalid};
    std::vector<FeatureMatch> matches;  // geometric inliers, ordered
    std::optional<Pose3D> relative_pose;
    std::optional<Mat3> E;
    std::optional<Mat3> F;
    float weight_spatial{0.F};
    float mean_ray_angle{0.F};  // radians
    bool active{true};

    ImagePair() = default;
    ImagePair(Index a, Index b) : id1(a), id2(b) {
        if (id1 > id2) std::swap(id1, id2);
    }

    [[nodiscard]] unsigned num_inliers() const {
        return static_cast<unsigned>(matches.size());
    }

    // openMVS-style composite weight (spatial * capped inliers); connectivity/triplet
    // filled later when available.
    [[nodiscard]] float composite_weight() const {
        if (!active) return 0.F;
        const unsigned capped = std::min(num_inliers(), 1000u);
        return static_cast<float>(capped) * std::max(weight_spatial, 0.05F);
    }

    [[nodiscard]] bool has_geometry() const {
        return relative_pose.has_value() || E.has_value() || F.has_value();
    }
};

// Track mirrors openMVS: inlier observations occupy [0, num_inliers).
struct Track {
    Vec3 position{Vec3::Zero()};
    std::vector<Observation> observations;
    std::uint8_t num_inliers{0};

    [[nodiscard]] bool is_valid() const { return observations.size() >= 2; }
    [[nodiscard]] bool is_triangulated() const { return num_inliers >= 2; }
};

struct Scene {
    std::vector<PinholeCamera> cameras;
    std::vector<Image> images;
    std::vector<ImagePair> pairs;
    std::vector<Track> tracks;
    unsigned thread_count{0};  // 0 = hardware concurrency

    void clear();
    [[nodiscard]] bool empty() const { return images.empty(); }

    ImagePair* find_pair(Index a, Index b);
    [[nodiscard]] const ImagePair* find_pair(Index a, Index b) const;

    [[nodiscard]] const PinholeCamera& camera_of(const Image& image) const {
        return cameras[image.camera_id];
    }

    [[nodiscard]] unsigned registered_count() const;
    bool invalidate_image(Index image_id);
};

}  // namespace aetherscan::sfm
