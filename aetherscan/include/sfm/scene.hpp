#pragma once

#include "features/types.hpp"
#include "sfm/types.hpp"

#include <algorithm>
#include <cstdint>
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
    std::optional<double> estimated_focal;
    std::optional<Mat3> H;
    float weight_spatial{0.F};
    float weight_geometry{1.F};  // <1 for planar/low-parallax pairs
    float weight_connectivity{1.F};  // local importance at both incident views
    float weight_triplet{0.F};  // consistent rotation-cycle support [0, 1]
    float weight_cycle{1.F};  // penalty for strongly inconsistent triplets
    float mean_ray_angle{0.F};  // radians
    float homography_ratio{0.F};  // H_inliers / E_inliers
    bool degenerate_planar{false};
    bool active{true};

    ImagePair() = default;
    ImagePair(Index a, Index b) : id1(a), id2(b) {
        if (id1 > id2) std::swap(id1, id2);
    }

    [[nodiscard]] unsigned num_inliers() const {
        return static_cast<unsigned>(matches.size());
    }

    // Preserve the historical scale for edges without triplet support while
    // boosting cycle-supported edges and suppressing cycle-inconsistent ones.
    [[nodiscard]] float composite_weight() const {
        if (!active) return 0.F;
        const unsigned capped = std::min(num_inliers(), 1000u);
        return static_cast<float>(capped) * std::max(weight_spatial, 0.05F) *
               std::max(weight_geometry, 0.F) *
               std::clamp(weight_connectivity, 0.F, 1.F) *
               (1.F + 2.F * std::clamp(weight_triplet, 0.F, 1.F)) *
               std::clamp(weight_cycle, 0.F, 1.F);
    }

    // Prefer non-planar pairs for star initialization / seed selection.
    [[nodiscard]] bool usable_for_init() const {
        return active && relative_pose.has_value() && !degenerate_planar;
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
    // Runtime-only scheduling metadata. Checkpoints deliberately omit this:
    // after resume tracks are conservatively checked once, without invalidating
    // the existing checkpoint schema and large frontend caches.
    std::uint32_t split_generation{0};

    [[nodiscard]] bool is_valid() const { return observations.size() >= 2; }
    [[nodiscard]] bool is_triangulated() const { return num_inliers >= 2; }
};

struct ImageTrackRef {
    Index track_id{k_invalid};
    Index feature_id{k_invalid};
};

struct ResectionProgress {
    unsigned since_full_ba{0};
    unsigned bundle_adjustment_stage{0};
    std::vector<Index> last_registered;
    std::vector<double> recent_inlier_ratios;
};

struct Scene {
    std::vector<PinholeCamera> cameras;
    std::vector<Image> images;
    std::vector<ImagePair> pairs;
    std::vector<Track> tracks;
    // image_id -> tracks observed in that image; rebuilt whenever track
    // topology changes and used by resection/covisibility hot paths.
    std::vector<std::vector<ImageTrackRef>> image_tracks;
    ResectionProgress resection_progress;
    // Incremented whenever another image becomes registered. A track whose
    // split_generation matches this value has no newly registered outliers.
    std::uint32_t registration_generation{0};
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
