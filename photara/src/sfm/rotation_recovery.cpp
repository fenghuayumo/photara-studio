#include "sfm/rotation_recovery.hpp"

#include "core/logging.hpp"
#include "sfm/global_rotation.hpp"
#include "sfm/submap_recovery.hpp"
#include "sfm/tracks.hpp"

#include <Eigen/SVD>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <set>

namespace photara::sfm {

std::vector<std::uint8_t> verified_rotation_conflicts(const Scene& scene) {
    std::vector<double> total(scene.images.size()), bad(scene.images.size());
    std::vector<std::set<Index>> bad_neighbors(scene.images.size());
    for (const ImagePair& pair : scene.pairs) {
        const double weight = pair.composite_weight();
        if (!pair.active || !pair.relative_pose || pair.zero_baseline ||
            pair.degenerate_planar || pair.num_inliers() < 100 ||
            pair.weight_triplet < 0.25F || pair.weight_cycle < 0.5F ||
            !std::isfinite(weight) || weight < 30.0 ||
            pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size() ||
            !scene.images[pair.id1].registered || !scene.images[pair.id2].registered)
            continue;
        const Mat3 actual = scene.images[pair.id2].pose.R *
            scene.images[pair.id1].pose.R.transpose();
        const double degrees = Eigen::AngleAxisd(
            actual * pair.relative_pose->R.transpose()).angle() * 180.0 / std::numbers::pi;
        for (const Index id : {pair.id1, pair.id2}) {
            total[id] += weight;
            if (degrees > 5.0) {
                bad[id] += weight;
                bad_neighbors[id].insert(id == pair.id1 ? pair.id2 : pair.id1);
            }
        }
    }
    std::vector<std::uint8_t> result(scene.images.size());
    for (Index id = 0; id < scene.images.size(); ++id)
        result[id] = bad_neighbors[id].size() >= 2 && bad[id] >= 0.8 * total[id];
    return result;
}

unsigned prune_verified_rotation_conflicts(Scene& scene) {
    const auto removed = verified_rotation_conflicts(scene);
    unsigned invalidated = 0;
    for (Index id = 0; id < scene.images.size(); ++id) {
        if (!removed[id]) continue;
        scene.images[id].registered = false;
        ++invalidated;
        core::Logger::instance().warning("final rotation audit: rejected image=",
            scene.images[id].path.filename());
    }
    if (invalidated == 0) return 0;
    for (Track& track : scene.tracks) {
        const auto end = track.observations.begin() + std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        const auto kept = std::stable_partition(track.observations.begin(), end,
            [&](const Observation& observation) {
                return observation.image_id < removed.size() && !removed[observation.image_id];
            });
        track.num_inliers = static_cast<std::uint8_t>(std::min<std::size_t>(
            std::distance(track.observations.begin(), kept), 255));
        if (track.num_inliers < 2) track.num_inliers = 0;
        track.split_generation = 0;
    }
    rebuild_track_index(scene);
    return invalidated;
}

RotationRecoveryReport recover_rotation_inconsistent_views(Scene& scene) {
    core::StageScope stage("sfm.rotation_recovery");
    RotationRecoveryReport report;
    if (scene.registered_count() < 4) return report;
    Scene graph;
    graph.images.resize(scene.images.size());
    for (Index id = 0; id < graph.images.size(); ++id) graph.images[id].id = id;
    graph.pairs.reserve(scene.pairs.size());
    for (const auto& pair : scene.pairs) {
        const float weight = pair.composite_weight();
        if (!pair.active || !pair.relative_pose || !std::isfinite(weight) || weight <= 0)
            continue;
        // Rotation averaging needs only the composite edge weight. Preserve
        // it without copying the potentially large feature-match arrays.
        ImagePair edge(pair.id1, pair.id2);
        edge.relative_pose = pair.relative_pose;
        edge.matches.resize(1);
        edge.weight_spatial = 1.F;
        edge.weight_geometry = weight;
        edge.degenerate_planar = pair.degenerate_planar;
        edge.zero_baseline = pair.zero_baseline;
        graph.pairs.push_back(std::move(edge));
    }
    if (!estimate_global_rotations(graph).success) return report;
    const auto angle = [](const Mat3& a, const Mat3& b) {
        return Eigen::AngleAxisd(a * b.transpose()).angle() * 180.0 / std::numbers::pi;
    };
    constexpr double threshold = 5.0;
    Mat3 gauge = Mat3::Identity();
    std::size_t common = 0, consensus = 0;
    for (unsigned pass = 0; pass < 3; ++pass) {
        Mat3 sum = Mat3::Zero();
        consensus = 0;
        for (Index id = 0; id < scene.images.size(); ++id) {
            const auto& image = scene.images[id];
            if (!image.registered || !graph.images[id].registered) continue;
            if (pass == 0) ++common;
            const Mat3& rotation = graph.images[id].pose.R;
            if (pass > 0 && angle(image.pose.R, rotation * gauge) > threshold) continue;
            sum += rotation.transpose() * image.pose.R;
            ++consensus;
        }
        if (consensus < 3) return report;
        const Eigen::JacobiSVD<Mat3> svd(sum, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat3 sign = Mat3::Identity();
        sign(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
        gauge = svd.matrixU() * sign * svd.matrixV().transpose();
    }
    // Without a dominant shared gauge, disagreement alone cannot identify
    // which reconstruction is trustworthy. Leave such scenes unchanged.
    if (consensus < 0.8 * common) return report;
    std::vector<std::uint8_t> stable(scene.images.size());
    for (Index id = 0; id < scene.images.size(); ++id) {
        const auto& image = scene.images[id];
        stable[id] = image.registered && graph.images[id].registered;
        if (!stable[id]) continue;
        const double error = angle(image.pose.R, graph.images[id].pose.R * gauge);
        if (error <= threshold) continue;
        stable[id] = 0;
        report.candidates.push_back(id);
        core::Logger::instance().info("rotation recovery candidate: image=",
            image.path.filename(), " error_deg=", error);
    }
    // Each pass freezes its anchors. Only independently validated results
    // become anchors for the next pass; unvalidated candidates never do.
    for (std::size_t pass = 0; pass < report.candidates.size(); ++pass) {
        const auto accepted = recover_stable_resections(scene, stable);
        if (accepted.empty()) break;
        for (const Index id : accepted) stable[id] = 1;
        report.recovered.insert(report.recovered.end(), accepted.begin(), accepted.end());
    }
    if (!report.candidates.empty())
        core::Logger::instance().info("rotation recovery: candidates=", report.candidates.size(),
            " recovered=", report.recovered.size());
    return report;
}

}  // namespace photara::sfm
