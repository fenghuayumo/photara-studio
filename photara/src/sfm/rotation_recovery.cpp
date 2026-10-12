#include "sfm/rotation_recovery.hpp"

#include "core/logging.hpp"
#include "sfm/global_rotation.hpp"
#include "sfm/submap_recovery.hpp"

#include <Eigen/SVD>
#include <cmath>
#include <numbers>

namespace photara::sfm {

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
