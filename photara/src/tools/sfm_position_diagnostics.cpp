#include "sfm/asfm.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/triangulation.hpp"
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

namespace {
using namespace photara::sfm;
constexpr double degrees = 57.29577951308232;
double quantile(std::vector<double> values, double fraction) {
    if (values.empty()) return -1;
    const auto index = static_cast<std::size_t>(fraction * (values.size() - 1));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}
std::string csv(const std::string& text) {
    std::string out = "\"";
    for (char c : text) { if (c == '"') out += '"'; out += c; }
    return out + '"';
}
struct Support {
    unsigned degree{}, parallax_degree{}, boundary_degree{}, stable_candidates{}, stable_depths{}, positive{}, inliers{};
    std::set<unsigned> cells;
    std::vector<double> errors;
    Vec3 sum{Vec3::Zero()};
    Mat3 scatter{Mat3::Zero()};
};
}

int main(int argc, char** argv) {
    try {
        using namespace photara::sfm;
        if (argc != 3) throw std::invalid_argument("Usage: photara_sfm_position_diagnostics reconstruction-HEX.bin NEW_OUTPUT_PREFIX");
        const std::filesystem::path prefix(argv[2]);
        const auto views_path = std::filesystem::path(prefix.string() + ".views.csv");
        const auto pairs_path = std::filesystem::path(prefix.string() + ".pairs.csv");
        if (std::filesystem::exists(views_path) || std::filesystem::exists(pairs_path))
            throw std::invalid_argument("Diagnostic output already exists");
        const std::filesystem::path input(argv[1]);
        const auto stem = input.stem().string();
        if (!stem.starts_with("reconstruction-")) throw std::invalid_argument("A final graph checkpoint is required; ASFM omits pair geometry");
        std::size_t consumed = 0;
        const auto key = std::stoull(stem.substr(15), &consumed, 16);
        if (consumed != stem.size() - 15) throw std::invalid_argument("Invalid reconstruction checkpoint key");
        CheckpointOptions checkpoint;
        checkpoint.directory = input.parent_path(); checkpoint.write = false;
        Scene scene;
        if (!CheckpointStore(checkpoint).load_scene(CheckpointStage::reconstruction, key, scene))
            throw std::runtime_error("Cannot load final graph checkpoint");
        if (scene.pairs.empty()) throw std::runtime_error("Position diagnostics require a matched pair graph");
        const auto audit = analyze_position_observability(scene);
        std::vector<Support> support(scene.images.size());
        std::vector<float> strongest(scene.images.size());
        const auto eligible = [&](const ImagePair& pair) {
            return pair.active && pair.relative_pose && !pair.zero_baseline && !pair.degenerate_planar &&
                pair.matches.size() >= 30 && pair.id1 < scene.images.size() && pair.id2 < scene.images.size() &&
                scene.images[pair.id1].registered && scene.images[pair.id2].registered &&
                scene.images[pair.id1].camera_id < scene.cameras.size() && scene.images[pair.id2].camera_id < scene.cameras.size() &&
                std::isfinite(pair.composite_weight()) && pair.composite_weight() > 0;
        };
        for (const auto& pair : scene.pairs) if (eligible(pair))
            for (auto id : {pair.id1, pair.id2}) strongest[id] = std::max(strongest[id], pair.composite_weight());
        std::ofstream pairs(pairs_path);
        if (!pairs) throw std::runtime_error("Cannot write pairs CSV");
        pairs << std::setprecision(12)
              << "id1,id2,name1,name2,active,planar,zero_baseline,matches,weight,strong_weight,parallax_samples,parallax_1deg,parallax_p50_deg,parallax_p90_deg,translation_error_deg,stable1,stable2\n";
        for (const auto& pair : scene.pairs) {
            if (pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size()) continue;
            const auto& a = scene.images[pair.id1]; const auto& b = scene.images[pair.id2];
            if (pair.active) { ++support[pair.id1].degree; ++support[pair.id2].degree; }
            const bool strong = eligible(pair) && pair.composite_weight() >= 0.1F * std::min(strongest[pair.id1], strongest[pair.id2]);
            std::vector<double> angles;
            unsigned useful = 0;
            if (a.camera_id < scene.cameras.size() && b.camera_id < scene.cameras.size()) {
                const std::size_t samples = std::min<std::size_t>(256, pair.matches.size());
                for (std::size_t index = 0; index < samples; ++index) {
                    const auto& match = pair.matches[index * pair.matches.size() / samples];
                    if (match.query >= a.features.keypoints.size() || match.train >= b.features.keypoints.size()) continue;
                    const auto& x = a.features.keypoints[match.query]; const auto& y = b.features.keypoints[match.train];
                    const Vec3 u = a.pose.R.transpose() * scene.camera_of(a).unproject_normalized(Vec2(x.x, x.y));
                    const Vec3 v = b.pose.R.transpose() * scene.camera_of(b).unproject_normalized(Vec2(y.x, y.y));
                    if (!u.allFinite() || !v.allFinite()) continue;
                    const double cosine = std::clamp(u.dot(v), -1.0, 1.0);
                    angles.push_back(degrees * std::acos(cosine));
                    if (cosine > 0 && cosine <= 0.9998476951563913) ++useful;
                }
            }
            if (strong && useful >= 20) {
                for (auto id : {pair.id1, pair.id2}) ++support[id].parallax_degree;
                if (audit.reliable[pair.id1] != audit.reliable[pair.id2]) {
                    ++support[pair.id1].boundary_degree; ++support[pair.id2].boundary_degree;
                }
            }
            if (audit.reliable[pair.id1] && audit.reliable[pair.id2]) continue;
            double direction_error = -1;
            if (pair.relative_pose) {
                const Vec3 expected = -(b.pose.R.transpose() * pair.relative_pose->translation());
                const Vec3 actual = b.pose.C - a.pose.C;
                if (expected.allFinite() && actual.allFinite() && expected.norm() > 1e-12 && actual.norm() > 1e-12)
                    direction_error = degrees * std::acos(std::clamp(expected.normalized().dot(actual.normalized()), -1.0, 1.0));
            }
            pairs << pair.id1 << ',' << pair.id2 << ',' << csv(a.path.filename().string()) << ',' << csv(b.path.filename().string())
                  << ',' << pair.active << ',' << pair.degenerate_planar << ',' << pair.zero_baseline << ',' << pair.matches.size()
                  << ',' << pair.composite_weight() << ',' << strong << ',' << angles.size() << ',' << useful
                  << ',' << quantile(angles, .5) << ',' << quantile(angles, .9) << ',' << direction_error
                  << ',' << unsigned(audit.reliable[pair.id1]) << ',' << unsigned(audit.reliable[pair.id2]) << '\n';
        }
        TriangulationOptions triangulation;
        triangulation.reproj_threshold_px = 2;
        triangulation.min_angle_deg = 1;
        triangulation.split_tracks = false;
        for (const auto& source : scene.tracks) {
            Track anchor;
            std::vector<Observation> targets;
            std::set<Index> distinct;
            for (std::size_t index = 0; index < std::min<std::size_t>(source.num_inliers, source.observations.size()); ++index) {
                const auto& obs = source.observations[index];
                if (obs.image_id >= scene.images.size() || !scene.images[obs.image_id].registered) continue;
                if (audit.reliable[obs.image_id]) {
                    if (distinct.insert(obs.image_id).second) anchor.observations.push_back(obs);
                } else targets.push_back(obs);
            }
            if (targets.empty() || anchor.observations.size() < 2) continue;
            for (const auto& obs : targets) ++support[obs.image_id].stable_candidates;
            if (triangulate_track(anchor, scene, triangulation) < 2) continue;
            for (const auto& obs : targets) {
                auto& counts = support[obs.image_id]; ++counts.stable_depths;
                const auto& image = scene.images[obs.image_id];
                if (image.camera_id >= scene.cameras.size() || obs.feature_id >= image.features.keypoints.size()) continue;
                const auto& camera = scene.camera_of(image); const auto& keypoint = image.features.keypoints[obs.feature_id];
                Vec2 predicted;
                if (!camera.project_checked(image.pose.transform_world_to_camera(anchor.position), predicted)) continue;
                ++counts.positive;
                const double error = (predicted - Vec2(keypoint.x, keypoint.y)).norm();
                counts.errors.push_back(error);
                if (error > 2) continue;
                ++counts.inliers;
                const unsigned x = static_cast<unsigned>(std::clamp(4.F * keypoint.x / std::max(1u, camera.width), 0.F, 3.F));
                const unsigned y = static_cast<unsigned>(std::clamp(4.F * keypoint.y / std::max(1u, camera.height), 0.F, 3.F));
                counts.cells.insert(4 * y + x);
                counts.sum += anchor.position; counts.scatter += anchor.position * anchor.position.transpose();
            }
        }
        std::ofstream views(views_path);
        if (!views) throw std::runtime_error("Cannot write views CSV");
        views << std::setprecision(12)
              << "id,name,registered,stable,active_degree,strong_parallax_degree,boundary_degree,stable_track_candidates,stable_depth_tracks,positive_predictions,stable_inliers_2px,cells_4x4,stable_depth_p50_px,stable_depth_p90_px,stable_depth_rank_ratio\n";
        for (Index id = 0; id < scene.images.size(); ++id) {
            const auto& image = scene.images[id]; const auto& counts = support[id];
            double rank = -1;
            if (counts.inliers >= 3) {
                const Vec3 mean = counts.sum / counts.inliers;
                const Mat3 covariance = counts.scatter / counts.inliers - mean * mean.transpose();
                Eigen::SelfAdjointEigenSolver<Mat3> eigen(covariance);
                if (eigen.info() == Eigen::Success && eigen.eigenvalues()[2] > 1e-12)
                    rank = std::max(0.0, eigen.eigenvalues()[0]) / eigen.eigenvalues()[2];
            }
            views << id << ',' << csv(image.path.filename().string()) << ',' << image.registered
                  << ',' << unsigned(audit.reliable[id])
                  << ',' << counts.degree << ',' << counts.parallax_degree << ',' << counts.boundary_degree
                  << ',' << counts.stable_candidates << ',' << counts.stable_depths << ',' << counts.positive
                  << ',' << counts.inliers << ',' << counts.cells.size() << ',' << quantile(counts.errors, .5)
                  << ',' << quantile(counts.errors, .9) << ',' << rank << '\n';
        }
        if (!pairs || !views) throw std::runtime_error("Failed writing diagnostic CSV");
        std::cout << "registered=" << scene.registered_count() << " reliable=" << audit.reliable_views
                  << " unverified=" << audit.unreliable_views << " bridge_edges=" << audit.bridge_edges << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
