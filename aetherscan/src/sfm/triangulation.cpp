#include "sfm/triangulation.hpp"

#include "core/logging.hpp"
#include "sfm/tracks.hpp"

#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace aetherscan::sfm {
namespace {

constexpr double k_rad2deg = 180.0 / 3.14159265358979323846;

Vec2 keypoint_xy(const Image& image, const Index feature_id) {
    const auto& kp = image.features.keypoints[feature_id];
    return {kp.x, kp.y};
}

struct ObservationGeometry {
    Index obs_index{};
    Index image_id{};
    Index feature_id{};
    Vec3 center{};
    Vec3 world_dir{};
    Mat3 DR{};
    Vec3 Dt{};
    Vec2 pixel{};
    const PinholeCamera* camera{nullptr};
    const Image* image{nullptr};
};

struct Consensus {
    Vec3 position{Vec3::Zero()};
    std::vector<std::size_t> inlier_indices;
    double score{std::numeric_limits<double>::infinity()};
};

bool collect_observation_geometry(
    const Track& track,
    const Scene& scene,
    std::vector<ObservationGeometry>& cams) {
    cams.clear();
    cams.reserve(track.observations.size());
    for (Index obs_index = 0; obs_index < track.observations.size(); ++obs_index) {
        const Observation& obs = track.observations[obs_index];
        if (obs.image_id >= scene.images.size()) continue;
        const Image& image = scene.images[obs.image_id];
        if (!image.registered) continue;
        if (obs.feature_id >= image.features.keypoints.size()) continue;
        const PinholeCamera& camera = scene.camera_of(image);
        const Vec2 pixel = keypoint_xy(image, obs.feature_id);
        const Vec3 bearing = camera.unproject_normalized(pixel);
        Mat3 Dcross;
        Dcross << 0, -bearing.z(), bearing.y(), bearing.z(), 0, -bearing.x(),
            -bearing.y(), bearing.x(), 0;
        ObservationGeometry row;
        row.obs_index = obs_index;
        row.image_id = obs.image_id;
        row.feature_id = obs.feature_id;
        row.center = image.pose.C;
        row.world_dir = (image.pose.R.transpose() * bearing).normalized();
        row.DR = Dcross * image.pose.R;
        row.Dt = -Dcross * image.pose.translation();
        row.pixel = pixel;
        row.camera = &camera;
        row.image = &image;
        cams.push_back(std::move(row));
    }
    return !cams.empty();
}

bool solve_linear_lls(
    const std::vector<ObservationGeometry>& cams,
    const std::vector<std::size_t>& indices,
    Vec3& position) {
    if (indices.size() < 2) return false;
    Mat3 normal = Mat3::Zero();
    Vec3 rhs = Vec3::Zero();
    for (const std::size_t index : indices) {
        const ObservationGeometry& cam = cams[index];
        for (Eigen::Index row = 0; row < 2; ++row) {
            const Eigen::Vector3d a = cam.DR.row(row).transpose();
            normal.noalias() += a * a.transpose();
            rhs.noalias() += a * cam.Dt(row);
        }
    }

    Eigen::LDLT<Mat3> ldlt(normal);
    if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
        const auto diagonal = ldlt.vectorD().cwiseAbs();
        const double largest = diagonal.maxCoeff();
        const double smallest = diagonal.minCoeff();
        // Normal equations are substantially faster for the common
        // well-conditioned case. Avoid squaring an already poor condition
        // number by retaining the previous SVD path as a guarded fallback.
        if (largest > 0.0 && smallest > largest * 1e-10) {
            position = ldlt.solve(rhs);
            if (ldlt.info() == Eigen::Success && position.allFinite())
                return true;
        }
    }

    Eigen::MatrixXd A(2 * static_cast<Eigen::Index>(indices.size()), 3);
    Eigen::VectorXd b(2 * static_cast<Eigen::Index>(indices.size()));
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const ObservationGeometry& cam = cams[indices[i]];
        A.row(static_cast<Eigen::Index>(2 * i)) = cam.DR.row(0);
        b(static_cast<Eigen::Index>(2 * i)) = cam.Dt(0);
        A.row(static_cast<Eigen::Index>(2 * i + 1)) = cam.DR.row(1);
        b(static_cast<Eigen::Index>(2 * i + 1)) = cam.Dt(1);
    }
    position =
        A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    return position.allFinite();
}

bool triangulate_two_view_midpoint(
    const ObservationGeometry& first,
    const ObservationGeometry& second,
    Vec3& position) {
    const Vec3& d1 = first.world_dir;
    const Vec3& d2 = second.world_dir;
    const Mat3 A = Mat3::Identity() - d1 * d1.transpose() + Mat3::Identity() -
                   d2 * d2.transpose();
    const Vec3 rhs =
        (Mat3::Identity() - d1 * d1.transpose()) * first.center +
        (Mat3::Identity() - d2 * d2.transpose()) * second.center;
    Eigen::LDLT<Mat3> ldlt(A);
    if (ldlt.info() != Eigen::Success) return false;
    position = ldlt.solve(rhs);
    return position.allFinite();
}

bool observation_supports_point(
    const ObservationGeometry& cam,
    const Vec3& position,
    const float reproj_threshold_px,
    double* reproj_error = nullptr) {
    Vec2 proj;
    const Vec3 camera_point =
        cam.image->pose.transform_world_to_camera(position);
    if (!cam.camera->project_checked(camera_point, proj)) return false;
    const double error = (proj - cam.pixel).norm();
    if (reproj_error) *reproj_error = error;
    return std::isfinite(error) &&
           error <= static_cast<double>(reproj_threshold_px);
}

double score_consensus(
    const std::vector<ObservationGeometry>& cams,
    const std::vector<std::size_t>& inliers,
    const Vec3& position,
    const float reproj_threshold_px) {
    if (inliers.empty()) return std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (const std::size_t index : inliers) {
        double error = 0.0;
        if (!observation_supports_point(
                cams[index], position, reproj_threshold_px, &error))
            return std::numeric_limits<double>::infinity();
        sum += error * error;
    }
    return sum / static_cast<double>(inliers.size());
}

std::vector<std::size_t> classify_inliers(
    const std::vector<ObservationGeometry>& cams,
    const Vec3& position,
    const float reproj_threshold_px) {
    std::vector<std::size_t> inliers;
    inliers.reserve(cams.size());
    for (std::size_t i = 0; i < cams.size(); ++i) {
        if (observation_supports_point(cams[i], position, reproj_threshold_px))
            inliers.push_back(i);
    }
    return inliers;
}

float min_ray_angle_deg_for_indices(
    const std::vector<ObservationGeometry>& cams,
    const std::vector<std::size_t>& indices,
    const Vec3& position) {
    if (indices.size() < 2) return 0.F;
    double best_cos = 1.0;
    for (std::size_t i = 0; i + 1 < indices.size(); ++i) {
        const Vec3 ray_i = (position - cams[indices[i]].center).normalized();
        for (std::size_t j = i + 1; j < indices.size(); ++j) {
            const Vec3 ray_j = (position - cams[indices[j]].center).normalized();
            best_cos = std::min(best_cos, ray_i.dot(ray_j));
        }
    }
    return static_cast<float>(
        std::acos(std::clamp(best_cos, -1.0, 1.0)) * k_rad2deg);
}

bool refine_point_nonlinear(
    const std::vector<ObservationGeometry>& cams,
    const std::vector<std::size_t>& indices,
    Vec3& position,
    const unsigned max_iterations) {
    if (indices.size() < 2 || !position.allFinite()) return false;
    double lambda = 1e-3;
    for (unsigned iteration = 0; iteration < max_iterations; ++iteration) {
        Mat3 H = Mat3::Zero();
        Vec3 g = Vec3::Zero();
        double cost = 0.0;
        for (const std::size_t index : indices) {
            const ObservationGeometry& cam = cams[index];
            const Vec3 Xc = cam.image->pose.transform_world_to_camera(position);
            if (Xc.z() <= 1e-8) return false;
            const double inv_z = 1.0 / Xc.z();
            const double x = Xc.x() * inv_z;
            const double y = Xc.y() * inv_z;
            // Ignore distortion derivatives; residual still uses full project().
            const Vec2 proj = cam.camera->project(Xc);
            const Vec2 residual = proj - cam.pixel;
            cost += residual.squaredNorm();

            Eigen::Matrix<double, 2, 3> dproj_dXc;
            dproj_dXc << cam.camera->fx * inv_z, 0.0,
                -cam.camera->fx * x * inv_z, 0.0, cam.camera->fy * inv_z,
                -cam.camera->fy * y * inv_z;
            const Eigen::Matrix<double, 2, 3> J = dproj_dXc * cam.image->pose.R;
            H += J.transpose() * J;
            g += J.transpose() * residual;
        }
        H += lambda * Mat3::Identity();
        Eigen::LDLT<Mat3> ldlt(H);
        if (ldlt.info() != Eigen::Success) return false;
        const Vec3 step = -ldlt.solve(g);
        if (!step.allFinite()) return false;
        const Vec3 candidate = position + step;
        double candidate_cost = 0.0;
        bool candidate_valid = true;
        for (const std::size_t index : indices) {
            const ObservationGeometry& cam = cams[index];
            Vec2 proj;
            if (!cam.camera->project_checked(
                    cam.image->pose.transform_world_to_camera(candidate),
                    proj)) {
                candidate_valid = false;
                break;
            }
            candidate_cost += (proj - cam.pixel).squaredNorm();
        }
        if (!candidate_valid || !(candidate_cost < cost)) {
            lambda *= 10.0;
            if (lambda > 1e8) break;
            continue;
        }
        position = candidate;
        lambda = std::max(1e-8, lambda * 0.3);
        if (step.norm() < 1e-8) break;
    }
    return position.allFinite();
}

void local_optimize(
    const std::vector<ObservationGeometry>& cams,
    Consensus& consensus,
    const TriangulationOptions& options) {
    for (unsigned pass = 0; pass < 2; ++pass) {
        if (!solve_linear_lls(cams, consensus.inlier_indices, consensus.position))
            return;
        if (options.refine_nonlinear)
            refine_point_nonlinear(
                cams, consensus.inlier_indices, consensus.position,
                options.refine_iterations);
        consensus.inlier_indices = classify_inliers(
            cams, consensus.position, options.reproj_threshold_px);
        if (consensus.inlier_indices.size() < options.min_inliers) return;
    }
    consensus.score = score_consensus(
        cams, consensus.inlier_indices, consensus.position,
        options.reproj_threshold_px);
}

Consensus run_lo_ransac(
    const std::vector<ObservationGeometry>& cams,
    const TriangulationOptions& options) {
    Consensus best;
    if (cams.size() < options.min_inliers) return best;

    std::mt19937 generator(options.ransac_seed);
    std::uniform_int_distribution<std::size_t> pick(0, cams.size() - 1);

    const unsigned iterations = std::max(1U, options.ransac_iterations);
    for (unsigned iter = 0; iter < iterations; ++iter) {
        const std::size_t i = pick(generator);
        std::size_t j = pick(generator);
        if (cams.size() > 1) {
            while (j == i) j = pick(generator);
        }
        // Prefer a non-trivial baseline when possible.
        if ((cams[i].center - cams[j].center).squaredNorm() < 1e-12) continue;

        Consensus candidate;
        if (!triangulate_two_view_midpoint(cams[i], cams[j], candidate.position))
            continue;
        candidate.inlier_indices = classify_inliers(
            cams, candidate.position, options.reproj_threshold_px);
        if (candidate.inlier_indices.size() < options.min_inliers) continue;
        local_optimize(cams, candidate, options);
        if (candidate.inlier_indices.size() < options.min_inliers) continue;
        if (min_ray_angle_deg_for_indices(
                cams, candidate.inlier_indices, candidate.position) <
            options.min_angle_deg)
            continue;
        if (candidate.inlier_indices.size() > best.inlier_indices.size() ||
            (candidate.inlier_indices.size() == best.inlier_indices.size() &&
             candidate.score < best.score))
            best = std::move(candidate);
    }
    return best;
}

Consensus triangulate_linear_consensus(
    const std::vector<ObservationGeometry>& cams,
    const TriangulationOptions& options) {
    Consensus consensus;
    std::vector<std::size_t> all(cams.size());
    std::iota(all.begin(), all.end(), 0);
    if (!solve_linear_lls(cams, all, consensus.position)) return consensus;
    consensus.inlier_indices = classify_inliers(
        cams, consensus.position, options.reproj_threshold_px);
    if (consensus.inlier_indices.size() < options.min_inliers) {
        consensus.inlier_indices.clear();
        return consensus;
    }
    local_optimize(cams, consensus, options);
    if (consensus.inlier_indices.size() < options.min_inliers ||
        min_ray_angle_deg_for_indices(
            cams, consensus.inlier_indices, consensus.position) <
            options.min_angle_deg) {
        consensus.inlier_indices.clear();
        return consensus;
    }
    return consensus;
}

void apply_consensus_to_track(
    Track& track,
    const std::vector<ObservationGeometry>& cams,
    Consensus& consensus,
    const TriangulationOptions& options) {
    if (options.refine_nonlinear)
        refine_point_nonlinear(
            cams, consensus.inlier_indices, consensus.position,
            options.refine_iterations);

    std::vector<char> used(track.observations.size(), 0);
    std::vector<Observation> ordered;
    ordered.reserve(track.observations.size());
    for (const std::size_t cam_index : consensus.inlier_indices) {
        const Index obs_index = cams[cam_index].obs_index;
        ordered.push_back(track.observations[obs_index]);
        used[obs_index] = 1;
    }
    for (std::size_t i = 0; i < track.observations.size(); ++i) {
        if (!used[i]) ordered.push_back(track.observations[i]);
    }
    track.observations = std::move(ordered);
    track.position = consensus.position;
    track.num_inliers = static_cast<std::uint8_t>(std::min<std::size_t>(
        consensus.inlier_indices.size(), 255));
}

unsigned triangulate_track_impl(
    Track& track,
    const Scene& scene,
    const TriangulationOptions& options) {
    if (!track.is_valid()) {
        track.num_inliers = 0;
        return 0;
    }

    std::vector<ObservationGeometry> cams;
    if (!collect_observation_geometry(track, scene, cams) ||
        cams.size() < options.min_inliers) {
        track.num_inliers = 0;
        return 0;
    }

    Consensus consensus;
    const bool try_ransac = options.use_lo_ransac &&
                            cams.size() >= options.min_observations_for_ransac;
    if (try_ransac) consensus = run_lo_ransac(cams, options);
    if (consensus.inlier_indices.size() < options.min_inliers)
        consensus = triangulate_linear_consensus(cams, options);
    if (consensus.inlier_indices.size() < options.min_inliers) {
        track.num_inliers = 0;
        return 0;
    }

    apply_consensus_to_track(track, cams, consensus, options);
    if (track_min_ray_angle_deg(track, scene) < options.min_angle_deg) {
        track.num_inliers = 0;
        return 0;
    }
    return track.num_inliers;
}

std::vector<Observation> extract_registered_outliers(
    const Track& track,
    const Scene& scene) {
    std::vector<Observation> outliers;
    if (track.num_inliers >= track.observations.size()) return outliers;
    for (std::size_t i = track.num_inliers; i < track.observations.size(); ++i) {
        const Observation& obs = track.observations[i];
        if (obs.image_id >= scene.images.size()) continue;
        const Image& image = scene.images[obs.image_id];
        if (!image.registered) continue;
        if (obs.feature_id >= image.features.keypoints.size()) continue;
        outliers.push_back(obs);
    }
    return outliers;
}

// Build a child track from registered outliers without mutating the parent.
bool build_child_from_registered_outliers(
    const Track& parent,
    const Scene& scene,
    Track& child) {
    child = Track{};
    if (parent.num_inliers >= parent.observations.size()) return false;
    for (std::size_t i = parent.num_inliers; i < parent.observations.size();
         ++i) {
        const Observation& obs = parent.observations[i];
        if (obs.image_id >= scene.images.size()) continue;
        const Image& image = scene.images[obs.image_id];
        if (!image.registered) continue;
        if (obs.feature_id >= image.features.keypoints.size()) continue;
        child.observations.push_back(obs);
    }
    return child.observations.size() >= 2;
}

// Commit a successful child peel: drop registered outliers from the parent.
void commit_peel_registered_outliers(Track& parent, const Scene& scene) {
    std::vector<Observation> kept;
    kept.reserve(parent.observations.size());
    for (unsigned i = 0; i < parent.num_inliers; ++i)
        kept.push_back(parent.observations[i]);
    for (std::size_t i = parent.num_inliers; i < parent.observations.size();
         ++i) {
        const Observation& obs = parent.observations[i];
        if (obs.image_id >= scene.images.size()) {
            kept.push_back(obs);
            continue;
        }
        const Image& image = scene.images[obs.image_id];
        if (!image.registered ||
            obs.feature_id >= image.features.keypoints.size()) {
            kept.push_back(obs);
            continue;
        }
        // Registered outlier peeled into the child track.
    }
    parent.observations = std::move(kept);
}

}  // namespace

float track_min_ray_angle_deg(const Track& track, const Scene& scene) {
    if (track.num_inliers < 2) return 0.F;
    double best_cos = 1.0;
    for (unsigned i = 0; i + 1 < track.num_inliers; ++i) {
        const Image& img_i = scene.images[track.observations[i].image_id];
        const Vec3 ray_i = (track.position - img_i.pose.C).normalized();
        for (unsigned j = i + 1; j < track.num_inliers; ++j) {
            const Image& img_j = scene.images[track.observations[j].image_id];
            const Vec3 ray_j = (track.position - img_j.pose.C).normalized();
            best_cos = std::min(best_cos, ray_i.dot(ray_j));
        }
    }
    return static_cast<float>(
        std::acos(std::clamp(best_cos, -1.0, 1.0)) * k_rad2deg);
}

unsigned triangulate_track(
    Track& track,
    const Scene& scene,
    const TriangulationOptions& options) {
    return triangulate_track_impl(track, scene, options);
}

unsigned triangulate_track(
    Track& track,
    const Scene& scene,
    const float reproj_threshold_px,
    const float min_angle_deg,
    const unsigned min_inliers) {
    TriangulationOptions options;
    options.reproj_threshold_px = reproj_threshold_px;
    options.min_angle_deg = min_angle_deg;
    options.min_inliers = min_inliers;
    // Preserve historical single-track call semantics: no recursive splitting.
    options.split_tracks = false;
    return triangulate_track_impl(track, scene, options);
}

unsigned triangulate_tracks(
    Scene& scene,
    const bool outliers_only,
    const TriangulationOptions& options) {
    unsigned inlier_tracks = 0;
    core::StageScope stage("sfm.triangulate_tracks", core::LogLevel::debug);
    const std::size_t initial_track_count = scene.tracks.size();
    core::ProgressReporter progress(
        "triangulate tracks", initial_track_count, std::chrono::seconds(1),
        core::LogLevel::debug);

#if defined(_OPENMP)
    const int worker_count = scene.thread_count > 0
        ? static_cast<int>(scene.thread_count)
        : omp_get_max_threads();
#pragma omp parallel for reduction(+ : inlier_tracks) schedule(dynamic, 16) \
    num_threads(worker_count)
#endif
    for (std::int64_t i = 0;
         i < static_cast<std::int64_t>(initial_track_count); ++i) {
        Track& track = scene.tracks[static_cast<std::size_t>(i)];
        if (outliers_only && track.is_triangulated()) {
            ++inlier_tracks;
            progress.advance();
            continue;
        }
        if (triangulate_track_impl(track, scene, options) >= options.min_inliers)
            ++inlier_tracks;
        progress.advance();
    }
    progress.finish();

    if (!options.split_tracks || options.max_splits_per_track == 0)
        return inlier_tracks;

    // Breadth-limited peeling of contaminated tracks into new consensus sets.
    std::vector<std::pair<Index, unsigned>> pending;
    pending.reserve(scene.tracks.size());
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        const Track& track = scene.tracks[track_id];
        if (!track.is_triangulated()) continue;
        if (extract_registered_outliers(track, scene).size() >=
            options.min_inliers)
            pending.push_back({track_id, 0});
    }

    unsigned split_tracks = 0;
    while (!pending.empty()) {
        const auto [parent_id, depth] = pending.back();
        pending.pop_back();
        if (parent_id >= scene.tracks.size()) continue;
        Track& parent = scene.tracks[parent_id];
        if (!parent.is_triangulated()) continue;
        if (depth >= options.max_splits_per_track) continue;

        Track child;
        if (!build_child_from_registered_outliers(parent, scene, child))
            continue;
        if (triangulate_track_impl(child, scene, options) < options.min_inliers)
            continue;

        commit_peel_registered_outliers(parent, scene);
        scene.tracks.push_back(std::move(child));
        const Index child_id = static_cast<Index>(scene.tracks.size() - 1);
        ++split_tracks;
        ++inlier_tracks;
        if (depth + 1 < options.max_splits_per_track &&
            extract_registered_outliers(scene.tracks[child_id], scene).size() >=
                options.min_inliers)
            pending.push_back({child_id, depth + 1});
        if (extract_registered_outliers(parent, scene).size() >=
            options.min_inliers)
            pending.push_back({parent_id, depth + 1});
    }

    if (split_tracks > 0) {
        rebuild_track_index(scene);
        core::Logger::instance().debug(
            "track split: created=", split_tracks,
            " total_tracks=", scene.tracks.size());
    }
    return inlier_tracks;
}

unsigned triangulate_tracks(
    Scene& scene,
    const bool outliers_only,
    const float reproj_threshold_px,
    const float min_angle_deg) {
    TriangulationOptions options;
    options.reproj_threshold_px = reproj_threshold_px;
    options.min_angle_deg = min_angle_deg;
    return triangulate_tracks(scene, outliers_only, options);
}

}  // namespace aetherscan::sfm
