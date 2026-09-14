#include "sfm/triangulation.hpp"

#include "core/logging.hpp"
#include "sfm/tracks.hpp"

#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
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

bool make_observation_ray_geometry(
    const Index obs_index,
    const Observation& obs,
    const Scene& scene,
    ObservationGeometry& row,
    Vec3* bearing_out = nullptr) {
    if (obs.image_id >= scene.images.size()) return false;
    const Image& image = scene.images[obs.image_id];
    if (!image.registered ||
        obs.feature_id >= image.features.keypoints.size())
        return false;
    const PinholeCamera& camera = scene.camera_of(image);
    const Vec2 pixel = keypoint_xy(image, obs.feature_id);
    const Vec3 bearing = camera.unproject_normalized(pixel);
    row.obs_index = obs_index;
    row.image_id = obs.image_id;
    row.feature_id = obs.feature_id;
    row.center = image.pose.C;
    row.world_dir = (image.pose.R.transpose() * bearing).normalized();
    row.pixel = pixel;
    row.camera = &camera;
    row.image = &image;
    if (bearing_out) *bearing_out = bearing;
    return row.world_dir.allFinite();
}

bool make_observation_geometry(
    const Index obs_index,
    const Observation& obs,
    const Scene& scene,
    ObservationGeometry& row) {
    Vec3 bearing;
    if (!make_observation_ray_geometry(
            obs_index, obs, scene, row, &bearing))
        return false;
    Mat3 Dcross;
    Dcross << 0, -bearing.z(), bearing.y(), bearing.z(), 0, -bearing.x(),
        -bearing.y(), bearing.x(), 0;
    row.DR = Dcross * row.image->pose.R;
    row.Dt = -Dcross * row.image->pose.translation();
    return true;
}

bool collect_observation_geometry(
    const Track& track,
    const Scene& scene,
    std::vector<ObservationGeometry>& cams) {
    cams.clear();
    cams.reserve(track.observations.size());
    for (Index obs_index = 0; obs_index < track.observations.size(); ++obs_index) {
        const Observation& obs = track.observations[obs_index];
        ObservationGeometry row;
        if (make_observation_geometry(obs_index, obs, scene, row))
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

void classify_inliers(
    const std::vector<ObservationGeometry>& cams,
    const Vec3& position,
    const float reproj_threshold_px,
    std::vector<std::size_t>& inliers) {
    inliers.clear();
    inliers.reserve(cams.size());
    for (std::size_t i = 0; i < cams.size(); ++i) {
        if (observation_supports_point(cams[i], position, reproj_threshold_px))
            inliers.push_back(i);
    }
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

bool local_optimize(
    const std::vector<ObservationGeometry>& cams,
    Consensus& consensus,
    const TriangulationOptions& options) {
    for (unsigned pass = 0; pass < 2; ++pass) {
        if (!solve_linear_lls(cams, consensus.inlier_indices, consensus.position))
            return false;
        if (options.refine_nonlinear)
            refine_point_nonlinear(
                cams, consensus.inlier_indices, consensus.position,
                options.refine_iterations);
        classify_inliers(cams, consensus.position, options.reproj_threshold_px,
            consensus.inlier_indices);
        if (consensus.inlier_indices.size() < options.min_inliers) return false;
    }
    consensus.score = score_consensus(
        cams, consensus.inlier_indices, consensus.position,
        options.reproj_threshold_px);
    return true;
}

Consensus run_lo_ransac(
    const std::vector<ObservationGeometry>& cams,
    const TriangulationOptions& options) {
    Consensus best;
    if (cams.size() < options.min_inliers) return best;

    std::mt19937 generator(options.ransac_seed);
    std::uniform_int_distribution<std::size_t> pick(0, cams.size() - 1);

    const unsigned iterations = std::max(1U, options.ransac_iterations);
    Consensus candidate, cached;
    std::uint64_t cached_support = 0;
    for (unsigned iter = 0; iter < iterations; ++iter) {
        const std::size_t i = pick(generator);
        std::size_t j = pick(generator);
        if (cams.size() > 1) {
            while (j == i) j = pick(generator);
        }
        // Prefer a non-trivial baseline when possible.
        if ((cams[i].center - cams[j].center).squaredNorm() < 1e-12) continue;

        candidate.score = std::numeric_limits<double>::infinity();
        if (!triangulate_two_view_midpoint(cams[i], cams[j], candidate.position))
            continue;
        classify_inliers(cams, candidate.position, options.reproj_threshold_px,
            candidate.inlier_indices);
        if (candidate.inlier_indices.size() < options.min_inliers) continue;
        // LO starts by solving the same linear system for a given support
        // set, discarding the sampled point. Reuse only fully completed LO
        // results; failed solves may retain their sampled point. Keep every
        // RANSAC draw, tie-break and iteration unchanged.
        std::uint64_t support = 0;
        if (cams.size() <= 64)
            for (const auto index : candidate.inlier_indices) support |= std::uint64_t{1} << index;
        if (support != 0 && support == cached_support) candidate = cached;
        else if (local_optimize(cams, candidate, options) && support != 0) {
            cached_support = support;
            cached = candidate;
        }
        if (candidate.inlier_indices.size() < options.min_inliers) continue;
        if (min_ray_angle_deg_for_indices(
                cams, candidate.inlier_indices, candidate.position) <
            options.min_angle_deg)
            continue;
        if (candidate.inlier_indices.size() > best.inlier_indices.size() ||
            (candidate.inlier_indices.size() == best.inlier_indices.size() &&
             candidate.score < best.score))
            std::swap(best, candidate);
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
    classify_inliers(cams, consensus.position, options.reproj_threshold_px,
        consensus.inlier_indices);
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

bool is_registered_observation(
    const Observation& observation, const Scene& scene) {
    if (observation.image_id >= scene.images.size()) return false;
    const Image& image = scene.images[observation.image_id];
    return image.registered &&
           observation.feature_id < image.features.keypoints.size();
}

bool observation_supports_position(
    const Observation& observation,
    const Scene& scene,
    const Vec3& position,
    const float reproj_threshold_px) {
    if (!is_registered_observation(observation, scene)) return false;
    const Image& image = scene.images[observation.image_id];
    const PinholeCamera& camera = scene.camera_of(image);
    Vec2 projected;
    if (!camera.project_checked(
            image.pose.transform_world_to_camera(position), projected))
        return false;
    const double error =
        (projected - keypoint_xy(image, observation.feature_id)).norm();
    return std::isfinite(error) &&
           error <= static_cast<double>(reproj_threshold_px);
}

// Newly registered observations commonly belong to the existing landmark.
// Move those directly into the inlier prefix so they never reach the split
// solver. This is intentionally allocation-free.
std::size_t absorb_parent_supported_observations(
    Track& track,
    const Scene& scene,
    const float reproj_threshold_px) {
    std::size_t write = track.num_inliers;
    const std::size_t max_inliers =
        std::numeric_limits<decltype(track.num_inliers)>::max();
    for (std::size_t i = track.num_inliers;
         i < track.observations.size() && write < max_inliers; ++i) {
        if (!observation_supports_position(
                track.observations[i], scene, track.position,
                reproj_threshold_px))
            continue;
        if (write != i)
            std::swap(track.observations[write], track.observations[i]);
        ++write;
    }
    const std::size_t absorbed = write - track.num_inliers;
    track.num_inliers = static_cast<decltype(track.num_inliers)>(write);
    return absorbed;
}

std::size_t count_registered_outliers(
    const Track& track, const Scene& scene) {
    std::size_t count = 0;
    for (std::size_t i = track.num_inliers; i < track.observations.size(); ++i) {
        if (is_registered_observation(track.observations[i], scene)) ++count;
    }
    return count;
}

// A pair always explains its own two rays, so a useful pre-gate must require
// at least one independent supporting observation. Candidate sets larger than
// the fixed stack workspace are conservatively allowed through to avoid false
// negatives on unusually long contaminated tracks.
bool has_alternate_geometric_consensus(
    const Track& track,
    const Scene& scene,
    const TriangulationOptions& options,
    const std::size_t registered_outliers) {
    if (!options.use_fast_split_gate || !options.use_lo_ransac ||
        registered_outliers < options.min_observations_for_ransac)
        return true;

    constexpr std::size_t k_max_stack_observations = 64;
    const std::size_t max_observations = std::min<std::size_t>(
        options.split_gate_max_observations, k_max_stack_observations);
    if (max_observations < 3 || registered_outliers > max_observations ||
        options.split_gate_max_pairs == 0)
        return true;

    std::array<ObservationGeometry, k_max_stack_observations> observations{};
    std::size_t observation_count = 0;
    for (std::size_t i = track.num_inliers; i < track.observations.size(); ++i) {
        if (!is_registered_observation(track.observations[i], scene)) continue;
        if (!make_observation_ray_geometry(
                static_cast<Index>(i), track.observations[i], scene,
                observations[observation_count]))
            return true;
        ++observation_count;
    }
    if (observation_count != registered_outliers) return true;

    const std::size_t required_support = std::max<std::size_t>(
        3, std::max<std::size_t>(
               options.min_inliers, options.split_gate_min_support));
    if (observation_count < required_support) return false;
    const float gate_threshold = options.reproj_threshold_px *
        std::max(1.F, options.split_gate_threshold_multiplier);

    const std::size_t total_pairs =
        observation_count * (observation_count - 1) / 2;
    const std::size_t pair_budget = std::min<std::size_t>(
        total_pairs, options.split_gate_max_pairs);
    std::size_t pair_ordinal = 0;
    std::size_t tested_pairs = 0;
    for (std::size_t first_index = 0;
         first_index + 1 < observation_count; ++first_index) {
        const ObservationGeometry& first = observations[first_index];
        for (std::size_t second_index = first_index + 1;
             second_index < observation_count; ++second_index) {
            const std::size_t target_ordinal =
                tested_pairs < pair_budget
                ? tested_pairs * total_pairs / pair_budget
                : total_pairs;
            if (pair_ordinal++ != target_ordinal) continue;
            ++tested_pairs;
            const ObservationGeometry& second = observations[second_index];
            Vec3 candidate;
            if (!triangulate_two_view_midpoint(first, second, candidate))
                continue;

            const Vec3 first_ray = candidate - first.center;
            const Vec3 second_ray = candidate - second.center;
            const double ray_norms = first_ray.norm() * second_ray.norm();
            if (!(ray_norms > 1e-12)) continue;
            const double cosine =
                std::clamp(first_ray.dot(second_ray) / ray_norms, -1.0, 1.0);
            const float angle =
                static_cast<float>(std::acos(cosine) * k_rad2deg);
            if (angle < options.min_angle_deg) continue;

            std::size_t support = 0;
            for (std::size_t i = 0; i < observation_count; ++i) {
                if (observation_supports_point(
                        observations[i], candidate, gate_threshold) &&
                    ++support >= required_support)
                    return true;
            }
        }
    }
    return false;
}

// Build a child track from registered outliers without mutating the parent.
bool build_child_from_registered_outliers(
    const Track& parent,
    const Scene& scene,
    Track& child) {
    child = Track{};
    if (parent.num_inliers >= parent.observations.size()) return false;
    child.observations.reserve(
        parent.observations.size() - parent.num_inliers);
    for (std::size_t i = parent.num_inliers; i < parent.observations.size();
         ++i) {
        const Observation& obs = parent.observations[i];
        if (is_registered_observation(obs, scene))
            child.observations.push_back(obs);
    }
    return child.observations.size() >= 2;
}

// Commit a successful child peel: drop registered outliers from the parent.
void commit_peel_registered_outliers(Track& parent, const Scene& scene) {
    std::size_t kept = parent.num_inliers;
    for (std::size_t i = parent.num_inliers; i < parent.observations.size();
         ++i) {
        if (!is_registered_observation(parent.observations[i], scene)) {
            if (kept != i)
                parent.observations[kept] =
                    std::move(parent.observations[i]);
            ++kept;
        }
    }
    parent.observations.resize(kept);
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

static unsigned triangulate_tracks_impl(
    Scene& scene,
    const std::vector<Index>* selected_track_ids,
    const bool outliers_only,
    const TriangulationOptions& options) {
    unsigned inlier_tracks = 0;
    core::StageScope stage("sfm.triangulate_tracks", core::LogLevel::debug);
    scene.registration_generation = std::max<std::uint32_t>(
        scene.registration_generation, scene.registered_count());
    const std::size_t initial_track_count = scene.tracks.size();
    const std::size_t candidate_count = selected_track_ids
        ? selected_track_ids->size()
        : initial_track_count;
    core::ProgressReporter progress(
        selected_track_ids ? "triangulate dirty tracks" : "triangulate tracks",
        candidate_count, std::chrono::seconds(1),
        core::LogLevel::debug);

#if defined(_OPENMP)
    const int worker_count = scene.thread_count > 0
        ? static_cast<int>(scene.thread_count)
        : omp_get_max_threads();
#pragma omp parallel for reduction(+ : inlier_tracks) schedule(dynamic, 16) \
    num_threads(worker_count)
#endif
    for (std::int64_t i = 0;
         i < static_cast<std::int64_t>(candidate_count); ++i) {
        const Index track_id = selected_track_ids
            ? (*selected_track_ids)[static_cast<std::size_t>(i)]
            : static_cast<Index>(i);
        Track& track = scene.tracks[track_id];
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
    pending.reserve(candidate_count);
    const bool force_split_check = !outliers_only;
    std::size_t split_checks = 0;
    std::size_t generation_skips = 0;
    std::size_t absorbed_parent_observations = 0;
    std::size_t split_gate_candidates = 0;
    std::size_t split_gate_passes = 0;
    std::size_t split_gate_rejections = 0;
    const auto schedule_split = [&](const Index track_id,
                                    const unsigned depth) {
        if (track_id >= scene.tracks.size()) return;
        Track& track = scene.tracks[track_id];
        if (!track.is_triangulated()) return;
        track.split_generation = scene.registration_generation;
        if (depth >= options.max_splits_per_track) return;
        absorbed_parent_observations +=
            absorb_parent_supported_observations(
                track, scene, options.reproj_threshold_px);
        const std::size_t registered_outliers =
            count_registered_outliers(track, scene);
        if (registered_outliers < options.min_inliers) return;
        const bool gated = options.use_fast_split_gate &&
            options.use_lo_ransac &&
            registered_outliers >= options.min_observations_for_ransac;
        if (gated) ++split_gate_candidates;
        if (!has_alternate_geometric_consensus(
                track, scene, options, registered_outliers)) {
            if (gated) ++split_gate_rejections;
            return;
        }
        if (gated) ++split_gate_passes;
        pending.push_back({track_id, depth});
    };
    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
        const Index track_id = selected_track_ids
            ? (*selected_track_ids)[candidate]
            : static_cast<Index>(candidate);
        Track& track = scene.tracks[track_id];
        if (!track.is_triangulated()) continue;
        if (!force_split_check &&
            track.split_generation == scene.registration_generation) {
            ++generation_skips;
            continue;
        }
        ++split_checks;
        schedule_split(track_id, 0);
    }
    core::Logger::instance().debug(
        "track split scan: checked=", split_checks,
        " skipped_generation=", generation_skips,
        " pending=", pending.size(),
        " absorbed_parent=", absorbed_parent_observations,
        " gate_candidates=", split_gate_candidates,
        " gate_pass=", split_gate_passes,
        " gate_rejected=", split_gate_rejections,
        " generation=", scene.registration_generation);

    unsigned split_tracks = 0;
    std::size_t child_consensus_solves = 0;
    std::size_t child_lo_ransac_launches = 0;
    while (!pending.empty()) {
        const auto [parent_id, depth] = pending.back();
        pending.pop_back();
        if (parent_id >= scene.tracks.size()) continue;
        if (!scene.tracks[parent_id].is_triangulated()) continue;
        if (depth >= options.max_splits_per_track) continue;

        Track child;
        if (!build_child_from_registered_outliers(
                scene.tracks[parent_id], scene, child))
            continue;
        ++child_consensus_solves;
        if (options.use_lo_ransac &&
            child.observations.size() >=
                options.min_observations_for_ransac)
            ++child_lo_ransac_launches;
        if (triangulate_track_impl(child, scene, options) < options.min_inliers)
            continue;

        // Do not keep a Track reference across push_back: appending the child
        // can reallocate scene.tracks and invalidate every reference into it.
        commit_peel_registered_outliers(scene.tracks[parent_id], scene);
        scene.tracks.push_back(std::move(child));
        const Index child_id = static_cast<Index>(scene.tracks.size() - 1);
        ++split_tracks;
        ++inlier_tracks;
        const unsigned next_depth = depth + 1;
        schedule_split(child_id, next_depth);
        schedule_split(parent_id, next_depth);
    }

    if (split_tracks > 0) rebuild_track_index(scene);
    core::Logger::instance().debug(
        "track split: created=", split_tracks,
        " child_solves=", child_consensus_solves,
        " lo_ransac_launches=", child_lo_ransac_launches,
        " absorbed_parent=", absorbed_parent_observations,
        " gate_candidates=", split_gate_candidates,
        " gate_pass=", split_gate_passes,
        " gate_rejected=", split_gate_rejections,
        " total_tracks=", scene.tracks.size());
    return inlier_tracks;
}

unsigned triangulate_tracks(
    Scene& scene,
    const bool outliers_only,
    const TriangulationOptions& options) {
    return triangulate_tracks_impl(scene, nullptr, outliers_only, options);
}

unsigned triangulate_tracks(
    Scene& scene,
    const std::vector<Index>& track_ids,
    const bool outliers_only,
    const TriangulationOptions& options) {
    std::vector<Index> candidates = track_ids;
    candidates.erase(
        std::remove_if(
            candidates.begin(), candidates.end(),
            [&](const Index track_id) {
                return track_id >= scene.tracks.size();
            }),
        candidates.end());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(
        std::unique(candidates.begin(), candidates.end()), candidates.end());
    return triangulate_tracks_impl(
        scene, &candidates, outliers_only, options);
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

unsigned triangulate_tracks(
    Scene& scene,
    const std::vector<Index>& track_ids,
    const bool outliers_only,
    const float reproj_threshold_px,
    const float min_angle_deg) {
    TriangulationOptions options;
    options.reproj_threshold_px = reproj_threshold_px;
    options.min_angle_deg = min_angle_deg;
    return triangulate_tracks(scene, track_ids, outliers_only, options);
}

}  // namespace aetherscan::sfm
