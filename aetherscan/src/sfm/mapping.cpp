#include "aetherscan/sfm/mapping.hpp"

#include "aetherscan/geometry/pose.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::sfm {
namespace {

geometry::Mat3 matrix(const std::array<double, 9>& value) {
    geometry::Mat3 result;
    result << value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
        value[8];
    return result;
}

geometry::Mat3 pose_rotation(const ba::Pose& pose) {
    const double w = pose.qw, x = pose.qx, y = pose.qy, z = pose.qz;
    geometry::Mat3 r;
    r << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
    return r;
}

void set_rotation(ba::Pose& pose, const geometry::Mat3& input) {
    const geometry::Mat3 r = geometry::rodrigues(geometry::inverse_rodrigues(input));
    const double trace = r(0, 0) + r(1, 1) + r(2, 2);
    double w, x, y, z;
    if (trace > 0) {
        const double s = 2 * std::sqrt(trace + 1);
        w = 0.25 * s;
        x = (r(2, 1) - r(1, 2)) / s;
        y = (r(0, 2) - r(2, 0)) / s;
        z = (r(1, 0) - r(0, 1)) / s;
    } else if (r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2)) {
        const double s = 2 * std::sqrt(1 + r(0, 0) - r(1, 1) - r(2, 2));
        w = (r(2, 1) - r(1, 2)) / s;
        x = 0.25 * s;
        y = (r(0, 1) + r(1, 0)) / s;
        z = (r(0, 2) + r(2, 0)) / s;
    } else if (r(1, 1) > r(2, 2)) {
        const double s = 2 * std::sqrt(1 + r(1, 1) - r(0, 0) - r(2, 2));
        w = (r(0, 2) - r(2, 0)) / s;
        x = (r(0, 1) + r(1, 0)) / s;
        y = 0.25 * s;
        z = (r(1, 2) + r(2, 1)) / s;
    } else {
        const double s = 2 * std::sqrt(1 + r(2, 2) - r(0, 0) - r(1, 1));
        w = (r(1, 0) - r(0, 1)) / s;
        x = (r(0, 2) + r(2, 0)) / s;
        y = (r(1, 2) + r(2, 1)) / s;
        z = 0.25 * s;
    }
    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    pose.qw = w / norm;
    pose.qx = x / norm;
    pose.qy = y / norm;
    pose.qz = z / norm;
}

int variable_offset(const Id view, const Id anchor) {
    if (view == anchor) return -1;
    return static_cast<int>(3 * (view - (view > anchor ? 1 : 0)));
}

std::size_t add_new_landmarks(Scene& scene, const TriangulationOptions& options) {
    std::vector<unsigned char> existing(scene.tracks.size(), 0);
    for (const auto& landmark : scene.landmarks)
        if (landmark.track_id < existing.size()) existing[landmark.track_id] = 1;
    std::vector<TriangulationResult> results(scene.tracks.size());
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 64) if (scene.tracks.size() > 256)
#endif
    for (int index = 0; index < static_cast<int>(scene.tracks.size()); ++index) {
        const auto& track = scene.tracks[static_cast<std::size_t>(index)];
        if (track.id >= existing.size() || existing[track.id]) continue;
        results[static_cast<std::size_t>(index)] = triangulate_track(scene, track, options);
    }
    std::size_t added = 0;
    for (std::size_t index = 0; index < scene.tracks.size(); ++index) {
        if (!results[index].valid) continue;
        const auto& track = scene.tracks[index];
        scene.landmarks.push_back(
            {static_cast<Id>(scene.landmarks.size()), results[index].position, track.id,
             results[index].mean_reprojection_error});
        existing[track.id] = 1;
        ++added;
    }
    return added;
}

std::size_t correspondence_count(const Scene& scene, const Id view_id) {
    std::size_t count = 0;
    for (const auto& landmark : scene.landmarks) {
        if (landmark.track_id >= scene.tracks.size()) continue;
        for (const auto& observation : scene.tracks[landmark.track_id].observations)
            if (observation.view_id == view_id) {
                ++count;
                break;
            }
    }
    return count;
}

std::vector<Id> local_views(const Scene& scene, const Id center, const std::size_t maximum) {
    std::vector<std::pair<std::size_t, Id>> scores;
    for (const auto& view : scene.views)
        if (view.registered) {
            std::size_t score = 0;
            for (const auto& landmark : scene.landmarks)
                if (landmark.track_id < scene.tracks.size()) {
                    bool has_center = false, has_view = false;
                    for (const auto& observation :
                         scene.tracks[landmark.track_id].observations) {
                        has_center |= observation.view_id == center;
                        has_view |= observation.view_id == view.id;
                    }
                    score += has_center && has_view;
                }
            scores.emplace_back(score, view.id);
        }
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    std::vector<Id> result;
    for (const auto& [score, id] : scores) {
        (void)score;
        if (result.size() >= maximum) break;
        result.push_back(id);
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<Id> landmarks_for_views(const Scene& scene, const std::vector<Id>& views) {
    std::unordered_set<Id> selected(views.begin(), views.end());
    std::vector<Id> result;
    for (const auto& landmark : scene.landmarks)
        if (landmark.track_id < scene.tracks.size()) {
            std::size_t observations = 0;
            for (const auto& observation : scene.tracks[landmark.track_id].observations)
                observations += selected.contains(observation.view_id);
            if (observations >= 2) result.push_back(landmark.id);
        }
    return result;
}

}  // namespace

PnPResult register_view_pnp(Scene& scene, const Id view_id, const PnPOptions& options) {
    if (view_id >= scene.views.size() || options.minimum_correspondences < 4 ||
        options.minimum_inliers < 4)
        throw std::invalid_argument("Invalid PnP view or options");
    View& view = scene.views[view_id];
    const Camera& camera = scene.cameras.at(view.camera_id);
    std::vector<geometry::Vec3> object;
    std::vector<geometry::Vec2> image;
    for (const auto& landmark : scene.landmarks) {
        if (landmark.track_id >= scene.tracks.size()) continue;
        for (const auto& observation : scene.tracks[landmark.track_id].observations)
            if (observation.view_id == view_id) {
                if (observation.feature_index >= view.features.keypoints.size())
                    throw std::out_of_range("PnP feature is invalid");
                const auto& point = view.features.keypoints[observation.feature_index];
                object.emplace_back(landmark.position[0], landmark.position[1], landmark.position[2]);
                image.emplace_back(point.x, point.y);
                break;
            }
    }
    PnPResult result;
    result.correspondence_count = object.size();
    if (object.size() < options.minimum_correspondences) return result;

    auto pose = geometry::estimate_pnp_ransac(
        object, image, camera.fx, camera.fy, camera.cx, camera.cy,
        options.maximum_reprojection_error, options.confidence, options.maximum_iterations, true);
    if (!pose.valid) return result;

    // Count inliers with the RANSAC pose before LM.
    std::vector<std::uint8_t> inliers(object.size(), 0);
    std::size_t inlier_count = 0;
    for (std::size_t i = 0; i < object.size(); ++i) {
        const auto projected =
            geometry::project_point(pose.R, pose.t, object[i], camera.fx, camera.fy, camera.cx, camera.cy);
        if ((projected - image[i]).norm() <= options.maximum_reprojection_error) {
            inliers[i] = 1;
            ++inlier_count;
        }
    }
    result.inlier_count = inlier_count;
    if (result.inlier_count < options.minimum_inliers) return result;

    pose = geometry::refine_pnp_lm(pose, object, image, camera.fx, camera.fy, camera.cx, camera.cy,
                                   &inliers);
    if (!pose.valid) return result;

    set_rotation(view.pose, pose.R);
    const geometry::Vec3 center = -pose.R.transpose() * pose.t;
    view.pose.cx = center.x();
    view.pose.cy = center.y();
    view.pose.cz = center.z();
    view.registered = true;

    double error_sum = 0.0;
    std::size_t used = 0;
    for (std::size_t i = 0; i < object.size(); ++i) {
        if (!inliers[i]) continue;
        const auto projected =
            geometry::project_point(pose.R, pose.t, object[i], camera.fx, camera.fy, camera.cx, camera.cy);
        error_sum += (projected - image[i]).norm();
        ++used;
    }
    result.mean_reprojection_error = used == 0 ? 0.0 : error_sum / static_cast<double>(used);
    result.valid = true;
    return result;
}

AveragingResult average_global_poses(
    const std::size_t view_count, const std::vector<RelativePoseEdge>& edges, const Id anchor,
    const AveragingOptions& options) {
    AveragingResult result;
    if (view_count == 0 || anchor >= view_count || edges.empty()) return result;
    std::vector<geometry::Mat3> rotations(view_count, geometry::Mat3::Identity());
    std::vector<bool> initialized(view_count, false);
    initialized[anchor] = true;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& edge : edges) {
            if (edge.source >= view_count || edge.target >= view_count)
                throw std::out_of_range("Pose edge view is invalid");
            const geometry::Mat3 relative = matrix(edge.rotation);
            if (initialized[edge.source] && !initialized[edge.target]) {
                rotations[edge.target] = relative * rotations[edge.source];
                initialized[edge.target] = true;
                changed = true;
            } else if (initialized[edge.target] && !initialized[edge.source]) {
                rotations[edge.source] = relative.transpose() * rotations[edge.target];
                initialized[edge.source] = true;
                changed = true;
            }
        }
    }
    if (std::find(initialized.begin(), initialized.end(), false) != initialized.end())
        return result;

    const int columns = static_cast<int>(3 * (view_count - 1));
    for (std::size_t iteration = 0; iteration < options.maximum_iterations; ++iteration) {
        Eigen::MatrixXd design = Eigen::MatrixXd::Zero(static_cast<int>(3 * edges.size()), columns);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<int>(3 * edges.size()));
        for (std::size_t e = 0; e < edges.size(); ++e) {
            const auto& edge = edges[e];
            const geometry::Mat3 predicted =
                rotations[edge.target] * rotations[edge.source].transpose();
            const geometry::Vec3 residual =
                geometry::inverse_rodrigues(matrix(edge.rotation) * predicted.transpose());
            const double norm = residual.norm();
            const double robust =
                norm > options.huber_threshold ? options.huber_threshold / norm : 1.0;
            const double weight = std::sqrt(std::max(0.0, edge.weight) * robust);
            const int source = variable_offset(edge.source, anchor);
            const int target = variable_offset(edge.target, anchor);
            for (int row = 0; row < 3; ++row) {
                rhs(static_cast<int>(3 * e) + row) = weight * residual(row);
                if (target >= 0) design(static_cast<int>(3 * e) + row, target + row) = weight;
                if (source >= 0)
                    for (int col = 0; col < 3; ++col)
                        design(static_cast<int>(3 * e) + row, source + col) =
                            -weight * predicted(row, col);
            }
        }
        if (rhs.norm() < options.convergence_tolerance) {
            result.iterations = iteration;
            break;
        }
        const Eigen::VectorXd delta =
            design.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
        const double step = delta.norm();
        for (Id view = 0; view < view_count; ++view)
            if (view != anchor) {
                const int offset = variable_offset(view, anchor);
                const geometry::Vec3 vector = delta.segment<3>(offset);
                rotations[view] = geometry::rodrigues(vector) * rotations[view];
            }
        result.iterations = iteration + 1;
        if (step < options.convergence_tolerance) break;
    }

    const int position_columns = static_cast<int>(3 * (view_count - 1) + edges.size());
    std::vector<double> weights(edges.size(), 1.0);
    Eigen::VectorXd solution;
    for (std::size_t iteration = 0; iteration < 5; ++iteration) {
        Eigen::MatrixXd design =
            Eigen::MatrixXd::Zero(static_cast<int>(3 * edges.size() + 1), position_columns);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<int>(3 * edges.size() + 1));
        for (std::size_t e = 0; e < edges.size(); ++e) {
            const auto& edge = edges[e];
            geometry::Vec3 t(edge.translation_direction[0], edge.translation_direction[1],
                             edge.translation_direction[2]);
            geometry::Vec3 direction = -rotations[edge.target].transpose() * t;
            direction.normalize();
            const double weight = std::sqrt(std::max(0.0, edge.weight) * weights[e]);
            const int source = variable_offset(edge.source, anchor);
            const int target = variable_offset(edge.target, anchor);
            for (int row = 0; row < 3; ++row) {
                const int equation = static_cast<int>(3 * e) + row;
                if (target >= 0) design(equation, target + row) = weight;
                if (source >= 0) design(equation, source + row) = -weight;
                design(equation, static_cast<int>(3 * (view_count - 1) + e)) =
                    -weight * direction(row);
            }
        }
        design(static_cast<int>(3 * edges.size()), static_cast<int>(3 * (view_count - 1))) = 1.0;
        rhs(static_cast<int>(3 * edges.size())) = 1.0;
        if (design.rows() < design.cols()) return result;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            design, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (iteration == 0) {
            const auto& singular = svd.singularValues();
            const double largest = singular(0);
            int rank = 0;
            for (int i = 0; i < singular.size(); ++i) rank += singular(i) > largest * 1e-10;
            if (rank < design.cols()) return result;
        }
        solution = svd.solve(rhs);
        for (std::size_t e = 0; e < edges.size(); ++e) {
            const auto& edge = edges[e];
            const int so = variable_offset(edge.source, anchor);
            const int to = variable_offset(edge.target, anchor);
            geometry::Vec3 cs = geometry::Vec3::Zero();
            geometry::Vec3 ct = geometry::Vec3::Zero();
            if (so >= 0) cs = solution.segment<3>(so);
            if (to >= 0) ct = solution.segment<3>(to);
            const double scale = solution(static_cast<int>(3 * (view_count - 1) + e));
            geometry::Vec3 t(edge.translation_direction[0], edge.translation_direction[1],
                             edge.translation_direction[2]);
            geometry::Vec3 d = -rotations[edge.target].transpose() * t;
            d.normalize();
            const geometry::Vec3 residual = ct - cs - scale * d;
            const double norm = residual.norm();
            weights[e] = norm > options.huber_threshold ? options.huber_threshold / norm : 1.0;
        }
    }

    result.poses.resize(view_count);
    double residual_sum = 0;
    for (Id view = 0; view < view_count; ++view) {
        set_rotation(result.poses[view], rotations[view]);
        const int offset = variable_offset(view, anchor);
        if (offset >= 0) {
            result.poses[view].cx = solution(offset);
            result.poses[view].cy = solution(offset + 1);
            result.poses[view].cz = solution(offset + 2);
        }
    }
    for (std::size_t e = 0; e < edges.size(); ++e) residual_sum += 1.0 - weights[e];
    result.mean_residual = residual_sum / edges.size();
    result.valid = true;
    return result;
}

ba::OptimizerSummary bundle_adjust_scene(
    Scene& scene, const std::vector<Id>& view_ids, const std::vector<Id>& landmark_ids,
    const ba::OptimizerOptions& options) {
    if (view_ids.empty() || landmark_ids.empty()) return {};
    std::unordered_map<Id, ba::Index> view_map;
    ba::Problem problem;
    for (const Id id : view_ids) {
        if (id >= scene.views.size() || !scene.views[id].registered)
            throw std::invalid_argument("BA view is invalid");
        view_map[id] = static_cast<ba::Index>(problem.poses.size());
        problem.poses.push_back(scene.views[id].pose);
        const auto& c = scene.cameras.at(scene.views[id].camera_id);
        problem.intrinsics.push_back({c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.p1, c.p2});
    }
    std::vector<Id> used_landmarks;
    for (const Id id : landmark_ids) {
        if (id >= scene.landmarks.size()) throw std::invalid_argument("BA landmark is invalid");
        const auto& landmark = scene.landmarks[id];
        if (landmark.track_id >= scene.tracks.size()) continue;
        std::vector<std::pair<ba::Index, const TrackObservation*>> observations;
        for (const auto& observation : scene.tracks[landmark.track_id].observations) {
            const auto found = view_map.find(observation.view_id);
            if (found != view_map.end()) observations.emplace_back(found->second, &observation);
        }
        if (observations.size() < 2) continue;
        const ba::Index point = static_cast<ba::Index>(problem.points.size());
        problem.points.push_back(
            {landmark.position[0], landmark.position[1], landmark.position[2]});
        used_landmarks.push_back(id);
        for (const auto& [camera, observation] : observations) {
            const auto& keypoint =
                scene.views[observation->view_id].features.keypoints.at(observation->feature_index);
            problem.observations.push_back(camera, point, keypoint.x, keypoint.y);
        }
    }
    if (problem.points.empty() || problem.observations.size() == 0) return {};
    auto summary = ba::optimize_cpu(problem, options);
    for (std::size_t i = 0; i < view_ids.size(); ++i) scene.views[view_ids[i]].pose = problem.poses[i];
    for (std::size_t i = 0; i < used_landmarks.size(); ++i)
        scene.landmarks[used_landmarks[i]].position = {
            problem.points[i].x, problem.points[i].y, problem.points[i].z};
    return summary;
}

MapperSummary run_incremental_mapping(
    Scene& scene, const RelativePoseEdge& seed, const IncrementalMapperOptions& options) {
    MapperSummary summary;
    if (seed.source >= scene.views.size() || seed.target >= scene.views.size()) return summary;
    scene.views[seed.source].pose = {};
    scene.views[seed.source].registered = true;
    const geometry::Mat3 rotation = matrix(seed.rotation);
    set_rotation(scene.views[seed.target].pose, rotation);
    geometry::Vec3 t(
        seed.translation_direction[0], seed.translation_direction[1],
        seed.translation_direction[2]);
    const geometry::Vec3 center = -rotation.transpose() * t;
    scene.views[seed.target].pose.cx = center.x();
    scene.views[seed.target].pose.cy = center.y();
    scene.views[seed.target].pose.cz = center.z();
    scene.views[seed.target].registered = true;
    add_new_landmarks(scene, options.triangulation);
    std::vector<bool> attempted(scene.views.size(), false);
    attempted[seed.source] = attempted[seed.target] = true;
    std::size_t registrations = 2;
    while (true) {
        Id candidate = invalid_id;
        std::size_t best = 0;
        for (const auto& view : scene.views)
            if (!view.registered && !attempted[view.id]) {
                const auto count = correspondence_count(scene, view.id);
                if (count > best) {
                    best = count;
                    candidate = view.id;
                }
            }
        if (candidate == invalid_id || best < options.pnp.minimum_correspondences) break;
        attempted[candidate] = true;
        const auto pnp = register_view_pnp(scene, candidate, options.pnp);
        if (!pnp.valid) {
            ++summary.failed_views;
            continue;
        }
        ++registrations;
        add_new_landmarks(scene, options.triangulation);
        if (options.local_ba_interval > 0 && registrations % options.local_ba_interval == 0) {
            const auto views = local_views(scene, candidate, options.bundle.maximum_local_views);
            const auto landmarks = landmarks_for_views(scene, views);
            if (!landmarks.empty())
                summary.bundle_summaries.push_back(
                    bundle_adjust_scene(scene, views, landmarks, options.bundle.optimizer));
        }
    }
    summary.registered_views = static_cast<std::size_t>(std::count_if(
        scene.views.begin(), scene.views.end(), [](const auto& v) { return v.registered; }));
    scene.landmarks.clear();
    add_new_landmarks(scene, options.triangulation);
    std::vector<Id> final_views;
    for (const auto& view : scene.views)
        if (view.registered) final_views.push_back(view.id);
    std::vector<Id> final_landmarks(scene.landmarks.size());
    std::iota(final_landmarks.begin(), final_landmarks.end(), 0);
    if (!final_landmarks.empty())
        summary.bundle_summaries.push_back(
            bundle_adjust_scene(scene, final_views, final_landmarks, options.bundle.optimizer));
    summary.landmarks = scene.landmarks.size();
    summary.valid = summary.registered_views >= 2 && !scene.landmarks.empty();
    return summary;
}

MapperSummary run_global_mapping(
    Scene& scene, const std::vector<RelativePoseEdge>& edges, const GlobalMapperOptions& options) {
    MapperSummary summary;
    const auto averaged = average_global_poses(scene.views.size(), edges, 0, options.averaging);
    if (!averaged.valid) return summary;
    for (std::size_t i = 0; i < scene.views.size(); ++i) {
        scene.views[i].pose = averaged.poses[i];
        scene.views[i].registered = true;
    }
    add_new_landmarks(scene, options.triangulation);
    std::vector<Id> views(scene.views.size()), landmarks(scene.landmarks.size());
    std::iota(views.begin(), views.end(), 0);
    std::iota(landmarks.begin(), landmarks.end(), 0);
    if (!landmarks.empty())
        summary.bundle_summaries.push_back(
            bundle_adjust_scene(scene, views, landmarks, options.bundle));
    scene.landmarks.clear();
    add_new_landmarks(scene, options.triangulation);
    landmarks.resize(scene.landmarks.size());
    std::iota(landmarks.begin(), landmarks.end(), 0);
    if (!landmarks.empty())
        summary.bundle_summaries.push_back(
            bundle_adjust_scene(scene, views, landmarks, options.bundle));
    summary.registered_views = scene.views.size();
    summary.landmarks = scene.landmarks.size();
    summary.valid = !scene.landmarks.empty();
    return summary;
}

}  // namespace aetherscan::sfm
