#include "aetherscan/sfm/triangulation.hpp"

#include "aetherscan/geometry/pose.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace aetherscan::sfm {
namespace {

geometry::Mat3 pose_rotation(const ba::Pose& pose) {
    const double w = pose.qw, x = pose.qx, y = pose.qy, z = pose.qz;
    geometry::Mat3 r;
    r << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
    return r;
}

}  // namespace

TriangulationResult triangulate_track(
    const Scene& scene, const Track& track, const TriangulationOptions& options) {
    if (options.minimum_depth <= 0.0 || options.maximum_reprojection_error <= 0.0 ||
        options.minimum_angle_degrees < 0.0)
        throw std::invalid_argument("Invalid triangulation options");

    struct Observation {
        const View* view;
        const Camera* camera;
        const features::Keypoint* point;
    };
    std::vector<Observation> observations;
    for (const auto& observation : track.observations) {
        if (observation.view_id >= scene.views.size())
            throw std::out_of_range("Track view is invalid");
        const View& view = scene.views[observation.view_id];
        if (!view.registered) continue;
        if (view.camera_id >= scene.cameras.size() ||
            observation.feature_index >= view.features.keypoints.size())
            throw std::out_of_range("Track feature or camera is invalid");
        observations.push_back(
            {&view, &scene.cameras[view.camera_id],
             &view.features.keypoints[observation.feature_index]});
    }

    TriangulationResult result;
    if (observations.size() < 2) return result;

    std::vector<geometry::Mat34> projections;
    std::vector<geometry::Vec2> pixels;
    projections.reserve(observations.size());
    pixels.reserve(observations.size());
    for (const auto& value : observations) {
        const geometry::Mat3 r = pose_rotation(value.view->pose);
        const geometry::Vec3 center(value.view->pose.cx, value.view->pose.cy, value.view->pose.cz);
        const geometry::Vec3 t = -r * center;
        geometry::Mat3 k = geometry::Mat3::Zero();
        k(0, 0) = value.camera->fx;
        k(1, 1) = value.camera->fy;
        k(0, 2) = value.camera->cx;
        k(1, 2) = value.camera->cy;
        k(2, 2) = 1.0;
        geometry::Mat34 rt;
        rt.leftCols<3>() = r;
        rt.col(3) = t;
        projections.push_back(k * rt);
        pixels.emplace_back(value.point->x, value.point->y);
    }

    geometry::Vec3 point;
    if (!geometry::triangulate_dlt(projections, pixels, point)) return result;
    result.position = {point.x(), point.y(), point.z()};

    double error_sum = 0.0;
    std::vector<geometry::Vec3> rays;
    for (const auto& value : observations) {
        const geometry::Mat3 r = pose_rotation(value.view->pose);
        const geometry::Vec3 center(value.view->pose.cx, value.view->pose.cy, value.view->pose.cz);
        const geometry::Vec3 delta = point - center;
        const geometry::Vec3 camera = r * delta;
        if (camera.z() <= options.minimum_depth) return result;
        const double ex = value.camera->fx * camera.x() / camera.z() + value.camera->cx -
                          value.point->x;
        const double ey = value.camera->fy * camera.y() / camera.z() + value.camera->cy -
                          value.point->y;
        error_sum += std::sqrt(ex * ex + ey * ey);
        rays.push_back(delta.normalized());
    }
    for (std::size_t first = 0; first < rays.size(); ++first)
        for (std::size_t second = first + 1; second < rays.size(); ++second) {
            const double cosine =
                std::clamp(rays[first].dot(rays[second]), -1.0, 1.0);
            result.maximum_angle_degrees = std::max(
                result.maximum_angle_degrees, std::acos(cosine) * 180.0 / 3.14159265358979323846);
        }
    result.supporting_views = observations.size();
    result.mean_reprojection_error = error_sum / static_cast<double>(observations.size());
    result.valid = result.mean_reprojection_error <= options.maximum_reprojection_error &&
                   result.maximum_angle_degrees >= options.minimum_angle_degrees;
    return result;
}

}  // namespace aetherscan::sfm
