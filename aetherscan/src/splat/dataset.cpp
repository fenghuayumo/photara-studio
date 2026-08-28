#include "splat/dataset.hpp"

#include "dataset_internal.hpp"
#include "mvs/export.hpp"
#include "mvs/internal.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace aetherscan::splat {
namespace {

void use_sparse_points_as_initial_cloud_impl(mvs::MvsScene& scene) {
    if (!scene.dense_cloud.points.empty()) return;
    scene.dense_cloud.points.reserve(scene.sparse_points.size());
    for (const mvs::SparsePoint& sparse : scene.sparse_points) {
        if (!sparse.position.allFinite()) continue;
        mvs::DensePoint point;
        point.position = sparse.position;
        point.normal = mvs::Vec3f::UnitZ();
        point.color = sparse.color;
        if (!point.color.allFinite())
            point.color = mvs::Vec3f::Constant(0.5F);
        else
            point.color = point.color.cwiseMax(0.F).cwiseMin(1.F);
        point.weight = static_cast<float>(sparse.view_ids.size());
        point.views = sparse.view_ids;
        scene.dense_cloud.points.push_back(std::move(point));
    }
}

float estimate_camera_scene_scale(const std::vector<mvs::MvsView>& views) {
    if (views.size() < 2) return 1.F;
    double total_nearest = 0.0;
    for (std::size_t left = 0; left < views.size(); ++left) {
        double nearest = (std::numeric_limits<double>::infinity)();
        for (std::size_t right = 0; right < views.size(); ++right) {
            if (left == right) continue;
            nearest = std::min(
                nearest,
                (views[left].pose.C - views[right].pose.C).norm());
        }
        if (std::isfinite(nearest)) total_nearest += nearest;
    }
    return std::max(
        static_cast<float>(3.0 * total_nearest / views.size()), 1.F);
}

void generate_camera_frustum_points(
    mvs::MvsScene& scene, const std::size_t count, const std::uint32_t seed) {
    if (count == 0 || scene.views.empty())
        throw std::runtime_error(
            "Camera-only splat datasets require random initial points");
    const float scene_scale = estimate_camera_scene_scale(scene.views);
    const float near_depth = 0.05F * scene_scale;
    const float log_near = std::log(near_depth);
    const float log_far = std::log(scene_scale);
    std::mt19937 random(seed);
    std::uniform_int_distribution<std::size_t> camera_distribution(
        0, scene.views.size() - 1);
    std::uniform_real_distribution<float> unit(0.F, 1.F);
    std::uniform_real_distribution<float> color(0.F, 1.F);

    scene.dense_cloud.points.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t view_index = camera_distribution(random);
        const auto& view = scene.views[view_index];
        const float half_fov_x = std::atan(
            0.5F * static_cast<float>(view.width) /
            std::max(view.fx, 1e-6F));
        const float half_fov_y = std::atan(
            0.5F * static_cast<float>(view.height) /
            std::max(view.fy, 1e-6F));
        const float angle_x = (2.F * unit(random) - 1.F) * half_fov_x;
        const float angle_y = (2.F * unit(random) - 1.F) * half_fov_y;
        const float depth = std::exp(
            log_near + unit(random) * (log_far - log_near));
        const Eigen::Vector3d camera_point(
            std::tan(angle_x) * depth,
            std::tan(angle_y) * depth,
            depth);
        mvs::DensePoint point;
        point.position =
            view.pose.transform_camera_to_world(camera_point).cast<float>();
        point.normal =
            (view.pose.R.transpose() * Eigen::Vector3d::UnitZ()).cast<float>();
        point.color = mvs::Vec3f(color(random), color(random), color(random));
        point.weight = 1.F;
        point.views.push_back(static_cast<mvs::Index>(view_index));
        scene.dense_cloud.points.push_back(std::move(point));
    }
}

void finalize_initial_cloud(
    DatasetLoadResult& result, const DatasetLoadRequest& request) {
    if (!request.initial_point_cloud.empty()) {
        result.scene.dense_cloud =
            mvs::load_dense_ply(request.initial_point_cloud);
        result.initial_point_cloud = request.initial_point_cloud;
        result.initial_points_dense = true;
    } else {
        use_sparse_points_as_initial_cloud_impl(result.scene);
    }
    if (result.scene.dense_cloud.points.empty()) {
        generate_camera_frustum_points(
            result.scene, request.random_initial_point_count, request.seed);
        result.generated_initial_points = true;
        result.initial_points_dense = false;
        result.warnings.emplace_back(
            "Dataset contains camera poses but no point cloud; generated " +
            std::to_string(request.random_initial_point_count) +
            " deterministic camera-frustum initialization points");
    }
    if (!result.scene.subject_bounds.valid)
        mvs::detail::estimate_subject_bounds(
            result.scene.sparse_points, result.scene.subject_bounds, 0, 1.15F);
}

}  // namespace

void initialize_scene_from_sparse_points(mvs::MvsScene& scene) {
    use_sparse_points_as_initial_cloud_impl(scene);
    if (!scene.subject_bounds.valid)
        mvs::detail::estimate_subject_bounds(
            scene.sparse_points, scene.subject_bounds,
            scene.thread_count, 1.15F);
}

DatasetFormat parse_dataset_format(const std::string_view value) {
    const std::string normalized =
        dataset_detail::lower_ascii(std::string(value));
    if (normalized.empty() || normalized == "auto")
        return DatasetFormat::auto_detect;
    if (normalized == "colmap") return DatasetFormat::colmap;
    if (normalized == "realitycapture" ||
        normalized == "reality_capture" ||
        normalized == "reality-capture" ||
        normalized == "rc")
        return DatasetFormat::reality_capture;
    if (normalized == "openmvs" || normalized == "mvs")
        return DatasetFormat::openmvs;
    throw std::invalid_argument(
        "Unknown splat dataset format '" + std::string(value) +
        "' (expected auto, colmap, realitycapture, or openmvs)");
}

std::string_view dataset_format_name(const DatasetFormat format) noexcept {
    switch (format) {
        case DatasetFormat::auto_detect: return "auto";
        case DatasetFormat::colmap: return "colmap";
        case DatasetFormat::reality_capture: return "realitycapture";
        case DatasetFormat::openmvs: return "openmvs";
    }
    return "unknown";
}

void DatasetLoader::register_reader(std::unique_ptr<DatasetReader> reader) {
    if (!reader)
        throw std::invalid_argument("Cannot register a null dataset reader");
    const DatasetFormat incoming = reader->format();
    if (incoming == DatasetFormat::auto_detect)
        throw std::invalid_argument(
            "Dataset readers must expose a concrete format");
    if (std::any_of(
            readers_.begin(), readers_.end(),
            [incoming](const auto& existing) {
                return existing->format() == incoming;
            }))
        throw std::invalid_argument(
            "A reader for dataset format " +
            std::string(dataset_format_name(incoming)) +
            " is already registered");
    readers_.push_back(std::move(reader));
}

DatasetLoadResult DatasetLoader::load(
    const DatasetLoadRequest& request) const {
    if (request.source.empty())
        throw std::invalid_argument(
            "External splat dataset source path is empty");
    for (const auto& reader : readers_) {
        if (request.format != DatasetFormat::auto_detect &&
            reader->format() != request.format)
            continue;
        if (request.format == DatasetFormat::auto_detect &&
            !reader->probe(request))
            continue;
        DatasetLoadResult result = reader->load(request);
        finalize_initial_cloud(result, request);
        return result;
    }
    if (request.format != DatasetFormat::auto_detect)
        throw std::invalid_argument(
            "No registered reader can load explicit format " +
            std::string(dataset_format_name(request.format)));
    throw std::invalid_argument(
        "Could not detect a COLMAP, RealityCapture, or OpenMVS dataset at " +
        request.source.string());
}

DatasetLoader make_default_dataset_loader() {
    DatasetLoader loader;
    loader.register_reader(dataset_detail::make_colmap_reader());
    loader.register_reader(dataset_detail::make_reality_capture_reader());
    loader.register_reader(dataset_detail::make_openmvs_reader());
    return loader;
}

DatasetLoadResult load_splat_dataset(const DatasetLoadRequest& request) {
    return make_default_dataset_loader().load(request);
}

}  // namespace aetherscan::splat
