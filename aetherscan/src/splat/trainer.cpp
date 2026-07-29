#include "splat/trainer.hpp"

#include "cuda_ops.hpp"
#include "core/logging.hpp"
#include "densification.hpp"
#include "io/image.hpp"
#include "training_data_loader.hpp"

#include <cuda_runtime.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace aetherscan::splat {
namespace {

namespace data = training_data;
namespace refine = densification;

constexpr float k_sh0 = 0.28209479177387814F;

class KdTree {
public:
    explicit KdTree(const std::vector<mvs::Vec3f>& points)
        : points_(points), order_(points.size()) {
        std::iota(order_.begin(), order_.end(), std::size_t{0});
        nodes_.reserve(points.size());
        root_ = build(0, order_.size());
    }

    [[nodiscard]] float three_neighbor_rms(const std::size_t query) const {
        if (points_.size() < 4) return 0.F;
        std::array<float, 4> best{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
        search(root_, points_[query], best);
        std::sort(best.begin(), best.end());
        return std::sqrt(std::max((best[1] + best[2] + best[3]) / 3.F, 0.F));
    }

    [[nodiscard]] float two_neighbor_half_average(
        const std::size_t query) const {
        if (points_.size() < 3) return 0.F;
        std::array<float, 4> best{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
        search(root_, points_[query], best);
        std::sort(best.begin(), best.end());
        return 0.25F * (
            std::sqrt(std::max(best[1], 0.F)) +
            std::sqrt(std::max(best[2], 0.F)));
    }

private:
    struct Node {
        std::size_t point{};
        int left{-1};
        int right{-1};
        std::uint8_t axis{};
    };

    int build(const std::size_t begin, const std::size_t end) {
        if (begin >= end) return -1;
        mvs::Vec3f minimum = points_[order_[begin]];
        mvs::Vec3f maximum = minimum;
        for (std::size_t i = begin + 1; i < end; ++i) {
            minimum = minimum.cwiseMin(points_[order_[i]]);
            maximum = maximum.cwiseMax(points_[order_[i]]);
        }
        Eigen::Index axis{};
        (maximum - minimum).maxCoeff(&axis);
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(
            order_.begin() + static_cast<std::ptrdiff_t>(begin),
            order_.begin() + static_cast<std::ptrdiff_t>(middle),
            order_.begin() + static_cast<std::ptrdiff_t>(end),
            [&](const std::size_t left, const std::size_t right) {
                return points_[left](axis) < points_[right](axis);
            });
        const int node = static_cast<int>(nodes_.size());
        nodes_.push_back({order_[middle], -1, -1, static_cast<std::uint8_t>(axis)});
        const int left = build(begin, middle);
        const int right = build(middle + 1, end);
        nodes_[static_cast<std::size_t>(node)].left = left;
        nodes_[static_cast<std::size_t>(node)].right = right;
        return node;
    }

    void search(
        const int node_index, const mvs::Vec3f& query,
        std::array<float, 4>& best) const {
        if (node_index < 0) return;
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        const mvs::Vec3f& point = points_[node.point];
        const float distance2 = (query - point).squaredNorm();
        const auto worst = std::max_element(best.begin(), best.end());
        if (distance2 < *worst) *worst = distance2;

        const float delta = query(node.axis) - point(node.axis);
        const int near = delta < 0.F ? node.left : node.right;
        const int far = delta < 0.F ? node.right : node.left;
        search(near, query, best);
        if (delta * delta <= *std::max_element(best.begin(), best.end()))
            search(far, query, best);
    }

    const std::vector<mvs::Vec3f>& points_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;
    int root_{-1};
};

float percentile_median_size(
    const std::vector<mvs::Vec3f>& points, const float percentile) {
    if (points.empty()) return 1.F;
    std::array<std::vector<float>, 3> axes;
    for (auto& axis : axes) axis.reserve(points.size());
    for (const auto& point : points) {
        if (!point.allFinite()) continue;
        for (int axis = 0; axis < 3; ++axis)
            axes[axis].push_back(point(axis));
    }
    std::array<float, 3> sizes{};
    const float p = std::clamp(percentile, 0.F, 1.F);
    for (int axis = 0; axis < 3; ++axis) {
        auto& values = axes[axis];
        if (values.empty()) return 1.F;
        std::sort(values.begin(), values.end());
        const std::size_t count = values.size();
        const std::size_t low = std::min(
            count - 1,
            static_cast<std::size_t>((1.F - p) * 0.5F * count));
        const std::size_t high = std::min(
            count - 1,
            static_cast<std::size_t>((1.F + p) * 0.5F * count));
        sizes[axis] = values[high] - values[low];
    }
    std::sort(sizes.begin(), sizes.end());
    return std::max(sizes[1], 0.01F);
}

template <typename T>
std::vector<T> download(const tinytensor::Tensor& tensor) {
    std::vector<T> values(tensor.numel());
    if (!values.empty()) {
        const cudaError_t error = cudaMemcpy(
            values.data(), tensor.data_ptr(), values.size() * sizeof(T),
            cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            throw std::runtime_error(
                std::string("Failed to download tensor: ") +
                cudaGetErrorString(error));
    }
    return values;
}

void write_float(std::ofstream& stream, const float value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}



}  // namespace

GaussianModel initialize_from_dense_cloud(
    const mvs::MvsScene& scene, const TrainingOptions& options) {
    if (scene.dense_cloud.points.empty())
        throw std::invalid_argument("GGGS initialization requires a non-empty dense cloud");
    if (options.sh_degree > 3)
        throw std::invalid_argument("The current GGGS CUDA backend supports SH degree <= 3");
    const std::size_t source_count = scene.dense_cloud.points.size();
    const std::size_t count = options.max_gaussians == 0
        ? source_count
        : std::min(source_count, options.max_gaussians);
    const std::size_t bases = static_cast<std::size_t>(options.sh_degree + 1U) *
                              (options.sh_degree + 1U);
    const bool brush_adc_plus = options.densification_strategy ==
                                DensificationStrategy::adc_plus;
    std::vector<float> means(count * 3);
    std::vector<float> scales(count * 3);
    std::vector<float> quaternions(count * 4);
    std::vector<float> opacities(count);
    std::vector<float> sh(count * bases * 3, 0.F);

    const auto source_index = [source_count, count](const std::size_t index) {
        return count == source_count
            ? index
            : std::min(source_count - 1, index * source_count / count);
    };
    std::vector<mvs::Vec3f> selected_positions(count);
    for (std::size_t index = 0; index < count; ++index)
        selected_positions[index] =
            scene.dense_cloud.points[source_index(index)].position;
    float initialization_extent{};
    if (options.input_is_dense) {
        mvs::Vec3f minimum = selected_positions.front();
        mvs::Vec3f maximum = minimum;
        for (const mvs::Vec3f& position : selected_positions) {
            minimum = minimum.cwiseMin(position);
            maximum = maximum.cwiseMax(position);
        }
        initialization_extent = std::max(
            (maximum - minimum).norm(), 1e-6F);
    } else {
        initialization_extent =
            refine::training_scene_geometry(scene, false).scale;
    }
    const float fallback_scale = std::max(
        initialization_extent /
            std::sqrt(static_cast<float>(std::max<std::size_t>(count, 1))),
        1e-6F);
    const float minimum_initial_scale = options.constrain_scale_range
        ? initialization_extent *
              std::max(options.minimum_scale_fraction, 1e-8F)
        : 1e-7F;
    const float maximum_initial_scale = options.constrain_scale_range
        ? initialization_extent *
              std::max(
                  options.maximum_scale_fraction,
                  options.minimum_scale_fraction)
        : std::numeric_limits<float>::infinity();
    const float opacity = std::clamp(
        brush_adc_plus ? 0.5F : options.initial_opacity,
        1e-6F, 1.F - 1e-6F);
    const float opacity_logit = std::log(opacity / (1.F - opacity));
    std::vector<float> knn_scales(count, 0.F);
    if (options.initialize_scale_from_knn &&
        count >= (brush_adc_plus ? 3U : 4U)) {
        const KdTree tree(selected_positions);
        const float brush_maximum_scale = 0.1F *
            percentile_median_size(selected_positions, 0.75F);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t index = 0;
             index < static_cast<std::int64_t>(count); ++index)
            knn_scales[static_cast<std::size_t>(index)] =
                brush_adc_plus
                ? std::clamp(
                      tree.two_neighbor_half_average(
                          static_cast<std::size_t>(index)),
                      1e-3F, brush_maximum_scale)
                : tree.three_neighbor_rms(static_cast<std::size_t>(index));
    }
    std::mt19937 quaternion_random(options.seed);
    std::uniform_real_distribution<float> quaternion_uniform(0.F, 1.F);

    for (std::size_t index = 0; index < count; ++index) {
        const auto& point = scene.dense_cloud.points[source_index(index)];
        for (int axis = 0; axis < 3; ++axis)
            means[3 * index + axis] = point.position(axis);
        float footprint = std::numeric_limits<float>::infinity();
        for (const auto view_id : point.views) {
            if (view_id >= scene.views.size()) continue;
            const auto& view = scene.views[view_id];
            const auto camera_point = view.pose.transform_world_to_camera(
                point.position.cast<double>());
            if (camera_point.z() > 0.0)
                footprint = std::min(
                    footprint,
                    static_cast<float>(camera_point.z()) /
                        std::max(view.fx, view.fy));
        }
        const float initial_spacing =
            options.initialize_scale_from_knn &&
                    std::isfinite(knn_scales[index]) &&
                    knn_scales[index] > 0.F
                ? knn_scales[index]
                : std::isfinite(footprint) ? footprint : fallback_scale;
        const float scale = std::clamp(
            initial_spacing * options.initial_scale,
            minimum_initial_scale, maximum_initial_scale);
        for (int axis = 0; axis < 3; ++axis)
            scales[3 * index + axis] = std::log(scale);

        if (!options.input_is_dense && !brush_adc_plus) {
            // Exact pygsplat sparse-SfM initialization: raw U[0,1) quaternion
            // parameters. The raster path normalizes them before use.
            for (int component = 0; component < 4; ++component)
                quaternions[4 * index + component] = quaternion_uniform(
                    quaternion_random);
        } else if (options.input_is_dense) {
            mvs::Vec3f normal = point.normal;
            if (!normal.allFinite() || normal.squaredNorm() < 1e-12F)
                normal = mvs::Vec3f::UnitZ();
            else
                normal.normalize();
            Eigen::Quaternionf rotation = Eigen::Quaternionf::FromTwoVectors(
                mvs::Vec3f::UnitZ(), normal).normalized();
            quaternions[4 * index + 0] = rotation.w();
            quaternions[4 * index + 1] = rotation.x();
            quaternions[4 * index + 2] = rotation.y();
            quaternions[4 * index + 3] = rotation.z();
        } else {
            quaternions[4 * index + 0] = 1.F;
            quaternions[4 * index + 1] = 0.F;
            quaternions[4 * index + 2] = 0.F;
            quaternions[4 * index + 3] = 0.F;
        }
        opacities[index] = opacity_logit;
        for (int channel = 0; channel < 3; ++channel)
            sh[(index * bases) * 3 + channel] =
                (std::clamp(point.color(channel), 0.F, 1.F) - 0.5F) / k_sh0;
    }

    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        means, {count, 3}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        scales, {count, 3}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        quaternions, {count, 4}, tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        opacities, {count, 1}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(
        sh, {count, bases, 3}, tinytensor::Device::CUDA);
    model.sh_degree = options.sh_degree;
    return model;
}


Trainer::Trainer(TrainingOptions options) : options_(std::move(options)) {}

GaussianModel Trainer::train(
    const mvs::MvsScene& scene, ProgressCallback progress,
    EvaluationCallback evaluate) const {
    if (scene.views.empty())
        throw std::invalid_argument("GGGS training requires at least one MVS view");
    GaussianModel model = initialize_from_dense_cloud(scene, options_);
    const bool use_3d_filter =
        options_.use_depth_normal_loss &&
        options_.depth_normal_weight > 0.F;
    std::vector<std::size_t> view_indices;
    view_indices.reserve(scene.views.size());
    for (std::size_t index = 0; index < scene.views.size(); ++index)
        if (options_.evaluation_split_every == 0 ||
            index % options_.evaluation_split_every != 0)
            view_indices.push_back(index);
    if (view_indices.empty())
        throw std::invalid_argument(
            "GGGS evaluation split left no training views");
    float active_resolution_scale =
        data::progressive_resolution_scale(1, options_);
    data::TrainingDataLoader view_cache(
        scene.views, options_, active_resolution_scale);
    if (options_.use_mask) {
        for (const std::size_t index : view_indices)
            if (!view_cache.has_mask(index))
                throw std::invalid_argument(
                    "GGGS subject-only training requires a matching mask "
                    "file or source alpha channel for every selected view");
    }

    std::vector<Camera> all_cameras;
    all_cameras.reserve(scene.views.size());
    for (const mvs::MvsView& view : scene.views)
        all_cameras.push_back(data::training_camera(
            view, options_, active_resolution_scale));
    std::vector<Camera> filter_cameras;
    std::vector<Camera> full_resolution_filter_cameras;
    if (use_3d_filter) {
        filter_cameras.reserve(view_indices.size());
        full_resolution_filter_cameras.reserve(view_indices.size());
        for (const std::size_t index : view_indices) {
            filter_cameras.push_back(all_cameras[index]);
            full_resolution_filter_cameras.push_back(
                data::training_camera(scene.views[index], options_, 1.F));
        }
    }
    const float filter_3d_factor =
        options_.densification_strategy ==
                DensificationStrategy::adc_plus
            ? 0.1F
            : 0.2F;
    const bool brush_filter =
        options_.densification_strategy ==
        DensificationStrategy::adc_plus;
    // Keep the Mip-Splatting floor separate from the canonical parameters.
    // pygsplat applies this filter only while rasterizing and recomputes it
    // after topology changes; repeatedly baking it into scale/opacity causes
    // a cumulative opacity loss.
    if (use_3d_filter)
        model.filter_3d = detail::compute_3d_filter(
            model.means, filter_cameras, filter_3d_factor, brush_filter);
    const bool use_multi_view = options_.multi_view_geo_weight > 0.F ||
                                options_.multi_view_ncc_weight > 0.F;
    const auto multi_view_neighbours = use_multi_view
        ? data::compute_multi_view_neighbours(
              all_cameras, view_indices, options_)
        : std::vector<std::vector<std::size_t>>(scene.views.size());
    std::future<void> evaluation_future;
    const auto finish_evaluation = [&] {
        if (evaluation_future.valid()) evaluation_future.get();
    };
    const auto launch_evaluation =
        [&](const unsigned iteration, const GaussianModel& current) {
            finish_evaluation();
            GaussianModel snapshot = refine::clone_model(current);
            if (use_3d_filter)
                snapshot.filter_3d = detail::compute_3d_filter(
                    snapshot.means, full_resolution_filter_cameras,
                    filter_3d_factor, brush_filter);
            evaluation_future = std::async(
                std::launch::async,
                [evaluate, iteration,
                 snapshot = std::move(snapshot)]() mutable {
                    evaluate(iteration, snapshot);
                });
        };

    detail::AdamState means_state = detail::make_adam_state(model.means);
    detail::AdamState scales_state = detail::make_adam_state(model.log_scales);
    detail::AdamState rotations_state = detail::make_adam_state(model.quaternions);
    detail::AdamState opacity_state = detail::make_adam_state(model.opacity_logits);
    detail::AdamState sh_state =
        options_.densification_strategy == DensificationStrategy::adc_plus
            ? detail::make_reduced_second_adam_state(model.sh)
            : detail::make_adam_state(model.sh);
    const refine::AdamStates adam_states{
        &means_state, &scales_state, &rotations_state, &opacity_state,
        &sh_state};
    const bool densification_enabled = refine::is_enabled(options_);
    detail::DensificationStats densification_stats =
        detail::make_densification_stats(model.size());
    refine::RefinementCounts latest_refinement;
    Rasterizer rasterizer;
    std::mt19937 random(options_.seed);
    std::vector<std::size_t> shuffled_views = view_indices;
    std::shuffle(shuffled_views.begin(), shuffled_views.end(), random);
    std::size_t shuffled_view_cursor = 0;
    const refine::SceneGeometry scene_geometry =
        refine::training_scene_geometry(scene, options_.input_is_dense);
    const float scene_extent = scene_geometry.scale;
    const mvs::Vec3f scene_center = scene_geometry.center;
    float means_learning_rate_scale = scene_extent;
    refine::SceneGeometry refinement_geometry = scene_geometry;
    if (options_.densification_strategy ==
        DensificationStrategy::adc_plus) {
        refinement_geometry = refine::brush_scene_geometry(
            download<float>(model.means));
        means_learning_rate_scale = refinement_geometry.scale;
    }
    const float minimum_log_scale = options_.constrain_scale_range
        ? std::log(
              scene_extent *
              std::max(options_.minimum_scale_fraction, 1e-8F))
        : -std::numeric_limits<float>::infinity();
    // pygsplat does not clamp scales to its 0.1*scene prune threshold between
    // refinements. Clamping exactly at that boundary and then comparing
    // exp(log_scale) > threshold is numerically unsafe: round-off caused tens
    // of thousands of boundary Gaussians to be pruned after opacity reset.
    // Fixed-topology diagnostics still need a ceiling because they have no
    // prune pass at all.
    const float fixed_maximum_scale_fraction =
        options_.structure_freeze_iter != 0
            ? 0.1F
            : std::max(
                  options_.maximum_scale_fraction,
                  options_.minimum_scale_fraction);
    const float maximum_log_scale =
        options_.constrain_scale_range && !densification_enabled
            ? std::log(std::max(
                  fixed_maximum_scale_fraction * scene_extent, 1e-6F))
            : std::numeric_limits<float>::infinity();

    for (unsigned iteration = 1; iteration <= options_.iterations; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        const float requested_resolution_scale =
            data::progressive_resolution_scale(iteration, options_);
        if (std::abs(
                requested_resolution_scale -
                active_resolution_scale) > 1e-6F) {
            active_resolution_scale = requested_resolution_scale;
            view_cache.set_resolution_scale(active_resolution_scale);
            all_cameras.clear();
            for (const mvs::MvsView& view : scene.views)
                all_cameras.push_back(data::training_camera(
                    view, options_, active_resolution_scale));
            if (use_3d_filter) {
                filter_cameras.clear();
                for (const std::size_t index : view_indices)
                    filter_cameras.push_back(all_cameras[index]);
                model.filter_3d = detail::compute_3d_filter(
                    model.means, filter_cameras, filter_3d_factor,
                    brush_filter);
            }
        }
        const bool report_progress = progress &&
            (iteration == 1 || iteration == options_.iterations ||
             (options_.log_interval != 0 &&
              iteration % options_.log_interval == 0));
        if (shuffled_view_cursor == shuffled_views.size()) {
            std::shuffle(shuffled_views.begin(), shuffled_views.end(), random);
            shuffled_view_cursor = 0;
        }
        const std::size_t prefetch_end = std::min(
            shuffled_views.size(),
            shuffled_view_cursor + options_.training_prefetch_views + 1);
        for (std::size_t cursor = shuffled_view_cursor;
             cursor < prefetch_end; ++cursor)
            view_cache.prefetch(shuffled_views[cursor]);
        const std::size_t view_index =
            shuffled_views[shuffled_view_cursor++];
        const TrainingView target = view_cache.get(view_index);
        RasterizeOptions raster_options;
        const unsigned active_sh_degree = std::min(
            options_.sh_degree,
            options_.sh_degree_interval == 0
                ? options_.sh_degree
                : iteration / options_.sh_degree_interval);
        raster_options.active_sh_degree = active_sh_degree;
        if (options_.background_noise_strength > 0.F) {
            std::uniform_real_distribution<float> background_noise(
                -options_.background_noise_strength,
                options_.background_noise_strength);
            for (float& channel : raster_options.background)
                channel = std::clamp(background_noise(random), 0.F, 1.F);
        }
        raster_options.kernel_size = options_.kernel_size;
        raster_options.scale_modifier = options_.scale_modifier;
        const bool depth_normal_active = options_.use_depth_normal_loss &&
            options_.depth_normal_weight > 0.F &&
            iteration >= options_.depth_normal_from_iter;
        const bool multi_view_active =
            (options_.multi_view_geo_weight > 0.F ||
             options_.multi_view_ncc_weight > 0.F) &&
            iteration >= options_.depth_normal_from_iter &&
            !multi_view_neighbours[view_index].empty();
        raster_options.require_depth = options_.use_mvs_depth ||
                                       options_.use_mvs_normals ||
                                       depth_normal_active ||
                                       multi_view_active;
        RenderResult rendered = rasterizer.forward(model, target.camera, raster_options);
        detail::LossGradients loss = detail::compute_training_loss(
            rendered, target, options_, report_progress,
            depth_normal_active);
        detail::MultiViewLoss multi_view_loss;
        DepthSampleGradients multi_view_sample_gradients;
        bool has_multi_view_sample_gradients = false;
        if (multi_view_active) {
            const auto& candidates = multi_view_neighbours[view_index];
            std::uniform_int_distribution<std::size_t> select_neighbour(
                0, candidates.size() - 1);
            const std::size_t neighbour_index =
                candidates[select_neighbour(random)];
            const TrainingView neighbour =
                view_cache.get(neighbour_index);
            const auto world_points = detail::unproject_depth_to_world(
                rendered.median_depth, target.camera);
            const DepthSampleResult sampled = rasterizer.sample_depth(
                model, world_points, neighbour.camera,
                raster_options);
            tinytensor::Tensor grad_sampled_points;
            multi_view_loss = detail::add_multi_view_loss(
                sampled.camera_points, sampled.inside, rendered, target,
                neighbour, options_, loss,
                grad_sampled_points, report_progress);
            multi_view_sample_gradients = rasterizer.sample_depth_backward(
                model, sampled, grad_sampled_points);
            detail::add_sample_depth_point_gradients(
                target.camera, multi_view_sample_gradients.points, loss);
            has_multi_view_sample_gradients = true;
            if (report_progress) {
                loss.total += options_.multi_view_geo_weight *
                                  multi_view_loss.geometry +
                              options_.multi_view_ncc_weight *
                                  multi_view_loss.ncc;
                loss.depth_value += options_.multi_view_geo_weight *
                                    multi_view_loss.geometry;
                loss.normal_value += options_.multi_view_ncc_weight *
                                     multi_view_loss.ncc;
            }
        }
        ModelGradients gradients = rasterizer.backward(
            model, rendered, loss.color, loss.alpha, loss.depth, loss.normal);
        if (has_multi_view_sample_gradients)
            detail::add_sample_depth_model_gradients(
                multi_view_sample_gradients, gradients);
        if (densification_enabled)
            detail::accumulate_densification_stats(
                gradients.refine_weight, rendered.visibility, rendered.radii,
                densification_stats, target.camera.width,
                target.camera.height,
                options_.densification_strategy !=
                    DensificationStrategy::default_strategy,
                options_.densification_strategy ==
                        DensificationStrategy::adc_plus ||
                    options_.densification_strategy ==
                        DensificationStrategy::adc_igs);

        // Dense MVS already provides accurate surface positions. Decaying the
        // position LR across the full 10k run keeps large geometric updates
        // active for too long and destroys that initialization. Match the
        // stable short-run trajectory, then retain the 1% tail for refinement.
        const unsigned means_decay_steps = options_.input_is_dense
            ? std::min(options_.iterations, 1'500U)
            : options_.iterations;
        const float progress_fraction = std::min(
            static_cast<float>(iteration - 1) /
                std::max(1U, means_decay_steps),
            1.F);
        const float means_lr = options_.means_lr *
                               means_learning_rate_scale *
                               std::pow(0.01F, progress_fraction);
        const bool global_structure_active =
            options_.structure_freeze_iter == 0 ||
            iteration <= options_.structure_freeze_iter;
        const bool dense_structure_active = !options_.input_is_dense ||
            options_.dense_structure_freeze_iter == 0 ||
            iteration <= options_.dense_structure_freeze_iter;
        const bool update_structure = global_structure_active &&
            dense_structure_active;
        if (update_structure) {
            detail::adam_step(
                model.means, gradients.means, means_state, means_lr,
                iteration, options_);
            detail::adam_step(
                model.log_scales, gradients.log_scales, scales_state,
                options_.scales_lr, iteration, options_, 0, 0.F,
                minimum_log_scale, maximum_log_scale);
            detail::constrain_scale_ratio(
                model.log_scales, options_.max_scale_ratio);
            detail::adam_step(
                model.quaternions, gradients.quaternions, rotations_state,
                options_.quaternions_lr, iteration, options_);
            detail::adam_step(
                model.opacity_logits, gradients.opacity_logits, opacity_state,
                options_.opacities_lr, iteration, options_, 0, 0.F,
                -12.F, 12.F);
        }
        const std::size_t full_sh_stride = model.sh.shape()[1] * 3;
        const std::size_t active_sh_stride =
            static_cast<std::size_t>(active_sh_degree + 1) *
            (active_sh_degree + 1) * 3;
        if (active_sh_stride < full_sh_stride)
            detail::adam_step_active_prefix(
                model.sh, gradients.sh, sh_state, options_.sh0_lr, iteration,
                options_, full_sh_stride, active_sh_stride,
                options_.sh_rest_lr);
        else if (options_.densification_strategy ==
                 DensificationStrategy::adc_plus)
            detail::adam_step_reduced_second(
                model.sh, gradients.sh, sh_state, options_.sh0_lr, iteration,
                options_, full_sh_stride, options_.sh_rest_lr);
        else
            detail::adam_step(
                model.sh, gradients.sh, sh_state, options_.sh0_lr, iteration,
                options_, full_sh_stride, options_.sh_rest_lr);

        if (densification_enabled &&
            (options_.densification_strategy ==
                 DensificationStrategy::adc_plus ||
             options_.densification_strategy ==
                 DensificationStrategy::adc_igs)) {
            const unsigned noise_stop = options_.densification_strategy ==
                    DensificationStrategy::adc_igs
                ? options_.grow_stop_iter
                : refine::strategy_schedule(options_).stop;
            if (iteration < noise_stop)
                detail::inject_adc_noise(
                    model, rendered.visibility,
                    means_lr * options_.mean_noise_weight,
                    options_.densification_strategy ==
                            DensificationStrategy::adc_plus
                        ? refinement_geometry.scale
                        : scene_extent,
                    options_.seed + iteration);
        }

        latest_refinement = {};
        if (densification_enabled) {
            const bool adc_plus =
                options_.densification_strategy ==
                DensificationStrategy::adc_plus;
            latest_refinement = refine::refine_gaussians(
                model, densification_stats, iteration,
                adc_plus ? refinement_geometry.maximum_extent : scene_extent,
                adc_plus ? refinement_geometry.center : scene_center,
                options_, random, adam_states);
            if (options_.densification_strategy ==
                    DensificationStrategy::default_strategy &&
                iteration < refine::strategy_schedule(options_).stop &&
                options_.opacity_reset_every != 0 && iteration > 0 &&
                iteration % options_.opacity_reset_every == 0) {
                detail::reset_opacity(
                    model, options_.prune_opacity * 2.F);
                opacity_state = detail::make_adam_state(
                    model.opacity_logits);
            }
            detail::constrain_scale_ratio(
                model.log_scales, options_.max_scale_ratio);
            const bool refined =
                refine::is_refinement_iteration(iteration, options_);
            if (refined && adc_plus) {
                refinement_geometry = refine::brush_scene_geometry(
                    download<float>(model.means));
                means_learning_rate_scale = refinement_geometry.scale;
            }
        }

        // The Mip-Splatting radius depends on Gaussian positions and count.
        // Refresh immediately after topology changes and periodically while
        // the means continue to move, matching pygsplat's GGGS schedule.
        if (use_3d_filter) {
            const bool adc_plus_refine =
                options_.densification_strategy ==
                    DensificationStrategy::adc_plus &&
                refine::is_refinement_iteration(iteration, options_);
            const float training_progress =
                static_cast<float>(iteration) /
                std::max(1.F, static_cast<float>(options_.iterations));
            // Match pygsplat: recompute after every ADC+ topology update
            // through 95%, then periodically while the fixed-topology tail
            // continues moving Gaussian means.
            const bool adc_plus_refresh =
                options_.densification_strategy ==
                    DensificationStrategy::adc_plus &&
                (adc_plus_refine ||
                 (training_progress > 0.95F &&
                  options_.filter_3d_update_interval != 0 &&
                  iteration % options_.filter_3d_update_interval == 0 &&
                  iteration + options_.filter_3d_update_interval <
                      options_.iterations));
            const bool other_refresh =
                options_.densification_strategy !=
                    DensificationStrategy::adc_plus &&
                (latest_refinement.grown != 0 ||
                 latest_refinement.pruned != 0 ||
                 (options_.filter_3d_update_interval != 0 &&
                  iteration % options_.filter_3d_update_interval == 0 &&
                  iteration + options_.filter_3d_update_interval <
                      options_.iterations));
            if (adc_plus_refresh || other_refresh)
                model.filter_3d = detail::compute_3d_filter(
                    model.means, filter_cameras, filter_3d_factor,
                    brush_filter);
        }

        bool continue_training = true;
        if (report_progress) {
            const cudaError_t report_error = cudaDeviceSynchronize();
            if (report_error != cudaSuccess)
                throw std::runtime_error(
                    std::string("GGGS training step failed: ") +
                    cudaGetErrorString(report_error));
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            const auto opacity_gradients = download<float>(
                gradients.opacity_logits);
            const auto opacity_logits = download<float>(model.opacity_logits);
            double opacity_gradient_sum = 0.0;
            double opacity_sum = 0.0;
            std::size_t positive_opacity_gradients = 0;
            for (std::size_t index = 0;
                 index < opacity_gradients.size(); ++index) {
                opacity_gradient_sum += opacity_gradients[index];
                positive_opacity_gradients += opacity_gradients[index] > 0.F;
                opacity_sum += 1.0 /
                    (1.0 + std::exp(-static_cast<double>(opacity_logits[index])));
            }
            const double inverse_gaussians = 1.0 /
                static_cast<double>(std::max<std::size_t>(
                    opacity_gradients.size(), 1));
            continue_training = progress({
                iteration, options_.iterations, model.size(),
                static_cast<std::size_t>(rendered.rendered_instances),
                view_index, latest_refinement.grown,
                latest_refinement.pruned, loss.total, loss.rgb,
                loss.alpha_value, loss.depth_value, loss.normal_value,
                static_cast<float>(opacity_gradient_sum * inverse_gaussians),
                static_cast<float>(positive_opacity_gradients *
                                   inverse_gaussians),
                static_cast<float>(opacity_sum * inverse_gaussians),
                milliseconds, multi_view_loss.geometry, multi_view_loss.ncc,
                multi_view_loss.geometry_pixels,
                multi_view_loss.ncc_pixels, active_resolution_scale,
                target.camera.width, target.camera.height,
                active_sh_degree});
        }
        if (!continue_training) break;
        if (evaluate &&
            std::find(
                options_.evaluation_iterations.begin(),
                options_.evaluation_iterations.end(),
                iteration) != options_.evaluation_iterations.end())
            launch_evaluation(iteration, model);
    }
    finish_evaluation();
    const cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string("GGGS training synchronization failed: ") +
            cudaGetErrorString(error));
    if (use_3d_filter)
        model.filter_3d = detail::compute_3d_filter(
            model.means, full_resolution_filter_cameras,
            filter_3d_factor, brush_filter);
    return model;
}

RenderMetrics render_evaluation_png(
    const GaussianModel& model, const mvs::MvsView& view,
    const std::filesystem::path& path, const TrainingOptions& training_options) {
    const TrainingView target = make_training_view(view, training_options);
    RasterizeOptions options;
    options.active_sh_degree = model.sh_degree;
    options.kernel_size = training_options.kernel_size;
    options.scale_modifier = training_options.scale_modifier;
    options.require_depth = false;
    const RenderResult rendered = Rasterizer().forward(
        model, target.camera, options);
    const std::vector<float> color = download<float>(rendered.color);
    const std::vector<float> alpha = download<float>(rendered.alpha);
    const std::vector<float> target_rgb = download<float>(target.rgb);
    const std::vector<float> mask =
        target.has_mask ? download<float>(target.mask)
                        : std::vector<float>{};
    const std::size_t pixels =
        static_cast<std::size_t>(target.camera.width) * target.camera.height;

    io::RgbImage image;
    image.width = target.camera.width;
    image.height = target.camera.height;
    image.pixels.resize(3 * pixels);
    double absolute_error = 0.0;
    double squared_error = 0.0;
    double alpha_bce = 0.0;
    std::size_t samples = 0;
    std::size_t covered = 0;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        if (alpha[pixel] > 0.01F) ++covered;
        const double predicted_alpha = std::clamp(
            static_cast<double>(alpha[pixel]), 1e-7, 1.0 - 1e-7);
        const bool foreground = !target.has_mask || mask[pixel] > 0.F;
        const double target_alpha = foreground ? 1.0 : 0.0;
        alpha_bce -= target_alpha * std::log(predicted_alpha) +
            (1.0 - target_alpha) * std::log(1.0 - predicted_alpha);
        for (int channel = 0; channel < 3; ++channel) {
            const std::size_t planar =
                static_cast<std::size_t>(channel) * pixels + pixel;
            const float prediction = std::clamp(color[planar], 0.F, 1.F);
            image.pixels[3 * pixel + channel] = static_cast<std::uint8_t>(
                std::lround(prediction * 255.F));
            if (foreground) {
                const double difference =
                    static_cast<double>(prediction - target_rgb[planar]);
                absolute_error += std::abs(difference);
                squared_error += difference * difference;
                ++samples;
            }
        }
    }
    io::save_rgb_png(image, path);
    const double inverse_samples = 1.0 / std::max<std::size_t>(samples, 1);
    const double mse = squared_error * inverse_samples;
    RenderMetrics metrics;
    metrics.mae = static_cast<float>(absolute_error * inverse_samples);
    metrics.psnr = mse > 0.0
        ? static_cast<float>(-10.0 * std::log10(mse))
        : std::numeric_limits<float>::infinity();
    const double masked_mse = squared_error /
        static_cast<double>(std::max<std::size_t>(3 * pixels, 1));
    metrics.masked_psnr = masked_mse > 0.0
        ? static_cast<float>(-10.0 * std::log10(masked_mse))
        : std::numeric_limits<float>::infinity();
    metrics.alpha_bce = static_cast<float>(alpha_bce /
        static_cast<double>(std::max<std::size_t>(pixels, 1)));
    metrics.alpha_coverage = pixels != 0
        ? static_cast<float>(covered) / static_cast<float>(pixels)
        : 0.F;
    return metrics;
}

void save_gaussians_ply(
    const GaussianModel& model, const std::filesystem::path& path) {
    const std::size_t count = model.size();
    const std::size_t bases = model.sh.shape()[1];
    const auto means = download<float>(model.means);
    const auto log_scales = download<float>(model.log_scales);
    const auto rotations = download<float>(model.quaternions);
    const auto opacities = download<float>(model.opacity_logits);
    const auto sh = download<float>(model.sh);
    const bool has_filter = model.filter_3d.is_valid() &&
        model.filter_3d.numel() == count;
    const auto filter_3d = has_filter
        ? download<float>(model.filter_3d)
        : std::vector<float>{};
    const auto require_finite = [](const std::vector<float>& values,
                                   const char* name) {
        const auto invalid = std::find_if(
            values.begin(), values.end(),
            [](const float value) { return !std::isfinite(value); });
        if (invalid != values.end()) {
            const auto index = static_cast<std::size_t>(
                std::distance(values.begin(), invalid));
            throw std::runtime_error(
                std::string("Refusing to write non-finite GGGS parameter ") +
                name + " at scalar index " + std::to_string(index));
        }
    };
    require_finite(means, "means");
    require_finite(log_scales, "log_scales");
    require_finite(rotations, "quaternions");
    require_finite(opacities, "opacity_logits");
    require_finite(sh, "SH");
    if (has_filter) require_finite(filter_3d, "filter_3D");
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Failed to create Gaussian PLY: " + path.string());
    output << "ply\nformat binary_little_endian 1.0\n"
           << "comment AetherScan GGGS (3DGS-compatible SH layout)\n"
           << "element vertex " << count << '\n'
           << "property float x\nproperty float y\nproperty float z\n"
           << "property float nx\nproperty float ny\nproperty float nz\n"
           << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    for (std::size_t i = 0; i < (bases - 1) * 3; ++i)
        output << "property float f_rest_" << i << '\n';
    output << "property float opacity\n"
           << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
           << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
           << (has_filter ? "property float filter_3D\n" : "")
           << "end_header\n";
    for (std::size_t gaussian = 0; gaussian < count; ++gaussian) {
        for (int axis = 0; axis < 3; ++axis) write_float(output, means[3 * gaussian + axis]);
        write_float(output, 0.F); write_float(output, 0.F); write_float(output, 0.F);
        for (int channel = 0; channel < 3; ++channel)
            write_float(output, sh[(gaussian * bases) * 3 + channel]);
        // Standard 3DGS PLY stores all remaining bases for R, then G, then B.
        for (int channel = 0; channel < 3; ++channel)
            for (std::size_t basis = 1; basis < bases; ++basis)
                write_float(output, sh[(gaussian * bases + basis) * 3 + channel]);
        write_float(output, opacities[gaussian]);
        for (int axis = 0; axis < 3; ++axis)
            write_float(output, log_scales[3 * gaussian + axis]);
        for (int component = 0; component < 4; ++component)
            write_float(output, rotations[4 * gaussian + component]);
        if (has_filter) write_float(output, filter_3d[gaussian]);
    }
    if (!output) throw std::runtime_error("Failed while writing Gaussian PLY: " + path.string());
}

GaussianModel load_gaussians_ply(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error(
            "Failed to open Gaussian PLY: " + path.string());

    std::string line;
    if (!std::getline(input, line) || line != "ply")
        throw std::runtime_error("Invalid Gaussian PLY header: " + path.string());
    std::size_t count = 0;
    bool binary_little_endian = false;
    bool reading_vertices = false;
    std::vector<std::string> properties;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") break;
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword == "format") {
            std::string format;
            tokens >> format;
            binary_little_endian = format == "binary_little_endian";
        } else if (keyword == "element") {
            std::string element;
            std::size_t element_count = 0;
            tokens >> element >> element_count;
            reading_vertices = element == "vertex";
            if (reading_vertices) count = element_count;
        } else if (keyword == "property" && reading_vertices) {
            std::string type;
            std::string name;
            tokens >> type >> name;
            if (type != "float" && type != "float32")
                throw std::runtime_error(
                    "Gaussian PLY requires float vertex properties: " +
                    path.string());
            properties.push_back(std::move(name));
        }
    }
    if (!binary_little_endian || count == 0 || properties.empty())
        throw std::runtime_error(
            "Gaussian PLY must contain binary little-endian vertices: " +
            path.string());

    std::unordered_map<std::string, std::size_t> property_index;
    property_index.reserve(properties.size());
    for (std::size_t index = 0; index < properties.size(); ++index)
        property_index.emplace(properties[index], index);
    const auto required = [&](const std::string& name) {
        const auto found = property_index.find(name);
        if (found == property_index.end())
            throw std::runtime_error(
                "Gaussian PLY is missing property " + name + ": " +
                path.string());
        return found->second;
    };
    const std::array<std::size_t, 3> position_index{
        required("x"), required("y"), required("z")};
    const std::array<std::size_t, 3> dc_index{
        required("f_dc_0"), required("f_dc_1"), required("f_dc_2")};
    const std::size_t opacity_index = required("opacity");
    const std::array<std::size_t, 3> scale_index{
        required("scale_0"), required("scale_1"), required("scale_2")};
    const std::array<std::size_t, 4> rotation_index{
        required("rot_0"), required("rot_1"), required("rot_2"),
        required("rot_3")};

    std::size_t rest_count = 0;
    while (property_index.contains("f_rest_" + std::to_string(rest_count)))
        ++rest_count;
    if (rest_count % 3U != 0)
        throw std::runtime_error(
            "Gaussian PLY has an invalid SH property count: " + path.string());
    const std::size_t bases = 1U + rest_count / 3U;
    const unsigned degree = static_cast<unsigned>(
        std::lround(std::sqrt(static_cast<double>(bases)))) - 1U;
    if (static_cast<std::size_t>(degree + 1U) * (degree + 1U) != bases ||
        degree > 3U)
        throw std::runtime_error(
            "Gaussian PLY has unsupported SH degree: " + path.string());
    std::vector<std::size_t> rest_index(rest_count);
    for (std::size_t rest = 0; rest < rest_count; ++rest)
        rest_index[rest] = required("f_rest_" + std::to_string(rest));

    std::vector<float> means(count * 3U);
    std::vector<float> scales(count * 3U);
    std::vector<float> rotations(count * 4U);
    std::vector<float> opacities(count);
    std::vector<float> sh(count * bases * 3U, 0.F);
    const auto filter_property = property_index.find("filter_3D");
    std::vector<float> filter;
    if (filter_property != property_index.end()) filter.resize(count);
    std::vector<float> row(properties.size());
    for (std::size_t gaussian = 0; gaussian < count; ++gaussian) {
        input.read(
            reinterpret_cast<char*>(row.data()),
            static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!input)
            throw std::runtime_error(
                "Gaussian PLY ended inside vertex data: " + path.string());
        for (int axis = 0; axis < 3; ++axis) {
            means[3U * gaussian + static_cast<std::size_t>(axis)] =
                row[position_index[static_cast<std::size_t>(axis)]];
            scales[3U * gaussian + static_cast<std::size_t>(axis)] =
                row[scale_index[static_cast<std::size_t>(axis)]];
        }
        for (int component = 0; component < 4; ++component)
            rotations[4U * gaussian + static_cast<std::size_t>(component)] =
                row[rotation_index[static_cast<std::size_t>(component)]];
        opacities[gaussian] = row[opacity_index];
        for (int channel = 0; channel < 3; ++channel) {
            sh[(gaussian * bases) * 3U +
               static_cast<std::size_t>(channel)] =
                row[dc_index[static_cast<std::size_t>(channel)]];
            for (std::size_t basis = 1; basis < bases; ++basis) {
                const std::size_t rest =
                    static_cast<std::size_t>(channel) * (bases - 1U) +
                    basis - 1U;
                sh[(gaussian * bases + basis) * 3U +
                   static_cast<std::size_t>(channel)] =
                    row[rest_index[rest]];
            }
        }
        if (!filter.empty()) filter[gaussian] = row[filter_property->second];
    }
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(
            values.begin(), values.end(),
            [](const float value) { return std::isfinite(value); });
    };
    if (!finite(means) || !finite(scales) || !finite(rotations) ||
        !finite(opacities) || !finite(sh) ||
        (!filter.empty() && !finite(filter)))
        throw std::runtime_error(
            "Gaussian PLY contains non-finite parameters: " + path.string());

    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        means, {count, 3U}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        scales, {count, 3U}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        rotations, {count, 4U}, tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        opacities, {count, 1U}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(
        sh, {count, bases, 3U}, tinytensor::Device::CUDA);
    if (!filter.empty())
        model.filter_3d = tinytensor::Tensor::from_vector(
            filter, {count, 1U}, tinytensor::Device::CUDA);
    model.sh_degree = degree;
    core::Logger::instance().info(
        "loaded GGGS PLY=", path, " gaussians=", count,
        " sh_degree=", degree, " filter_3d=", !filter.empty());
    return model;
}

}  // namespace aetherscan::splat
