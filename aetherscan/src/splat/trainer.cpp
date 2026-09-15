#include "splat/trainer.hpp"
#include "splat/visualize.hpp"

#include "bilateral_grid.hpp"
#include "cuda_ops.hpp"
#include "fused_ssim.hpp"
#include "ppisp.hpp"
#include "core/camera_projection.hpp"
#include "core/logging.hpp"
#include "densification.hpp"
#include "io/format_version.hpp"
#include "io/image.hpp"
#include "multi_view_scheduler.hpp"
#include "training_data_loader.hpp"

#include <cuda_runtime.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace aetherscan::splat {
namespace {

namespace data = training_data;
namespace refine = densification;

constexpr float k_sh0 = 0.28209479177387814F;

// Logging iterations also run densify / live preview / a full GPU sync, so a
// single step's wall time is a poor ETA rate. Smooth the mean ms/iter over
// each log interval instead.
constexpr double k_step_time_ema = 0.85;

std::size_t requested_preview_view(
    const TrainingOptions& options, const std::size_t view_count) {
    unsigned requested = options.preview_view_index;
    if (!options.preview_view_file.empty()) {
        std::ifstream input(options.preview_view_file);
        unsigned from_file{};
        if (input >> from_file) requested = from_file;
    }
    if (view_count == 0) return 0;
    return std::min<std::size_t>(requested, view_count - 1);
}

bool load_preview_camera_file(
    const std::filesystem::path& path, Camera& camera,
    std::uint64_t& revision, VisualizeOptions* vis = nullptr) {
    return load_preview_camera_sidecar(path, camera, revision, vis);
}

bool preview_camera_equivalent(const Camera& left, const Camera& right) {
    if (left.width != right.width || left.height != right.height ||
        left.model != right.model)
        return false;
    const auto close = [](const float a, const float b) {
        return std::abs(a - b) <=
            1e-4F * std::max(1.F, std::abs(a) + std::abs(b));
    };
    if (!close(left.fx, right.fx) || !close(left.fy, right.fy) ||
        !close(left.cx, right.cx) || !close(left.cy, right.cy))
        return false;
    for (std::size_t i = 0; i < left.world_to_camera.size(); ++i)
        if (!close(left.world_to_camera[i], right.world_to_camera[i]))
            return false;
    return true;
}

// Interactive preview pacing. The orbit camera is polled at k_preview_poll_ms
// so a drag lands within a frame or two, an interactive preview is only worth a
// raster once the pivot moved k_preview_pose_min_px on screen, and interactive
// previews are kept inside a fixed share of the training loop.
constexpr unsigned k_preview_poll_ms = 15;
constexpr float k_preview_pose_min_px = 0.35F;
constexpr unsigned k_preview_pose_hold_ms = 500;
constexpr unsigned k_preview_interactive_min_ms = 33;
constexpr float k_preview_interactive_budget = 5.F;

// Screen-space shift of `reference` between two cameras, in raster pixels.
// Cameras the reference cannot be projected into (non-pinhole models, pivots
// behind the camera) report an infinite shift so the caller always redraws.
[[nodiscard]] float preview_pose_shift_px(
    const Camera& left, const Camera& right,
    const std::array<float, 3>& reference) {
    if (left.model != CameraModel::pinhole ||
        right.model != CameraModel::pinhole)
        return std::numeric_limits<float>::infinity();
    const auto project = [&reference](
                             const Camera& camera, float& u, float& v) {
        const float* matrix = camera.world_to_camera.data();
        const float x = matrix[0] * reference[0] + matrix[4] * reference[1] +
                        matrix[8] * reference[2] + matrix[12];
        const float y = matrix[1] * reference[0] + matrix[5] * reference[1] +
                        matrix[9] * reference[2] + matrix[13];
        const float z = matrix[2] * reference[0] + matrix[6] * reference[1] +
                        matrix[10] * reference[2] + matrix[14];
        if (!(z > 1e-4F)) return false;
        u = camera.fx * x / z + camera.cx;
        v = camera.fy * y / z + camera.cy;
        return true;
    };
    float left_u = 0.F, left_v = 0.F, right_u = 0.F, right_v = 0.F;
    if (!project(left, left_u, left_v) || !project(right, right_u, right_v))
        return std::numeric_limits<float>::infinity();
    return std::max(std::abs(left_u - right_u), std::abs(left_v - right_v));
}

// Intrinsics changes (zoom, raster resize, camera model) always need a redraw;
// only a pure pose change is subject to the pixel threshold.
[[nodiscard]] bool preview_intrinsics_close(
    const Camera& left, const Camera& right) {
    if (left.width != right.width || left.height != right.height ||
        left.model != right.model)
        return false;
    const auto close = [](const float a, const float b) {
        return std::abs(a - b) <=
            1e-4F * std::max(1.F, std::abs(a) + std::abs(b));
    };
    return close(left.fx, right.fx) && close(left.fy, right.fy) &&
           close(left.cx, right.cx) && close(left.cy, right.cy);
}

// The editor publishes how many preview frames it has copied out of the shared
// image. Returns false when the sidecar is absent or unreadable, which keeps
// the pre-acknowledgement behaviour of waiting on the shared image.
[[nodiscard]] bool read_preview_ack_frames(
    const std::filesystem::path& path, std::uint64_t& frames) {
    if (path.empty()) return false;
    std::ifstream input(path);
    std::uint64_t parsed{};
    if (!(input >> parsed)) return false;
    frames = parsed;
    return true;
}

tinytensor::Tensor render_preview_color(
    const GaussianModel& model, const Camera& camera,
    const TrainingOptions& options, const unsigned active_sh_degree,
    VisualizeOptions vis, std::uint64_t& vis_revision) {
    vis.active_sh_degree = active_sh_degree;
    vis.kernel_size = options.kernel_size;
    vis.scale_modifier = options.scale_modifier;
    load_visualization_sidecar(options.preview_vis_file, vis, vis_revision);
    return visualize(model, camera, vis);
}

enum class CudaTrainingStage : std::size_t {
    data_load,
    raster_forward,
    preview,
    training_loss,
    multi_view_unproject,
    multi_view_sample_forward,
    multi_view_loss,
    multi_view_sample_backward,
    raster_backward,
    multi_view_gradient_merge,
    densification_stats,
    optimizer,
    adc_noise,
    refinement,
    filter_3d,
    count
};

constexpr std::size_t k_cuda_training_stage_count =
    static_cast<std::size_t>(CudaTrainingStage::count);

void check_profile_cuda(const cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    throw std::runtime_error(
        std::string{"Splat CUDA profiler "} + operation + " failed: " +
        cudaGetErrorString(error));
}

struct CudaProfileSample {
    std::array<cudaEvent_t, k_cuda_training_stage_count + 1> boundaries{};
};

class CudaTrainingProfiler {
public:
    explicit CudaTrainingProfiler(const TrainingOptions& options)
        : enabled_(options.profile_cuda),
          interval_(std::clamp(options.cuda_profile_interval, 1U, 1'000U)) {
        if (!enabled_) return;
        samples_.resize(interval_);
        try {
            for (CudaProfileSample& sample : samples_)
                for (cudaEvent_t& event : sample.boundaries)
                    check_profile_cuda(
                        cudaEventCreateWithFlags(&event, cudaEventDefault),
                        "event creation");
        } catch (...) {
            destroy_events();
            throw;
        }
        core::Logger::instance().info(
            "splat_cuda_profile enabled=1 interval=", interval_,
            " stages=data_load,raster_forward,preview,training_loss,multi_view_unproject,"
            "multi_view_sample_forward,multi_view_loss,"
            "multi_view_sample_backward,raster_backward,"
            "multi_view_gradient_merge,densification_stats,optimizer,"
            "adc_noise,refinement,filter_3d");
    }

    ~CudaTrainingProfiler() { destroy_events(); }

    CudaTrainingProfiler(const CudaTrainingProfiler&) = delete;
    CudaTrainingProfiler& operator=(const CudaTrainingProfiler&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    void begin_iteration(
        const unsigned iteration, const std::size_t gaussian_count) {
        if (!enabled_) return;
        if (sample_cursor_ >= samples_.size())
            throw std::logic_error("Splat CUDA profiler sample window overflow");
        if (sample_cursor_ == 0) {
            first_iteration_ = iteration;
            first_gaussians_ = gaussian_count;
        }
        active_stage_ = 0;
        check_profile_cuda(
            cudaEventRecord(
                samples_[sample_cursor_].boundaries.front(), nullptr),
            "start record");
    }

    void mark(const CudaTrainingStage stage) {
        if (!enabled_) return;
        const std::size_t index = static_cast<std::size_t>(stage);
        if (index != active_stage_)
            throw std::logic_error(
                "Splat CUDA profiler stage order is inconsistent");
        check_profile_cuda(
            cudaEventRecord(
                samples_[sample_cursor_].boundaries[index + 1], nullptr),
            "stage record");
        ++active_stage_;
    }

    void end_iteration(
        const unsigned iteration, const std::size_t gaussian_count,
        const std::size_t rendered_instances, const bool depth_normal_active,
        const bool multi_view_active, const bool refined,
        const bool filter_refreshed) {
        if (!enabled_) return;
        if (active_stage_ != k_cuda_training_stage_count)
            throw std::logic_error(
                "Splat CUDA profiler iteration ended before every stage");
        last_iteration_ = iteration;
        last_gaussians_ = gaussian_count;
        rendered_instances_sum_ += rendered_instances;
        depth_normal_steps_ += depth_normal_active;
        multi_view_steps_ += multi_view_active;
        refinement_steps_ += refined;
        filter_refresh_steps_ += filter_refreshed;
        ++sample_cursor_;
        if (sample_cursor_ == samples_.size()) report_and_reset();
    }

    void flush() {
        if (enabled_ && sample_cursor_ != 0) report_and_reset();
    }

private:
    void report_and_reset() {
        check_profile_cuda(
            cudaEventSynchronize(
                samples_[sample_cursor_ - 1].boundaries.back()),
            "window synchronization");
        std::array<double, k_cuda_training_stage_count> totals{};
        for (std::size_t sample_index = 0;
             sample_index < sample_cursor_; ++sample_index) {
            const CudaProfileSample& sample = samples_[sample_index];
            for (std::size_t stage = 0;
                 stage < k_cuda_training_stage_count; ++stage) {
                float elapsed_ms = 0.F;
                check_profile_cuda(
                    cudaEventElapsedTime(
                        &elapsed_ms, sample.boundaries[stage],
                        sample.boundaries[stage + 1]),
                    "elapsed-time query");
                totals[stage] += elapsed_ms;
            }
        }
        const double inverse_samples =
            1.0 / static_cast<double>(sample_cursor_);
        std::array<double, k_cuda_training_stage_count> averages{};
        double cuda_timeline_average_ms = 0.0;
        for (std::size_t stage = 0;
             stage < k_cuda_training_stage_count; ++stage) {
            averages[stage] = totals[stage] * inverse_samples;
            cuda_timeline_average_ms += averages[stage];
        }
        const auto value = [&](const CudaTrainingStage stage) {
            return averages[static_cast<std::size_t>(stage)];
        };
        const auto percent = [&](const CudaTrainingStage stage) {
            return cuda_timeline_average_ms > 0.0
                ? value(stage) * 100.0 / cuda_timeline_average_ms
                : 0.0;
        };
        const double multi_view_average_ms =
            value(CudaTrainingStage::multi_view_unproject) +
            value(CudaTrainingStage::multi_view_sample_forward) +
            value(CudaTrainingStage::multi_view_loss) +
            value(CudaTrainingStage::multi_view_sample_backward) +
            value(CudaTrainingStage::multi_view_gradient_merge);
        const double multi_view_percent =
            cuda_timeline_average_ms > 0.0
            ? multi_view_average_ms * 100.0 / cuda_timeline_average_ms
            : 0.0;
        core::Logger::instance().info(
            "splat_cuda_profile iterations=", first_iteration_, '-',
            last_iteration_, " samples=", sample_cursor_,
            " gaussians=[", first_gaussians_, ',', last_gaussians_, ']',
            " avg_tile_instances=",
            rendered_instances_sum_ * inverse_samples,
            " depth_normal_steps=", depth_normal_steps_,
            " multi_view_steps=", multi_view_steps_,
            " refinement_steps=", refinement_steps_,
            " filter_refresh_steps=", filter_refresh_steps_,
            " cuda_timeline_avg_ms=", cuda_timeline_average_ms,
            " data_load_ms=", value(CudaTrainingStage::data_load),
            " preview_ms=", value(CudaTrainingStage::preview),
            " raster_forward_ms=",
            value(CudaTrainingStage::raster_forward),
            " raster_forward_pct=",
            percent(CudaTrainingStage::raster_forward),
            " training_loss_ms=", value(CudaTrainingStage::training_loss),
            " training_loss_pct=",
            percent(CudaTrainingStage::training_loss),
            " multi_view_ms=", multi_view_average_ms,
            " multi_view_pct=", multi_view_percent,
            " multi_view_unproject_ms=",
            value(CudaTrainingStage::multi_view_unproject),
            " multi_view_unproject_pct=",
            percent(CudaTrainingStage::multi_view_unproject),
            " multi_view_sample_forward_ms=",
            value(CudaTrainingStage::multi_view_sample_forward),
            " multi_view_sample_forward_pct=",
            percent(CudaTrainingStage::multi_view_sample_forward),
            " multi_view_loss_ms=",
            value(CudaTrainingStage::multi_view_loss),
            " multi_view_loss_pct=",
            percent(CudaTrainingStage::multi_view_loss),
            " multi_view_sample_backward_ms=",
            value(CudaTrainingStage::multi_view_sample_backward),
            " multi_view_sample_backward_pct=",
            percent(CudaTrainingStage::multi_view_sample_backward),
            " raster_backward_ms=",
            value(CudaTrainingStage::raster_backward),
            " raster_backward_pct=",
            percent(CudaTrainingStage::raster_backward),
            " multi_view_gradient_merge_ms=",
            value(CudaTrainingStage::multi_view_gradient_merge),
            " multi_view_gradient_merge_pct=",
            percent(CudaTrainingStage::multi_view_gradient_merge),
            " densification_stats_ms=",
            value(CudaTrainingStage::densification_stats),
            " densification_stats_pct=",
            percent(CudaTrainingStage::densification_stats),
            " optimizer_ms=", value(CudaTrainingStage::optimizer),
            " optimizer_pct=", percent(CudaTrainingStage::optimizer),
            " adc_noise_ms=", value(CudaTrainingStage::adc_noise),
            " adc_noise_pct=", percent(CudaTrainingStage::adc_noise),
            " refinement_ms=", value(CudaTrainingStage::refinement),
            " refinement_pct=", percent(CudaTrainingStage::refinement),
            " filter_3d_ms=", value(CudaTrainingStage::filter_3d),
            " filter_3d_pct=", percent(CudaTrainingStage::filter_3d));
        sample_cursor_ = 0;
        rendered_instances_sum_ = 0;
        depth_normal_steps_ = 0;
        multi_view_steps_ = 0;
        refinement_steps_ = 0;
        filter_refresh_steps_ = 0;
    }

    void destroy_events() noexcept {
        for (CudaProfileSample& sample : samples_)
            for (cudaEvent_t& event : sample.boundaries) {
                if (event != nullptr) cudaEventDestroy(event);
                event = nullptr;
            }
    }

    bool enabled_{false};
    unsigned interval_{};
    std::vector<CudaProfileSample> samples_;
    std::size_t sample_cursor_{};
    std::size_t active_stage_{};
    unsigned first_iteration_{};
    unsigned last_iteration_{};
    std::size_t first_gaussians_{};
    std::size_t last_gaussians_{};
    std::uint64_t rendered_instances_sum_{};
    std::uint64_t depth_normal_steps_{};
    std::uint64_t multi_view_steps_{};
    std::uint64_t refinement_steps_{};
    std::uint64_t filter_refresh_steps_{};
};

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

tinytensor::Tensor normal_features_from_smallest_axis(
    const GaussianModel& model) {
    const detail::ActivatedParameters activated =
        detail::activate_parameters(model);
    const auto scales = activated.scales.to_vector();
    const auto quaternions = activated.quaternions.to_vector();
    std::vector<float> features(model.size() * 4, 0.F);
    for (std::size_t index = 0; index < model.size(); ++index) {
        int minimum_axis = 0;
        if (scales[3 * index + 1] < scales[3 * index])
            minimum_axis = 1;
        if (scales[3 * index + 2] <
            scales[3 * index + static_cast<std::size_t>(minimum_axis)])
            minimum_axis = 2;
        const Eigen::Quaternionf rotation(
            quaternions[4 * index], quaternions[4 * index + 1],
            quaternions[4 * index + 2], quaternions[4 * index + 3]);
        const mvs::Vec3f direction =
            rotation.toRotationMatrix().col(minimum_axis);
        for (int axis = 0; axis < 3; ++axis)
            features[4 * index + static_cast<std::size_t>(axis)] =
                direction(axis);
        // GaussianWrapping resets the learnable orientation sign to zero.
        features[4 * index + 3] = 0.F;
    }
    return tinytensor::Tensor::from_vector(
        features, {model.size(), std::size_t{4}},
        tinytensor::Device::CUDA);
}



}  // namespace

GaussianModel initialize_from_dense_cloud(
    const mvs::MvsScene& scene, const TrainingOptions& options) {
    if (scene.dense_cloud.points.empty())
        throw std::invalid_argument("Splat initialization requires a non-empty dense cloud");
    if (options.sh_degree > 3)
        throw std::invalid_argument("The current splat CUDA backend supports SH degree <= 3");
    const std::size_t count = scene.dense_cloud.points.size();
    const std::size_t bases = static_cast<std::size_t>(options.sh_degree + 1U) *
                              (options.sh_degree + 1U);
    const bool brush_adc_plus = is_adc_strategy(options.densification_strategy);
    std::vector<float> means(count * 3);
    std::vector<float> scales(count * 3);
    std::vector<float> quaternions(count * 4);
    std::vector<float> opacities(count);
    std::vector<float> sh(count * bases * 3, 0.F);
    std::vector<float> normal_features(count * 4, 0.F);

    std::vector<mvs::Vec3f> selected_positions(count);
    for (std::size_t index = 0; index < count; ++index)
        selected_positions[index] = scene.dense_cloud.points[index].position;
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
        const auto& point = scene.dense_cloud.points[index];
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
        mvs::Vec3f field_normal = point.normal;
        if (!field_normal.allFinite() ||
            field_normal.squaredNorm() < 1e-12F)
            field_normal = mvs::Vec3f::UnitZ();
        else
            field_normal.normalize();
        for (int axis = 0; axis < 3; ++axis)
            normal_features[4 * index + axis] = field_normal(axis);
        // GaussianWrapping resets orientation signs to zero when normal-field
        // regularization starts, so tanh(w) initially contributes no bias.
        normal_features[4 * index + 3] = 0.F;
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
    model.normal_features = tinytensor::Tensor::from_vector(
        normal_features, {count, 4}, tinytensor::Device::CUDA);
    model.sh_degree = options.sh_degree;
    return model;
}


Trainer::Trainer(TrainingOptions options) : options_(std::move(options)) {}

GaussianModel Trainer::train(
    const mvs::MvsScene& scene, ProgressCallback progress,
    EvaluationCallback evaluate, PreviewCallback preview,
    DevicePreviewCallback device_preview) const {
    if (scene.views.empty())
        throw std::invalid_argument("Splat training requires at least one MVS view");
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
            "Splat evaluation split left no training views");
    float active_resolution_scale =
        data::progressive_resolution_scale(1, options_);
    data::TrainingDataLoader view_cache(
        scene.views, options_, active_resolution_scale);
    if (options_.use_mask) {
        for (const std::size_t index : view_indices)
            if (!view_cache.has_mask(index))
                throw std::invalid_argument(
                    "Splat subject-only training requires a matching mask "
                    "file or source alpha channel for every selected view");
    }
    std::mt19937 random(options_.seed);
    std::vector<std::size_t> shuffled_views = view_indices;
    std::shuffle(shuffled_views.begin(), shuffled_views.end(), random);
    std::size_t shuffled_view_cursor = 0;
    const std::size_t initial_prefetch_end = std::min(
        shuffled_views.size(),
        shuffled_view_cursor + options_.training_prefetch_views + 1);
    for (std::size_t cursor = shuffled_view_cursor;
         cursor < initial_prefetch_end; ++cursor)
        view_cache.prefetch(shuffled_views[cursor]);

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
        is_adc_strategy(options_.densification_strategy)
            ? 0.1F
            : 0.2F;
    const bool brush_filter =
        is_adc_strategy(options_.densification_strategy);
    // Keep the Mip-Splatting floor separate from the canonical parameters.
    // pygsplat applies this filter only while rasterizing and recomputes it
    // after topology changes; repeatedly baking it into scale/opacity causes
    // a cumulative opacity loss.
    if (use_3d_filter)
        model.filter_3d = detail::compute_3d_filter(
            model.means, filter_cameras, filter_3d_factor, brush_filter);
    const bool native_non_pinhole = std::any_of(
        all_cameras.begin(), all_cameras.end(), [](const Camera& camera) {
            return uses_native_splat_projection(camera.model);
        });
    if (native_non_pinhole &&
        (options_.multi_view_geo_weight > 0.F ||
         options_.multi_view_ncc_weight > 0.F ||
         options_.use_depth_normal_loss || options_.use_normal_field ||
         options_.use_mvs_depth || options_.use_mvs_normals)) {
        core::Logger::instance().info(
            "Disabling pinhole-only depth-normal, normal-field and multi-view losses; "
            "native views do not reuse MVS depth/normal maps without reprojection. Cameras: "
            "fisheye/equirectangular splat cameras");
    }
    const bool use_multi_view =
        !native_non_pinhole &&
        (options_.multi_view_geo_weight > 0.F ||
         options_.multi_view_ncc_weight > 0.F);
    const unsigned multi_view_tail_interval =
        std::max(1U, options_.multi_view_tail_interval);
    const bool adaptive_multi_view =
        use_multi_view && options_.multi_view_adaptive_frequency &&
        options_.multi_view_adaptive_max_interval > 1;
    detail::MultiViewStabilityScheduler multi_view_scheduler(options_);
    TrainingOptions tail_multi_view_options = options_;
    tail_multi_view_options.multi_view_geo_weight *=
        static_cast<float>(multi_view_tail_interval);
    tail_multi_view_options.multi_view_ncc_weight *=
        static_cast<float>(multi_view_tail_interval);
    TrainingOptions adaptive_multi_view_options = options_;
    auto multi_view_stability_accumulator = adaptive_multi_view
        ? tinytensor::Tensor::zeros(
              {std::size_t{3}}, tinytensor::Device::CUDA)
        : tinytensor::Tensor{};
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
        is_adc_strategy(options_.densification_strategy)
            ? detail::make_reduced_second_adam_state(model.sh)
            : detail::make_adam_state(model.sh);
    detail::AdamState normal_features_state =
        detail::make_adam_state(model.normal_features);
    const refine::AdamStates adam_states{
        &means_state, &scales_state, &rotations_state, &opacity_state,
        &sh_state, &normal_features_state};
    // Per-view photometric compensation. State is indexed by the source view,
    // so it survives densification: rows change, cameras do not. PPISP owns
    // exposure and white-balance; the bilateral grid adds spatial variation.
    const bool ppisp_enabled = options_.use_ppisp;
    const bool bilagrid_enabled = options_.use_bilateral_grid;
    const bool ppisp_before_bilagrid = options_.ppisp_before_bilagrid;
    detail::PpispState ppisp_state = ppisp_enabled
        ? detail::make_ppisp_state(scene.views.size(), options_)
        : detail::PpispState{};
    detail::BilateralGridState bilagrid_state = bilagrid_enabled
        ? detail::make_bilateral_grid_state(scene.views.size(), options_)
        : detail::BilateralGridState{};
    const bool densification_enabled = refine::is_enabled(options_);
    detail::DensificationStats densification_stats =
        detail::make_densification_stats(model.size());
    refine::RefinementCounts latest_refinement;
    Rasterizer rasterizer;
    CudaTrainingProfiler cuda_profiler(options_);
    std::size_t last_preview_view = std::numeric_limits<std::size_t>::max();
    std::uint64_t last_preview_camera_revision =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t last_preview_vis_revision =
        std::numeric_limits<std::uint64_t>::max();
    const refine::SceneGeometry scene_geometry =
        refine::training_scene_geometry(scene, options_.input_is_dense);
    const float scene_extent = scene_geometry.scale;
    const mvs::Vec3f scene_center = scene_geometry.center;
    float means_learning_rate_scale = scene_extent;
    refine::SceneGeometry refinement_geometry = scene_geometry;
    if (is_adc_strategy(options_.densification_strategy)) {
        refinement_geometry = refine::brush_scene_geometry_cuda(model.means);
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

    auto interval_started = std::chrono::steady_clock::now();
    unsigned last_progress_iteration = 0;
    double ema_step_ms = 0.0;
    auto next_preview_poll = std::chrono::steady_clock::time_point::min();
    auto next_extra_preview = std::chrono::steady_clock::time_point::min();
    Camera last_preview_camera{};
    bool has_last_preview_camera = false;
    // Interactive previews are gated on the pivot's screen shift, so the pivot
    // is the scene centre the reconstruction is framed around.
    const std::array<float, 3> preview_pivot{
        scene_center.x(), scene_center.y(), scene_center.z()};
    auto next_pose_hold = std::chrono::steady_clock::time_point::min();
    // Preview frames handed to the device transport; the editor acknowledges
    // them through the ack sidecar.
    std::uint64_t preview_frames_queued = 0;
    // Last acknowledgement the editor published. Once it has published one, a
    // transiently unreadable sidecar keeps this value instead of dropping back
    // to the blocking handshake.
    std::uint64_t preview_ack_frames = 0;
    bool has_preview_ack = false;
    bool reported_preview_ack = false;
    auto next_preview_drop_log = std::chrono::steady_clock::time_point::min();
    const bool preview_has_sidecars = !options_.preview_camera_file.empty() ||
        !options_.preview_view_file.empty() || !options_.preview_vis_file.empty();

    for (unsigned iteration = 1; iteration <= options_.iterations; ++iteration) {
        cuda_profiler.begin_iteration(iteration, model.size());
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
        cuda_profiler.mark(CudaTrainingStage::data_load);
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
        const bool depth_normal_active = !native_non_pinhole &&
            options_.use_depth_normal_loss &&
            options_.depth_normal_weight > 0.F &&
            iteration >= options_.depth_normal_from_iter;
        const bool normal_field_active = !native_non_pinhole && options_.use_normal_field &&
            options_.normal_field_weight > 0.F &&
            iteration >= options_.normal_field_from_iter;
        const bool multi_view_eligible = use_multi_view &&
            iteration >= options_.depth_normal_from_iter &&
            !multi_view_neighbours[view_index].empty();
        const bool multi_view_tail = !adaptive_multi_view &&
            iteration > options_.grow_stop_iter;
        const unsigned active_multi_view_interval = adaptive_multi_view
            ? multi_view_scheduler.interval()
            : multi_view_tail ? multi_view_tail_interval : 1U;
        const bool multi_view_scheduled =
            active_multi_view_interval == 1U ||
            (adaptive_multi_view
                ? iteration % active_multi_view_interval == 0
                : (iteration - options_.grow_stop_iter) %
                        active_multi_view_interval == 0);
        const bool multi_view_active =
            multi_view_eligible && multi_view_scheduled;
        std::size_t multi_view_neighbour_index = 0;
        if (multi_view_eligible) {
            const auto& candidates = multi_view_neighbours[view_index];
            std::uniform_int_distribution<std::size_t> select_neighbour(
                0, candidates.size() - 1);
            multi_view_neighbour_index =
                candidates[select_neighbour(random)];
        }
        raster_options.require_depth = options_.use_mvs_depth ||
                                       options_.use_mvs_normals ||
                                       depth_normal_active ||
                                       normal_field_active ||
                                       multi_view_active;
        RenderResult rendered = rasterizer.forward(model, target.camera, raster_options);
        cuda_profiler.mark(CudaTrainingStage::raster_forward);
        const bool preview_enabled = (preview || device_preview) &&
            options_.preview_interval != 0;
        const bool preview_scheduled = preview_enabled &&
            (iteration == 1 || iteration == options_.iterations ||
             iteration % options_.preview_interval == 0);
        // Sidecar polls are for interactive orbit/vis changes. They must not
        // launch a second full raster every optimizer step, so the pixel gate
        // and the interactive pacing below decide what is worth a raster: the
        // sidecars are tiny and only rewritten when the editor has news.
        if (preview_enabled && (preview_scheduled ||
                (preview_has_sidecars &&
                 std::chrono::steady_clock::now() >= next_preview_poll))) {
            const auto preview_now = std::chrono::steady_clock::now();
            next_preview_poll =
                preview_now + std::chrono::milliseconds(k_preview_poll_ms);
            Camera preview_camera;
            std::uint64_t camera_revision = 0;
            VisualizeOptions camera_vis;
            const bool custom_camera = load_preview_camera_file(
                options_.preview_camera_file, preview_camera, camera_revision,
                &camera_vis);
            const std::size_t preview_index =
                requested_preview_view(options_, scene.views.size());
            std::uint64_t vis_revision = last_preview_vis_revision;
            VisualizeOptions vis_peek = camera_vis;
            load_visualization_sidecar(
                options_.preview_vis_file, vis_peek, vis_revision);
            const bool any_pose_change = custom_camera &&
                (!has_last_preview_camera ||
                 !preview_camera_equivalent(
                     preview_camera, last_preview_camera));
            // A sub-pixel orbit update is not worth a raster, but it must not
            // stall the live view either: force one once the change has been
            // pending for k_preview_pose_hold_ms. The shift accumulates against
            // the camera the editor last received, so a slow but visible drag
            // still reaches the pixel threshold on its own.
            bool pose_changed = any_pose_change;
            if (any_pose_change && has_last_preview_camera &&
                preview_intrinsics_close(
                    preview_camera, last_preview_camera)) {
                const float shift = preview_pose_shift_px(
                    last_preview_camera, preview_camera, preview_pivot);
                pose_changed = shift >= k_preview_pose_min_px ||
                    preview_now >= next_pose_hold;
            }
            if (pose_changed)
                next_pose_hold = preview_now +
                    std::chrono::milliseconds(k_preview_pose_hold_ms);
            const bool extra_preview_allowed =
                preview_now >= next_extra_preview;
            const bool due =
                iteration == 1 ||
                iteration == options_.iterations ||
                iteration % options_.preview_interval == 0 ||
                vis_revision != last_preview_vis_revision ||
                (custom_camera
                     ? pose_changed && extra_preview_allowed
                     : preview_index != last_preview_view);
            // The editor publishes how many preview frames it has copied out of
            // the shared image. While that sidecar is readable, drop this
            // preview instead of entering the handshake: the shared image still
            // holds the frame the editor is working on, and the semaphore wait
            // inside submit() remains the real barrier, so a stale
            // acknowledgement can cost an update but can never tear the image.
            std::uint64_t ack_frames = 0;
            if (device_preview &&
                read_preview_ack_frames(
                    options_.preview_ack_file, ack_frames)) {
                has_preview_ack = true;
                preview_ack_frames = ack_frames;
                if (!reported_preview_ack) {
                    reported_preview_ack = true;
                    core::Logger::instance().info(
                        "splat_preview_ack_frames=", ack_frames,
                        " file=\"", options_.preview_ack_file, '"');
                }
            }
            const bool shared_image_free =
                !has_preview_ack || preview_ack_frames >= preview_frames_queued;
            if (due && !shared_image_free && preview_now >= next_preview_drop_log) {
                // The editor is still copying the previous frame. Dropping this
                // preview keeps the optimizer moving; the next poll retries.
                next_preview_drop_log = preview_now + std::chrono::seconds(2);
                core::Logger::instance().info(
                    "splat_preview_dropped iteration=", iteration,
                    " editor_frames=", preview_ack_frames,
                    " queued_frames=", preview_frames_queued,
                    " action=skip");
            }
            if (due && shared_image_free) {
                const auto preview_started = std::chrono::steady_clock::now();
                if (!custom_camera) {
                    preview_camera = all_cameras[preview_index];
                }
                const tinytensor::Tensor preview_color = render_preview_color(
                    model, preview_camera, options_,
                    raster_options.active_sh_degree, camera_vis, vis_revision);
                last_preview_view = preview_index;
                last_preview_camera_revision = camera_revision;
                last_preview_vis_revision = vis_revision;
                last_preview_camera = preview_camera;
                has_last_preview_camera = true;
                if (device_preview) {
                    device_preview(
                        iteration, preview_index, preview_camera,
                        preview_color);
                    ++preview_frames_queued;
                } else {
                    const std::vector<float> planar =
                        download<float>(preview_color);
                    const std::size_t pixels =
                        static_cast<std::size_t>(preview_camera.width) *
                        preview_camera.height;
                    TrainingPreview frame;
                    frame.iteration = iteration;
                    frame.view_index = preview_index;
                    frame.width = preview_camera.width;
                    frame.height = preview_camera.height;
                    frame.rgb.resize(3 * pixels);
                    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                        for (std::size_t channel = 0; channel < 3; ++channel) {
                            const float value = std::clamp(
                                planar[channel * pixels + pixel], 0.F, 1.F);
                            frame.rgb[3 * pixel + channel] =
                                static_cast<std::uint8_t>(
                                    std::lround(value * 255.F));
                        }
                    }
                    preview(std::move(frame));
                }
                if (pose_changed && !preview_scheduled) {
                    // Keep interactive previews inside a fixed share of the
                    // loop: never sooner than the render that just happened.
                    const auto cost = std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - preview_started);
                    next_extra_preview = preview_now + std::max(
                        std::chrono::milliseconds(
                            k_preview_interactive_min_ms),
                        std::chrono::milliseconds(static_cast<long long>(
                            static_cast<float>(cost.count()) *
                            k_preview_interactive_budget)));
                }
            }
        }
        cuda_profiler.mark(CudaTrainingStage::preview);
        // Colour correction is applied to the plane the photometric loss sees.
        // Previews, held-out evaluation and exported models keep the canonical
        // appearance, so these transforms never leave training.
        RenderResult loss_render = rendered;
        const tinytensor::Tensor* photo_color = &rendered.color;
        const tinytensor::Tensor* ppisp_input = nullptr;
        const tinytensor::Tensor* bilagrid_input = nullptr;
        if (ppisp_enabled && ppisp_before_bilagrid) {
            ppisp_input = photo_color;
            detail::apply_ppisp(
                *ppisp_input, ppisp_state, target.camera, view_index);
            photo_color = &ppisp_state.output;
        }
        if (bilagrid_enabled) {
            bilagrid_input = photo_color;
            detail::apply_bilateral_grid(
                *bilagrid_input, bilagrid_state, view_index);
            photo_color = &bilagrid_state.output;
        }
        if (ppisp_enabled && !ppisp_before_bilagrid) {
            ppisp_input = photo_color;
            detail::apply_ppisp(
                *ppisp_input, ppisp_state, target.camera, view_index);
            photo_color = &ppisp_state.output;
        }
        loss_render.color = *photo_color;
        detail::LossGradients loss = detail::compute_training_loss(
            loss_render, target, options_, report_progress,
            depth_normal_active);
        RenderResult normal_field_render;
        detail::LossGradients normal_field_loss;
        ModelGradients normal_field_gradients;
        if (options_.use_normal_field &&
            iteration == std::max(options_.normal_field_from_iter, 1U)) {
            model.normal_features =
                normal_features_from_smallest_axis(model);
            normal_features_state =
                detail::make_adam_state(model.normal_features);
        }
        if (normal_field_active) {
            RasterizeOptions normal_options = raster_options;
            normal_options.colors_precomp =
                detail::normal_features_to_normals(model.normal_features);
            normal_options.require_depth = true;
            normal_field_render = rasterizer.forward(
                model, target.camera, normal_options);
            normal_field_loss = detail::compute_normal_field_loss(
                normal_field_render, target.camera,
                options_.normal_field_weight *
                    options_.normal_field_depth_ratio,
                report_progress);
            normal_field_gradients = rasterizer.backward(
                model, normal_field_render, normal_field_loss.color,
                normal_field_loss.alpha, normal_field_loss.depth,
                normal_field_loss.normal);
            normal_field_gradients.normal_features =
                detail::normal_features_backward(
                    model.normal_features,
                    normal_field_gradients.colors_precomp);
            if (report_progress) {
                loss.total += normal_field_loss.total;
                loss.normal_value += normal_field_loss.normal_value;
            }
        }
        cuda_profiler.mark(CudaTrainingStage::training_loss);
        detail::MultiViewLoss multi_view_loss;
        DepthSampleGradients multi_view_sample_gradients;
        bool has_multi_view_sample_gradients = false;
        if (multi_view_active) {
            const TrainingView neighbour =
                view_cache.get(multi_view_neighbour_index);
            const TrainingOptions& multi_view_options =
                adaptive_multi_view
                ? adaptive_multi_view_options
                : multi_view_tail ? tail_multi_view_options : options_;
            const auto world_points = detail::unproject_depth_to_world(
                rendered.median_depth, target.camera);
            cuda_profiler.mark(CudaTrainingStage::multi_view_unproject);
            const DepthSampleResult sampled = rasterizer.sample_depth(
                model, world_points, neighbour.camera,
                raster_options);
            cuda_profiler.mark(
                CudaTrainingStage::multi_view_sample_forward);
            tinytensor::Tensor grad_sampled_points;
            multi_view_loss = detail::add_multi_view_loss(
                sampled.camera_points, sampled.inside, rendered, target,
                neighbour, multi_view_options, loss,
                grad_sampled_points, report_progress,
                adaptive_multi_view
                    ? &multi_view_stability_accumulator
                    : nullptr);
            cuda_profiler.mark(CudaTrainingStage::multi_view_loss);
            multi_view_sample_gradients = rasterizer.sample_depth_backward(
                model, sampled, grad_sampled_points);
            detail::add_sample_depth_point_gradients(
                target.camera, multi_view_sample_gradients.points, loss);
            cuda_profiler.mark(
                CudaTrainingStage::multi_view_sample_backward);
            has_multi_view_sample_gradients = true;
            if (report_progress) {
                loss.total += multi_view_options.multi_view_geo_weight *
                                  multi_view_loss.geometry +
                              multi_view_options.multi_view_ncc_weight *
                                  multi_view_loss.ncc;
                loss.depth_value += multi_view_options.multi_view_geo_weight *
                                    multi_view_loss.geometry;
                loss.normal_value += multi_view_options.multi_view_ncc_weight *
                                     multi_view_loss.ncc;
            }
        } else {
            cuda_profiler.mark(CudaTrainingStage::multi_view_unproject);
            cuda_profiler.mark(
                CudaTrainingStage::multi_view_sample_forward);
            cuda_profiler.mark(CudaTrainingStage::multi_view_loss);
            cuda_profiler.mark(
                CudaTrainingStage::multi_view_sample_backward);
        }
        tinytensor::Tensor densify_map;
        if (densification_enabled && options_.densify_use_error_map) {
            const bool mask_enabled =
                target.has_mask && (options_.use_mask || target.mask_is_validity);
            densify_map = detail::compute_ssim_cs_error_map(
                loss_render.color, target.rgb, target.mask, mask_enabled,
                options_.densify_loss_map_power);
        }
        tinytensor::Tensor* photo_grad = &loss.color;
        if (ppisp_enabled && !ppisp_before_bilagrid) {
            detail::backward_ppisp(
                ppisp_state, *ppisp_input, *photo_grad, target.camera,
                view_index);
            detail::step_ppisp(ppisp_state, options_, iteration);
            photo_grad = &ppisp_state.input_grad;
        }
        if (bilagrid_enabled) {
            detail::backward_bilateral_grid(
                bilagrid_state, *bilagrid_input, *photo_grad, view_index);
            detail::step_bilateral_grid(bilagrid_state, options_, iteration);
            photo_grad = &bilagrid_state.input_grad;
        }
        if (ppisp_enabled && ppisp_before_bilagrid) {
            detail::backward_ppisp(
                ppisp_state, *ppisp_input, *photo_grad, target.camera,
                view_index);
            detail::step_ppisp(ppisp_state, options_, iteration);
            photo_grad = &ppisp_state.input_grad;
        }
        if (ppisp_enabled && report_progress) {
            // How much exposure / white-balance drift the capture carried.
            // The layout decides which parameters are gains.
            const std::array<float, 2> deviation =
                detail::ppisp_identity_deviation(ppisp_state);
            core::Logger::instance().info(
                "splat_ppisp view=", view_index,
                " layout=", static_cast<int>(ppisp_state.type),
                " mean_gain_deviation=", deviation[0],
                " maximum_gain_deviation=", deviation[1]);
        }
        ModelGradients gradients = rasterizer.backward(
            model, rendered, *photo_grad, loss.alpha, loss.depth, loss.normal,
            densify_map);
        if (normal_field_active)
            detail::add_model_gradients(
                normal_field_gradients, gradients, false);
        cuda_profiler.mark(CudaTrainingStage::raster_backward);
        if (has_multi_view_sample_gradients)
            detail::add_sample_depth_model_gradients(
                multi_view_sample_gradients, gradients);
        cuda_profiler.mark(CudaTrainingStage::multi_view_gradient_merge);
        if (densification_enabled) {
            tinytensor::Tensor step_score = gradients.refine_weight;
            bool use_maximum = true;
            if (options_.densify_use_error_map &&
                gradients.densify_weight.is_valid()) {
                step_score = detail::densify_avg_scores(
                    gradients.densify_weight, gradients.densify_weight_den);
                step_score = detail::densify_blend_world_gradient(
                    step_score, gradients.means, model.log_scales,
                    options_.densify_world_gradient_blend);
                use_maximum = false;
            }
            detail::accumulate_densification_stats(
                step_score, rendered.visibility, rendered.radii,
                densification_stats, target.camera.width,
                target.camera.height,
                use_maximum,
                is_adc_strategy(options_.densification_strategy),
                options_.densify_use_error_map ? options_.densify_score_power : 1.F,
                options_.densification_strategy == DensificationStrategy::adc_igs
                    ? options_.densify_screen_threshold : 0.F,
                gradients.refine_weight, static_cast<int>(view_index));
        }
        cuda_profiler.mark(CudaTrainingStage::densification_stats);

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
            detail::adam_step_structure(model, gradients, means_state, scales_state,
                rotations_state, opacity_state, means_lr, iteration, options_,
                minimum_log_scale, maximum_log_scale);
        }
        const std::size_t full_sh_stride = model.sh.shape()[1] * 3;
        detail::add_sh_regularization(
            model.sh, gradients.sh, options_.sh_regularization_weight);
        const std::size_t active_sh_stride =
            static_cast<std::size_t>(active_sh_degree + 1) *
            (active_sh_degree + 1) * 3;
        if (active_sh_stride < full_sh_stride)
            detail::adam_step_active_prefix(
                model.sh, gradients.sh, sh_state, options_.sh0_lr, iteration,
                options_, full_sh_stride, active_sh_stride,
                options_.sh_rest_lr);
        else if (is_adc_strategy(options_.densification_strategy))
            detail::adam_step_reduced_second(
                model.sh, gradients.sh, sh_state, options_.sh0_lr, iteration,
                options_, full_sh_stride, options_.sh_rest_lr);
        else
            detail::adam_step(
                model.sh, gradients.sh, sh_state, options_.sh0_lr, iteration,
                options_, full_sh_stride, options_.sh_rest_lr);
        if (normal_field_active)
            detail::adam_step(
                model.normal_features,
                normal_field_gradients.normal_features,
                normal_features_state, options_.normal_features_lr,
                iteration, options_);
        cuda_profiler.mark(CudaTrainingStage::optimizer);

        if (densification_enabled &&
            is_adc_strategy(options_.densification_strategy)) {
            const unsigned noise_stop = refine::strategy_schedule(options_).stop;
            if (iteration < noise_stop)
                detail::inject_adc_noise(
                    model, rendered.visibility,
                    options_.densify_revised_noise
                        ? options_.mean_noise_weight * std::pow(0.01F, progress_fraction)
                        : means_lr * options_.mean_noise_weight,
                    is_adc_strategy(options_.densification_strategy)
                        ? refinement_geometry.scale
                        : scene_extent,
                    options_.seed + iteration,
                    rendered.radii,
                    options_.densify_revised_noise);
        }
        cuda_profiler.mark(CudaTrainingStage::adc_noise);

        // Reduce opacity stats before densify remaps rows. Gradients still
        // match the pre-refinement model; a later download would not.
        detail::OpacityProgressStats opacity_stats{};
        if (report_progress)
            opacity_stats = detail::summarize_opacity_progress(
                model.opacity_logits, gradients.opacity_logits);

        latest_refinement = {};
        bool refinement_happened = false;
        if (densification_enabled) {
            const bool adc_plus =
                is_adc_strategy(options_.densification_strategy);
            if (refine::is_refinement_iteration(iteration, options_)) {
                const std::size_t remaining =
                    options_.densification_cap > model.size()
                        ? options_.densification_cap - model.size()
                        : 0;
                const std::size_t expected_growth = std::min(
                    remaining,
                    std::max<std::size_t>(
                        std::size_t{65'536}, model.size() / 4));
                view_cache.ensure_device_headroom(
                    std::size_t{512} * 1024 * 1024 +
                    expected_growth * std::size_t{2} * 1024);
            }
            latest_refinement = refine::refine_gaussians(
                model, densification_stats, iteration,
                adc_plus ? refinement_geometry.maximum_extent : scene_extent,
                adc_plus ? refinement_geometry.center : scene_center,
                options_, random, adam_states);
            detail::constrain_scale_ratio(
                model.log_scales, options_.max_scale_ratio);
            refinement_happened =
                refine::is_refinement_iteration(iteration, options_);
            if (refinement_happened && adc_plus) {
                refinement_geometry = refine::brush_scene_geometry_cuda(model.means);
                means_learning_rate_scale = refinement_geometry.scale;
            }
        }
        if (adaptive_multi_view && refinement_happened) {
            const std::vector<float> window =
                multi_view_stability_accumulator.to_vector();
            multi_view_stability_accumulator.zero_();
            const float consistent_pixels = window[0];
            const float candidate_pixels = window[1];
            const float active_steps = window[2];
            if (candidate_pixels > 0.F && active_steps > 0.F) {
                const float depth_consistency =
                    consistent_pixels / candidate_pixels;
                const detail::GeometryDistributionSummary distribution =
                    detail::summarize_geometry_distribution(model);
                const detail::GeometryStabilityDecision decision =
                    multi_view_scheduler.update({
                        model.size(), latest_refinement.grown,
                        latest_refinement.pruned, depth_consistency,
                        distribution});
                adaptive_multi_view_options = options_;
                adaptive_multi_view_options.multi_view_geo_weight *=
                    static_cast<float>(decision.interval);
                adaptive_multi_view_options.multi_view_ncc_weight *=
                    static_cast<float>(decision.interval);
                core::Logger::instance().info(
                    "splat_mv_stability iteration=", iteration,
                    " reference_ready=", decision.reference_ready,
                    " stable=", decision.stable,
                    " stable_refinements=", decision.stable_refinements,
                    " interval=", decision.interval,
                    " reduced=", decision.reduced,
                    " recovered=", decision.recovered,
                    " gaussian_count=", model.size(),
                    " count_delta=", decision.count_delta,
                    " churn=", decision.churn,
                    " depth_consistency=", depth_consistency,
                    " depth_delta=", decision.depth_delta,
                    " opacity_mean=", distribution.opacity_mean,
                    " opacity_std=", distribution.opacity_stddev,
                    " log_scale_mean=", distribution.log_scale_mean,
                    " log_scale_std=", distribution.log_scale_stddev,
                    " log_anisotropy_mean=",
                    distribution.log_anisotropy_mean,
                    " distribution_delta=",
                    decision.distribution_delta,
                    " window_steps=", active_steps,
                    " consistent_pixels=", consistent_pixels,
                    " candidate_pixels=", candidate_pixels);
            } else {
                const detail::GeometryStabilityDecision decision =
                    multi_view_scheduler.invalidate();
                adaptive_multi_view_options = options_;
                core::Logger::instance().info(
                    "splat_mv_stability iteration=", iteration,
                    " reference_ready=false stable=false interval=",
                    decision.interval,
                    " recovered=", decision.recovered,
                    " reason=no_depth_candidates window_steps=",
                    active_steps);
            }
        }
        cuda_profiler.mark(CudaTrainingStage::refinement);

        // The Mip-Splatting radius depends on Gaussian positions and count.
        // Refresh immediately after topology changes and periodically while
        // the means continue to move, matching pygsplat's GGGS schedule.
        bool filter_refreshed = false;
        if (use_3d_filter) {
            const bool adc_plus_refine =
                is_adc_strategy(options_.densification_strategy) &&
                refine::is_refinement_iteration(iteration, options_);
            const float training_progress =
                static_cast<float>(iteration) /
                std::max(1.F, static_cast<float>(options_.iterations));
            // Match pygsplat: recompute after every ADC+ topology update
            // through 95%, then periodically while the fixed-topology tail
            // continues moving Gaussian means.
            const bool adc_plus_refresh =
                is_adc_strategy(options_.densification_strategy) &&
                (adc_plus_refine ||
                 (training_progress > 0.95F &&
                  options_.filter_3d_update_interval != 0 &&
                  iteration % options_.filter_3d_update_interval == 0 &&
                  iteration + options_.filter_3d_update_interval <
                      options_.iterations));
            const bool other_refresh =
                !is_adc_strategy(options_.densification_strategy) &&
                (latest_refinement.grown != 0 ||
                 latest_refinement.pruned != 0 ||
                 (options_.filter_3d_update_interval != 0 &&
                  iteration % options_.filter_3d_update_interval == 0 &&
                  iteration + options_.filter_3d_update_interval <
                      options_.iterations));
            filter_refreshed = adc_plus_refresh || other_refresh;
            if (filter_refreshed)
                model.filter_3d = detail::compute_3d_filter(
                    model.means, filter_cameras, filter_3d_factor,
                    brush_filter);
        }
        cuda_profiler.mark(CudaTrainingStage::filter_3d);
        cuda_profiler.end_iteration(
            iteration, model.size(),
            static_cast<std::size_t>(rendered.rendered_instances),
            depth_normal_active, multi_view_active, refinement_happened,
            filter_refreshed);

        bool continue_training = true;
        if (report_progress) {
            const auto now = std::chrono::steady_clock::now();
            const double interval_ms =
                std::chrono::duration<double, std::milli>(
                    now - interval_started).count();
            const unsigned steps = iteration - last_progress_iteration;
            double milliseconds = 0.0;
            if (last_progress_iteration > 0 && steps > 0) {
                const double mean_ms =
                    interval_ms / static_cast<double>(steps);
                ema_step_ms = ema_step_ms <= 0.0
                    ? mean_ms
                    : k_step_time_ema * ema_step_ms +
                          (1.0 - k_step_time_ema) * mean_ms;
                milliseconds = ema_step_ms;
            }
            continue_training = progress({
                iteration, options_.iterations, model.size(),
                static_cast<std::size_t>(rendered.rendered_instances),
                view_index, latest_refinement.grown,
                latest_refinement.pruned, loss.total, loss.rgb,
                loss.alpha_value, loss.depth_value, loss.normal_value,
                opacity_stats.gradient_mean,
                opacity_stats.positive_gradient_fraction,
                opacity_stats.opacity_mean,
                milliseconds, multi_view_loss.geometry, multi_view_loss.ncc,
                multi_view_loss.geometry_pixels,
                multi_view_loss.ncc_pixels, active_resolution_scale,
                target.camera.width, target.camera.height,
                active_sh_degree, active_multi_view_interval,
                multi_view_loss.geometry_candidates != 0
                    ? static_cast<float>(
                          multi_view_loss.geometry_pixels) /
                          static_cast<float>(
                              multi_view_loss.geometry_candidates)
                    : 0.F});
            interval_started = now;
            last_progress_iteration = iteration;
        }
        if (!continue_training) break;
        if (evaluate &&
            std::find(
                options_.evaluation_iterations.begin(),
                options_.evaluation_iterations.end(),
                iteration) != options_.evaluation_iterations.end())
            launch_evaluation(iteration, model);
    }
    cuda_profiler.flush();
    if (device_preview) {
        core::Logger::instance().info(
            "splat_preview_frames_queued=", preview_frames_queued,
            " editor_ack_frames=", preview_ack_frames,
            " acknowledged=", has_preview_ack ? 1 : 0);
    }
    if (options_.profile_cuda) {
        const auto cache = view_cache.stats();
        core::Logger::instance().info(
            "splat_data_cache requests=", cache.requests,
            " device_hits=", cache.device_hits,
            " device_prefetch_hits=", cache.device_prefetch_hits,
            " uploaded_bytes=", cache.uploaded_bytes,
            " device_resident_bytes=", cache.device_resident_bytes,
            " device_budget_bytes=", cache.device_budget_bytes,
            " device_prefetch_pending=", cache.device_prefetch_pending,
            " device_prefetch_bytes=", cache.device_prefetch_bytes,
            " host_budget_bytes=", cache.host_budget_bytes,
            " dataset_packed_bytes=", cache.dataset_packed_bytes);
    }
    finish_evaluation();
    const cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string("Splat training synchronization failed: ") +
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
    const float ssim = detail::fused_ssim_metric(
        rendered.color, target.rgb, target.mask, target.has_mask,
        target.camera.width, target.camera.height);
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
    metrics.ssim = ssim;
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
    const bool has_normal_features = model.normal_features.is_valid() &&
        model.normal_features.shape().rank() == 2 &&
        model.normal_features.shape()[0] == count &&
        model.normal_features.shape()[1] == 4;
    const auto normal_features = has_normal_features
        ? download<float>(model.normal_features)
        : std::vector<float>{};
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
                std::string("Refusing to write non-finite splat parameter ") +
                name + " at scalar index " + std::to_string(index));
        }
    };
    require_finite(means, "means");
    require_finite(log_scales, "log_scales");
    require_finite(rotations, "quaternions");
    require_finite(opacities, "opacity_logits");
    require_finite(sh, "SH");
    if (has_normal_features)
        require_finite(normal_features, "normal_features");
    if (has_filter) require_finite(filter_3d, "filter_3D");
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Failed to create Gaussian PLY: " + path.string());
    output << "ply\nformat binary_little_endian 1.0\n"
           << "comment AetherScan splat (GGGS-derived geometry; 3DGS-compatible SH layout)\n"
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
           << (has_normal_features
                   ? "property float gaussian_features_0\n"
                     "property float gaussian_features_1\n"
                     "property float gaussian_features_2\n"
                     "property float gaussian_features_3\n"
                   : "")
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
        if (has_normal_features)
            for (int component = 0; component < 4; ++component)
                write_float(
                    output,
                    normal_features[4 * gaussian + component]);
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
    std::vector<float> normal_features(count * 4U, 0.F);
    const auto filter_property = property_index.find("filter_3D");
    std::vector<float> filter;
    if (filter_property != property_index.end()) filter.resize(count);
    std::array<std::size_t, 4> normal_feature_index{};
    bool has_normal_features = true;
    for (std::size_t component = 0; component < 4; ++component) {
        const auto found = property_index.find(
            "gaussian_features_" + std::to_string(component));
        if (found == property_index.end()) {
            has_normal_features = false;
            break;
        }
        normal_feature_index[component] = found->second;
    }
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
        if (has_normal_features)
            for (std::size_t component = 0; component < 4; ++component)
                normal_features[4U * gaussian + component] =
                    row[normal_feature_index[component]];
    }
    if (!has_normal_features) {
        // Legacy 3DGS PLY: seed the field from the thinnest covariance axis.
        // Its orientation is necessarily ambiguous without learned features;
        // w=1 keeps the field usable for PAM while further training can flip it.
        for (std::size_t gaussian = 0; gaussian < count; ++gaussian) {
            const auto scale = scales.begin() +
                static_cast<std::ptrdiff_t>(3U * gaussian);
            const int axis = static_cast<int>(std::distance(
                scale, std::min_element(scale, scale + 3)));
            float w = rotations[4U * gaussian];
            float x = rotations[4U * gaussian + 1U];
            float y = rotations[4U * gaussian + 2U];
            float z = rotations[4U * gaussian + 3U];
            const float inverse_norm = 1.F / std::max(
                std::sqrt(w * w + x * x + y * y + z * z), 1e-12F);
            w *= inverse_norm;
            x *= inverse_norm;
            y *= inverse_norm;
            z *= inverse_norm;
            const std::array<std::array<float, 3>, 3> columns{{
                {{1.F - 2.F * (y * y + z * z),
                  2.F * (x * y + w * z),
                  2.F * (x * z - w * y)}},
                {{2.F * (x * y - w * z),
                  1.F - 2.F * (x * x + z * z),
                  2.F * (y * z + w * x)}},
                {{2.F * (x * z + w * y),
                  2.F * (y * z - w * x),
                  1.F - 2.F * (x * x + y * y)}}}};
            for (int component = 0; component < 3; ++component)
                normal_features[4U * gaussian + component] =
                    columns[static_cast<std::size_t>(axis)]
                           [static_cast<std::size_t>(component)];
            normal_features[4U * gaussian + 3U] = 1.F;
        }
    }
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(
            values.begin(), values.end(),
            [](const float value) { return std::isfinite(value); });
    };
    if (!finite(means) || !finite(scales) || !finite(rotations) ||
        !finite(opacities) || !finite(sh) || !finite(normal_features) ||
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
    model.normal_features = tinytensor::Tensor::from_vector(
        normal_features, {count, 4U}, tinytensor::Device::CUDA);
    if (!filter.empty())
        model.filter_3d = tinytensor::Tensor::from_vector(
            filter, {count, 1U}, tinytensor::Device::CUDA);
    model.sh_degree = degree;
    core::Logger::instance().info(
        "loaded splat PLY=", path, " gaussians=", count,
        " sh_degree=", degree, " filter_3d=", !filter.empty(),
        " learned_normal_field=", has_normal_features);
    return model;
}

std::vector<std::uint8_t> encode_gaussians(const GaussianModel& model) {
    const std::size_t count = model.size();
    if (count == 0) {
        const std::uint32_t version = k_gaussian_chunk_version;
        const std::uint64_t stored_count = 0;
        const std::uint32_t sh_degree = model.sh_degree;
        const std::uint32_t flags = 0;
        std::vector<std::uint8_t> bytes(4 + 8 + 4 + 4);
        std::memcpy(bytes.data(), &version, 4);
        std::memcpy(bytes.data() + 4, &stored_count, 8);
        std::memcpy(bytes.data() + 12, &sh_degree, 4);
        std::memcpy(bytes.data() + 16, &flags, 4);
        return bytes;
    }
    const std::size_t bases = count == 0 ? 1 : model.sh.shape()[1];
    const auto means = download<float>(model.means);
    const auto log_scales = download<float>(model.log_scales);
    const auto rotations = download<float>(model.quaternions);
    const auto opacities = download<float>(model.opacity_logits);
    const auto sh = count == 0 ? std::vector<float>{} : download<float>(model.sh);
    const bool has_normal_features = model.normal_features.is_valid() &&
        model.normal_features.shape().rank() == 2 &&
        model.normal_features.shape()[0] == count &&
        model.normal_features.shape()[1] == 4;
    const auto normal_features = has_normal_features
        ? download<float>(model.normal_features)
        : std::vector<float>{};
    const bool has_filter = model.filter_3d.is_valid() &&
        model.filter_3d.numel() == count;
    const auto filter_3d = has_filter
        ? download<float>(model.filter_3d)
        : std::vector<float>{};
    std::uint32_t flags = 0;
    if (has_filter) flags |= 1U;
    if (has_normal_features) flags |= 2U;
    const std::uint32_t version = k_gaussian_chunk_version;
    const std::uint64_t stored_count = count;
    const std::uint32_t sh_degree = model.sh_degree;
    std::vector<std::uint8_t> bytes(
        sizeof(version) + sizeof(stored_count) + sizeof(sh_degree) +
        sizeof(flags) +
        (means.size() + log_scales.size() + rotations.size() + opacities.size() +
         sh.size() + normal_features.size() + filter_3d.size()) *
            sizeof(float));
    std::uint8_t* cursor = bytes.data();
    const auto append = [&](const void* data, const std::size_t size) {
        if (size == 0) return;
        std::memcpy(cursor, data, size);
        cursor += size;
    };
    append(&version, sizeof(version));
    append(&stored_count, sizeof(stored_count));
    append(&sh_degree, sizeof(sh_degree));
    append(&flags, sizeof(flags));
    append(means.data(), means.size() * sizeof(float));
    append(log_scales.data(), log_scales.size() * sizeof(float));
    append(rotations.data(), rotations.size() * sizeof(float));
    append(opacities.data(), opacities.size() * sizeof(float));
    append(sh.data(), sh.size() * sizeof(float));
    append(normal_features.data(), normal_features.size() * sizeof(float));
    append(filter_3d.data(), filter_3d.size() * sizeof(float));
    return bytes;
}

GaussianModel decode_gaussians(const std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 4 + 8 + 4 + 4)
        throw std::runtime_error("Gaussian chunk is too small");
    const std::uint8_t* cursor = bytes.data();
    const std::uint8_t* end = bytes.data() + bytes.size();
    const auto take = [&](const std::size_t size) {
        if (cursor + size > end)
            throw std::runtime_error("Truncated Gaussian chunk");
        const std::uint8_t* data = cursor;
        cursor += size;
        return data;
    };
    std::uint32_t version = 0;
    std::uint64_t count64 = 0;
    std::uint32_t sh_degree = 0;
    std::uint32_t flags = 0;
    std::memcpy(&version, take(sizeof(version)), sizeof(version));
    std::memcpy(&count64, take(sizeof(count64)), sizeof(count64));
    std::memcpy(&sh_degree, take(sizeof(sh_degree)), sizeof(sh_degree));
    std::memcpy(&flags, take(sizeof(flags)), sizeof(flags));
    if (version == 0 || version > k_gaussian_chunk_version)
        throw std::runtime_error(io::unsupported_payload_version(
            "Gaussian chunk", version, k_gaussian_chunk_version));
    if (count64 > 50'000'000ULL || sh_degree > 3)
        throw std::runtime_error("Gaussian chunk header is invalid");
    const std::size_t count = static_cast<std::size_t>(count64);
    const std::size_t bases =
        static_cast<std::size_t>(sh_degree + 1U) * (sh_degree + 1U);
    const bool has_filter = (flags & 1U) != 0;
    const bool has_normal_features = (flags & 2U) != 0;
    std::vector<float> means(count * 3U);
    std::vector<float> scales(count * 3U);
    std::vector<float> rotations(count * 4U);
    std::vector<float> opacities(count);
    std::vector<float> sh(count * bases * 3U);
    std::vector<float> normal_features(has_normal_features ? count * 4U : 0);
    std::vector<float> filter(has_filter ? count : 0);
    const auto take_floats = [&](std::vector<float>& values) {
        if (values.empty()) return;
        std::memcpy(
            values.data(), take(values.size() * sizeof(float)),
            values.size() * sizeof(float));
    };
    take_floats(means);
    take_floats(scales);
    take_floats(rotations);
    take_floats(opacities);
    take_floats(sh);
    take_floats(normal_features);
    take_floats(filter);

    GaussianModel model;
    if (count == 0) return model;
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
    if (has_normal_features)
        model.normal_features = tinytensor::Tensor::from_vector(
            normal_features, {count, 4U}, tinytensor::Device::CUDA);
    if (has_filter)
        model.filter_3d = tinytensor::Tensor::from_vector(
            filter, {count, 1U}, tinytensor::Device::CUDA);
    model.sh_degree = sh_degree;
    return model;
}

void run_orbit_preview(
    const GaussianModel& model,
    const std::filesystem::path& camera_file,
    DevicePreviewCallback device_preview,
    const float kernel_size,
    const std::filesystem::path& vis_file) {
    if (model.size() == 0)
        throw std::runtime_error("Orbit preview requires a trained Gaussian model");
    if (!device_preview)
        throw std::runtime_error("Orbit preview requires a live display callback");
    if (camera_file.empty())
        throw std::runtime_error("Orbit preview requires a camera sidecar");

    std::uint64_t last_revision = ~0ULL;
    std::uint64_t last_vis_revision = ~0ULL;
    while (true) {
        Camera next;
        std::uint64_t revision = 0;
        VisualizeOptions vis;
        vis.active_sh_degree = model.sh_degree;
        vis.kernel_size = kernel_size;
        std::uint64_t vis_revision = last_vis_revision;
        load_visualization_sidecar(vis_file, vis, vis_revision);
        if (load_preview_camera_file(camera_file, next, revision, &vis) &&
            (revision != last_revision || vis_revision != last_vis_revision)) {
            device_preview(0, 0, next, visualize(model, next, vis));
            last_revision = revision;
            last_vis_revision = vis_revision;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
}

}  // namespace aetherscan::splat
