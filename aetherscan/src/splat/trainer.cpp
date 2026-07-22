#include "splat/trainer.hpp"

#include "cuda_ops.hpp"
#include "io/image.hpp"

#include <cuda_runtime.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace aetherscan::splat {
namespace {

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

Camera make_camera_impl(const mvs::MvsView& view) {
    Camera camera;
    // Force evaluation: translation() returns a temporary, so retaining the
    // lazy Eigen cast expression with `auto` would leave a dangling operand.
    const Eigen::Vector3f translation =
        view.pose.translation().cast<float>();
    const Eigen::Matrix3f rotation = view.pose.R.cast<float>();
    // Transpose a conventional row-major W2C into the contiguous layout used
    // by the reference CUDA kernels (the same conversion as torch .t()).
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            float value = 0.F;
            if (row < 3 && column < 3) value = rotation(row, column);
            else if (row < 3 && column == 3) value = translation(row);
            else if (row == 3 && column == 3) value = 1.F;
            camera.world_to_camera[static_cast<std::size_t>(column) * 4 + row] = value;
        }
    }
    const Eigen::Vector3f position = view.pose.C.cast<float>();
    camera.position = {position.x(), position.y(), position.z()};
    camera.fx = view.fx;
    camera.fy = view.fy;
    camera.cx = view.cx;
    camera.cy = view.cy;
    camera.width = view.width;
    camera.height = view.height;
    return camera;
}

float sample_rgb(
    const io::RgbImage& image, const float x, const float y, const int channel) {
    if (x < 0.F || y < 0.F || x > static_cast<float>(image.width - 1) ||
        y > static_cast<float>(image.height - 1))
        return 0.F;
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, static_cast<int>(image.width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(image.height) - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const auto at = [&](const int px, const int py) {
        return static_cast<float>(image.pixels[
            (static_cast<std::size_t>(py) * image.width + px) * 3 + channel]);
    };
    return ((at(x0, y0) * (1.F - tx) + at(x1, y0) * tx) * (1.F - ty) +
            (at(x0, y1) * (1.F - tx) + at(x1, y1) * tx) * ty) /
           255.F;
}

std::pair<float, float> source_coordinate(
    const mvs::MvsView& view, const std::uint32_t x, const std::uint32_t y,
    const Camera& output_camera, const io::RgbImage& source) {
    const bool distorted = view.k1 != 0.F || view.k2 != 0.F ||
                           view.p1 != 0.F || view.p2 != 0.F;
    if (!distorted) {
        const float scale_x =
            static_cast<float>(output_camera.width) / source.width;
        const float scale_y =
            static_cast<float>(output_camera.height) / source.height;
        return {(static_cast<float>(x) + 0.5F) / scale_x - 0.5F,
                (static_cast<float>(y) + 0.5F) / scale_y - 0.5F};
    }
    const double xn =
        (static_cast<double>(x) - output_camera.cx) / output_camera.fx;
    const double yn =
        (static_cast<double>(y) - output_camera.cy) / output_camera.fy;
    const double radius2 = xn * xn + yn * yn;
    const double radial = 1.0 + view.k1 * radius2 +
                          view.k2 * radius2 * radius2;
    const double xd = xn * radial + 2.0 * view.p1 * xn * yn +
                      view.p2 * (radius2 + 2.0 * xn * xn);
    const double yd = yn * radial + view.p1 * (radius2 + 2.0 * yn * yn) +
                      2.0 * view.p2 * xn * yn;
    return {static_cast<float>(view.src_fx * xd + view.src_cx),
            static_cast<float>(view.src_fy * yd + view.src_cy)};
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

std::filesystem::path find_mask_path(
    const std::filesystem::path& directory,
    const std::filesystem::path& image_path) {
    if (directory.empty() || !std::filesystem::is_directory(directory))
        return {};
    const auto exact = directory / image_path.filename();
    if (std::filesystem::is_regular_file(exact)) return exact;
    static constexpr std::array<const char*, 6> extensions{
        ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"};
    for (const char* extension : extensions) {
        const auto candidate = directory / (image_path.stem().string() + extension);
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    return {};
}

std::filesystem::path resolve_mask_path(
    const mvs::MvsView& view, const TrainingOptions& options) {
    if (!options.mask_dir.empty())
        return find_mask_path(options.mask_dir, view.path);
    const auto nested = find_mask_path(view.path.parent_path() / "masks", view.path);
    if (!nested.empty()) return nested;
    return find_mask_path(
        view.path.parent_path().parent_path() / "masks", view.path);
}

float sample_mask_coverage(
    const io::GrayImage& mask, const io::RgbImage& source,
    const float source_x, const float source_y) {
    const float x = (source_x + 0.5F) * mask.width / source.width - 0.5F;
    const float y = (source_y + 0.5F) * mask.height / source.height - 0.5F;
    const int px = static_cast<int>(std::round(x));
    const int py = static_cast<int>(std::round(y));
    if (px < 0 || py < 0 || px >= static_cast<int>(mask.width) ||
        py >= static_cast<int>(mask.height))
        return 0.F;
    return mask.pixels[
               static_cast<std::size_t>(py) * mask.width +
               static_cast<std::size_t>(px)] > 127
        ? 1.F
        : 0.F;
}

struct RefinementCounts {
    std::size_t grown{};
    std::size_t pruned{};
};

struct StrategyPreset {
    unsigned start{};
    unsigned stop{};
    unsigned every{};
};

struct SceneGeometry {
    float scale{1.F};
    mvs::Vec3f center{mvs::Vec3f::Zero()};
};

SceneGeometry training_scene_geometry(
    const mvs::MvsScene& scene, const bool dense_input) {
    mvs::Vec3f minimum = scene.dense_cloud.points.front().position;
    mvs::Vec3f maximum = minimum;
    for (const auto& point : scene.dense_cloud.points) {
        minimum = minimum.cwiseMin(point.position);
        maximum = maximum.cwiseMax(point.position);
    }
    SceneGeometry geometry{
        std::max((maximum - minimum).norm(), 1e-6F),
        0.5F * (minimum + maximum)};
    if (dense_input || scene.views.empty()) return geometry;

    // Match pygsplat's COLMAP convention: optimization scale is measured from
    // camera centers, not the untrimmed sparse-point AABB. A handful of bad
    // triangulations can make that AABB tens of times too large, which in turn
    // inflates both the means LR and the minimum Gaussian scale.
    mvs::Vec3f camera_center = mvs::Vec3f::Zero();
    std::size_t finite_cameras = 0;
    for (const auto& view : scene.views) {
        const mvs::Vec3f position = view.pose.C.cast<float>();
        if (!position.allFinite()) continue;
        camera_center += position;
        ++finite_cameras;
    }
    if (finite_cameras == 0)
        throw std::invalid_argument(
            "Sparse GGGS training requires at least one finite camera center");
    camera_center /= static_cast<float>(finite_cameras);
    float camera_radius = 0.F;
    for (const auto& view : scene.views) {
        const mvs::Vec3f position = view.pose.C.cast<float>();
        if (position.allFinite())
            camera_radius = std::max(
                camera_radius, (position - camera_center).norm());
    }
    if (!(camera_radius > 1e-6F) || !std::isfinite(camera_radius))
        throw std::invalid_argument(
            "Sparse GGGS training requires non-degenerate camera coverage");
    return {1.1F * camera_radius, camera_center};
}

StrategyPreset strategy_preset(const TrainingOptions& options) {
    const bool dense_adaptive = options.densification_strategy ==
                                DensificationStrategy::dense_adaptive;
    const bool adc = options.densification_strategy ==
                         DensificationStrategy::adc_plus ||
                     options.densification_strategy ==
                         DensificationStrategy::adc_igs;
    const unsigned preset_stop = dense_adaptive
        ? 5'000U
        : options.densification_strategy == DensificationStrategy::adc_igs
            ? 25'000U
            : 15'000U;
    return {
        options.refine_start_iter != 0
            ? options.refine_start_iter
            : dense_adaptive ? 750U : adc ? 600U : 500U,
        std::min(
            options.refine_stop_iter != 0
                ? options.refine_stop_iter
                : preset_stop,
            options.iterations),
        options.refine_every != 0
            ? options.refine_every
            : dense_adaptive ? 500U : adc ? 200U : 100U};
}

tinytensor::Tensor index_tensor(const std::vector<int>& indices) {
    return tinytensor::Tensor::from_vector(
        indices, {indices.size()}, tinytensor::Device::CUDA);
}

GaussianModel select_model_rows(
    const GaussianModel& model, const tinytensor::Tensor& indices) {
    GaussianModel selected;
    selected.means = model.means.index_select(0, indices);
    selected.log_scales = model.log_scales.index_select(0, indices);
    selected.quaternions = model.quaternions.index_select(0, indices);
    selected.opacity_logits = model.opacity_logits.index_select(0, indices);
    selected.sh = model.sh.index_select(0, indices);
    selected.sh_degree = model.sh_degree;
    return selected;
}

void append_model(GaussianModel& model, const GaussianModel& added) {
    if (added.size() == 0) return;
    model.means = tinytensor::Tensor::cat({model.means, added.means}, 0);
    model.log_scales = tinytensor::Tensor::cat(
        {model.log_scales, added.log_scales}, 0);
    model.quaternions = tinytensor::Tensor::cat(
        {model.quaternions, added.quaternions}, 0);
    model.opacity_logits = tinytensor::Tensor::cat(
        {model.opacity_logits, added.opacity_logits}, 0);
    model.sh = tinytensor::Tensor::cat({model.sh, added.sh}, 0);
}

void select_adam_rows(
    detail::AdamState& state, const tinytensor::Tensor& indices) {
    state.first = state.first.index_select(0, indices);
    state.second = state.second.index_select(0, indices);
}

void append_zero_adam(detail::AdamState& state, const std::size_t count) {
    if (count == 0) return;
    std::vector<std::size_t> dimensions = state.first.shape().dims();
    dimensions[0] = count;
    const auto shape = tinytensor::TensorShape(dimensions);
    state.first = tinytensor::Tensor::cat(
        {state.first, tinytensor::Tensor::zeros(
                          shape, tinytensor::Device::CUDA)}, 0);
    state.second = tinytensor::Tensor::cat(
        {state.second, tinytensor::Tensor::zeros(
                           shape, tinytensor::Device::CUDA)}, 0);
}

using AdamStates = std::array<detail::AdamState*, 5>;

void zero_adam_rows(
    const std::vector<int>& rows, const AdamStates& states) {
    if (rows.empty()) return;
    const auto indices = index_tensor(rows);
    for (detail::AdamState* state : states) {
        std::vector<std::size_t> dimensions = state->first.shape().dims();
        dimensions[0] = rows.size();
        const auto zeros = tinytensor::Tensor::zeros(
            tinytensor::TensorShape(dimensions), tinytensor::Device::CUDA);
        state->first.index_copy_(0, indices, zeros);
        state->second.index_copy_(0, indices, zeros);
    }
}

void select_training_rows(
    GaussianModel& model, const std::vector<int>& keep,
    const AdamStates& states) {
    const auto indices = index_tensor(keep);
    model = select_model_rows(model, indices);
    for (detail::AdamState* state : states) select_adam_rows(*state, indices);
}

void grow_training_model(
    GaussianModel& model, const std::vector<int>& parents,
    const int split_mode, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states) {
    if (parents.empty()) return;
    const auto indices = index_tensor(parents);
    GaussianModel children = select_model_rows(model, indices);
    if (split_mode != 0) {
        std::normal_distribution<float> normal(0.F, 1.F);
        std::vector<float> samples(3 * parents.size());
        for (float& value : samples) value = normal(random);
        auto random_tensor = tinytensor::Tensor::from_vector(
            samples, {parents.size(), 3}, tinytensor::Device::CUDA);
        detail::split_gaussians(
            model, children, indices, random_tensor, split_mode,
            options.prune_opacity);
        // Splitting mutates the retained parent as well as creating a child.
        // Both are new primitives and must start with clean optimizer moments,
        // matching pygsplat's replacement-based split.
        zero_adam_rows(parents, states);
    }
    append_model(model, children);
    for (detail::AdamState* state : states)
        append_zero_adam(*state, parents.size());
}

std::vector<std::size_t> weighted_unique_sample(
    const std::vector<std::pair<std::size_t, float>>& candidates,
    const std::size_t requested, const bool gumbel,
    std::mt19937& random) {
    if (requested == 0 || candidates.empty()) return {};
    const std::size_t count = std::min(requested, candidates.size());
    if (gumbel) {
        std::uniform_real_distribution<float> uniform(1e-7F, 1.F - 1e-7F);
        std::vector<std::pair<float, std::size_t>> scores;
        scores.reserve(candidates.size());
        for (const auto& [index, weight] : candidates) {
            if (weight <= 0.F) continue;
            const float u = uniform(random);
            scores.emplace_back(
                std::log(weight) - std::log(-std::log(u)), index);
        }
        const std::size_t selected = std::min(count, scores.size());
        std::partial_sort(
            scores.begin(), scores.begin() + selected, scores.end(),
            std::greater<>());
        std::vector<std::size_t> result;
        result.reserve(selected);
        for (std::size_t i = 0; i < selected; ++i)
            result.push_back(scores[i].second);
        return result;
    }
    std::vector<double> weights;
    weights.reserve(candidates.size());
    for (const auto& candidate : candidates)
        weights.push_back(std::max(candidate.second, 0.F));
    std::discrete_distribution<std::size_t> distribution(
        weights.begin(), weights.end());
    std::unordered_set<std::size_t> selected;
    const std::size_t attempts_limit = candidates.size() * 8 + requested * 4;
    for (std::size_t attempt = 0;
         attempt < attempts_limit && selected.size() < count; ++attempt)
        selected.insert(candidates[distribution(random)].first);
    if (selected.size() < count) {
        std::vector<std::pair<float, std::size_t>> sorted;
        for (const auto& [index, weight] : candidates)
            if (!selected.contains(index)) sorted.emplace_back(weight, index);
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        for (const auto& entry : sorted) {
            if (selected.size() >= count) break;
            selected.insert(entry.second);
        }
    }
    return {selected.begin(), selected.end()};
}

RefinementCounts refine_gaussians(
    GaussianModel& model, detail::DensificationStats& stats,
    const unsigned iteration, const float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states) {
    const StrategyPreset preset = strategy_preset(options);
    const bool dense_adaptive = options.densification_strategy ==
                                DensificationStrategy::dense_adaptive;
    const bool adc = options.densification_strategy ==
                         DensificationStrategy::adc_plus ||
                     options.densification_strategy ==
                         DensificationStrategy::adc_igs;
    const bool managed = adc || dense_adaptive;
    if (iteration <= preset.start || iteration >= preset.stop ||
        preset.every == 0 || iteration % preset.every != 0 ||
        (managed && static_cast<float>(iteration) /
                    std::max(1.F, static_cast<float>(options.iterations)) >
                0.95F) ||
        model.size() == 0)
        return {};

    const std::size_t old_count = model.size();
    const auto gradients = download<float>(stats.gradient);
    const auto counts = download<float>(stats.count);
    const auto screen = download<float>(stats.max_screen_radius);
    const auto priorities = download<float>(stats.priority);
    const auto opacities = download<float>(model.opacity_logits);
    const auto log_scales = download<float>(model.log_scales);
    const auto means = download<float>(model.means);
    std::vector<bool> prune(old_count, false);
    std::vector<bool> hard_prune(old_count, false);
    std::vector<float> opacity_values(old_count);
    std::size_t best = 0;
    float best_opacity = -1.F;
    for (std::size_t index = 0; index < old_count; ++index) {
        const float opacity = 1.F / (1.F + std::exp(-opacities[index]));
        opacity_values[index] = opacity;
        if (opacity > best_opacity) {
            best_opacity = opacity;
            best = index;
        }
        float min_scale = std::numeric_limits<float>::infinity();
        float max_scale = 0.F;
        for (int axis = 0; axis < 3; ++axis) {
            const float scale = std::exp(log_scales[3 * index + axis]);
            min_scale = std::min(min_scale, scale);
            max_scale = std::max(max_scale, scale);
        }
        bool remove = opacity < options.prune_opacity;
        if (managed) {
            const bool invalid = min_scale < 1e-10F ||
                                 max_scale > 100.F * scene_extent;
            const mvs::Vec3f position(
                means[3 * index], means[3 * index + 1], means[3 * index + 2]);
            hard_prune[index] = invalid ||
                (position - scene_center).cwiseAbs().maxCoeff() >
                    100.F * scene_extent;
            remove = remove || hard_prune[index];
        } else if (iteration > options.opacity_reset_every) {
            remove = remove || max_scale > 0.1F * scene_extent;
        }
        prune[index] = remove;
    }
    if (dense_adaptive) {
        std::vector<std::pair<float, std::size_t>> low_opacity;
        std::size_t hard_count = 0;
        for (std::size_t index = 0; index < old_count; ++index) {
            if (hard_prune[index]) {
                ++hard_count;
            } else if (prune[index]) {
                low_opacity.emplace_back(opacity_values[index], index);
                prune[index] = false;
            }
        }
        std::sort(low_opacity.begin(), low_opacity.end());
        const std::size_t recycle_limit = static_cast<std::size_t>(std::ceil(
            old_count * std::clamp(
                options.dense_recycle_fraction, 0.F, 1.F)));
        const std::size_t opacity_budget = recycle_limit > hard_count
            ? recycle_limit - hard_count
            : 0;
        for (std::size_t index = 0;
             index < std::min(opacity_budget, low_opacity.size()); ++index)
            prune[low_opacity[index].second] = true;
    }
    prune[best] = false;
    std::size_t retained = static_cast<std::size_t>(
        std::count(prune.begin(), prune.end(), false));
    if (retained > options.densification_cap) {
        std::vector<std::pair<float, std::size_t>> by_opacity;
        by_opacity.reserve(retained);
        for (std::size_t index = 0; index < old_count; ++index)
            if (!prune[index] && index != best)
                by_opacity.emplace_back(opacity_values[index], index);
        std::sort(by_opacity.begin(), by_opacity.end());
        const std::size_t remove_count = retained - options.densification_cap;
        for (std::size_t index = 0; index < remove_count; ++index)
            prune[by_opacity[index].second] = true;
    }

    std::vector<int> keep;
    keep.reserve(old_count);
    std::vector<int> remap(old_count, -1);
    for (std::size_t index = 0; index < old_count; ++index) {
        if (!prune[index]) {
            remap[index] = static_cast<int>(keep.size());
            keep.push_back(static_cast<int>(index));
        }
    }
    const std::size_t pruned = old_count - keep.size();

    struct Candidate {
        std::size_t old_index{};
        std::size_t new_index{};
        float score{};
        float max_scale{};
        bool oversized{};
    };
    std::vector<Candidate> candidates;
    std::vector<float> positive_priorities;
    for (std::size_t index = 0; index < old_count; ++index) {
        if (remap[index] < 0 || counts[index] <= 0.F) continue;
        const float score = managed
            ? gradients[index]
            : gradients[index] / std::max(counts[index], 1.F);
        float max_scale = 0.F;
        for (int axis = 0; axis < 3; ++axis)
            max_scale = std::max(
                max_scale, std::exp(log_scales[3 * index + axis]));
        candidates.push_back({
            index, static_cast<std::size_t>(remap[index]), score, max_scale,
            screen[index] > options.densify_screen_threshold});
        const float priority = priorities[index] /
                               std::max(counts[index], 1.F);
        if (priority > 0.F) positive_priorities.push_back(priority);
    }
    select_training_rows(model, keep, states);

    const std::size_t capacity = options.densification_cap > model.size()
        ? options.densification_cap - model.size()
        : 0;
    if (capacity == 0) {
        stats = detail::make_densification_stats(model.size());
        return {0, pruned};
    }

    std::vector<int> duplicate_parents;
    std::vector<int> split_parents;
    if (!managed) {
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.score > b.score;
                  });
        for (const Candidate& candidate : candidates) {
            if (duplicate_parents.size() + split_parents.size() >= capacity) break;
            if (candidate.score <= options.densify_gradient_threshold) break;
            if (candidate.max_scale <=
                options.densify_scale_threshold * scene_extent)
                duplicate_parents.push_back(
                    static_cast<int>(candidate.new_index));
            else
                split_parents.push_back(
                    static_cast<int>(candidate.new_index));
        }
    } else {
        float priority_median = 1.F;
        if (!positive_priorities.empty()) {
            const auto middle = positive_priorities.begin() +
                                positive_priorities.size() / 2;
            std::nth_element(
                positive_priorities.begin(), middle,
                positive_priorities.end());
            priority_median = std::max(*middle, 1e-9F);
        }
        std::vector<std::pair<std::size_t, float>> replacement_weights;
        std::vector<std::pair<std::size_t, float>> growth_weights;
        std::unordered_set<std::size_t> forced;
        const bool allow_growth = dense_adaptive ||
            options.densification_strategy != DensificationStrategy::adc_igs ||
            iteration < options.grow_stop_iter;
        for (const Candidate& candidate : candidates) {
            const float opacity = 1.F /
                (1.F + std::exp(-opacities[candidate.old_index]));
            float edge_factor = 1.F;
            if (options.densification_strategy ==
                DensificationStrategy::adc_igs) {
                const float priority = priorities[candidate.old_index] /
                    std::max(counts[candidate.old_index], 1.F);
                if (priority > 0.F)
                    edge_factor += 0.25F * std::min(
                        priority / priority_median, 10.F);
            }
            const float screen_factor = candidate.oversized ? 2.F : 1.F;
            replacement_weights.emplace_back(
                candidate.new_index,
                dense_adaptive
                    ? std::max(candidate.score, 1e-12F) * screen_factor
                    : opacity * edge_factor);
            if (allow_growth &&
                candidate.score > options.densify_gradient_threshold)
                growth_weights.emplace_back(
                    candidate.new_index,
                    candidate.score * edge_factor * screen_factor);
            if (!dense_adaptive && allow_growth && candidate.oversized)
                forced.insert(candidate.new_index);
        }
        const bool gumbel = options.densification_strategy ==
                            DensificationStrategy::adc_igs;
        auto selected = weighted_unique_sample(
            replacement_weights, std::min(pruned, capacity), gumbel, random);
        forced.insert(selected.begin(), selected.end());
        std::size_t desired_growth = static_cast<std::size_t>(std::llround(
            growth_weights.size() * options.densify_select_fraction));
        if (dense_adaptive) {
            desired_growth = std::min(
                desired_growth,
                static_cast<std::size_t>(std::ceil(
                    model.size() * std::clamp(
                        options.dense_growth_fraction, 0.F, 1.F))));
            growth_weights.erase(
                std::remove_if(
                    growth_weights.begin(), growth_weights.end(),
                    [&](const auto& candidate) {
                        return forced.contains(candidate.first);
                    }),
                growth_weights.end());
        }
        const std::size_t remaining = capacity > forced.size()
            ? capacity - forced.size()
            : 0;
        selected = weighted_unique_sample(
            growth_weights,
            std::min(remaining, desired_growth), gumbel, random);
        forced.insert(selected.begin(), selected.end());
        split_parents.reserve(std::min(capacity, forced.size()));
        for (const std::size_t parent : forced) {
            if (split_parents.size() >= capacity) break;
            split_parents.push_back(static_cast<int>(parent));
        }
    }

    grow_training_model(
        model, duplicate_parents, 0, options, random, states);
    // Parent row indices still refer to the original retained prefix after
    // duplicates are appended, so they remain valid here.
    const int split_mode = options.densification_strategy ==
            DensificationStrategy::adc_igs
        ? 3
        : dense_adaptive
            ? 4
            : options.densification_strategy == DensificationStrategy::adc_plus
            ? 2
            : 1;
    grow_training_model(
        model, split_parents, split_mode, options, random, states);
    if (adc) {
        const float remaining_progress = 1.F -
            static_cast<float>(iteration) /
                std::max(1.F, static_cast<float>(options.iterations));
        detail::apply_adc_decay(
            model,
            options.opacity_decay * std::max(remaining_progress, 0.F),
            options.scale_decay * std::max(remaining_progress, 0.F));
    }
    stats = detail::make_densification_stats(model.size());
    return {duplicate_parents.size() + split_parents.size(), pruned};
}

}  // namespace

Camera camera_from_mvs_view(const mvs::MvsView& view) {
    return make_camera_impl(view);
}

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
            training_scene_geometry(scene, false).scale;
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
    const float opacity = std::clamp(options.initial_opacity, 1e-6F, 1.F - 1e-6F);
    const float opacity_logit = std::log(opacity / (1.F - opacity));
    std::vector<float> knn_scales(count, 0.F);
    if (options.initialize_scale_from_knn && count >= 4) {
        const KdTree tree(selected_positions);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t index = 0;
             index < static_cast<std::int64_t>(count); ++index)
            knn_scales[static_cast<std::size_t>(index)] =
                tree.three_neighbor_rms(static_cast<std::size_t>(index));
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

        if (!options.input_is_dense) {
            // Exact pygsplat sparse-SfM initialization: raw U[0,1) quaternion
            // parameters. The raster path normalizes them before use.
            for (int component = 0; component < 4; ++component)
                quaternions[4 * index + component] = quaternion_uniform(
                    quaternion_random);
        } else {
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

TrainingView make_training_view(
    const mvs::MvsView& view, const TrainingOptions& options) {
    if (view.width == 0 || view.height == 0)
        throw std::invalid_argument("Cannot build a GGGS training view with empty dimensions");
    const io::RgbImage source = io::load_rgb(view.path);
    Camera camera = camera_from_mvs_view(view);
    if (options.use_source_resolution && view.src_width != 0 &&
        view.src_height != 0) {
        camera.fx = view.src_fx;
        camera.fy = view.src_fy;
        camera.cx = view.src_cx;
        camera.cy = view.src_cy;
        camera.width = view.src_width;
        camera.height = view.src_height;
    }
    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    io::GrayImage source_mask;
    bool has_source_mask = false;
    if (options.use_mask) {
        const auto mask_path = resolve_mask_path(view, options);
        if (!mask_path.empty()) {
            source_mask = io::load_gray(mask_path);
            has_source_mask = true;
        } else {
            source_mask = io::load_alpha(view.path);
            has_source_mask = !source_mask.pixels.empty();
        }
    }
    std::vector<float> rgb(3 * pixels);
    std::vector<float> mask(pixels, 1.F);
    for (std::uint32_t y = 0; y < camera.height; ++y) {
        for (std::uint32_t x = 0; x < camera.width; ++x) {
            const auto [sx, sy] =
                source_coordinate(view, x, y, camera, source);
            const std::size_t pixel =
                static_cast<std::size_t>(y) * camera.width + x;
            for (int channel = 0; channel < 3; ++channel)
                rgb[static_cast<std::size_t>(channel) * pixels + pixel] =
                    sample_rgb(source, sx, sy, channel);
            if (has_source_mask)
                mask[pixel] = sample_mask_coverage(source_mask, source, sx, sy);
        }
    }

    std::vector<float> depth(pixels, 0.F);
    std::vector<float> normals(3 * pixels, 0.F);
    if (camera.width == view.width && camera.height == view.height &&
        view.depth_map.depth.size() == pixels)
        depth = view.depth_map.depth;
    if (camera.width == view.width && camera.height == view.height &&
        view.depth_map.normal.size() == pixels) {
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            for (int axis = 0; axis < 3; ++axis)
                normals[static_cast<std::size_t>(axis) * pixels + pixel] =
                    view.depth_map.normal[pixel](axis);
        }
    }
    bool has_mask = has_source_mask;
    if (options.use_mask && !has_source_mask &&
        camera.width == view.width && camera.height == view.height &&
        view.foreground_mask.size() == pixels) {
        has_mask = true;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel)
            mask[pixel] = view.foreground_mask[pixel] != 0 ? 1.F : 0.F;
    }

    TrainingView result;
    result.camera = camera;
    result.rgb = tinytensor::Tensor::from_vector(
        rgb, {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    result.depth = tinytensor::Tensor::from_vector(
        depth, {camera.height, camera.width}, tinytensor::Device::CUDA);
    result.normal = tinytensor::Tensor::from_vector(
        normals, {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    result.mask = tinytensor::Tensor::from_vector(
        mask, {camera.height, camera.width}, tinytensor::Device::CUDA);
    result.has_mask = has_mask;
    return result;
}

Trainer::Trainer(TrainingOptions options) : options_(std::move(options)) {}

GaussianModel Trainer::train(
    const mvs::MvsScene& scene, ProgressCallback progress,
    EvaluationCallback evaluate) const {
    if (scene.views.empty())
        throw std::invalid_argument("GGGS training requires at least one MVS view");
    GaussianModel model = initialize_from_dense_cloud(scene, options_);
    std::vector<TrainingView> views;
    std::vector<std::size_t> view_indices(scene.views.size());
    std::iota(view_indices.begin(), view_indices.end(), std::size_t{0});
    views.reserve(view_indices.size());
    for (const std::size_t index : view_indices)
        views.push_back(make_training_view(scene.views[index], options_));
    if (options_.use_mask &&
        std::any_of(views.begin(), views.end(),
                    [](const TrainingView& view) { return !view.has_mask; }))
        throw std::invalid_argument(
            "GGGS subject-only training requires a matching mask file or "
            "source alpha channel for every selected view");

    detail::AdamState means_state = detail::make_adam_state(model.means);
    detail::AdamState scales_state = detail::make_adam_state(model.log_scales);
    detail::AdamState rotations_state = detail::make_adam_state(model.quaternions);
    detail::AdamState opacity_state = detail::make_adam_state(model.opacity_logits);
    detail::AdamState sh_state = detail::make_adam_state(model.sh);
    const AdamStates adam_states{
        &means_state, &scales_state, &rotations_state, &opacity_state,
        &sh_state};
    const bool dense_adaptive = options_.densification_strategy ==
                                DensificationStrategy::dense_adaptive;
    const bool densification_enabled = options_.enable_densification &&
        ((!options_.input_is_dense && !dense_adaptive) ||
         (options_.input_is_dense && dense_adaptive)) &&
        strategy_preset(options_).stop > 0;
    detail::DensificationStats densification_stats =
        detail::make_densification_stats(model.size());
    RefinementCounts latest_refinement;
    Rasterizer rasterizer;
    std::mt19937 random(options_.seed);
    std::vector<std::size_t> shuffled_views(views.size());
    std::iota(
        shuffled_views.begin(), shuffled_views.end(), std::size_t{0});
    std::shuffle(shuffled_views.begin(), shuffled_views.end(), random);
    std::size_t shuffled_view_cursor = 0;
    const SceneGeometry scene_geometry =
        training_scene_geometry(scene, options_.input_is_dense);
    const float scene_extent = scene_geometry.scale;
    const mvs::Vec3f scene_center = scene_geometry.center;
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
        const bool report_progress = progress &&
            (iteration == 1 || iteration == options_.iterations ||
             (options_.log_interval != 0 &&
              iteration % options_.log_interval == 0));
        if (shuffled_view_cursor == shuffled_views.size()) {
            std::shuffle(shuffled_views.begin(), shuffled_views.end(), random);
            shuffled_view_cursor = 0;
        }
        const std::size_t view_index =
            shuffled_views[shuffled_view_cursor++];
        const auto& target = views[view_index];
        RasterizeOptions raster_options;
        raster_options.active_sh_degree = std::min(
            options_.sh_degree,
            options_.sh_degree_interval == 0
                ? options_.sh_degree
                : (iteration - 1) / options_.sh_degree_interval);
        raster_options.kernel_size = options_.kernel_size;
        raster_options.scale_modifier = options_.scale_modifier;
        const bool depth_normal_active = options_.use_depth_normal_loss &&
            options_.depth_normal_weight > 0.F &&
            iteration >= options_.depth_normal_from_iter;
        raster_options.require_depth = options_.use_mvs_depth ||
                                       options_.use_mvs_normals ||
                                       depth_normal_active;
        RenderResult rendered = rasterizer.forward(model, target.camera, raster_options);
        detail::LossGradients loss = detail::compute_training_loss(
            rendered, target, options_, report_progress,
            depth_normal_active);
        ModelGradients gradients = rasterizer.backward(
            model, rendered, loss.color, loss.alpha, loss.depth, loss.normal);
        if (densification_enabled)
            detail::accumulate_densification_stats(
                gradients.refine_weight, rendered.radii,
                densification_stats, target.camera.width,
                target.camera.height,
                options_.densification_strategy !=
                    DensificationStrategy::default_strategy);

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
        const float means_lr = options_.means_lr * scene_extent *
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
        detail::adam_step(
            model.sh, gradients.sh, sh_state, options_.sh0_lr, iteration,
            options_, model.sh.shape()[1] * 3, options_.sh_rest_lr);

        if (densification_enabled &&
            (options_.densification_strategy ==
                 DensificationStrategy::adc_plus ||
             options_.densification_strategy ==
                 DensificationStrategy::adc_igs)) {
            const unsigned noise_stop = options_.densification_strategy ==
                    DensificationStrategy::adc_igs
                ? options_.grow_stop_iter
                : strategy_preset(options_).stop;
            if (iteration < noise_stop)
                detail::inject_adc_noise(
                    model, rendered.radii,
                    means_lr * options_.mean_noise_weight,
                    scene_extent, options_.seed + iteration);
        }

        latest_refinement = {};
        if (densification_enabled) {
            latest_refinement = refine_gaussians(
                model, densification_stats, iteration, scene_extent,
                scene_center, options_, random, adam_states);
            if (options_.densification_strategy ==
                    DensificationStrategy::default_strategy &&
                iteration < strategy_preset(options_).stop &&
                options_.opacity_reset_every != 0 && iteration > 0 &&
                iteration % options_.opacity_reset_every == 0) {
                detail::reset_opacity(
                    model, options_.prune_opacity * 2.F);
                opacity_state = detail::make_adam_state(
                    model.opacity_logits);
            }
            detail::constrain_scale_ratio(
                model.log_scales, options_.max_scale_ratio);
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
                view_indices[view_index], latest_refinement.grown,
                latest_refinement.pruned, loss.total, loss.rgb,
                loss.alpha_value, loss.depth_value, loss.normal_value,
                static_cast<float>(opacity_gradient_sum * inverse_gaussians),
                static_cast<float>(positive_opacity_gradients *
                                   inverse_gaussians),
                static_cast<float>(opacity_sum * inverse_gaussians),
                milliseconds});
        }
        if (!continue_training) break;
        if (evaluate &&
            std::find(
                options_.evaluation_iterations.begin(),
                options_.evaluation_iterations.end(),
                iteration) != options_.evaluation_iterations.end())
            evaluate(iteration, model);
    }
    const cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string("GGGS training synchronization failed: ") +
            cudaGetErrorString(error));
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
    const std::vector<float> mask = download<float>(target.mask);
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
        const double target_alpha = mask[pixel] > 0.F ? 1.0 : 0.0;
        alpha_bce -= target_alpha * std::log(predicted_alpha) +
            (1.0 - target_alpha) * std::log(1.0 - predicted_alpha);
        for (int channel = 0; channel < 3; ++channel) {
            const std::size_t planar =
                static_cast<std::size_t>(channel) * pixels + pixel;
            const float prediction = std::clamp(color[planar], 0.F, 1.F);
            image.pixels[3 * pixel + channel] = static_cast<std::uint8_t>(
                std::lround(prediction * 255.F));
            if (mask[pixel] > 0.F) {
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
    }
    if (!output) throw std::runtime_error("Failed while writing Gaussian PLY: " + path.string());
}

}  // namespace aetherscan::splat
