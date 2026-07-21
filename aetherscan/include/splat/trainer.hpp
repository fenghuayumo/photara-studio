#pragma once

#include "mvs/types.hpp"
#include "splat/options.hpp"
#include "splat/rasterizer.hpp"

#include <filesystem>
#include <functional>
#include <vector>

namespace aetherscan::splat {

struct TrainingProgress {
    unsigned iteration{};
    unsigned total_iterations{};
    std::size_t gaussian_count{};
    std::size_t view_index{};
    std::size_t grown_count{};
    std::size_t pruned_count{};
    float loss{};
    float rgb_loss{};
    float alpha_loss{};
    float depth_loss{};
    float normal_loss{};
    double milliseconds{};
};

struct RenderMetrics {
    float mae{};
    float psnr{};
    float alpha_coverage{};
};

using ProgressCallback = std::function<bool(const TrainingProgress&)>;

Camera camera_from_mvs_view(const mvs::MvsView& view);

GaussianModel initialize_from_dense_cloud(
    const mvs::MvsScene& scene, const TrainingOptions& options = {});

TrainingView make_training_view(
    const mvs::MvsView& view, const TrainingOptions& options = {});

class Trainer {
public:
    explicit Trainer(TrainingOptions options = {});

    GaussianModel train(
        const mvs::MvsScene& scene, ProgressCallback progress = {}) const;

private:
    TrainingOptions options_;
};

void save_gaussians_ply(
    const GaussianModel& model, const std::filesystem::path& path);

RenderMetrics render_evaluation_png(
    const GaussianModel& model, const mvs::MvsView& view,
    const std::filesystem::path& path,
    const TrainingOptions& options = {});

}  // namespace aetherscan::splat
