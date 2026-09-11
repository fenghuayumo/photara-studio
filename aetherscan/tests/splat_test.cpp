#include "core/camera_projection.hpp"
#include "splat/trainer.hpp"
#include "splat/colmap.hpp"
#include "splat/dataset.hpp"
#include "splat/formats.hpp"
#include "../src/splat/cuda_ops.hpp"
#include "../src/splat/densification.hpp"
#include "../src/splat/multi_view_scheduler.hpp"
#include "../src/splat/training_data_loader.hpp"
#include "io/image.hpp"
#include "sfm/export_mvs.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void write_binary(std::ostream& stream, const T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void require_finite(
    const tinytensor::Tensor& tensor, const char* message) {
    const std::vector<float> values = tensor.to_vector();
    require(!values.empty(), message);
    require(
        std::all_of(values.begin(), values.end(), [](const float value) {
            return std::isfinite(value);
        }),
        message);
}

void test_mvs_camera_conversion() {
    aetherscan::mvs::MvsView view;
    view.pose.R = Eigen::AngleAxisd(
        0.73, Eigen::Vector3d(0.2, 1.0, -0.1).normalized()).toRotationMatrix();
    view.pose.C = Eigen::Vector3d(-0.3, 0.25, 1.2);
    const aetherscan::splat::Camera camera =
        aetherscan::splat::camera_from_mvs_view(view);
    const Eigen::Vector3d world(0.4, -0.2, 2.1);
    const Eigen::Vector3d expected =
        view.pose.transform_world_to_camera(world);
    const auto& matrix = camera.world_to_camera;
    const Eigen::Vector3d actual(
        matrix[0] * world.x() + matrix[4] * world.y() +
            matrix[8] * world.z() + matrix[12],
        matrix[1] * world.x() + matrix[5] * world.y() +
            matrix[9] * world.z() + matrix[13],
        matrix[2] * world.x() + matrix[6] * world.y() +
            matrix[10] * world.z() + matrix[14]);
    require(
        (actual - expected).norm() < 1e-5,
        "MVS to GGGS camera conversion changed projection");
}

void test_gggs_3d_filter() {
    using namespace aetherscan::splat;
    Camera camera;
    camera.world_to_camera[0] = 1.F;
    camera.world_to_camera[5] = 1.F;
    camera.world_to_camera[10] = 1.F;
    camera.world_to_camera[15] = 1.F;
    camera.fx = camera.fy = 50.F;
    camera.width = camera.height = 100;
    const auto means = tinytensor::Tensor::from_vector(
        std::vector<float>{0.F, 0.F, 2.F, 0.F, 0.F, 4.F,
                           100.F, 0.F, 1.F},
        {3, 3}, tinytensor::Device::CUDA);
    const auto filter = detail::compute_3d_filter(means, {camera}).to_vector();
    const float unit = std::sqrt(0.2F) / 50.F;
    require(
        filter.size() == 3 && std::abs(filter[0] - 2.F * unit) < 1e-6F &&
            std::abs(filter[1] - 4.F * unit) < 1e-6F &&
            std::abs(filter[2] - 4.F * unit) < 1e-6F,
        "GGGS 3D filter differs from pygsplat visibility/focal formula");
    const auto invisible = tinytensor::Tensor::from_vector(
        std::vector<float>{0.F, 0.F, -2.F}, {1, 3}, tinytensor::Device::CUDA);
    require(std::abs(detail::compute_3d_filter(invisible, {camera}).to_vector()[0] -
                     unit) < 1e-6F,
            "GPU filter maximum lost the all-invisible fallback");

    GaussianModel model;
    model.means = means;
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(0.1F), std::log(0.2F), std::log(0.3F),
                           std::log(0.1F), std::log(0.2F), std::log(0.3F),
                           std::log(0.1F), std::log(0.2F), std::log(0.3F)},
        {3, 3}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F,
                           1.F, 0.F, 0.F, 0.F},
        {3, 4}, tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::zeros(
        {3, 1}, tinytensor::Device::CUDA);
    model.filter_3d = filter.size() == 3
        ? tinytensor::Tensor::from_vector(filter, {3, 1}, tinytensor::Device::CUDA)
        : tinytensor::Tensor{};
    const auto activated = detail::activate_parameters(model);
    const auto scales = activated.scales.to_vector();
    const auto opacities = activated.opacities.to_vector();
    for (std::size_t row = 0; row < 3; ++row) {
        float determinant_ratio = 1.F;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const float raw = 0.1F * static_cast<float>(axis + 1);
            const float expected = std::sqrt(raw * raw + filter[row] * filter[row]);
            require(std::abs(scales[3 * row + axis] - expected) < 1e-6F,
                    "GGGS filtered scale formula differs from pygsplat");
            determinant_ratio *= raw / expected;
        }
        require(std::abs(opacities[row] - 0.5F * determinant_ratio) < 1e-6F,
                "GGGS filtered opacity compensation differs from pygsplat");
    }
    ModelGradients chained;
    detail::chain_parameter_gradients(
        model, activated,
        tinytensor::Tensor::from_vector(
            std::vector<float>(9, 1.F), {3, 3}, tinytensor::Device::CUDA),
        tinytensor::Tensor::zeros({3, 4}, tinytensor::Device::CUDA),
        tinytensor::Tensor::from_vector(
            std::vector<float>(3, 1.F), {3, 1}, tinytensor::Device::CUDA),
        chained);
    const auto scale_gradients = chained.log_scales.to_vector();
    const auto opacity_gradients = chained.opacity_logits.to_vector();
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const float raw = 0.1F * static_cast<float>(axis + 1);
            const float filtered = scales[3 * row + axis];
            const float expected = raw * raw / filtered +
                opacities[row] * filter[row] * filter[row] /
                    (filtered * filtered);
            require(std::abs(scale_gradients[3 * row + axis] - expected) < 1e-6F,
                    "GGGS 3D-filter scale/opacity chain gradient is incorrect");
        }
        require(std::abs(opacity_gradients[row] - 0.5F * opacities[row]) < 1e-6F,
                "GGGS filtered-opacity logit gradient is incorrect");
    }

    detail::bake_3d_filter(model);
    require(
        !model.filter_3d.is_valid(),
        "GGGS 3D-filter bake did not clear the separate floor");
    const auto baked = detail::activate_parameters(model);
    const auto baked_scales = baked.scales.to_vector();
    const auto baked_opacities = baked.opacities.to_vector();
    for (std::size_t index = 0; index < scales.size(); ++index)
        require(
            std::abs(baked_scales[index] - scales[index]) < 1e-6F,
            "GGGS 3D-filter bake changed the rendered scale");
    for (std::size_t index = 0; index < opacities.size(); ++index)
        require(
            std::abs(baked_opacities[index] - opacities[index]) < 1e-6F,
            "GGGS 3D-filter bake changed the rendered opacity");
}

void test_gggs_multi_view_geometry_and_ncc() {
    using namespace aetherscan::splat;
    constexpr std::uint32_t width = 32;
    constexpr std::uint32_t height = 32;
    constexpr std::size_t pixels = width * height;
    auto make_camera = [](const float center_x) {
        Camera camera;
        camera.world_to_camera[0] = 1.F;
        camera.world_to_camera[5] = 1.F;
        camera.world_to_camera[10] = 1.F;
        camera.world_to_camera[15] = 1.F;
        camera.world_to_camera[12] = -center_x;
        camera.position[0] = center_x;
        camera.fx = camera.fy = 40.F;
        camera.cx = camera.cy = 15.5F;
        camera.width = width;
        camera.height = height;
        return camera;
    };
    TrainingView reference;
    TrainingView neighbour;
    reference.camera = make_camera(0.F);
    neighbour.camera = make_camera(0.1F);
    std::vector<float> reference_rgb(3 * pixels);
    std::vector<float> neighbour_rgb(3 * pixels);
    std::vector<float> reference_gray(pixels, 0.F);
    std::vector<float> neighbour_gray(pixels, 0.F);
    constexpr std::array<float, 3> gray_weights{
        0.299F, 0.587F, 0.114F};
    constexpr std::array<float, 3> channel_scales{
        0.8F, 1.F, 0.6F};
    constexpr std::array<float, 3> channel_offsets{
        0.05F, 0.F, 0.15F};
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            const float ref_value = 0.5F + 0.22F * std::sin(0.37F * x) +
                                    0.18F * std::cos(0.29F * y);
            // A fronto-parallel plane at z=2 moves left by fx*b/z=2 pixels
            // in the translated camera.
            const float world_x_pixel = static_cast<float>(x) + 2.F;
            const float neighbour_value =
                0.5F + 0.22F * std::sin(0.37F * world_x_pixel) +
                0.18F * std::cos(0.29F * y);
            for (int channel = 0; channel < 3; ++channel) {
                const float reference_channel =
                    channel_offsets[channel] +
                    channel_scales[channel] * ref_value;
                const float neighbour_channel =
                    channel_offsets[channel] +
                    channel_scales[channel] * neighbour_value;
                reference_rgb[
                    static_cast<std::size_t>(channel) * pixels + pixel] =
                    reference_channel;
                neighbour_rgb[
                    static_cast<std::size_t>(channel) * pixels + pixel] =
                    neighbour_channel;
                reference_gray[pixel] +=
                    gray_weights[channel] * reference_channel;
                neighbour_gray[pixel] +=
                    gray_weights[channel] * neighbour_channel;
            }
        }
    }
    reference.rgb = tinytensor::Tensor::from_vector(
        reference_rgb, {3, height, width}, tinytensor::Device::CUDA);
    neighbour.rgb = tinytensor::Tensor::from_vector(
        neighbour_rgb, {3, height, width}, tinytensor::Device::CUDA);
    reference.gray = tinytensor::Tensor::from_vector(
        reference_gray,
        {height, width}, tinytensor::Device::CUDA);
    neighbour.gray = tinytensor::Tensor::from_vector(
        neighbour_gray,
        {height, width}, tinytensor::Device::CUDA);
    RenderResult reference_render;
    reference_render.median_depth = tinytensor::Tensor::from_vector(
        std::vector<float>(pixels, 2.F), {height, width},
        tinytensor::Device::CUDA);
    std::vector<float> normals(3 * pixels, 0.F);
    std::fill(normals.begin() + 2 * pixels, normals.end(), 1.F);
    reference_render.normal = tinytensor::Tensor::from_vector(
        normals, {3, height, width}, tinytensor::Device::CUDA);
    detail::LossGradients gradients;
    gradients.depth = tinytensor::Tensor::zeros(
        {height, width}, tinytensor::Device::CUDA);
    gradients.normal = tinytensor::Tensor::zeros(
        {3, height, width}, tinytensor::Device::CUDA);
    TrainingOptions options;
    options.multi_view_geo_weight = 0.02F;
    options.multi_view_ncc_weight = 0.6F;
    std::vector<float> sampled_points(3 * pixels);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            sampled_points[3 * pixel] =
                (static_cast<float>(x) - 15.5F) / 40.F * 2.F - 0.1F;
            sampled_points[3 * pixel + 1] =
                (static_cast<float>(y) - 15.5F) / 40.F * 2.F;
            sampled_points[3 * pixel + 2] = 2.F;
        }
    }
    const auto sampled = tinytensor::Tensor::from_vector(
        sampled_points, {pixels, std::size_t{3}}, tinytensor::Device::CUDA);
    const auto inside = tinytensor::Tensor::ones_bool(
        {pixels}, tinytensor::Device::CUDA);
    auto stability_accumulator = tinytensor::Tensor::zeros(
        {std::size_t{3}}, tinytensor::Device::CUDA);
    tinytensor::Tensor grad_sampled;
    const auto loss = detail::add_multi_view_loss(
        sampled, inside, reference_render, reference, neighbour,
        options, gradients, grad_sampled, true,
        &stability_accumulator);
    require(loss.geometry_pixels > 0 && loss.ncc_pixels > 0,
            "GGGS multi-view consistency rejected a valid planar pair");
    require(
        loss.geometry_candidates >= loss.geometry_pixels &&
            loss.geometry_candidates > 0,
        "GGGS multi-view consistency candidate count is invalid");
    const std::vector<float> stability =
        stability_accumulator.to_vector();
    require(
        static_cast<std::size_t>(stability[0]) == loss.geometry_pixels &&
            static_cast<std::size_t>(stability[1]) ==
                loss.geometry_candidates &&
            stability[2] == 1.F,
        "GGGS multi-view stability window did not accumulate loss terms");
    require(loss.geometry < 1e-3F && loss.ncc < 2e-3F,
            "GGGS multi-view geometry/NCC does not preserve a consistent plane");
    require_finite(gradients.depth, "Non-finite multi-view depth gradient");
    require_finite(gradients.normal, "Non-finite multi-view normal gradient");
    require_finite(grad_sampled, "Non-finite multi-view sampled-point gradient");

    reference.has_mask = true;
    neighbour.has_mask = true;
    reference.mask = tinytensor::Tensor::zeros(
        {height, width}, tinytensor::Device::CUDA);
    neighbour.mask = tinytensor::Tensor::ones(
        {height, width}, tinytensor::Device::CUDA);
    tinytensor::Tensor masked_grad_sampled;
    const auto masked_loss = detail::add_multi_view_loss(
        sampled, inside, reference_render, reference, neighbour,
        options, gradients, masked_grad_sampled, true);
    require(
        masked_loss.geometry_pixels == 0 &&
            masked_loss.geometry_candidates == 0 &&
            masked_loss.ncc_pixels == 0,
        "GGGS multi-view loss ignored the coarse foreground mask");
}

void test_geometry_stability_scheduler() {
    using namespace aetherscan::splat;
    GaussianModel model;
    model.means = tinytensor::Tensor::zeros(
        {std::size_t{2}, std::size_t{3}}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{
            std::log(0.1F), std::log(0.2F), std::log(0.4F),
            std::log(0.2F), std::log(0.2F), std::log(0.2F)},
        {std::size_t{2}, std::size_t{3}}, tinytensor::Device::CUDA);
    const auto logit = [](const float value) {
        return std::log(value / (1.F - value));
    };
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>{logit(0.2F), logit(0.8F)},
        {std::size_t{2}, std::size_t{1}}, tinytensor::Device::CUDA);
    const detail::GeometryDistributionSummary distribution =
        detail::summarize_geometry_distribution(model);
    require(
        std::abs(distribution.opacity_mean - 0.5F) < 1e-5F &&
            std::abs(distribution.opacity_stddev - 0.3F) < 1e-5F,
        "GGGS opacity distribution summary is incorrect");
    require(
        std::abs(distribution.log_scale_mean - std::log(0.2F)) <
                1e-5F &&
            distribution.log_scale_stddev < 1e-5F &&
            std::abs(
                distribution.log_anisotropy_mean - std::log(2.F)) <
                1e-5F &&
            std::abs(
                distribution.log_anisotropy_stddev - std::log(2.F)) <
                1e-5F,
        "GGGS scale distribution summary is incorrect");

    TrainingOptions options;
    options.multi_view_adaptive_max_interval = 4;
    options.multi_view_adaptive_stable_refinements = 2;
    options.multi_view_adaptive_count_threshold = 0.01F;
    options.multi_view_adaptive_churn_threshold = 0.01F;
    options.multi_view_adaptive_depth_threshold = 0.03F;
    options.multi_view_adaptive_min_depth_consistency = 0.5F;
    options.multi_view_adaptive_distribution_threshold = 0.03F;
    detail::MultiViewStabilityScheduler scheduler(options);
    detail::GeometryStabilitySample sample{
        1000, 0, 0, 0.8F, distribution};
    require(
        !scheduler.update(sample).reference_ready &&
            scheduler.interval() == 1,
        "GGGS stability scheduler reduced before a reference window");
    sample.gaussian_count = 1002;
    sample.depth_consistency = 0.805F;
    require(
        !scheduler.update(sample).reduced && scheduler.interval() == 1,
        "GGGS stability scheduler reduced too early");
    sample.gaussian_count = 1004;
    sample.depth_consistency = 0.81F;
    require(
        scheduler.update(sample).reduced && scheduler.interval() == 2,
        "GGGS stability scheduler did not reduce after stable windows");
    sample.gaussian_count = 1006;
    sample.depth_consistency = 0.815F;
    scheduler.update(sample);
    sample.gaussian_count = 1008;
    sample.depth_consistency = 0.82F;
    require(
        scheduler.update(sample).reduced && scheduler.interval() == 4,
        "GGGS stability scheduler did not reduce gradually");
    sample.depth_consistency = 0.35F;
    const detail::GeometryStabilityDecision recovery =
        scheduler.update(sample);
    require(
        recovery.recovered && scheduler.interval() == 1,
        "GGGS stability scheduler did not recover on geometry drift");
    scheduler.update(sample);
    sample.depth_consistency = 0.8F;
    scheduler.update(sample);
    sample.depth_consistency = 0.805F;
    scheduler.update(sample);
    sample.depth_consistency = 0.81F;
    require(
        scheduler.update(sample).reduced && scheduler.interval() == 2,
        "GGGS stability scheduler did not resume after recovery");
    const detail::GeometryStabilityDecision invalidated =
        scheduler.invalidate();
    require(
        invalidated.recovered && scheduler.interval() == 1 &&
            !scheduler.update(sample).reference_ready,
        "GGGS stability scheduler did not reset after missing geometry");
}

void test_camera_projection_roundtrip() {
    using namespace aetherscan;
    const auto fish = project_fisheye_camera(
        0.0, 0.0, 1.0, 400.0, 400.0, 640.0, 360.0, 0.0, 0.0, 0.0, 0.0);
    require(fish.valid && std::abs(fish.u - 640.0) < 1e-9 &&
                std::abs(fish.v - 360.0) < 1e-9,
            "fisheye optical axis should land on the principal point");
    const double x = 0.8, y = 0.6, z = 1.0;
    const auto off = project_fisheye_camera(
        x, y, z, 400.0, 400.0, 640.0, 360.0, 0.025, -0.002, 0.0002, -0.00001);
    require(off.valid, "fisheye off-axis projection failed");
    const auto plane = project_camera_plane(
        CameraModel::opencv_fisheye, x / z, y / z,
        0.025, -0.002, 0.0002, -0.00001);
    require(std::abs(off.u - (400.0 * plane.x + 640.0)) < 1e-8 &&
                std::abs(off.v - (400.0 * plane.y + 360.0)) < 1e-8,
            "fisheye camera projection must match the normalized-plane model");
    const auto back = unproject_fisheye_camera(
        off.u, off.v, 400.0, 400.0, 640.0, 360.0,
        0.025, -0.002, 0.0002, -0.00001);
    require(back.valid, "fisheye unproject failed");
    const double scale = z / back.z;
    require(std::hypot(back.x * scale - x, back.y * scale - y, back.z * scale - z) < 1e-6,
            "fisheye project/unproject roundtrip");

    const auto front = project_equirectangular_camera(0.0, 0.0, 1.0, 800, 400);
    require(front.valid && std::abs(front.u - 400.0) < 1e-6 &&
                std::abs(front.v - 200.0) < 1e-6,
            "equirect +Z should land at the image center");
    const auto right = project_equirectangular_camera(1.0, 0.0, 0.0, 800, 400);
    require(right.valid && std::abs(right.u - 600.0) < 1e-6 &&
                std::abs(right.v - 200.0) < 1e-6,
            "equirect +X should land at 0.75 width");
    const auto behind = project_equirectangular_camera(0.0, 0.0, -1.0, 800, 400);
    require(behind.valid &&
                (std::abs(behind.u) < 1e-5 || std::abs(behind.u - 800.0) < 1e-5),
            "equirect -Z should land on the azimuth seam");
    const auto ray = unproject_equirectangular_camera(600.0, 200.0, 800, 400);
    require(ray.valid && std::abs(ray.x - 1.0) < 1e-6 &&
                std::abs(ray.y) < 1e-6 && std::abs(ray.z) < 1e-6,
            "equirect unproject of +X");
}

void test_fisheye_equirect_rasterize() {
    using namespace aetherscan::splat;
    auto identity_camera = []() {
        Camera camera;
        camera.world_to_camera[0] = 1.F;
        camera.world_to_camera[5] = 1.F;
        camera.world_to_camera[10] = 1.F;
        camera.world_to_camera[15] = 1.F;
        camera.width = 64;
        camera.height = 48;
        return camera;
    };
    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        std::vector<float>{0.F, 0.F, 2.F}, {1, 3},
        tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(0.08F), std::log(0.08F), std::log(0.08F)},
        {1, 3}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F, 0.F, 0.F}, {1, 4},
        tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>{2.F}, {1, 1}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(
        std::vector<float>{0.5F, 0.25F, 0.1F}, {1, 1, 3},
        tinytensor::Device::CUDA);
    model.sh_degree = 0;
    Rasterizer rasterizer;

    Camera fisheye = identity_camera();
    fisheye.model = aetherscan::CameraModel::opencv_fisheye;
    fisheye.fx = fisheye.fy = 28.F;
    fisheye.cx = 31.5F;
    fisheye.cy = 23.5F;
    const auto fish_render = rasterizer.forward(model, fisheye);
    require(fish_render.rendered_instances > 0, "fisheye Gaussian was not rasterized");
    const auto fish_alpha = fish_render.alpha.to_vector();
    std::size_t fish_peak = 0;
    for (std::size_t i = 1; i < fish_alpha.size(); ++i)
        if (fish_alpha[i] > fish_alpha[fish_peak]) fish_peak = i;
    require(
        fish_peak % fisheye.width == 31 && fish_peak / fisheye.width == 23 &&
            fish_alpha[fish_peak] > 0.F,
        "fisheye on-axis splat should peak at the principal point");
    const std::size_t fish_pixels =
        static_cast<std::size_t>(fisheye.width) * fisheye.height;
    const auto fish_grad = tinytensor::Tensor::from_vector(
        std::vector<float>(3 * fish_pixels, 1.F / static_cast<float>(fish_pixels)),
        {3, fisheye.height, fisheye.width}, tinytensor::Device::CUDA);
    const auto zero_a = tinytensor::Tensor::zeros(
        {fisheye.height, fisheye.width}, tinytensor::Device::CUDA);
    const auto zero_n = tinytensor::Tensor::zeros(
        {3, fisheye.height, fisheye.width}, tinytensor::Device::CUDA);
    const auto fish_grad_out = rasterizer.backward(
        model, fish_render, fish_grad, zero_a, zero_a, zero_n);
    require_finite(fish_grad_out.means, "fisheye mean gradient");

    Camera equirect = identity_camera();
    equirect.model = aetherscan::CameraModel::equirectangular;
    equirect.fx = equirect.fy =
        static_cast<float>(equirect.width) / (2.F * 3.14159265358979323846F);
    equirect.cx = 0.5F * static_cast<float>(equirect.width);
    equirect.cy = 0.5F * static_cast<float>(equirect.height);
    const auto eq_render = rasterizer.forward(model, equirect);
    require(eq_render.rendered_instances > 0, "equirect Gaussian was not rasterized");
    const auto eq_alpha = eq_render.alpha.to_vector();
    std::size_t eq_peak = 0;
    for (std::size_t i = 1; i < eq_alpha.size(); ++i)
        if (eq_alpha[i] > eq_alpha[eq_peak]) eq_peak = i;
    require(
        eq_peak % equirect.width == 32 && eq_peak / equirect.width == 24 &&
            eq_alpha[eq_peak] > 0.F,
        "equirect +Z splat should peak at the image center");

    model.means = tinytensor::Tensor::from_vector(
        std::vector<float>{2.F, 0.F, 0.F}, {1, 3},
        tinytensor::Device::CUDA);
    const auto right_render = rasterizer.forward(model, equirect);
    const auto right_alpha = right_render.alpha.to_vector();
    std::size_t right_peak = 0;
    for (std::size_t i = 1; i < right_alpha.size(); ++i)
        if (right_alpha[i] > right_alpha[right_peak]) right_peak = i;
    require(
        right_peak % equirect.width == 48 &&
            right_peak / equirect.width == 24,
        "equirect +X splat should peak at 0.75 width");
}

void test_fisheye_parameter_finite_differences() {
    using namespace aetherscan::splat;
    using tinytensor::Tensor;
    const auto gpu=tinytensor::Device::CUDA;
    Camera camera;
    camera.model=aetherscan::CameraModel::opencv_fisheye;
    camera.width=80; camera.height=64;
    camera.fx=29.F; camera.fy=31.F; camera.cx=39.2F; camera.cy=31.1F;
    camera.k1=0.06F; camera.k2=-0.012F; camera.k3=0.003F; camera.k4=-0.0004F;
    // Non-identity camera rotation exercises covariance/world gradient transforms.
    const float angle=0.23F, c=std::cos(angle), s=std::sin(angle);
    camera.world_to_camera={c,0.F,-s,0.F, 0.F,1.F,0.F,0.F, s,0.F,c,0.F, 0.F,0.F,0.F,1.F};
    const std::size_t pixels=camera.width*camera.height;
    Rasterizer rasterizer;
    for(int scene=0;scene<4;++scene) {
        std::vector<float> parameters={0.9F,0.32F,1.3F, std::log(.16F),std::log(.10F),std::log(.22F),
                                       0.94F,0.13F,-0.18F,0.21F, 3.F, 0.4F,0.2F,0.3F};
        if(scene==1) { // Close to the horizon, with camera Z below the old 0.2 cutoff.
            parameters[0]=1.15F; parameters[1]=.10F;
            parameters[2]=(.16F+s*parameters[0])/c;
        }
        if(scene>=2) { // Exactly on-axis, including the Taylor branch.
            parameters[0]=-s*1.4F; parameters[1]=0.F; parameters[2]=c*1.4F;
        }
        if(scene==3) {parameters[10]=8.F;camera.cx=39.F;camera.cy=31.F;}
        auto make_model=[&](const std::vector<float>& p) {
            GaussianModel m;
            auto upload=[&](int offset,int count,std::initializer_list<std::size_t> shape) {
                return Tensor::from_vector(std::vector<float>(p.begin()+offset,p.begin()+offset+count),shape,gpu);
            };
            m.means=upload(0,3,{1,3}); m.log_scales=upload(3,3,{1,3});
            m.quaternions=upload(6,4,{1,4}); m.opacity_logits=upload(10,1,{1,1});
            m.sh=upload(11,3,{1,1,3}); m.sh_degree=0;
            return m;
        };
        for(float kernel : {0.F,0.3F}) {
            RasterizeOptions options;
            options.kernel_size=kernel; options.require_depth=true;
            options.background={.07F,.11F,.03F};
            auto model=make_model(parameters);
            auto rendered=rasterizer.forward(model,camera,options);
            require(rendered.rendered_instances>0,"Fisheye test Gaussian culled");
            const auto depth=rendered.median_depth.to_vector();
            const auto alpha=rendered.alpha.to_vector();
            std::size_t pixel=std::max_element(alpha.begin(),alpha.end())-alpha.begin();
            require(depth[pixel]>0.F,"Fisheye test needs a valid median depth");
            // Independent double-precision projection and covariance oracle.
            Eigen::Matrix3d view_rotation;
            for(int row=0;row<3;++row) for(int col=0;col<3;++col)
                view_rotation(row,col)=camera.world_to_camera[4*col+row];
            const Eigen::Vector3d position=view_rotation*Eigen::Vector3d(parameters[0],parameters[1],parameters[2]);
            const Eigen::Matrix3d rotation=Eigen::Quaterniond(parameters[6],parameters[7],parameters[8],parameters[9]).normalized().toRotationMatrix();
            const Eigen::Vector3d variance(std::exp(2*parameters[3]),std::exp(2*parameters[4]),std::exp(2*parameters[5]));
            const Eigen::Matrix3d covariance=view_rotation*rotation*variance.asDiagonal()*rotation.transpose()*view_rotation.transpose();
            auto project=[&](Eigen::Vector3d t) {
                const auto p=aetherscan::project_fisheye_camera(t.x(),t.y(),t.z(),camera.fx,camera.fy,camera.cx,camera.cy,
                    camera.k1,camera.k2,camera.k3,camera.k4);
                require(p.valid,"CPU fish projection failed");
                return Eigen::Vector2d(p.u,p.v);
            };
            Eigen::Matrix<double,2,3> jacobian;
            for(int i=0;i<3;++i) {
                auto plus=position,minus=position; plus[i]+=1e-5;minus[i]-=1e-5;
                jacobian.col(i)=(project(plus)-project(minus))/(2e-5);
            }
            const Eigen::Matrix2d raw=jacobian*covariance*jacobian.transpose();
            const Eigen::Matrix2d filtered=raw+kernel*Eigen::Matrix2d::Identity();
            const Eigen::Vector2d delta=project(position)-Eigen::Vector2d(pixel%camera.width,pixel/camera.width);
            const double expected_alpha=std::min(.99,1.0/(1.0+std::exp(-parameters[10]))*
                std::sqrt(raw.determinant()/filtered.determinant())*
                std::exp(-.5*delta.dot(filtered.inverse()*delta)));
            require(std::abs(expected_alpha-alpha[pixel])<2e-4,"Fisheye forward covariance differs from CPU oracle");
            const Eigen::Vector3d expected_normal=-(covariance.inverse()*position).normalized();
            const auto normals=rendered.normal.to_vector();
            for(int i=0;i<3;++i)
                require(std::abs(normals[i*pixels+pixel]-expected_normal[i])<2e-4,"Fisheye forward normal differs from CPU oracle");
            const Eigen::Vector3d center_ray=position.normalized();
            const Eigen::Vector3d precision_ray=covariance.inverse()*center_ray;
            const double precision=center_ray.dot(precision_ray);
            const Eigen::Matrix<double,3,2> inverse_jacobian=jacobian.transpose()*(jacobian*jacobian.transpose()).inverse();
            const Eigen::Vector2d slope=inverse_jacobian.transpose()*precision_ray/precision;
            const double peak=position.norm()+slope.dot(delta);
            const double g=expected_alpha>=.75 ? .75/expected_alpha : (1-4*std::pow(1-expected_alpha,2))/expected_alpha;
            const double median=peak+(expected_alpha>=.75 ? -1 : 1)*std::sqrt(-2*std::log(g)/precision);
            const auto ray=aetherscan::unproject_fisheye_camera(pixel%camera.width,pixel/camera.width,
                camera.fx,camera.fy,camera.cx,camera.cy,camera.k1,camera.k2,camera.k3,camera.k4);
            require(ray.valid && std::abs(depth[pixel]-median*ray.z)<3e-4,"Fisheye median depth differs from CPU oracle");
            // Probe a fixed interior pixel, so visibility/support discontinuities
            // are not confused with derivatives of the smooth raster expression.
            for(int channel=0;channel<4;++channel) {
                std::vector<float> gc(3*pixels,0.F),ga(pixels,0.F),gd(pixels,0.F),gn(3*pixels,0.F);
                if(channel==0) { gc[pixel]=.7F;gc[pixels+pixel]=-.3F;gc[2*pixels+pixel]=.2F; }
                if(channel==1) ga[pixel]=1.F;
                if(channel==2) gd[pixel]=1.F;
                if(channel==3) {gn[pixel]=.3F;gn[pixels+pixel]=-.7F;gn[2*pixels+pixel]=.2F;}
                const auto gradients=rasterizer.backward(model,rendered,
                    Tensor::from_vector(gc,{3,camera.height,camera.width},gpu),
                    Tensor::from_vector(ga,{camera.height,camera.width},gpu),
                    Tensor::from_vector(gd,{camera.height,camera.width},gpu),
                    Tensor::from_vector(gn,{3,camera.height,camera.width},gpu));
                if(channel==0) {
                    auto rgb_options=options; rgb_options.require_depth=false;
                    auto rgb=rasterizer.forward(model,camera,rgb_options);
                    const auto rgb_grad=rasterizer.backward(model,rgb,
                        Tensor::from_vector(gc,{3,camera.height,camera.width},gpu),
                        Tensor::from_vector(ga,{camera.height,camera.width},gpu),
                        Tensor::from_vector(gd,{camera.height,camera.width},gpu),
                        Tensor::from_vector(gn,{3,camera.height,camera.width},gpu));
                    const auto a=gradients.means.to_vector(),b=rgb_grad.means.to_vector();
                    for(int i=0;i<3;++i) {
                        if(!std::isfinite(a[i]) || !std::isfinite(b[i]))
                            std::cerr<<"Geometry/RGB-only mean gradients: "<<a[i]<<", "<<b[i]<<'\n';
                        require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::abs(a[i]-b[i])<2e-5,
                            "Geometry outputs changed RGB-only training gradients");
                    }
                }
                std::vector<float> analytic;
                for(const auto* tensor : {&gradients.means,&gradients.log_scales,&gradients.quaternions,
                                         &gradients.opacity_logits,&gradients.sh}) {
                    const auto values=tensor->to_vector();analytic.insert(analytic.end(),values.begin(),values.end());
                }
                auto objective=[&](const std::vector<float>& p) {
                    const auto output=rasterizer.forward(make_model(p),camera,options);
                    if(channel==0) {auto v=output.color.to_vector();return .7F*v[pixel]-.3F*v[pixels+pixel]+.2F*v[2*pixels+pixel];}
                    if(channel==1) return output.alpha.to_vector()[pixel];
                    if(channel==2) return output.median_depth.to_vector()[pixel];
                    auto v=output.normal.to_vector();return .3F*v[pixel]-.7F*v[pixels+pixel]+.2F*v[2*pixels+pixel];
                };
                for(std::size_t i=0;i<parameters.size();++i) {
                    auto plus=parameters,minus=parameters;
                    const float h=channel==2 ? 5e-4F : 2e-4F;
                    plus[i]+=h;minus[i]-=h;
                    const float numerical=(objective(plus)-objective(minus))/(2*h);
                    const float tolerance=(channel==2 ? .025F : .004F)+.025F*std::abs(numerical);
                    if(!std::isfinite(analytic[i]) || std::abs(analytic[i]-numerical)>tolerance) {
                        std::cerr<<"fish scene="<<scene<<" kernel="<<kernel<<" channel="<<channel
                                 <<" parameter="<<i<<" analytic="<<analytic[i]<<" numerical="<<numerical<<'\n';
                        throw std::runtime_error("Fisheye parameter finite difference mismatch");
                    }
                }
            }
            // Depth sampling returns a point on the inverse-projected ray.
            // Check derivatives with respect to both the query and the model.
            const double query_u=static_cast<double>(pixel%camera.width)-.2;
            const double query_v=static_cast<double>(pixel/camera.width)-.2;
            const auto query_ray=aetherscan::unproject_fisheye_camera(query_u,query_v,
                camera.fx,camera.fy,camera.cx,camera.cy,camera.k1,camera.k2,camera.k3,camera.k4);
            const Eigen::Vector3d query_world=view_rotation.transpose()*
                (position.norm()*Eigen::Vector3d(query_ray.x,query_ray.y,query_ray.z));
            std::vector<float> query={static_cast<float>(query_world.x()),static_cast<float>(query_world.y()),static_cast<float>(query_world.z())};
            auto upload_query=[&](const std::vector<float>& q) {return Tensor::from_vector(q,{1,3},gpu);};
            const auto sampled=rasterizer.sample_depth(model,upload_query(query),camera,options);
            const auto point=sampled.camera_points.to_vector();
            const Eigen::Vector3d query_camera=view_rotation*Eigen::Vector3d(query[0],query[1],query[2]);
            require(Eigen::Vector3d(point[0],point[1],point[2]).norm()>0.,"Fisheye sample depth returned no intersection");
            require(Eigen::Vector3d(point[0],point[1],point[2]).normalized().dot(query_camera.normalized())>1.-1e-6,
                "Fisheye sample depth point is not on the camera ray");
            const auto sample_grad=rasterizer.sample_depth_backward(model,sampled,
                Tensor::from_vector(std::vector<float>{.3F,-.2F,.7F},{1,3},gpu));
            auto sample_objective=[&](const std::vector<float>& p,const std::vector<float>& q) {
                const auto v=rasterizer.sample_depth(make_model(p),upload_query(q),camera,options).camera_points.to_vector();
                return .3F*v[0]-.2F*v[1]+.7F*v[2];
            };
            const auto query_gradient=sample_grad.points.to_vector();
            std::vector<float> model_gradient;
            for(const auto* tensor : {&sample_grad.model.means,&sample_grad.model.log_scales,
                                     &sample_grad.model.quaternions,&sample_grad.model.opacity_logits}) {
                const auto v=tensor->to_vector();model_gradient.insert(model_gradient.end(),v.begin(),v.end());
            }
            for(int i=0;i<14;++i) {
                auto pp=parameters,pm=parameters,qp=query,qm=query;
                constexpr float h=5e-4F;
                if(i<11) {pp[i]+=h;pm[i]-=h;} else {qp[i-11]+=h;qm[i-11]-=h;}
                const float numerical=(sample_objective(pp,qp)-sample_objective(pm,qm))/(2*h);
                const float analytic=i<11 ? model_gradient[i] : query_gradient[i-11];
                if(!std::isfinite(analytic) || std::abs(numerical-analytic)>.025F+.025F*std::abs(numerical)) {
                    std::cerr<<"fish sample scene="<<scene<<" kernel="<<kernel<<" parameter="<<i
                             <<" analytic="<<analytic<<" numerical="<<numerical<<'\n';
                    throw std::runtime_error("Fisheye sample depth gradient mismatch");
                }
            }
        }
    }
}

void test_fisheye_filter_and_supervision() {
    using namespace aetherscan;
    using namespace aetherscan::splat;
    Camera camera;
    camera.model=CameraModel::opencv_fisheye;
    camera.world_to_camera={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    camera.width=128; camera.height=128;
    camera.fx=camera.fy=30.F;camera.cx=camera.cy=63.5F;
    const auto means=tinytensor::Tensor::from_vector(
        std::vector<float>{2.F,0.F,.1F}, {1,3},tinytensor::Device::CUDA);
    const auto filtered=detail::compute_3d_filter(means,{camera},.2F,false).to_vector();
    // For an equidistant camera, tangential angular magnification is f*theta/r.
    // This is larger than the radial magnification f/length at this test point.
    const float expected=std::sqrt(.2F)*2.F/(30.F*std::atan2(2.F,.1F));
    require(std::abs(filtered[0]-expected)<1e-5F,"Fisheye filter ignored the local projection scale");
    mvs::MvsView view;
    view.path=std::filesystem::temp_directory_path()/
        ("aetherscan_fisheye_supervision_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".png");
    io::save_rgb_png(io::RgbImage{16,16,std::vector<std::uint8_t>(16*16*3,128)},view.path);
    view.width=view.src_width=view.height=view.src_height=16;
    view.fx=view.fy=view.src_fx=view.src_fy=12.F;
    view.cx=view.cy=view.src_cx=view.src_cy=7.5F;
    view.source_model=CameraModel::opencv_fisheye;
    view.depth_map.depth.assign(256,2.F);
    view.depth_map.normal.assign(256,mvs::Vec3f(0.F,0.F,1.F));
    TrainingOptions options;
    options.use_mvs_depth=options.use_mvs_normals=true;
    const auto native=make_training_view(view,options);
    require(native.depth.numel()==1 && native.normal.numel()==1,
        "Native fisheye reused pinhole MVS geometry based only on matching image dimensions");
    options.undistort_to_pinhole=true;
    const auto pinhole=make_training_view(view,options);
    require(pinhole.depth.numel()==256 && pinhole.normal.numel()==768,
        "Pinhole working camera lost its matching MVS geometry");
    std::filesystem::remove(view.path);
}

void test_colmap_fisheye_and_equirect_loading() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_colmap_fisheye_equirect_test";
    const auto model = root / "sparse" / "0";
    const auto images = root / "images";
    std::filesystem::create_directories(model);
    std::filesystem::create_directories(images);
    io::save_rgb_png(
        io::RgbImage{8, 6, std::vector<std::uint8_t>(144, 127)},
        images / "frame.png");
    {
        std::ofstream stream(model / "cameras.txt");
        stream << "1 OPENCV_FISHEYE 8 6 200 201 4 3 0.01 -0.002 0.0001 -0.00001\n";
    }
    {
        std::ofstream stream(model / "images.txt");
        stream << "7 1 0 0 0 0 0 0 1 frame.png\n\n";
    }
    {
        std::ofstream stream(model / "points3D.txt");
        stream << "42 0.1 0.2 4 10 20 30 0.5 7 0\n";
    }
    const auto loaded = splat::load_colmap_scene(root, images);
    require(
        loaded.scene.views.front().source_model == CameraModel::opencv_fisheye &&
            std::abs(loaded.scene.views.front().fx - 200.F) < 1e-5F &&
            std::abs(loaded.scene.views.front().p2 + 0.00001F) < 1e-6F,
        "COLMAP OPENCV_FISHEYE was not loaded natively");
    {
        std::ofstream stream(model / "cameras.txt");
        stream << "1 EQUIRECTANGULAR 8 6 8 6\n";
    }
    const auto equirect = splat::load_colmap_scene(root, images);
    require(
        equirect.scene.views.front().source_model ==
            CameraModel::equirectangular,
        "COLMAP EQUIRECTANGULAR was not loaded");
    {
        std::ofstream stream(model / "cameras.txt");
        stream << "1 FOV 8 6 100 101 4 3 0.7\n";
    }
    bool refused_fov = false;
    try {
        (void)splat::load_colmap_scene(root, images);
    } catch (const std::runtime_error&) {
        refused_fov = true;
    }
    require(refused_fov, "COLMAP FOV must still be rejected");
    std::filesystem::remove_all(root);
}

void test_forward_backward() {
    using namespace aetherscan::splat;
    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        std::vector<float>{0.5F, 0.F, 2.F}, {1, 3},
        tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(0.15F), std::log(0.15F),
                           std::log(0.15F)},
        {1, 3}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F, 0.F, 0.F}, {1, 4},
        tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>{2.F}, {1, 1}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(
        std::vector<float>{0.5F, 0.25F, 0.1F}, {1, 1, 3},
        tinytensor::Device::CUDA);
    model.sh_degree = 0;

    Camera camera;
    camera.world_to_camera[0] = 1.F;
    camera.world_to_camera[5] = 1.F;
    camera.world_to_camera[10] = 1.F;
    camera.world_to_camera[15] = 1.F;
    camera.fx = 40.F;
    camera.fy = 40.F;
    // Deliberately use an off-centre Gaussian on a non-square tile grid.
    // This catches x/y-transposed AccuTile keys that square-image tests hide.
    camera.cx = 23.5F;
    camera.cy = 15.5F;
    camera.width = 48;
    camera.height = 32;

    Rasterizer rasterizer;
    RenderResult rendered = rasterizer.forward(model, camera);
    require(rendered.rendered_instances > 0, "Gaussian was not rasterized");
    const std::vector<float> alpha = rendered.alpha.to_vector();
    require(
        *std::max_element(alpha.begin(), alpha.end()) > 0.F,
        "Rasterized alpha is empty");
    require(
        rendered.visibility.to_vector() == std::vector<float>{1.F},
        "Contributing Gaussian was not marked visible");

    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    RasterizeOptions colored_background;
    colored_background.background = {0.1F, 0.3F, 0.7F};
    const auto colored = rasterizer.forward(model, camera, colored_background);
    const auto black_rgb = rendered.color.to_vector();
    const auto colored_rgb = colored.color.to_vector();
    for (std::size_t channel = 0; channel < 3; ++channel)
        for (std::size_t pixel = 0; pixel < pixels; ++pixel)
            require(std::abs(colored_rgb[channel * pixels + pixel] -
                        black_rgb[channel * pixels + pixel] -
                        (1.F - alpha[pixel]) * colored_background.background[channel]) < 5e-6F,
                    "Raster camera constants changed the background blend");
    const auto grad_color = tinytensor::Tensor::from_vector(
        std::vector<float>(3 * pixels, 1.F / static_cast<float>(pixels)),
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    const auto grad_alpha = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    const auto grad_depth = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    const auto grad_normal = tinytensor::Tensor::zeros(
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    const ModelGradients gradients = rasterizer.backward(
        model, rendered, grad_color, grad_alpha, grad_depth, grad_normal);
    require_finite(gradients.means, "Non-finite mean gradient");
    require_finite(gradients.log_scales, "Non-finite scale gradient");
    require_finite(gradients.quaternions, "Non-finite quaternion gradient");
    require_finite(gradients.opacity_logits, "Non-finite opacity gradient");
    require_finite(gradients.sh, "Non-finite SH gradient");

    // The GGGS kernels atomicAdd into a backward-only geometry chunk. Running
    // the same pass twice must not retain values from the memory pool.
    const RenderResult rendered_again = rasterizer.forward(model, camera);
    const ModelGradients gradients_again = rasterizer.backward(
        model, rendered_again, grad_color, grad_alpha, grad_depth, grad_normal);
    const auto first_opacity_gradient = gradients.opacity_logits.to_vector();
    const auto second_opacity_gradient =
        gradients_again.opacity_logits.to_vector();
    require(
        first_opacity_gradient.size() == second_opacity_gradient.size() &&
            std::equal(
                first_opacity_gradient.begin(), first_opacity_gradient.end(),
                second_opacity_gradient.begin(),
                [](const float first, const float second) {
                    return std::abs(first - second) < 1e-6F;
                }),
        "GGGS backward retained stale geometry gradients between calls");
}

void test_normal_field_parameterization_and_occupancy() {
    using namespace aetherscan::splat;
    const float sign_logit = std::atanh(0.5F);
    const auto features = tinytensor::Tensor::from_vector(
        std::vector<float>{2.F, 0.F, 0.F, sign_logit}, {1, 4},
        tinytensor::Device::CUDA);
    const auto normals =
        detail::normal_features_to_normals(features).to_vector();
    require(
        normals.size() == 3 && std::abs(normals[0] - 0.5F) < 1e-6F &&
            std::abs(normals[1]) < 1e-6F &&
            std::abs(normals[2]) < 1e-6F,
        "GaussianWrapping normal feature conversion changed");
    const auto feature_gradients = detail::normal_features_backward(
        features,
        tinytensor::Tensor::from_vector(
            std::vector<float>{1.F, 2.F, 3.F}, {1, 3},
            tinytensor::Device::CUDA)).to_vector();
    require(
        feature_gradients.size() == 4 &&
            std::abs(feature_gradients[0]) < 1e-6F &&
            std::abs(feature_gradients[1] - 0.5F) < 1e-6F &&
            std::abs(feature_gradients[2] - 0.75F) < 1e-6F &&
            std::abs(feature_gradients[3] - 0.75F) < 1e-6F,
        "GaussianWrapping normal feature chain rule changed");

    Camera loss_camera;
    loss_camera.world_to_camera[0] = 1.F;
    loss_camera.world_to_camera[5] = 1.F;
    loss_camera.world_to_camera[10] = 1.F;
    loss_camera.world_to_camera[15] = 1.F;
    loss_camera.fx = loss_camera.fy = 20.F;
    loss_camera.cx = loss_camera.cy = 2.F;
    loss_camera.width = loss_camera.height = 5;
    RenderResult normal_render;
    normal_render.median_depth = tinytensor::Tensor::from_vector(
        std::vector<float>(25, 2.F), {5, 5},
        tinytensor::Device::CUDA);
    std::vector<float> oriented(75, 0.F);
    std::fill(oriented.begin() + 50, oriented.end(), 1.F);
    normal_render.color = tinytensor::Tensor::from_vector(
        oriented, {3, 5, 5}, tinytensor::Device::CUDA);
    normal_render.alpha = tinytensor::Tensor::zeros(
        {5, 5}, tinytensor::Device::CUDA);
    normal_render.normal = tinytensor::Tensor::zeros(
        {3, 5, 5}, tinytensor::Device::CUDA);
    const detail::LossGradients normal_loss =
        detail::compute_normal_field_loss(
            normal_render, loss_camera, 0.05F, true);
    require(
        normal_loss.total > 0.F && std::isfinite(normal_loss.total),
        "GaussianWrapping normal-field alignment loss was not evaluated");
    require_finite(
        normal_loss.color,
        "Normal-field alignment produced non-finite normal gradients");
    require_finite(
        normal_loss.depth,
        "Normal-field alignment produced non-finite depth gradients");

    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        std::vector<float>{0.F, 0.F, 2.F}, {1, 3},
        tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(0.25F), std::log(0.25F),
                           std::log(0.08F)},
        {1, 3}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F, 0.F, 0.F}, {1, 4},
        tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>{5.F}, {1, 1}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::zeros(
        {1, 1, 3}, tinytensor::Device::CUDA);
    model.normal_features = features;
    model.sh_degree = 0;
    Camera camera;
    camera.world_to_camera[0] = 1.F;
    camera.world_to_camera[5] = 1.F;
    camera.world_to_camera[10] = 1.F;
    camera.world_to_camera[15] = 1.F;
    camera.fx = camera.fy = 40.F;
    camera.cx = camera.cy = 15.5F;
    camera.width = camera.height = 32;
    const auto query = tinytensor::Tensor::from_vector(
        std::vector<float>{0.F, 0.F, 2.F}, {1, 3},
        tinytensor::Device::CUDA);
    const OccupancyResult occupancy =
        Rasterizer().evaluate_occupancy(model, query, camera);
    require(
        occupancy.inside.to_vector_bool() == std::vector<bool>{true} &&
            occupancy.occupancy.to_vector()[0] > 0.5F,
        "PAM integrated occupancy did not classify a Gaussian center");

    const auto ply = std::filesystem::temp_directory_path() /
        "aetherscan_normal_field_roundtrip.ply";
    save_gaussians_ply(model, ply);
    const GaussianModel loaded = load_gaussians_ply(ply);
    std::filesystem::remove(ply);
    const auto loaded_features = loaded.normal_features.to_vector();
    const auto original_features = features.to_vector();
    require(
        loaded_features == original_features,
        "GaussianWrapping gaussian_features PLY round trip changed values");
}

void test_gaussian_format_roundtrip() {
    using namespace aetherscan::splat;
    constexpr std::size_t count = 3;
    constexpr unsigned degree = 1;
    const std::vector<float> means{
        0.F, 0.F, 2.F, 1.25F, -0.5F, 3.5F, -1.75F, 0.75F, 1.2F};
    const std::vector<float> scales{
        -1.4F, -1.1F, -1.8F, -0.3F, -0.9F, -1.2F, -2.1F, -1.7F, -0.6F};
    const std::vector<float> rotations{
        1.F, 0.F, 0.F, 0.F, 0.9238795F, 0.F, 0.3826834F, 0.F,
        0.8660254F, 0.2886751F, -0.2886751F, 0.2886751F};
    const std::vector<float> opacities{-2.F, 0.5F, 3.F};
    const std::vector<float> sh{
        0.2F, -0.1F, 0.35F, 0.04F, -0.08F, 0.12F, -0.2F, 0.16F, -0.1F,
        -0.3F, 0.25F, 0.1F, 0.45F, 0.05F, -0.25F, -0.12F, -0.2F, 0.28F,
        0.1F, 0.4F, 0.2F, -0.18F, 0.09F, -0.14F, 0.02F, -0.3F, 0.22F,
        -0.05F, 0.18F, 0.31F, 0.12F, -0.16F, 0.07F, -0.22F, 0.14F,
        0.19F};
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
        sh, {count, 4U, 3U}, tinytensor::Device::CUDA);
    model.sh_degree = degree;

    const auto verify = [&](const std::filesystem::path& path,
                            const GaussianFormat format, const float tolerance,
                            const char* label) {
        save_gaussians(model, path, format);
        const GaussianModel loaded = load_gaussians(path);
        require(loaded.size() == count, label);
        require(loaded.sh_degree == degree, label);
        const auto loaded_means = loaded.means.to_vector();
        const auto loaded_scales = loaded.log_scales.to_vector();
        const auto loaded_rotations = loaded.quaternions.to_vector();
        const auto loaded_opacities = loaded.opacity_logits.to_vector();
        const auto loaded_sh = loaded.sh.to_vector();
        require(loaded_means.size() == means.size(), label);
        require(loaded_sh.size() == sh.size(), label);
        for (std::size_t index = 0; index < means.size(); ++index)
            if (std::abs(loaded_means[index] - means[index]) >= tolerance)
                throw std::runtime_error(
                    std::string(label) + " (means index " +
                    std::to_string(index) + ", error=" +
                    std::to_string(std::abs(loaded_means[index] - means[index])) +
                    ", actual=" + std::to_string(loaded_means[index]) +
                    ")");
        for (std::size_t index = 0; index < sh.size(); ++index)
            if (std::abs(loaded_sh[index] - sh[index]) >= tolerance)
                throw std::runtime_error(
                    std::string(label) + " (SH index " +
                    std::to_string(index) + ", error=" +
                    std::to_string(std::abs(loaded_sh[index] - sh[index])) +
                    ")");
        for (std::size_t index = 0; index < scales.size(); ++index)
            require(
                std::abs(loaded_scales[index] - scales[index]) < tolerance,
                label);
        for (std::size_t index = 0; index < opacities.size(); ++index)
            require(
                std::abs(loaded_opacities[index] - opacities[index]) < tolerance,
                label);
        for (std::size_t index = 0; index < count; ++index) {
            float distance = 0.F;
            float antipodal_distance = 0.F;
            for (std::size_t component = 0; component < 4U; ++component) {
                const float actual = loaded_rotations[index * 4U + component];
                const float expected = rotations[index * 4U + component];
                distance += (actual - expected) * (actual - expected);
                antipodal_distance += (actual + expected) * (actual + expected);
            }
            require(
                std::sqrt(std::min(distance, antipodal_distance)) < tolerance,
                label);
        }
        std::error_code error;
        std::filesystem::remove(path, error);
    };

    verify(
        std::filesystem::temp_directory_path() /
            "aetherscan_splat_roundtrip.sog",
        GaussianFormat::sog, 0.03F, "SOG Gaussian round trip changed values");
    verify(
        std::filesystem::temp_directory_path() /
            "aetherscan_splat_roundtrip.spz",
        GaussianFormat::spz, 0.04F, "SPZ Gaussian round trip changed values");
    verify(
        std::filesystem::temp_directory_path() /
            "aetherscan_splat_roundtrip.glb",
        GaussianFormat::glb, 1e-5F, "GLB Gaussian round trip changed values");

    restrict_sh_degree(model, 0);
    require(model.sh_degree == 0, "SH restrict did not lower the degree");
    require(
        model.sh.shape()[1] == 1, "SH restrict did not drop higher bands");
    const auto ply = std::filesystem::temp_directory_path() /
                     "aetherscan_splat_sh0.ply";
    save_gaussians(model, ply, GaussianFormat::ply);
    const GaussianModel truncated = load_gaussians(ply);
    std::filesystem::remove(ply);
    require(truncated.sh_degree == 0, "PLY export did not keep SH degree 0");
    require(
        truncated.sh.shape()[1] == 1, "PLY export kept extra SH bands");
}

void test_pam_smoke() {
#if defined(AETHERSCAN_HAS_CGAL)
    namespace mvs = aetherscan::mvs;
    namespace splat = aetherscan::splat;
    const std::vector<mvs::Vec3f> shell{
        {1.F, 0.F, 0.F}, {-1.F, 0.F, 0.F},
        {0.F, 1.F, 0.F}, {0.F, -1.F, 0.F},
        {0.F, 0.F, 1.F}, {0.F, 0.F, -1.F}};
    std::vector<float> means;
    std::vector<float> features;
    for (const mvs::Vec3f& point : shell) {
        means.insert(means.end(), point.data(), point.data() + 3);
        features.insert(features.end(), point.data(), point.data() + 3);
        features.push_back(3.F);
    }
    splat::GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        means, {shell.size(), std::size_t{3}}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>(shell.size() * 3, std::log(0.65F)),
        {shell.size(), std::size_t{3}}, tinytensor::Device::CUDA);
    std::vector<float> quaternions(shell.size() * 4, 0.F);
    for (std::size_t index = 0; index < shell.size(); ++index)
        quaternions[4 * index] = 1.F;
    model.quaternions = tinytensor::Tensor::from_vector(
        quaternions, {shell.size(), std::size_t{4}},
        tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>(shell.size(), 8.F),
        {shell.size(), std::size_t{1}}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::zeros(
        {shell.size(), std::size_t{1}, std::size_t{3}},
        tinytensor::Device::CUDA);
    model.normal_features = tinytensor::Tensor::from_vector(
        features, {shell.size(), std::size_t{4}},
        tinytensor::Device::CUDA);

    mvs::MvsScene scene;
    mvs::MvsView view;
    view.pose.R = Eigen::Matrix3d::Identity();
    view.pose.C = Eigen::Vector3d(0.0, 0.0, -4.0);
    view.fx = view.fy = 50.F;
    view.cx = view.cy = 31.5F;
    view.width = view.height = 64;
    scene.views.push_back(view);
    mvs::Mesh seed;
    seed.vertices = shell;
    seed.faces = {
        {0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4},
        {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}};
    splat::PamMeshOptions options;
    options.max_points = 64;
    options.oversampling_factor = 8;
    options.max_resample_rounds = 4;
    options.refinement_steps = 0;
    options.vector_field_neighbors = shell.size();
    options.points_per_tetrahedron = 1;
    options.occupancy_iso_value = 0.05F;
    options.vacancy_threshold = 1.F;
    options.occupancy_chunk_size = 512;
    const auto result = splat::extract_pam_mesh(
        model, scene, seed, splat::TrainingOptions{}, options);
    require(
        result.candidate_cloud.points.size() == options.max_points &&
            result.tetrahedron_count > 0 &&
            result.occupied_tetrahedron_count > 0 &&
            !result.mesh.vertices.empty() && !result.mesh.faces.empty(),
        "PAM smoke extraction did not produce a boundary mesh");
    std::vector<std::pair<int, int>> edges;
    edges.reserve(result.mesh.faces.size() * 3);
    for (const Eigen::Vector3i& face : result.mesh.faces) {
        for (int corner = 0; corner < 3; ++corner) {
            int a = face[corner];
            int b = face[(corner + 1) % 3];
            if (a > b) std::swap(a, b);
            edges.emplace_back(a, b);
        }
    }
    std::sort(edges.begin(), edges.end());
    for (std::size_t begin = 0; begin < edges.size();) {
        std::size_t end = begin + 1;
        while (end < edges.size() && edges[end] == edges[begin]) ++end;
        require(
            end - begin <= 2,
            "PAM topology orientation retained a non-manifold edge");
        begin = end;
    }
#endif
}

void test_sample_depth_batch_boundary() {
    using namespace aetherscan::splat;
    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        std::vector<float>{0.F, 0.F, 2.F}, {1, 3},
        tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{
            std::log(0.25F), std::log(0.25F), std::log(0.08F)},
        {1, 3}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F, 0.F, 0.F}, {1, 4},
        tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>{5.F}, {1, 1}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::zeros(
        {1, 1, 3}, tinytensor::Device::CUDA);
    model.sh_degree = 0;

    Camera camera;
    camera.world_to_camera[0] = 1.F;
    camera.world_to_camera[5] = 1.F;
    camera.world_to_camera[10] = 1.F;
    camera.world_to_camera[15] = 1.F;
    camera.fx = camera.fy = 40.F;
    camera.cx = camera.cy = 15.5F;
    camera.width = camera.height = 32;

    struct SampleRun {
        std::vector<float> camera_points;
        std::vector<bool> inside;
        std::vector<float> point_gradients;
    };
    Rasterizer rasterizer;
    const auto run = [&](const std::size_t count) {
        std::vector<float> points(3 * count, 0.F);
        for (std::size_t index = 0; index < count; ++index)
            points[3 * index + 2] = 2.F;
        const auto world_points = tinytensor::Tensor::from_vector(
            points, {count, std::size_t{3}}, tinytensor::Device::CUDA);
        const DepthSampleResult sampled =
            rasterizer.sample_depth(model, world_points, camera);
        std::vector<float> output_gradient(3 * count, 0.F);
        for (std::size_t index = 0; index < count; ++index)
            output_gradient[3 * index + 2] =
                1.F / static_cast<float>(count);
        const auto gradients = rasterizer.sample_depth_backward(
            model, sampled,
            tinytensor::Tensor::from_vector(
                output_gradient, {count, std::size_t{3}},
                tinytensor::Device::CUDA));
        require_finite(
            gradients.points,
            "Non-finite sample-depth point gradient at tail boundary");
        return SampleRun{
            sampled.camera_points.to_vector(),
            sampled.inside.to_vector_bool(),
            gradients.points.to_vector()};
    };

    // Exercise both sides of the per-tile 256-thread sample-slot boundary.
    const SampleRun one_sample_tail = run(255);
    const SampleRun two_sample_tail = run(257);
    require(
        std::all_of(
            one_sample_tail.inside.begin(), one_sample_tail.inside.end(),
            [](const bool value) { return value; }) &&
        std::all_of(
            two_sample_tail.inside.begin(), two_sample_tail.inside.end(),
            [](const bool value) { return value; }),
        "GGGS sample-depth tail test did not intersect the Gaussian surface");
    for (std::size_t index = 0; index < 255 * 3; ++index) {
        require(
            std::abs(
                one_sample_tail.camera_points[index] -
                two_sample_tail.camera_points[index]) < 1e-6F,
            "GGGS sample-depth batch-boundary forward results differ");
    }
    // The loss above is averaged over the number of points, so remove that
    // known scale before comparing the two batch-boundary cases.
    for (std::size_t index = 0; index < 255 * 3; ++index) {
        require(
            std::abs(
                255.F * one_sample_tail.point_gradients[index] -
                257.F * two_sample_tail.point_gradients[index]) < 2e-5F,
            "GGGS sample-depth batch-boundary point gradients differ");
    }
}

void test_contribution_visibility_rejects_occluded_gaussians() {
    using namespace aetherscan::splat;
    constexpr std::size_t count = 4;
    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        std::vector<float>{
            0.F, 0.F, 1.F,
            0.F, 0.F, 1.1F,
            0.F, 0.F, 1.2F,
            0.F, 0.F, 2.F},
        {count, std::size_t{3}}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>(count * 3, std::log(10.F)),
        {count, std::size_t{3}}, tinytensor::Device::CUDA);
    std::vector<float> rotations(count * 4, 0.F);
    for (std::size_t index = 0; index < count; ++index)
        rotations[4 * index] = 1.F;
    model.quaternions = tinytensor::Tensor::from_vector(
        rotations, {count, std::size_t{4}}, tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>(count, 12.F), {count, std::size_t{1}},
        tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::zeros(
        {count, std::size_t{1}, std::size_t{3}},
        tinytensor::Device::CUDA);
    model.sh_degree = 0;

    Camera camera;
    camera.world_to_camera[0] = 1.F;
    camera.world_to_camera[5] = 1.F;
    camera.world_to_camera[10] = 1.F;
    camera.world_to_camera[15] = 1.F;
    camera.fx = camera.fy = 40.F;
    camera.cx = camera.cy = 15.5F;
    camera.width = camera.height = 32;

    const RenderResult rendered = Rasterizer().forward(model, camera);
    const auto radii = rendered.radii.to_vector();
    const auto visibility = rendered.visibility.to_vector();
    require(
        radii.back() > 0,
        "Occlusion test Gaussian did not project into the camera");
    require(
        visibility.front() == 1.F && visibility.back() == 0.F,
        "Projected-but-occluded Gaussian was incorrectly marked visible");
}

void test_alpha_parameter_gradients() {
    using namespace aetherscan::splat;
    const auto make_model = [](const float log_scale_x,
                               const float opacity_logit) {
        GaussianModel model;
        model.means = tinytensor::Tensor::from_vector(
            std::vector<float>{0.F, 0.F, 2.F}, {1, 3},
            tinytensor::Device::CUDA);
        model.log_scales = tinytensor::Tensor::from_vector(
            std::vector<float>{log_scale_x, std::log(0.15F),
                               std::log(0.15F)},
            {1, 3}, tinytensor::Device::CUDA);
        model.quaternions = tinytensor::Tensor::from_vector(
            std::vector<float>{1.F, 0.F, 0.F, 0.F}, {1, 4},
            tinytensor::Device::CUDA);
        model.opacity_logits = tinytensor::Tensor::from_vector(
            std::vector<float>{opacity_logit}, {1, 1},
            tinytensor::Device::CUDA);
        model.sh = tinytensor::Tensor::from_vector(
            std::vector<float>{0.5F, 0.25F, 0.1F}, {1, 1, 3},
            tinytensor::Device::CUDA);
        model.sh_degree = 0;
        return model;
    };
    Camera camera;
    camera.world_to_camera[0] = 1.F;
    camera.world_to_camera[5] = 1.F;
    camera.world_to_camera[10] = 1.F;
    camera.world_to_camera[15] = 1.F;
    camera.fx = 40.F;
    camera.fy = 40.F;
    camera.cx = 15.5F;
    camera.cy = 15.5F;
    camera.width = 32;
    camera.height = 32;
    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    const std::size_t sample_pixel =
        static_cast<std::size_t>(camera.cy) * camera.width +
        static_cast<std::size_t>(camera.cx) + 2;
    const auto sampled_alpha = [&](
                                   GaussianModel model,
                                   const RasterizeOptions& options) {
        const auto values =
            Rasterizer().forward(model, camera, options).alpha.to_vector();
        return values[sample_pixel];
    };

    constexpr float base_log_scale = -1.8971199849F;  // log(0.15)
    // Match training initialization and stay away from the rasterizer's
    // per-splat alpha=0.99 clamp, whose derivative is intentionally clipped.
    constexpr float base_opacity_logit = -2.19722458F;
    {
        Camera centered_camera = camera;
        centered_camera.cx = 16.F;
        centered_camera.cy = 16.F;
        RasterizeOptions classic_options;
        RasterizeOptions mip_options;
        mip_options.kernel_size = 0.1F;
        const GaussianModel centered_model =
            make_model(base_log_scale, base_opacity_logit);
        const RenderResult classic =
            Rasterizer().forward(centered_model, centered_camera, classic_options);
        const RenderResult mip =
            Rasterizer().forward(centered_model, centered_camera, mip_options);
        const std::size_t center_pixel =
            16U * centered_camera.width + 16U;
        const auto classic_alpha_image = classic.alpha.to_vector();
        const auto mip_alpha_image = mip.alpha.to_vector();
        const float classic_alpha = classic_alpha_image[center_pixel];
        const float mip_alpha = mip_alpha_image[center_pixel];
        // Projected std-dev is fx/z*scale = 3px on both axes, so the
        // determinant compensation is sqrt(9*9 / (9.1*9.1)) = 9/9.1.
        const float expected_mip_alpha = 0.1F * 9.F / 9.1F;
        require(
            std::abs(classic_alpha - 0.1F) < 1e-5F &&
                std::abs(mip_alpha - expected_mip_alpha) < 1e-5F,
            "GGGS screen-space opacity compensation is incorrect");
        const std::size_t tail_pixel = center_pixel + 6U;
        require(
            mip_alpha_image[tail_pixel] > classic_alpha_image[tail_pixel],
            "GGGS screen-space covariance low-pass did not expand the support");
    }
    const auto zero_color = tinytensor::Tensor::zeros(
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    std::vector<float> alpha_chain(pixels, 0.F);
    alpha_chain[sample_pixel] = 1.F;
    const auto grad_alpha = tinytensor::Tensor::from_vector(
        alpha_chain,
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    const auto zero_scalar = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    const auto zero_normal = tinytensor::Tensor::zeros(
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    constexpr float epsilon = 1e-3F;
    for (const float kernel_size : {0.F, 0.1F}) {
        RasterizeOptions options;
        options.kernel_size = kernel_size;
        GaussianModel model = make_model(base_log_scale, base_opacity_logit);
        Rasterizer rasterizer;
        const RenderResult rendered =
            rasterizer.forward(model, camera, options);
        const ModelGradients gradients = rasterizer.backward(
            model, rendered, zero_color, grad_alpha, zero_scalar, zero_normal);
        const float analytic_opacity =
            gradients.opacity_logits.to_vector()[0];
        const float analytic_scale = gradients.log_scales.to_vector()[0];

        const float numeric_opacity =
            (sampled_alpha(
                 make_model(
                     base_log_scale, base_opacity_logit + epsilon),
                 options) -
             sampled_alpha(
                 make_model(
                     base_log_scale, base_opacity_logit - epsilon),
                 options)) /
            (2.F * epsilon);
        const float numeric_scale =
            (sampled_alpha(
                 make_model(
                     base_log_scale + epsilon, base_opacity_logit),
                 options) -
             sampled_alpha(
                 make_model(
                     base_log_scale - epsilon, base_opacity_logit),
                 options)) /
            (2.F * epsilon);
        require(
            analytic_opacity > 0.F && numeric_opacity > 0.F &&
                std::isfinite(analytic_opacity) &&
                std::isfinite(numeric_opacity),
            "GGGS alpha-to-opacity-logit gradient is not a descent direction");
        require(
            analytic_scale > 0.F && numeric_scale > 0.F &&
                std::isfinite(analytic_scale) &&
                std::isfinite(numeric_scale),
            "GGGS alpha-to-log-scale gradient is not a descent direction");
        const auto relative_error = [](const float analytic,
                                       const float numeric) {
            return std::abs(analytic - numeric) /
                std::max(std::abs(numeric), 1e-6F);
        };
        require(
            relative_error(analytic_opacity, numeric_opacity) < 2e-2F,
            "GGGS opacity-compensation opacity gradient differs from finite "
            "differences");
        // Brush/Faster-GS intentionally detach Mip opacity compensation from
        // the covariance gradient. The unfiltered path still provides a
        // strict end-to-end finite-difference check for scale derivatives.
        if (kernel_size == 0.F)
            require(
                relative_error(analytic_scale, numeric_scale) < 2e-2F,
                "GGGS covariance scale gradient differs from finite "
                "differences");
    }
}

void test_adam_rejects_non_finite_gradients() {
    using namespace aetherscan::splat;
    auto parameter = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 2.F, 3.F}, {3}, tinytensor::Device::CUDA);
    const auto gradient = tinytensor::Tensor::from_vector(
        std::vector<float>{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            -std::numeric_limits<float>::infinity()},
        {3}, tinytensor::Device::CUDA);
    auto state = detail::make_adam_state(parameter);
    detail::adam_step(
        parameter, gradient, state, 1e-3F, 1, TrainingOptions{});
    require_finite(parameter, "Adam accepted a non-finite gradient");
    require_finite(state.first, "Adam first moment became non-finite");
    require_finite(state.second, "Adam second moment became non-finite");
    const auto values = parameter.to_vector();
    require(
        values == std::vector<float>({1.F, 2.F, 3.F}),
        "Adam changed parameters for rejected gradients");
}

void test_fused_adam_parity() {
    using namespace aetherscan::splat;
    auto parameter = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, -2.F}, {2}, tinytensor::Device::CUDA);
    const auto gradient = tinytensor::Tensor::from_vector(
        std::vector<float>{0.25F, -0.5F}, {2}, tinytensor::Device::CUDA);
    auto state = detail::make_adam_state(parameter);
    TrainingOptions options;
    options.adam_epsilon = 1e-15F;
    detail::adam_step(parameter, gradient, state, 1e-3F, 1, options);
    const auto values = parameter.to_vector();
    const auto first = state.first.to_vector();
    const auto second = state.second.to_vector();
    require(
        std::abs(values[0] - 0.999F) < 1e-6F &&
            std::abs(values[1] + 1.999F) < 1e-6F &&
            std::abs(first[0] - 0.025F) < 1e-7F &&
            std::abs(first[1] + 0.05F) < 1e-7F &&
            std::abs(second[0] - 0.0000625F) < 1e-8F &&
            std::abs(second[1] - 0.00025F) < 1e-8F,
        "CUDA Adam differs from FasterGS FusedAdam on its first step");
}

void test_reduced_second_sh_adam() {
    using namespace aetherscan::splat;
    auto parameter = tinytensor::Tensor::zeros(
        {2, 2, 3}, tinytensor::Device::CUDA);
    const std::vector<float> gradient_values{
        1.F, 2.F, 3.F, 4.F, 5.F, 6.F,
        2.F, 2.F, 2.F, 2.F, 2.F, 2.F};
    const auto gradient = tinytensor::Tensor::from_vector(
        gradient_values, {2, 2, 3}, tinytensor::Device::CUDA);
    auto state = detail::make_reduced_second_adam_state(parameter);
    require(
        state.first.numel() == parameter.numel() &&
            state.second.numel() == 2,
        "reduced-second Adam did not compact its second moment per row");

    TrainingOptions options;
    options.adam_epsilon = 1e-15F;
    constexpr float primary_lr = 1e-3F;
    constexpr float secondary_lr = 1e-4F;
    detail::adam_step_reduced_second(
        parameter, gradient, state, primary_lr, 1, options, 6,
        secondary_lr);

    const auto values = parameter.to_vector();
    const auto first = state.first.to_vector();
    const auto second = state.second.to_vector();
    const std::array<float, 2> square_means{91.F / 6.F, 4.F};
    for (std::size_t row = 0; row < 2; ++row) {
        require(
            std::abs(
                second[row] -
                (1.F - options.beta2) * square_means[row]) < 1e-7F,
            "reduced-second Adam stored the wrong row moment");
        for (std::size_t column = 0; column < 6; ++column) {
            const std::size_t index = row * 6 + column;
            const float lr = column < 3 ? primary_lr : secondary_lr;
            const float expected =
                -lr * gradient_values[index] /
                std::sqrt(square_means[row]);
            require(
                std::abs(values[index] - expected) < 1e-6F &&
                    std::abs(
                        first[index] -
                        (1.F - options.beta1) *
                            gradient_values[index]) < 1e-6F,
                "reduced-second Adam differs from row-reduced AdamScaled");
        }
    }

    auto prefix_parameter = tinytensor::Tensor::zeros(
        {2, 4, 3}, tinytensor::Device::CUDA);
    const auto prefix_gradient = tinytensor::Tensor::from_vector(
        std::vector<float>(24, 0.25F), {2, 4, 3},
        tinytensor::Device::CUDA);
    auto prefix_state =
        detail::make_reduced_second_adam_state(prefix_parameter);
    detail::adam_step_active_prefix(
        prefix_parameter, prefix_gradient, prefix_state, primary_lr, 1,
        options, 12, 3, secondary_lr);
    const auto prefix_values = prefix_parameter.to_vector();
    const auto prefix_first = prefix_state.first.to_vector();
    const auto prefix_second = prefix_state.second.to_vector();
    require(
        prefix_second.size() == 2,
        "active-prefix Adam expanded the compact second moment");
    for (std::size_t index = 0; index < prefix_values.size(); ++index) {
        const bool active = index < 3 || (index >= 12 && index < 15);
        require(
            active
                ? std::abs(prefix_values[index] + primary_lr) < 1e-6F &&
                      std::abs(prefix_first[index] - 0.025F) < 1e-6F
                : prefix_values[index] == 0.F &&
                      prefix_first[index] == 0.F,
            "active-prefix reduced-second Adam touched the wrong SH band");
    }
}

void test_active_sh_prefix_adam() {
    using namespace aetherscan::splat;
    std::vector<float> initial(24);
    std::iota(initial.begin(), initial.end(), 1.F);
    auto parameter = tinytensor::Tensor::from_vector(
        initial, {2, 4, 3}, tinytensor::Device::CUDA);
    const auto gradient = tinytensor::Tensor::from_vector(
        std::vector<float>(24, 0.25F), {2, 4, 3},
        tinytensor::Device::CUDA);
    auto state = detail::make_adam_state(parameter);
    TrainingOptions options;
    options.adam_epsilon = 1e-15F;
    detail::adam_step_active_prefix(
        parameter, gradient, state, 1e-3F, 1, options, 12, 3, 1e-4F);

    const auto values = parameter.to_vector();
    const auto first = state.first.to_vector();
    const auto second = state.second.to_vector();
    for (std::size_t index = 0; index < values.size(); ++index) {
        const bool active = index < 3 || (index >= 12 && index < 15);
        if (active) {
            require(
                std::abs(values[index] - (initial[index] - 1e-3F)) < 1e-6F &&
                    std::abs(first[index] - 0.025F) < 1e-7F &&
                    std::abs(second[index] - 0.0000625F) < 1e-8F,
                "active-prefix Adam did not update an active SH coefficient");
        } else {
            require(
                values[index] == initial[index] && first[index] == 0.F &&
                    second[index] == 0.F,
                "active-prefix Adam touched an inactive SH coefficient");
        }
    }
}

void test_mask_loading() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_splat_mask_test";
    const auto masks = root / "masks";
    std::filesystem::create_directories(masks);
    io::RgbImage source{2, 2, std::vector<std::uint8_t>(12, 128)};
    io::RgbImage mask{2, 2, {
        255, 255, 255, 0, 0, 0,
        0, 0, 0, 255, 255, 255}};
    const auto image_path = root / "frame.png";
    io::save_rgb_png(source, image_path);
    io::save_rgb_png(mask, masks / "frame.png");
    mvs::MvsView view;
    view.path = image_path;
    view.width = view.src_width = 2;
    view.height = view.src_height = 2;
    view.fx = view.fy = view.src_fx = view.src_fy = 1.F;
    view.cx = view.cy = view.src_cx = view.src_cy = 0.5F;
    view.foreground_mask = {255, 255, 255, 0};
    splat::TrainingOptions options;
    options.use_mask = true;
    options.mask_dir = masks;
    const splat::TrainingView training =
        splat::make_training_view(view, options);
    require(training.has_mask, "GGGS did not load the matching mask file");
    require(
        training.mask.to_vector() == std::vector<float>({1.F, 0.F, 0.F, 0.F}),
        "GGGS did not intersect the input and coarse-mesh masks");
    view.foreground_mask = {1, 1, 1, 0};
    const splat::TrainingView binary_training =
        splat::make_training_view(view, options);
    require(
        binary_training.mask.to_vector() ==
            std::vector<float>({1.F, 0.F, 0.F, 0.F}),
        "GGGS interpreted a binary MVS mask as 8-bit fractional coverage");
    io::RgbImage soft_mask{2, 2, {
        128, 128, 128, 0, 0, 0,
        0, 0, 0, 255, 255, 255}};
    io::save_rgb_png(soft_mask, masks / "frame.png");
    view.foreground_mask.clear();
    const splat::TrainingView soft_training =
        splat::make_training_view(view, options);
    const auto soft_values = soft_training.mask.to_vector();
    require(
        soft_values.size() == 4 &&
            std::abs(soft_values[0] - 128.F / 255.F) < 1e-6F &&
            soft_values[1] == 0.F && soft_values[2] == 0.F &&
            soft_values[3] == 1.F,
        "GGGS discarded grayscale coverage from an aether_drender mesh mask");
    std::filesystem::remove_all(root);
}

void test_training_device_cache() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
        "aetherscan_training_device_cache_test";
    std::filesystem::create_directories(root);
    std::vector<mvs::MvsView> views(2);
    for (std::size_t i = 0; i < views.size(); ++i) {
        auto& view = views[i];
        view.path = root / (std::to_string(i) + ".png");
        io::save_rgb_png(
            io::RgbImage{16, 16, std::vector<std::uint8_t>(
                16 * 16 * 3, static_cast<std::uint8_t>(60 + 90 * i))},
            view.path);
        view.width = view.src_width = view.height = view.src_height = 16;
        view.fx = view.fy = view.src_fx = view.src_fy = 12.F;
        view.cx = view.cy = view.src_cx = view.src_cy = 7.5F;
        view.foreground_mask.assign(256, 255);
        view.foreground_mask[0] = 0;
        view.depth_map.depth.assign(256, 2.F + static_cast<float>(i));
        view.depth_map.normal.assign(256, mvs::Vec3f(0.F, 0.F, 1.F));
    }
    splat::TrainingOptions options;
    options.use_mask = options.use_mvs_depth = options.use_mvs_normals = true;
    options.multi_view_ncc_weight = 0.6F;
    options.training_prefetch_views = 0;
    options.training_view_cache_bytes = 0;
    options.adaptive_training_cache = false;
    // Exactly one RGBA8 + depth + normal frame; the next view must evict it.
    options.training_device_cache_bytes = 256 * (4 + 4 + 12);
    splat::training_data::TrainingDataLoader cache(views, options);
    const auto reference = splat::make_training_view(views[0], options);
    const auto first = cache.get(0);
    const auto hit = cache.get(0);
    const auto compare = [](const splat::TrainingView& a,
                            const splat::TrainingView& b,
                            const char* stage) {
        require(a.rgb.to_vector() == b.rgb.to_vector() &&
                a.gray.to_vector() == b.gray.to_vector() &&
                a.mask.to_vector() == b.mask.to_vector() &&
                a.depth.to_vector() == b.depth.to_vector() &&
                a.normal.to_vector() == b.normal.to_vector() &&
                a.has_mask == b.has_mask,
                stage);
    };
    compare(reference, first, "first changed supervision");
    compare(reference, hit, "device hit changed supervision");
    require(cache.stats().device_hits == 1 &&
            cache.stats().uploaded_bytes == options.training_device_cache_bytes,
            "Training cache hit uploaded the frame again");
    const auto neighbour = cache.get(1);
    compare(first, reference,
            "first changed after LRU eviction");  // Tensors must survive eviction.
    compare(neighbour, splat::make_training_view(views[1], options),
            "neighbour changed supervision");
    compare(cache.get(0), reference, "reload changed supervision");
    require(cache.stats().device_hits == 1 &&
            cache.stats().device_resident_bytes <= options.training_device_cache_bytes,
            "Training device LRU did not enforce its budget");
    cache.set_resolution_scale(0.5F);
    require(cache.stats().device_resident_bytes == 0,
            "Resolution transition retained stale CUDA views");
    const auto resized = cache.get(0);
    require(resized.camera.width == 8 && resized.camera.height == 8 &&
            resized.rgb.numel() == 3 * 8 * 8 &&
            cache.stats().device_hits == 1,
            "Resolution transition reused old-size CUDA supervision");
    options.adaptive_training_cache = true;
    options.training_device_cache_bytes = 1;
    splat::training_data::TrainingDataLoader adaptive(views, options);
    require(adaptive.stats().device_budget_bytes ==
                2 * 256 * (4 + 4 + 12),
            "Adaptive CUDA cache did not cover this small dataset");
    const auto adaptive_first = adaptive.get(0);
    const auto adaptive_hit = adaptive.get(0);
    const auto adaptive_second = adaptive.get(1);
    require(adaptive_first.rgb.numel() == 3 * 16 * 16 &&
                    adaptive_hit.rgb.numel() == 3 * 16 * 16 &&
                    adaptive_second.rgb.numel() == 3 * 16 * 16 &&
                    adaptive.stats().device_hits == 1,
            "Adaptive CUDA cache did not retain the packed views");
    for (const std::size_t budget : {std::size_t{0}, std::size_t{1}}) {
        options.adaptive_training_cache = false;
        options.training_device_cache_bytes = budget;
        splat::training_data::TrainingDataLoader uncached(views, options);
        compare(uncached.get(0), reference,
                "uncached first changed supervision");
        compare(uncached.get(0), reference,
                "uncached second changed supervision");
        require(uncached.stats().device_hits == 0 &&
                uncached.stats().device_resident_bytes == 0,
                "Disabled/undersized CUDA cache retained an oversized view");
    }

    options.adaptive_training_cache = false;
    options.training_prefetch_views = 1;
    options.training_view_cache_bytes = 64 * 1024;
    options.training_device_cache_bytes = 256 * (4 + 4 + 12);
    splat::training_data::TrainingDataLoader asynchronous(views, options);
    const auto async_first = asynchronous.get(0);
    const auto async_neighbour = asynchronous.get(1);
    asynchronous.prefetch(0);
    const auto async_prefetched = asynchronous.get(0);
    compare(async_first, reference,
            "asynchronous first changed supervision");
    compare(async_prefetched, reference,
            "asynchronous prefetch changed supervision");
    compare(async_neighbour, splat::make_training_view(views[1], options),
            "asynchronous neighbour changed supervision");
    require(asynchronous.stats().device_prefetch_hits == 1 &&
                    asynchronous.stats().device_prefetch_pending == 0 &&
                    asynchronous.stats().device_prefetch_bytes == 0 &&
                    asynchronous.stats().device_resident_bytes <=
                        options.training_device_cache_bytes,
            "Asynchronous packed CUDA prefetch did not preserve the view");
    asynchronous.set_resolution_scale(0.5F);
    require(asynchronous.stats().device_prefetch_pending == 0 &&
                    asynchronous.stats().device_prefetch_bytes == 0 &&
                    asynchronous.stats().device_resident_bytes == 0,
            "Resolution transition retained asynchronous CUDA work");
    std::filesystem::remove_all(root);
}

void test_brush_quantile_selection() {
    using namespace aetherscan;
    for (const std::size_t count : {1U, 2U, 11U, 257U, 100003U}) {
        std::vector<float> xyz(count * 3);
        for (std::size_t i = 0; i < xyz.size(); ++i)
            xyz[i] = i % 37 == 0 ? 0.F
                : static_cast<float>((i * 73 + 19) % (count * 3)) -
                    static_cast<float>(count);
        // Include ties and independently missing coordinates, as the original
        // full-sort implementation filters non-finite values per axis.
        if (count > 2) xyz[4] = std::numeric_limits<float>::quiet_NaN();
        for (const float p : {0.F, 0.5F, 0.8F, 0.95F, 1.F}) {
            mvs::Vec3f low, high;
            for (std::size_t axis = 0; axis < 3; ++axis) {
                std::vector<float> sorted;
                for (std::size_t i = 0; i < count; ++i)
                    if (std::isfinite(xyz[3 * i + axis]))
                        sorted.push_back(xyz[3 * i + axis]);
                std::sort(sorted.begin(), sorted.end());
                low[axis] = sorted[static_cast<std::size_t>(
                    (1.F - p) * 0.5F * static_cast<float>(sorted.size()))];
                high[axis] = sorted[std::min(sorted.size() - 1,
                    static_cast<std::size_t>(
                        (1.F + p) * 0.5F * static_cast<float>(sorted.size())))];
            }
            const auto actual = splat::densification::brush_scene_geometry(xyz, p);
            const auto device = splat::densification::brush_scene_geometry_cuda(
                tinytensor::Tensor::from_vector(
                    xyz, {count, 3}, tinytensor::Device::CUDA), p);
            require(actual.center == device.center && actual.scale == device.scale &&
                    actual.maximum_extent == device.maximum_extent,
                    "GPU ADC quantiles differ from finite CPU order statistics");
            const mvs::Vec3f half = 0.5F * (high - low);
            std::array<float, 3> extents{half.x(), half.y(), half.z()};
            std::sort(extents.begin(), extents.end());
            require(actual.center == 0.5F * (low + high) &&
                    actual.scale == 2.F * extents[1] &&
                    actual.maximum_extent == extents[2],
                    "ADC quantile selection differs from full-sort reference");
        }
    }
    for (const std::vector<float>& xyz : {
             std::vector<float>{},
             std::vector<float>{std::numeric_limits<float>::infinity(), 1.F, 2.F}}) {
        const auto actual = splat::densification::brush_scene_geometry_cuda(
            tinytensor::Tensor::from_vector(
                xyz, {xyz.size() / 3, 3}, tinytensor::Device::CUDA));
        require(actual.center.isZero() && actual.scale == 2.F &&
                actual.maximum_extent == 1.F,
                "GPU ADC quantiles lost the empty/non-finite fallback");
    }
}

void test_source_resolution_and_knn_initialization() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_splat_source_resolution_test";
    std::filesystem::create_directories(root);
    io::RgbImage source;
    source.width = source.height = 4;
    source.pixels.resize(4 * 4 * 3);
    std::iota(source.pixels.begin(), source.pixels.end(), std::uint8_t{0});
    const auto image_path = root / "frame.png";
    io::save_rgb_png(source, image_path);
    const io::RgbImage loaded_source = io::load_rgb(image_path);

    mvs::MvsView view;
    view.path = image_path;
    view.width = view.height = 2;
    view.fx = view.fy = 2.F;
    view.cx = view.cy = 0.5F;
    view.src_width = view.src_height = 4;
    view.src_fx = view.src_fy = 4.F;
    view.src_cx = view.src_cy = 1.5F;
    view.k1 = 0.1F;
    view.foreground_mask = {255, 0, 0, 255};
    splat::TrainingOptions options;
    options.use_source_resolution = true;
    const auto training = splat::make_training_view(view, options);
    require(
        training.camera.width == 4 && training.camera.height == 4 &&
            std::abs(training.camera.fx - 4.F) < 1e-6F,
        "GGGS training did not restore source-resolution intrinsics");
    const auto projected_mask = training.mask.to_vector();
    require(
        training.has_mask && projected_mask.size() == 16 &&
            projected_mask[0] > 0.5F && projected_mask[15] > 0.5F &&
            projected_mask[3] < 0.1F && projected_mask[12] < 0.1F &&
            std::any_of(
                projected_mask.begin(), projected_mask.end(),
                [](const float coverage) {
                    return coverage > 0.F && coverage < 1.F;
                }),
        "GGGS did not softly reproject the coarse MVS mask to source resolution");
    const auto rgb = training.rgb.to_vector();
    const float xn = (2.F - view.src_cx) / view.src_fx;
    const float yn = (1.F - view.src_cy) / view.src_fy;
    const float radial = 1.F + view.k1 * (xn * xn + yn * yn);
    const float sx = view.src_fx * xn * radial + view.src_cx;
    const float sy = view.src_fy * yn * radial + view.src_cy;
    const int x0 = static_cast<int>(sx);
    const int y0 = static_cast<int>(sy);
    const int x1 = std::min(x0 + 1, 3);
    const int y1 = std::min(y0 + 1, 3);
    const float tx = sx - x0;
    const float ty = sy - y0;
    const auto red = [&](const int x, const int y) {
        return loaded_source.pixels[
                   (static_cast<std::size_t>(y) * 4 + x) * 3] /
               255.F;
    };
    const float expected_red =
        (red(x0, y0) * (1.F - tx) + red(x1, y0) * tx) * (1.F - ty) +
        (red(x0, y1) * (1.F - tx) + red(x1, y1) * tx) * ty;
    // Training targets are cached as packed RGBA8, matching Brush's
    // GPU-ready representation. Brown resampling therefore has one final
    // 8-bit quantization step after bilinear interpolation.
    if (std::abs(rgb[6] - expected_red) > 0.5F / 255.F + 1e-6F)
        throw std::runtime_error(
            "GGGS packed source-resolution Brown mapping differs: actual=" +
            std::to_string(rgb[6]) +
            " expected=" + std::to_string(expected_red));

    options.max_image_dimension = 2;
    const auto resized = splat::make_training_view(view, options);
    require(
        resized.camera.width == 2 && resized.camera.height == 2 &&
            std::abs(resized.camera.fx - 2.F) < 1e-6F &&
            resized.rgb.numel() == 12,
        "GGGS max image dimension did not resize the camera and target");

    mvs::MvsScene scene;
    for (const mvs::Vec3f& position : {
             mvs::Vec3f(0.F, 0.F, 0.F),
             mvs::Vec3f(1.F, 0.F, 0.F),
             mvs::Vec3f(0.F, 1.F, 0.F),
             mvs::Vec3f(0.F, 0.F, 1.F)}) {
        mvs::DensePoint point;
        point.position = position;
        point.color = mvs::Vec3f::Constant(0.5F);
        scene.dense_cloud.points.push_back(point);
    }
    options.max_gaussians = 0;
    options.sh_degree = 0;
    options.initialize_scale_from_knn = true;
    options.constrain_scale_range = false;
    const auto model = splat::initialize_from_dense_cloud(scene, options);
    const auto scales = model.log_scales.to_vector();
    const float expected_offset_scale = std::sqrt(5.F / 3.F);
    require(
        std::abs(std::exp(scales[0]) - 1.F) < 1e-5F &&
            std::abs(std::exp(scales[1]) - 1.F) < 1e-5F &&
            std::abs(std::exp(scales[2]) - 1.F) < 1e-5F &&
            std::abs(std::exp(scales[3]) - expected_offset_scale) < 1e-5F &&
            std::abs(std::exp(scales[6]) - expected_offset_scale) < 1e-5F &&
            std::abs(std::exp(scales[9]) - expected_offset_scale) < 1e-5F,
        "GGGS KNN initialization does not match three-neighbour RMS scale");
    options.densification_strategy =
        splat::DensificationStrategy::adc_plus;
    const auto brush_model =
        splat::initialize_from_dense_cloud(scene, options);
    const auto brush_scales = brush_model.log_scales.to_vector();
    const auto brush_rotations = brush_model.quaternions.to_vector();
    require(
        std::all_of(
            brush_scales.begin(), brush_scales.end(),
            [](const float log_scale) {
                return std::abs(std::exp(log_scale) - 0.1F) < 1e-5F;
            }) &&
            brush_rotations[0] == 1.F && brush_rotations[1] == 0.F &&
            brush_rotations[2] == 0.F && brush_rotations[3] == 0.F &&
            std::abs(brush_model.opacity_logits.to_vector()[0]) < 1e-6F,
        "ADC+ sparse initialization does not match brush");
    options.densification_strategy =
        splat::DensificationStrategy::default_strategy;
    options.constrain_scale_range = true;
    options.minimum_scale_fraction = 1e-4F;
    options.maximum_scale_fraction = 0.1F;
    const auto clamped_model =
        splat::initialize_from_dense_cloud(scene, options);
    const auto clamped_scales = clamped_model.log_scales.to_vector();
    const float maximum_scale = std::sqrt(3.F) * 0.1F;
    require(
        std::all_of(
            clamped_scales.begin(), clamped_scales.end(),
            [maximum_scale](const float log_scale) {
                return std::exp(log_scale) <= maximum_scale + 1e-5F;
            }),
        "GGGS KNN initialization ignored the configured maximum scale");
    mvs::MvsView left_camera;
    left_camera.pose.C = sfm::Vec3(-1.0, 0.0, 0.0);
    mvs::MvsView right_camera;
    right_camera.pose.C = sfm::Vec3(1.0, 0.0, 0.0);
    scene.views = {left_camera, right_camera};
    options.input_is_dense = false;
    const auto sparse_model =
        splat::initialize_from_dense_cloud(scene, options);
    const auto sparse_scales = sparse_model.log_scales.to_vector();
    const float sparse_maximum_scale = 1.1F * 0.1F;
    const float observed_sparse_maximum = std::exp(
        *std::max_element(sparse_scales.begin(), sparse_scales.end()));
    require(
        std::all_of(
            sparse_scales.begin(), sparse_scales.end(),
            [sparse_maximum_scale](const float log_scale) {
                return std::exp(log_scale) <= sparse_maximum_scale + 1e-5F;
            }) &&
            std::abs(observed_sparse_maximum - sparse_maximum_scale) < 1e-5F,
        "Sparse GGGS scale bounds did not use the camera-based scene scale");
    std::filesystem::remove_all(root);
}

void test_colmap_text_loading() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_colmap_splat_test";
    const auto model = root / "sparse" / "0";
    const auto images = root / "images";
    std::filesystem::create_directories(model);
    std::filesystem::create_directories(images);
    io::save_rgb_png(
        io::RgbImage{4, 3, std::vector<std::uint8_t>(36, 127)},
        images / "frame.png");
    {
        std::ofstream stream(model / "cameras.txt");
        stream << "1 OPENCV 4 3 100 101 2 1.5 0.01 -0.02 0.001 -0.002\n";
    }
    {
        std::ofstream stream(model / "images.txt");
        stream << "7 1 0 0 0 1 2 3 1 frame.png\n";
        stream << "2 1 42\n";
    }
    {
        std::ofstream stream(model / "points3D.txt");
        stream << "42 0.1 0.2 4 10 20 30 0.5 7 0\n";
    }
    const auto loaded = splat::load_colmap_scene(root, images);
    require(!loaded.binary, "COLMAP text model was reported as binary");
    require(loaded.scene.views.size() == 1, "COLMAP camera pose was not loaded");
    require(
        loaded.scene.dense_cloud.points.size() == 1 &&
            loaded.scene.sparse_points.size() == 1,
        "COLMAP sparse point was not loaded");
    const auto& view = loaded.scene.views.front();
    require(
        std::abs(view.fx - 100.F) < 1e-6F &&
            std::abs(view.fy - 101.F) < 1e-6F &&
            std::abs(view.k1 - 0.01F) < 1e-6F,
        "COLMAP intrinsics were decoded incorrectly");
    require(
        (view.pose.C - Eigen::Vector3d(-1, -2, -3)).norm() < 1e-9,
        "COLMAP world-to-camera translation was decoded incorrectly");
    const auto& point = loaded.scene.dense_cloud.points.front();
    require(
        point.views == std::vector<mvs::Index>{0} &&
            (point.position - mvs::Vec3f(0.1F, 0.2F, 4.F)).norm() < 1e-6F,
        "COLMAP track mapping or sparse position is wrong");

    const auto images_2 = root / "images_2";
    std::filesystem::create_directories(images_2);
    io::save_rgb_png(
        io::RgbImage{2, 2, std::vector<std::uint8_t>(12, 127)},
        images_2 / "frame.png");
    const auto half_loaded = splat::load_colmap_scene(root, images_2);
    const auto& half_view = half_loaded.scene.views.front();
    require(
        half_view.width == 2 && half_view.height == 2 &&
            half_view.src_width == 2 && half_view.src_height == 2 &&
            std::abs(half_view.fx - 50.F) < 1e-5F &&
            std::abs(half_view.fy - 101.F * (2.F / 3.F)) < 1e-5F &&
            std::abs(half_view.cx - 0.75F) < 1e-5F &&
            std::abs(half_view.cy - (5.F / 6.F)) < 1e-5F,
        "COLMAP intrinsics did not follow the selected image resolution");

    {
        std::ofstream stream(model / "cameras.bin", std::ios::binary);
        write_binary<std::uint64_t>(stream, 1);
        write_binary<std::uint32_t>(stream, 1);
        write_binary<std::int32_t>(stream, 1);  // PINHOLE
        write_binary<std::uint64_t>(stream, 4);
        write_binary<std::uint64_t>(stream, 3);
        for (const double value : {100.0, 101.0, 2.0, 1.5})
            write_binary(stream, value);
    }
    {
        std::ofstream stream(model / "images.bin", std::ios::binary);
        write_binary<std::uint64_t>(stream, 1);
        write_binary<std::uint32_t>(stream, 7);
        for (const double value : {1.0, 0.0, 0.0, 0.0})
            write_binary(stream, value);
        for (const double value : {1.0, 2.0, 3.0})
            write_binary(stream, value);
        write_binary<std::uint32_t>(stream, 1);
        stream.write("frame.png", 10);
        write_binary<std::uint64_t>(stream, 0);
    }
    {
        std::ofstream stream(model / "points3D.bin", std::ios::binary);
        write_binary<std::uint64_t>(stream, 1);
        write_binary<std::uint64_t>(stream, 42);
        for (const double value : {0.1, 0.2, 4.0})
            write_binary(stream, value);
        for (const std::uint8_t value : {10, 20, 30})
            write_binary(stream, value);
        write_binary(stream, 0.5);
        write_binary<std::uint64_t>(stream, 1);
        write_binary<std::uint32_t>(stream, 7);
        write_binary<std::uint32_t>(stream, 0);
    }
    const auto binary_loaded = splat::load_colmap_scene(root, images);
    require(
        binary_loaded.binary && binary_loaded.scene.views.size() == 1 &&
            binary_loaded.scene.dense_cloud.points.size() == 1,
        "COLMAP binary model was not loaded");
    std::filesystem::remove_all(root);
}

void test_reality_capture_dataset_loading() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_reality_capture_splat_test";
    const auto images = root / "images";
    std::filesystem::create_directories(images);
    for (const char* name : {"left.png", "right.png"})
        io::save_rgb_png(
            io::RgbImage{8, 6, std::vector<std::uint8_t>(8 * 6 * 3, 127)},
            images / name);
    const auto csv = root / "cameras.csv";
    {
        std::ofstream stream(csv);
        stream << "#name,x,y,alt,heading,pitch,roll,f,px,py,k1,k2,t1,t2\n";
        stream << "left.png,1,2,3,0,0,0,18,0,0,0.01,-0.02,0.001,-0.002\n";
        stream << "right.png,2,2,3,0,0,0,18,0,0,0,0,0,0\n";
    }
    splat::DatasetLoadRequest request;
    request.source = root;
    request.image_directory = images;
    request.random_initial_point_count = 16;
    const auto loaded = splat::load_splat_dataset(request);
    require(
        loaded.format == splat::DatasetFormat::reality_capture &&
            loaded.scene.views.size() == 2 &&
            loaded.generated_initial_points &&
            loaded.scene.dense_cloud.points.size() == 16,
        "RealityCapture dataset was not detected or initialized");
    const auto& view = loaded.scene.views.front();
    require(
        std::abs(view.fx - 4.F) < 1e-6F &&
            std::abs(view.cx - 4.F) < 1e-6F &&
            std::abs(view.k1 - 0.01F) < 1e-6F &&
            (view.pose.C - Eigen::Vector3d(1, 2, 3)).norm() < 1e-9,
        "RealityCapture intrinsics or pose conversion is incorrect");
    const Eigen::Vector3d forward_world =
        view.pose.R.transpose() * Eigen::Vector3d::UnitZ();
    require(
        (forward_world - Eigen::Vector3d(0, 0, -1)).norm() < 1e-9,
        "RealityCapture OpenGL-to-pinhole basis conversion is incorrect");
    std::filesystem::remove_all(root);
}

void test_sparse_init_keeps_photo_colors() {
    using namespace aetherscan;
    mvs::MvsScene scene;
    mvs::SparsePoint sparse;
    sparse.position = mvs::Vec3f(0.F, 0.F, 1.F);
    sparse.color = mvs::Vec3f(0.8F, 0.1F, 0.25F);
    sparse.view_ids = {0, 1};
    scene.sparse_points.push_back(sparse);
    splat::initialize_scene_from_sparse_points(scene);
    require(
        scene.dense_cloud.points.size() == 1 &&
            (scene.dense_cloud.points[0].color - sparse.color).norm() < 1e-6F,
        "SfM sparse colours were not copied into the Gaussian seed cloud");
}

void test_openmvs_dataset_loading() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_openmvs_splat_test";
    std::filesystem::create_directories(root);
    const auto first_path = root / "first.png";
    const auto second_path = root / "second.png";
    io::save_rgb_png(
        io::RgbImage{8, 6, std::vector<std::uint8_t>(8 * 6 * 3, 64)},
        first_path);
    io::save_rgb_png(
        io::RgbImage{8, 6, std::vector<std::uint8_t>(8 * 6 * 3, 192)},
        second_path);

    sfm::Scene source;
    sfm::PinholeCamera camera;
    camera.id = 0;
    camera.width = 8;
    camera.height = 6;
    camera.fx = 7;
    camera.fy = 7.5;
    camera.cx = 4;
    camera.cy = 3;
    source.cameras.push_back(camera);
    for (std::uint32_t index = 0; index < 2; ++index) {
        sfm::Image image;
        image.id = index;
        image.camera_id = 0;
        image.path = index == 0 ? first_path : second_path;
        image.registered = true;
        image.pose.C = Eigen::Vector3d(index, 0, 0);
        image.features.keypoints.push_back({4.F, 3.F});
        source.images.push_back(std::move(image));
    }
    sfm::Track track;
    track.position = Eigen::Vector3d(0.5, 0, 4);
    track.observations = {{0, 0}, {1, 0}};
    track.num_inliers = 2;
    source.tracks.push_back(track);
    const auto mvs_path = root / "scene.mvs";
    sfm::ExportMvsOptions export_options;
    export_options.image_path_base = root;
    export_options.sample_colors = false;
    sfm::export_openmvs_interface(source, mvs_path, export_options);

    splat::DatasetLoadRequest request;
    request.source = mvs_path;
    request.image_directory = root;
    const auto loaded = splat::load_splat_dataset(request);
    require(
        loaded.format == splat::DatasetFormat::openmvs &&
            loaded.scene.views.size() == 2 &&
            loaded.scene.dense_cloud.points.size() == 1 &&
            !loaded.generated_initial_points,
        "OpenMVS interface dataset was not loaded");
    require(
        std::abs(loaded.scene.views[0].fx - 7.F) < 1e-6F &&
            (loaded.scene.views[1].pose.C -
             Eigen::Vector3d(1, 0, 0)).norm() < 1e-9 &&
            (loaded.scene.dense_cloud.points[0].position -
             mvs::Vec3f(0.5F, 0.F, 4.F)).norm() < 1e-6F,
        "OpenMVS intrinsics, pose, or initial point conversion is incorrect");
    require(
        loaded.scene.sparse_points.size() == 1 &&
            loaded.scene.sparse_points[0].view_ids ==
                std::vector<mvs::Index>{0, 1},
        "OpenMVS vertex observations were not read in archive order");
    std::filesystem::remove_all(root);
}

void test_mask_loss_modes() {
    using namespace aetherscan::splat;
    RenderResult rendered;
    rendered.color = tinytensor::Tensor::from_vector(
        std::vector<float>(6, 0.8F), {3, 1, 2}, tinytensor::Device::CUDA);
    rendered.alpha = tinytensor::Tensor::from_vector(
        std::vector<float>{0.8F, 0.8F}, {1, 2}, tinytensor::Device::CUDA);
    rendered.median_depth = tinytensor::Tensor::zeros(
        {1, 2}, tinytensor::Device::CUDA);
    rendered.normal = tinytensor::Tensor::zeros(
        {3, 1, 2}, tinytensor::Device::CUDA);
    TrainingView target;
    target.camera.width = 2;
    target.camera.height = 1;
    target.rgb = tinytensor::Tensor::from_vector(
        std::vector<float>(6, 0.2F), {3, 1, 2}, tinytensor::Device::CUDA);
    target.depth = tinytensor::Tensor::zeros({1, 2}, tinytensor::Device::CUDA);
    target.normal = tinytensor::Tensor::zeros(
        {3, 1, 2}, tinytensor::Device::CUDA);
    target.mask = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F}, {1, 2}, tinytensor::Device::CUDA);
    target.has_mask = true;
    TrainingOptions options;
    options.use_mask = true;
    options.use_mvs_depth = false;
    options.use_mvs_normals = false;

    options.alpha_mode = AlphaMode::masked;
    auto loss = detail::compute_training_loss(rendered, target, options, true);
    auto rgb_gradient = loss.color.to_vector();
    auto alpha_gradient = loss.alpha.to_vector();
    require(rgb_gradient[0] != 0.F && rgb_gradient[1] == 0.F,
            "masked mode did not restrict RGB supervision to foreground");
    require(std::abs(alpha_gradient[0]) < 1e-6F &&
                std::abs(alpha_gradient[1] - 0.5F) < 1e-6F,
            "masked mode background-alpha penalty differs from pygsplat");
    require(std::abs(loss.alpha_value - 0.4F) < 1e-6F,
            "masked mode reported the wrong alpha loss");

    options.alpha_mode = AlphaMode::transparent;
    options.match_alpha_weight = 0.25F;
    loss = detail::compute_training_loss(rendered, target, options, true);
    alpha_gradient = loss.alpha.to_vector();
    require(std::abs(alpha_gradient[0] + 0.15625F) < 1e-5F &&
                std::abs(alpha_gradient[1] - 0.625F) < 1e-5F,
            "transparent mode BCE gradient differs from pygsplat");
    constexpr float finite_difference_step = 1e-3F;
    const auto alpha_loss_at = [&](const float foreground_alpha) {
        rendered.alpha = tinytensor::Tensor::from_vector(
            std::vector<float>{foreground_alpha, 0.8F}, {1, 2},
            tinytensor::Device::CUDA);
        return detail::compute_training_loss(
                   rendered, target, options, true).alpha_value;
    };
    const float numerical_gradient =
        (alpha_loss_at(0.8F + finite_difference_step) -
         alpha_loss_at(0.8F - finite_difference_step)) /
        (2.F * finite_difference_step);
    require(
        std::abs(numerical_gradient - alpha_gradient[0]) < 2e-4F,
        "transparent BCE analytic gradient failed finite differences");
    rendered.alpha = tinytensor::Tensor::from_vector(
        std::vector<float>{0.F, 1.F}, {1, 2},
        tinytensor::Device::CUDA);
    loss = detail::compute_training_loss(rendered, target, options, true);
    alpha_gradient = loss.alpha.to_vector();
    require(
        alpha_gradient[0] == 0.F && alpha_gradient[1] == 0.F,
        "transparent BCE did not match torch.clamp's saturated gradient");
    rendered.alpha = tinytensor::Tensor::from_vector(
        std::vector<float>{0.8F, 0.8F}, {1, 2},
        tinytensor::Device::CUDA);

    target.has_mask = false;
    loss = detail::compute_training_loss(rendered, target, options, false);
    rgb_gradient = loss.color.to_vector();
    alpha_gradient = loss.alpha.to_vector();
    require(rgb_gradient[0] != 0.F && rgb_gradient[1] != 0.F,
            "missing per-view mask incorrectly suppressed RGB training");
    require(std::all_of(alpha_gradient.begin(), alpha_gradient.end(),
                        [](float value) { return value == 0.F; }),
            "missing per-view mask incorrectly enabled alpha supervision");
}

void test_ssim_loss_and_scale_constraint() {
    using namespace aetherscan::splat;
    constexpr std::size_t side = 13;
    constexpr std::size_t pixels = side * side;
    std::vector<float> prediction(3 * pixels, 0.4F);
    std::vector<float> reference(3 * pixels, 0.4F);
    prediction[6 * side + 6] = 0.8F;
    RenderResult rendered;
    rendered.color = tinytensor::Tensor::from_vector(
        prediction, {3, side, side}, tinytensor::Device::CUDA);
    rendered.alpha = tinytensor::Tensor::zeros({side, side}, tinytensor::Device::CUDA);
    rendered.median_depth = tinytensor::Tensor::zeros(
        {side, side}, tinytensor::Device::CUDA);
    rendered.normal = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    TrainingView target;
    target.camera.width = target.camera.height = side;
    target.rgb = tinytensor::Tensor::from_vector(
        reference, {3, side, side}, tinytensor::Device::CUDA);
    target.depth = tinytensor::Tensor::zeros({side, side}, tinytensor::Device::CUDA);
    target.normal = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    target.mask = tinytensor::Tensor::from_vector(
        std::vector<float>(pixels, 1.F), {side, side}, tinytensor::Device::CUDA);
    TrainingOptions options;
    options.ssim_weight = 1.F;
    options.use_mvs_depth = false;
    options.use_mvs_normals = false;
    const auto loss = detail::compute_training_loss(
        rendered, target, options, true);
    require(
        std::abs(loss.rgb - 0.299324721F) < 2e-5F,
        "fused SSIM forward does not match the Python CUDA reference");
    require_finite(loss.color, "SSIM produced a non-finite gradient");
    const auto gradients = loss.color.to_vector();
    require(
        std::any_of(gradients.begin(), gradients.end(),
                    [](float value) { return std::abs(value) > 1e-6F; }),
        "SSIM did not backpropagate into rendered color");
    const std::size_t center = 6 * side + 6;
    require(
        std::abs(gradients[center] - 0.152631640F) < 2e-5F,
        "fused SSIM backward does not match the Python CUDA reference");
    const float gradient_sum = std::accumulate(
        gradients.begin(), gradients.end(), 0.F);
    const float gradient_abs_sum = std::accumulate(
        gradients.begin(), gradients.end(), 0.F,
        [](float total, float value) { return total + std::abs(value); });
    require(
        std::abs(gradient_sum - 0.004020121F) < 2e-5F &&
            std::abs(gradient_abs_sum - 0.301243156F) < 2e-5F,
        "fused SSIM gradient field differs from the Python CUDA reference");

    auto log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(0.01F), std::log(1.F), std::log(0.001F)},
        {1, 3}, tinytensor::Device::CUDA);
    detail::constrain_scale_ratio(log_scales, 10.F);
    const auto constrained = log_scales.to_vector();
    const auto [minimum, maximum] = std::minmax_element(
        constrained.begin(), constrained.end());
    require(
        std::exp(*maximum - *minimum) <= 10.0001F,
        "Gaussian scale-ratio constraint was not applied");
}

void test_gggs_depth_normal_consistency() {
    using namespace aetherscan::splat;
    constexpr std::size_t side = 5;
    constexpr std::size_t pixels = side * side;
    RenderResult rendered;
    rendered.color = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    rendered.alpha = tinytensor::Tensor::zeros(
        {side, side}, tinytensor::Device::CUDA);
    rendered.median_depth = tinytensor::Tensor::from_vector(
        std::vector<float>(pixels, 1.F), {side, side},
        tinytensor::Device::CUDA);
    std::vector<float> raster_normals(3 * pixels, 0.F);
    std::fill(
        raster_normals.begin(), raster_normals.begin() + pixels, 1.F);
    rendered.normal = tinytensor::Tensor::from_vector(
        raster_normals, {3, side, side}, tinytensor::Device::CUDA);

    TrainingView target;
    target.camera.width = target.camera.height = side;
    target.camera.fx = target.camera.fy = 5.F;
    target.camera.cx = target.camera.cy = 2.F;
    target.rgb = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    target.depth = tinytensor::Tensor::zeros(
        {side, side}, tinytensor::Device::CUDA);
    target.normal = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    target.mask = tinytensor::Tensor::from_vector(
        std::vector<float>(pixels, 1.F), {side, side},
        tinytensor::Device::CUDA);

    TrainingOptions options;
    options.ssim_weight = 0.F;
    options.use_depth_normal_loss = true;
    options.depth_normal_weight = 0.05F;
    const auto loss = detail::compute_training_loss(
        rendered, target, options, true, true);
    require(
        std::abs(loss.normal_value - 0.018F) < 1e-5F,
        "GGGS depth-normal forward differs from the Python definition");
    const auto normal_gradient = loss.normal.to_vector();
    const std::size_t center = 2 * side + 2;
    require(
        std::abs(normal_gradient[2 * pixels + center] - 0.002F) < 1e-6F,
        "GGGS depth-normal raster-normal gradient is incorrect");
    const auto depth_gradient = loss.depth.to_vector();
    const float maximum_depth_gradient = *std::max_element(
        depth_gradient.begin(), depth_gradient.end(),
        [](const float a, const float b) {
            return std::abs(a) < std::abs(b);
        });
    require(
        std::abs(std::abs(maximum_depth_gradient) - 0.005F) < 1e-6F,
        "GGGS median-depth gradient differs from Python autograd");
}

void test_gggs_depth_normal_parameter_gradients() {
    using namespace aetherscan::splat;
    constexpr std::size_t side = 32;
    constexpr std::size_t pixels = side * side;
    const auto make_model = [](const float z_offset) {
        GaussianModel model;
        model.means = tinytensor::Tensor::from_vector(
            std::vector<float>{
                -0.07F, -0.02F, 2.00F + z_offset,
                 0.08F,  0.03F, 2.08F},
            {2, 3}, tinytensor::Device::CUDA);
        model.log_scales = tinytensor::Tensor::from_vector(
            std::vector<float>{
                std::log(0.20F), std::log(0.16F), std::log(0.035F),
                std::log(0.18F), std::log(0.15F), std::log(0.030F)},
            {2, 3}, tinytensor::Device::CUDA);
        model.quaternions = tinytensor::Tensor::from_vector(
            std::vector<float>{
                0.98480775F, 0.F, 0.17364818F, 0.F,
                0.97629601F, -0.08583165F, -0.17298739F, 0.01513444F},
            {2, 4}, tinytensor::Device::CUDA);
        model.opacity_logits = tinytensor::Tensor::from_vector(
            std::vector<float>{1.5F, 1.2F}, {2, 1},
            tinytensor::Device::CUDA);
        model.sh = tinytensor::Tensor::from_vector(
            std::vector<float>{0.4F, 0.3F, 0.2F, 0.2F, 0.3F, 0.4F},
            {2, 1, 3}, tinytensor::Device::CUDA);
        model.sh_degree = 0;
        return model;
    };

    Camera camera;
    camera.world_to_camera[0] = 1.F;
    camera.world_to_camera[5] = 1.F;
    camera.world_to_camera[10] = 1.F;
    camera.world_to_camera[15] = 1.F;
    camera.fx = camera.fy = 45.F;
    camera.cx = camera.cy = 15.5F;
    camera.width = camera.height = side;

    TrainingView target;
    target.camera = camera;
    target.rgb = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    target.depth = tinytensor::Tensor::zeros(
        {side, side}, tinytensor::Device::CUDA);
    target.normal = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    target.mask = tinytensor::Tensor::from_vector(
        std::vector<float>(pixels, 1.F), {side, side},
        tinytensor::Device::CUDA);

    TrainingOptions options;
    options.photometric_weight = 0.F;
    options.ssim_weight = 0.F;
    options.use_depth_normal_loss = true;
    options.depth_normal_weight = 0.05F;
    RasterizeOptions raster_options;
    raster_options.require_depth = true;

    GaussianModel model = make_model(0.F);
    Rasterizer rasterizer;
    const RenderResult rendered = rasterizer.forward(
        model, camera, raster_options);
    const auto loss = detail::compute_training_loss(
        rendered, target, options, true, true);
    const ModelGradients gradients = rasterizer.backward(
        model, rendered, loss.color, loss.alpha, loss.depth, loss.normal);
    const float analytic = gradients.means.to_vector()[2];

    const auto evaluate = [&](const float offset) {
        const auto candidate = make_model(offset);
        const auto candidate_rendered = Rasterizer().forward(
            candidate, camera, raster_options);
        return detail::compute_training_loss(
            candidate_rendered, target, options, true, true).normal_value;
    };
    constexpr float epsilon = 2e-4F;
    const float numeric =
        (evaluate(epsilon) - evaluate(-epsilon)) / (2.F * epsilon);
    const float scale = std::max({std::abs(analytic), std::abs(numeric), 1e-5F});
    require(
        std::isfinite(analytic) && std::isfinite(numeric) &&
            std::abs(analytic - numeric) / scale < 0.08F,
        "GGGS depth-normal loss does not backpropagate through median depth "
        "and raster normals into Gaussian means");
}

void require_close(
    const std::vector<float>& actual, const std::vector<float>& expected,
    const float tolerance, const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t index = 0; index < actual.size(); ++index)
        require(std::abs(actual[index] - expected[index]) <= tolerance, message);
}

struct RefineHarness {
    aetherscan::splat::GaussianModel model;
    aetherscan::splat::detail::AdamState means;
    aetherscan::splat::detail::AdamState scales;
    aetherscan::splat::detail::AdamState rotations;
    aetherscan::splat::detail::AdamState opacity;
    aetherscan::splat::detail::AdamState sh;
    aetherscan::splat::detail::AdamState normals;
    aetherscan::splat::densification::AdamStates states() {
        return {&means, &scales, &rotations, &opacity, &sh, &normals};
    }
};

RefineHarness make_refine_harness(aetherscan::splat::GaussianModel model) {
    RefineHarness harness;
    harness.model = std::move(model);
    harness.means = aetherscan::splat::detail::make_adam_state(
        harness.model.means);
    harness.scales = aetherscan::splat::detail::make_adam_state(
        harness.model.log_scales);
    harness.rotations = aetherscan::splat::detail::make_adam_state(
        harness.model.quaternions);
    harness.opacity = aetherscan::splat::detail::make_adam_state(
        harness.model.opacity_logits);
    harness.sh = aetherscan::splat::detail::make_reduced_second_adam_state(
        harness.model.sh);
    harness.normals = aetherscan::splat::detail::make_adam_state(
        harness.model.normal_features.is_valid()
            ? harness.model.normal_features
            : tinytensor::Tensor::zeros(
                  {harness.model.size(), std::size_t{4}},
                  tinytensor::Device::CUDA));
    return harness;
}

aetherscan::splat::GaussianModel make_default_refine_model(
    const std::vector<float>& sh_values) {
    using tinytensor::Tensor;
    constexpr auto gpu = tinytensor::Device::CUDA;
    aetherscan::splat::GaussianModel model;
    model.means = Tensor::from_vector(
        std::vector<float>{0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0},
        {4, 3}, gpu);
    model.log_scales = Tensor::from_vector(
        std::vector<float>{
            std::log(0.001F), std::log(0.001F), std::log(0.001F),
            std::log(0.001F), std::log(0.001F), std::log(0.001F),
            std::log(0.05F), std::log(0.05F), std::log(0.05F),
            std::log(0.001F), std::log(0.001F), std::log(0.001F)},
        {4, 3}, gpu);
    model.quaternions = Tensor::from_vector(
        std::vector<float>{
            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
        {4, 4}, gpu);
    model.opacity_logits = Tensor::from_vector(
        std::vector<float>{-3.F, 0.F, 0.F, 0.F}, {4, 1}, gpu);
    model.sh = Tensor::from_vector(sh_values, {4, 1, 3}, gpu);
    model.sh_degree = 0;
    return model;
}

aetherscan::splat::detail::DensificationStats make_default_refine_stats() {
    using tinytensor::Tensor;
    constexpr auto gpu = tinytensor::Device::CUDA;
    auto stats = aetherscan::splat::detail::make_densification_stats(4);
    stats.gradient = Tensor::from_vector(
        std::vector<float>{1.F, 10.F, 10.F, 0.1F}, {4}, gpu);
    stats.count = Tensor::full({4}, 10.F, gpu);
    stats.priority = Tensor::full({4}, 10.F, gpu);
    stats.max_screen_radius = Tensor::zeros({4}, gpu);
    return stats;
}

aetherscan::splat::TrainingOptions make_default_refine_options() {
    aetherscan::splat::TrainingOptions options;
    options.densification_strategy =
        aetherscan::splat::DensificationStrategy::default_strategy;
    options.refine_start_iter = 1;
    options.refine_stop_iter = 200;
    options.refine_every = 100;
    options.prune_opacity = 0.1F;
    options.densify_gradient_threshold = 0.1F;
    options.densify_scale_threshold = 0.01F;
    options.densification_cap = 100;
    return options;
}

void test_opacity_progress_summary_matches_host() {
    using namespace aetherscan::splat;
    constexpr auto gpu = tinytensor::Device::CUDA;
    const auto empty = detail::summarize_opacity_progress(
        tinytensor::Tensor{}, tinytensor::Tensor{});
    require(
        empty.gradient_mean == 0.F &&
            empty.positive_gradient_fraction == 0.F &&
            empty.opacity_mean == 0.F,
        "empty opacity progress summary is not zero");

    const std::vector<float> logits{0.F, 2.F, -2.F};
    const std::vector<float> gradients{-1.F, 2.F, 0.5F};
    const auto stats = detail::summarize_opacity_progress(
        tinytensor::Tensor::from_vector(logits, {3, 1}, gpu),
        tinytensor::Tensor::from_vector(gradients, {3, 1}, gpu));
    double gradient_sum = 0.0;
    double opacity_sum = 0.0;
    std::size_t positive = 0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        gradient_sum += gradients[index];
        positive += gradients[index] > 0.F;
        opacity_sum += 1.0 / (1.0 + std::exp(-static_cast<double>(logits[index])));
    }
    const double inverse = 1.0 / static_cast<double>(logits.size());
    require(
        std::abs(stats.gradient_mean -
                 static_cast<float>(gradient_sum * inverse)) < 1e-6F,
        "opacity gradient mean does not match host reduction");
    require(
        std::abs(stats.positive_gradient_fraction -
                 static_cast<float>(positive * inverse)) < 1e-6F,
        "positive opacity-gradient fraction does not match host reduction");
    require(
        std::abs(stats.opacity_mean -
                 static_cast<float>(opacity_sum * inverse)) < 1e-6F,
        "opacity mean does not match host sigmoid reduction");

    std::vector<float> many_logits(4096, 0.F);
    std::vector<float> many_gradients(4096);
    for (std::size_t index = 0; index < many_gradients.size(); ++index)
        many_gradients[index] = (index % 2U) == 0U ? 1.F : -1.F;
    const auto many = detail::summarize_opacity_progress(
        tinytensor::Tensor::from_vector(many_logits, {4096}, gpu),
        tinytensor::Tensor::from_vector(many_gradients, {4096}, gpu));
    require(
        std::abs(many.gradient_mean) < 1e-5F &&
            std::abs(many.positive_gradient_fraction - 0.5F) < 1e-5F &&
            std::abs(many.opacity_mean - 0.5F) < 1e-5F,
        "large opacity progress summary drifted from the host pattern");

    bool threw = false;
    try {
        detail::summarize_opacity_progress(
            tinytensor::Tensor::zeros({4, 1}, gpu),
            tinytensor::Tensor::zeros({3, 1}, gpu));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "mismatched opacity progress tensors were accepted");
}

void test_default_refine_ignores_unused_host_downloads() {
    using namespace aetherscan::splat;
    auto options = make_default_refine_options();
    const std::vector<float> finite_sh(12, 0.25F);
    const std::vector<float> nan_sh(
        12, std::numeric_limits<float>::quiet_NaN());

    auto finite = make_refine_harness(make_default_refine_model(finite_sh));
    auto poisoned = make_refine_harness(make_default_refine_model(nan_sh));
    auto finite_stats = make_default_refine_stats();
    auto poisoned_stats = make_default_refine_stats();
    std::mt19937 finite_random(42);
    std::mt19937 poisoned_random(42);
    const auto finite_result = densification::refine_gaussians(
        finite.model, finite_stats, 100, 1.F, aetherscan::mvs::Vec3f::Zero(),
        options, finite_random, finite.states());
    const auto poisoned_result = densification::refine_gaussians(
        poisoned.model, poisoned_stats, 100, 1.F,
        aetherscan::mvs::Vec3f::Zero(), options, poisoned_random,
        poisoned.states());

    require(
        finite_result.pruned == 1 && finite_result.grown == 2 &&
            finite.model.size() == 5,
        "default refine did not prune the low-opacity row and clone/split");
    require(
        poisoned_result.pruned == finite_result.pruned &&
            poisoned_result.grown == finite_result.grown &&
            poisoned.model.size() == finite.model.size(),
        "default refine used SH values that should stay on the device");
    require_close(
        poisoned.model.means.to_vector(), finite.model.means.to_vector(),
        1e-5F, "default refine changed means when SH was NaN on device");
    require_close(
        poisoned.model.log_scales.to_vector(),
        finite.model.log_scales.to_vector(), 1e-5F,
        "default refine changed scales when SH was NaN on device");
    require_close(
        poisoned.model.opacity_logits.to_vector(),
        finite.model.opacity_logits.to_vector(), 1e-5F,
        "default refine changed opacity when SH was NaN on device");
}

void test_dense_adaptive_still_prunes_nonfinite_geometry() {
    using namespace aetherscan::splat;
    using tinytensor::Tensor;
    constexpr auto gpu = tinytensor::Device::CUDA;
    GaussianModel model = make_default_refine_model(std::vector<float>(12, 0.F));
    auto sh = model.sh.to_vector();
    sh[9] = std::numeric_limits<float>::quiet_NaN();
    sh[10] = std::numeric_limits<float>::quiet_NaN();
    sh[11] = std::numeric_limits<float>::quiet_NaN();
    model.sh = Tensor::from_vector(sh, {4, 1, 3}, gpu);
    model.opacity_logits = Tensor::zeros({4, 1}, gpu);
    auto harness = make_refine_harness(std::move(model));
    auto stats = make_default_refine_stats();
    TrainingOptions options = make_default_refine_options();
    options.densification_strategy = DensificationStrategy::dense_adaptive;
    options.dense_recycle_fraction = 0.F;
    options.dense_growth_fraction = 0.F;
    options.densify_gradient_threshold = 100.F;
    options.densification_cap = 3;
    std::mt19937 random(42);
    const auto result = densification::refine_gaussians(
        harness.model, stats, 100, 1.F, aetherscan::mvs::Vec3f::Zero(),
        options, random, harness.states());
    require(
        result.pruned == 1 && result.grown == 0 && harness.model.size() == 3,
        "dense_adaptive no longer hard-prunes non-finite SH rows");
    require_finite(
        harness.model.sh, "dense_adaptive kept a non-finite SH row");
}

void test_igs_growth_budget() {
    using namespace aetherscan::splat;
    using tinytensor::Tensor;
    constexpr auto gpu = tinytensor::Device::CUDA;
    GaussianModel model;
    model.means = Tensor::zeros({4, 3}, gpu);
    model.log_scales = Tensor::full({4, 3}, -3.F, gpu);
    model.quaternions = Tensor::from_vector(
        std::vector<float>{1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0}, {4,4}, gpu);
    model.opacity_logits = Tensor::zeros({4,1}, gpu);
    model.sh = Tensor::zeros({4,1,3}, gpu);
    model.normal_features = Tensor::zeros({4,3}, gpu);
    auto means = detail::make_adam_state(model.means);
    auto scales = detail::make_adam_state(model.log_scales);
    auto rotations = detail::make_adam_state(model.quaternions);
    auto opacity = detail::make_adam_state(model.opacity_logits);
    auto sh = detail::make_reduced_second_adam_state(model.sh);
    auto normals = detail::make_adam_state(model.normal_features);
    densification::AdamStates states{&means, &scales, &rotations, &opacity, &sh, &normals};
    auto stats = detail::make_densification_stats(4);
    stats.gradient = Tensor::full({4}, 1.F, gpu);
    stats.count = Tensor::full({4}, 10.F, gpu);
    stats.priority = Tensor::full({4}, 10.F, gpu);
    stats.max_screen_radius = Tensor::from_vector(
        std::vector<float>{0.5F,0.01F,0.01F,0.01F}, {4}, gpu);
    TrainingOptions options;
    options.densification_strategy = DensificationStrategy::adc_igs;
    options.densification_cap = 7;
    options.densify_select_fraction = 0.5F;
    options.densify_screen_threshold = 0.1F;
    options.densify_gradient_threshold = 0.01F;
    std::mt19937 random(42);
    const auto result = densification::refine_gaussians(
        model, stats, 200, 1.F, aetherscan::mvs::Vec3f::Zero(), options, random, states);
    require(result.grown == 3 && model.size() == 7,
            "IGS lost growth budget to an already selected oversized parent");
    for (const auto* state : states)
        require(state->first.shape()[0] == 7 && state->second.shape()[0] == 7,
                "IGS topology and Adam rows diverged");
    require_finite(model.means, "IGS produced non-finite means");
    require_finite(model.log_scales, "IGS produced non-finite scales");
}

void test_adc_plus_split_matches_brush() {
    using namespace aetherscan::splat;
    GaussianModel parents;
    parents.means = tinytensor::Tensor::zeros(
        {1, 3}, tinytensor::Device::CUDA);
    parents.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(2.F), 0.F, std::log(0.5F)},
        {1, 3}, tinytensor::Device::CUDA);
    parents.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F, 0.F, 0.F},
        {1, 4}, tinytensor::Device::CUDA);
    parents.opacity_logits = tinytensor::Tensor::zeros(
        {1, 1}, tinytensor::Device::CUDA);
    parents.sh = tinytensor::Tensor::zeros(
        {1, 1, 3}, tinytensor::Device::CUDA);
    parents.sh_degree = 0;
    GaussianModel children;
    children.means = tinytensor::Tensor::zeros(
        {1, 3}, tinytensor::Device::CUDA);
    children.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(2.F), 0.F, std::log(0.5F)},
        {1, 3}, tinytensor::Device::CUDA);
    children.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F, 0.F, 0.F},
        {1, 4}, tinytensor::Device::CUDA);
    children.opacity_logits = tinytensor::Tensor::zeros(
        {1, 1}, tinytensor::Device::CUDA);
    children.sh = tinytensor::Tensor::zeros(
        {1, 1, 3}, tinytensor::Device::CUDA);
    children.sh_degree = 0;
    const auto indices = tinytensor::Tensor::from_vector(
        std::vector<int>{0}, {1}, tinytensor::Device::CUDA);
    const auto unused_random = tinytensor::Tensor::zeros(
        {1, 3}, tinytensor::Device::CUDA);
    const auto screen_sizes = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F}, {1}, tinytensor::Device::CUDA);

    auto igs_parents = densification::clone_model(parents);
    auto igs_children = densification::clone_model(children);
    detail::split_gaussians(
        igs_parents, igs_children, indices, unused_random, screen_sizes, 5,
        1.F / 255.F, 0.5F);
    const auto igs_parent_mean = igs_parents.means.to_vector();
    const auto igs_child_mean = igs_children.means.to_vector();
    const auto igs_scale = igs_parents.log_scales.to_vector();
    const float original_variance[3]{4.F, 1.F, 0.25F};
    for (int row = 0; row < 3; ++row) {
        require(std::abs(igs_parent_mean[row] + igs_child_mean[row]) < 1e-5F,
                "IGS split changed the mixture centroid");
        for (int col = 0; col < 3; ++col) {
            const float covariance = (row == col ? std::exp(2.F * igs_scale[row]) : 0.F)
                + igs_child_mean[row] * igs_child_mean[col];
            require(std::abs(covariance - (row == col ? original_variance[row] : 0.F)) < 1e-5F,
                    "IGS split changed the mixture covariance");
        }
    }

    detail::split_gaussians(
        parents, children, indices, unused_random, screen_sizes, 2,
        1.F / 255.F, 0.5F);

    const auto parent_means = parents.means.to_vector();
    const auto child_means = children.means.to_vector();
    const auto parent_scales = parents.log_scales.to_vector();
    const auto child_scales = children.log_scales.to_vector();
    const float k[3]{0.5F, 0.875F, 0.96875F};
    const float scale[3]{2.F, 1.F, 0.5F};
    for (int axis = 0; axis < 3; ++axis) {
        const float offset =
            std::sqrt(1.F - k[axis] * k[axis]) * scale[axis];
        require(
            std::abs(parent_means[axis] + offset) < 1e-5F &&
                std::abs(child_means[axis] - offset) < 1e-5F,
            "ADC+ split does not match brush's centroid-preserving offset");
        const float expected_log_scale =
            std::log(scale[axis] * k[axis]);
        require(
            std::abs(parent_scales[axis] - expected_log_scale) < 1e-5F &&
                std::abs(child_scales[axis] - expected_log_scale) < 1e-5F,
            "ADC+ split does not match brush's covariance-aware shrink");
    }
    const float expected_opacity =
        1.F - std::pow(0.5F, std::numbers::sqrt2_v<float> / 2.F);
    const float expected_logit =
        std::log(expected_opacity / (1.F - expected_opacity));
    require(
        std::abs(parents.opacity_logits.to_vector()[0] - expected_logit) <
                1e-5F &&
            std::abs(children.opacity_logits.to_vector()[0] -
                     expected_logit) < 1e-5F,
        "ADC+ split opacity does not match brush's transmittance power");
}

void test_densification_strategies_and_dense_bypass() {
    using namespace aetherscan;
    require(
        splat::TrainingOptions{}.adam_epsilon == 1e-15F,
        "ADC+ Adam epsilon no longer matches Brush");

    std::vector<float> brush_points;
    for (const float value : {
             -100.F, 0.F, 1.F, 2.F, 3.F, 4.F,
             5.F, 6.F, 7.F, 8.F, 100.F})
        brush_points.insert(
            brush_points.end(), {value, 2.F * value, 3.F * value});
    const auto brush_bounds =
        splat::densification::brush_scene_geometry(brush_points);
    require(
        (brush_bounds.center - mvs::Vec3f(4.F, 8.F, 12.F)).norm() <
                1e-6F &&
            std::abs(brush_bounds.scale - 16.F) < 1e-6F &&
            std::abs(brush_bounds.maximum_extent - 12.F) < 1e-6F,
        "ADC+ percentile bounds no longer match Brush");
    const auto fallback_bounds =
        splat::densification::brush_scene_geometry({});
    require(
        fallback_bounds.center.isZero() &&
            fallback_bounds.scale == 2.F &&
            fallback_bounds.maximum_extent == 1.F,
        "ADC+ bounds fallback no longer matches Brush");

    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_densification_test";
    std::filesystem::create_directories(root);
    const auto image_path = root / "frame.png";
    io::save_rgb_png(
        io::RgbImage{32, 32, std::vector<std::uint8_t>(32 * 32 * 3, 128)},
        image_path);
    mvs::MvsScene scene;
    mvs::MvsView view;
    view.id = 0;
    view.path = image_path;
    view.width = view.src_width = 32;
    view.height = view.src_height = 32;
    view.fx = view.fy = view.src_fx = view.src_fy = 20.F;
    view.cx = view.cy = view.src_cx = view.src_cy = 15.5F;
    scene.views.push_back(view);
    view.id = 1;
    view.pose.C.x() = 0.1;
    scene.views.push_back(view);
    for (const float x : {-1.F, 1.F}) {
        mvs::DensePoint point;
        point.position = mvs::Vec3f(x, 0.F, 2.F);
        point.normal = mvs::Vec3f::UnitZ();
        point.color = mvs::Vec3f::Constant(0.5F);
        point.views = {0};
        scene.dense_cloud.points.push_back(point);
    }
    splat::TrainingOptions options;
    options.iterations = 3;
    options.sh_degree = 0;
    options.use_mvs_depth = false;
    options.use_mvs_normals = false;
    options.input_is_dense = false;
    options.maximum_scale_fraction = 1.F;
    options.densification_cap = 8;
    options.refine_start_iter = 1;
    options.refine_stop_iter = 3;
    options.refine_every = 2;
    options.densify_gradient_threshold = -1.F;
    options.densify_select_fraction = 1.F;
    options.log_interval = 1;
    for (const auto strategy : {
             splat::DensificationStrategy::default_strategy,
             splat::DensificationStrategy::adc_plus,
             splat::DensificationStrategy::adc_igs}) {
        options.densification_strategy = strategy;
        unsigned progress_steps = 0;
        const auto model = splat::Trainer(options).train(
            scene, [&](const splat::TrainingProgress& progress) {
                ++progress_steps;
                require(
                    std::isfinite(progress.opacity_mean) &&
                        std::isfinite(progress.opacity_gradient_mean),
                    "progress callback received non-finite opacity stats "
                    "across a densify step");
                return true;
            });
        require(
            progress_steps == options.iterations,
            "progress callback skipped a densify logging iteration");
        require(
            !model.filter_3d.is_valid(),
            "appearance-only sparse 3DGS unexpectedly enabled filter_3d");
        require(
            model.size() > scene.dense_cloud.points.size(),
            strategy == splat::DensificationStrategy::adc_plus
                ? "sparse-input ADC+ did not grow Gaussians"
                : strategy == splat::DensificationStrategy::adc_igs
                    ? "sparse-input ADC-IGS did not grow Gaussians"
                    : "sparse-input default strategy did not grow Gaussians");
        require(model.size() <= options.densification_cap,
                "densification exceeded its hard Gaussian cap");
    }
    // brush ADC+ refines from iteration 200 with no warm-up delay.
    auto brush_schedule = options;
    brush_schedule.iterations = 220;
    brush_schedule.densification_strategy =
        splat::DensificationStrategy::adc_plus;
    brush_schedule.refine_start_iter = 0;
    brush_schedule.refine_stop_iter = 0;
    brush_schedule.refine_every = 0;
    brush_schedule.densify_gradient_threshold = -1.F;
    brush_schedule.densify_select_fraction = 1.F;
    auto full_brush_schedule = brush_schedule;
    full_brush_schedule.iterations = 30'000;
    require(
        splat::densification::is_refinement_iteration(
            28'400, full_brush_schedule),
        "ADC+ stopped before Brush's final 30k refinement");
    require(
        !splat::densification::is_refinement_iteration(
            28'600, full_brush_schedule) &&
            !splat::densification::is_refinement_iteration(
                29'800, full_brush_schedule),
        "ADC+ refined after Brush's 95% cutoff");
    auto igs_schedule = full_brush_schedule;
    igs_schedule.densification_strategy = splat::DensificationStrategy::adc_igs;
    splat::apply_strategy_defaults(igs_schedule);
    require(igs_schedule.grow_stop_iter >= igs_schedule.iterations,
            "IGS unexpectedly truncates ADC+ growth budget");
    for (unsigned step : {200U, 600U, 24000U, 28400U, 28600U})
        require(splat::densification::is_refinement_iteration(step, igs_schedule) ==
                    splat::densification::is_refinement_iteration(step, full_brush_schedule),
                "IGS and ADC+ refinement schedules differ");
    const auto brush_schedule_model =
        splat::Trainer(brush_schedule).train(scene);
    require(
        brush_schedule_model.size() > scene.dense_cloud.points.size(),
        "ADC+ did not use brush's first refinement at iteration 200");

    options.input_is_dense = true;
    options.densification_strategy = splat::DensificationStrategy::adc_igs;
    options.evaluation_iterations = {2};
    options.evaluation_split_every = 2;
    options.preview_interval = 2;
    std::size_t evaluations = 0;
    std::vector<std::size_t> trained_views;
    std::vector<unsigned> preview_steps;
    const auto dense_model = splat::Trainer(options).train(
        scene,
        [&](const splat::TrainingProgress& progress) {
            trained_views.push_back(progress.view_index);
            return true;
        },
        [&](const unsigned iteration, const splat::GaussianModel&) {
            require(iteration == 2, "GGGS evaluation callback used wrong iteration");
            ++evaluations;
        }, {},
        [&](const unsigned iteration, const std::size_t view,
            const splat::Camera& camera, const tinytensor::Tensor& color) {
            require(view == 0 && color.numel() == 3 * camera.width * camera.height,
                    "Training preview changed camera or image shape");
            preview_steps.push_back(iteration);
        });
    require(
        dense_model.size() == scene.dense_cloud.points.size(),
        "dense point-cloud initialization incorrectly enabled densification");
    require(
        !dense_model.filter_3d.is_valid(),
        "appearance-only dense 3DGS unexpectedly enabled filter_3d");
    require(evaluations == 1, "GGGS evaluation callback was not invoked");
    require(preview_steps == std::vector<unsigned>{1, 2, 3},
            "Preview polling dropped the first, scheduled, or final frame");
    options.preview_interval = 0;
    require(
        !trained_views.empty() &&
            std::all_of(
                trained_views.begin(), trained_views.end(),
                [](const std::size_t index) { return index == 1; }),
        "GGGS evaluation split leaked a held-out view into training");

    options.evaluation_iterations.clear();
    options.evaluation_split_every = 0;
    options.use_depth_normal_loss = true;
    options.depth_normal_from_iter = 1;
    options.use_normal_field = true;
    options.normal_field_from_iter = 1;
    options.multi_view_geo_weight = 0.02F;
    options.profile_cuda = true;
    options.cuda_profile_interval = 2;
    const auto geometry_dense_model = splat::Trainer(options).train(scene);
    require(
        geometry_dense_model.filter_3d.is_valid(),
        "depth-normal mesh training did not enable filter_3d");
    require_finite(
        geometry_dense_model.normal_features,
        "normal-field training produced non-finite features");
    require(
        geometry_dense_model.normal_features.shape() ==
            tinytensor::TensorShape{
                geometry_dense_model.size(), std::size_t{4}},
        "normal-field training changed the feature row layout");
    options.profile_cuda = false;
    options.multi_view_geo_weight = 0.F;
    options.use_depth_normal_loss = false;
    options.use_normal_field = false;
    options.densification_strategy =
        splat::DensificationStrategy::dense_adaptive;
    options.dense_growth_fraction = 1.F;
    const auto adaptive_dense_model = splat::Trainer(options).train(scene);
    require(
        adaptive_dense_model.size() > scene.dense_cloud.points.size(),
        "dense_adaptive did not allocate Gaussians to high-gradient regions");
    require(adaptive_dense_model.size() <= options.densification_cap,
            "dense_adaptive exceeded its hard Gaussian cap");
    std::filesystem::remove_all(root);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "SKIP: no CUDA device\n";
            return 0;
        }
        if(argc>1 && std::string(argv[1])=="--fisheye-only") {
            test_fisheye_parameter_finite_differences();
            test_fisheye_filter_and_supervision();
            test_fisheye_equirect_rasterize();
            std::cout<<"Fisheye tests passed\n";
            return 0;
        }
        test_fisheye_parameter_finite_differences();
        test_fisheye_filter_and_supervision();
        test_mvs_camera_conversion();
        test_camera_projection_roundtrip();
        test_fisheye_equirect_rasterize();
        test_gggs_3d_filter();
        test_gggs_multi_view_geometry_and_ncc();
        test_geometry_stability_scheduler();
        test_forward_backward();
        test_normal_field_parameterization_and_occupancy();
        test_gaussian_format_roundtrip();
        test_pam_smoke();
        test_sample_depth_batch_boundary();
        test_contribution_visibility_rejects_occluded_gaussians();
        test_alpha_parameter_gradients();
        test_adam_rejects_non_finite_gradients();
        test_fused_adam_parity();
        test_reduced_second_sh_adam();
        test_active_sh_prefix_adam();
        test_mask_loading();
        test_training_device_cache();
        test_brush_quantile_selection();
        test_source_resolution_and_knn_initialization();
        test_colmap_text_loading();
        test_colmap_fisheye_and_equirect_loading();
        test_reality_capture_dataset_loading();
        test_openmvs_dataset_loading();
        test_sparse_init_keeps_photo_colors();
        test_mask_loss_modes();
        test_ssim_loss_and_scale_constraint();
        test_gggs_depth_normal_consistency();
        test_gggs_depth_normal_parameter_gradients();
        test_adc_plus_split_matches_brush();
        test_opacity_progress_summary_matches_host();
        test_default_refine_ignores_unused_host_downloads();
        test_dense_adaptive_still_prunes_nonfinite_geometry();
        test_igs_growth_budget();
        test_densification_strategies_and_dense_bypass();
        std::cout << "splat tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "splat test failed: " << error.what() << '\n';
        return 1;
    }
}
