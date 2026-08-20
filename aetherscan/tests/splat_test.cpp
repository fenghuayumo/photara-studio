#include "splat/trainer.hpp"
#include "splat/colmap.hpp"
#include "splat/dataset.hpp"
#include "../src/splat/cuda_ops.hpp"
#include "../src/splat/densification.hpp"
#include "../src/splat/multi_view_scheduler.hpp"
#include "io/image.hpp"
#include "sfm/export_mvs.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
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
        "GGGS discarded grayscale coverage from an asdiff mesh mask");
    std::filesystem::remove_all(root);
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
        const auto model = splat::Trainer(options).train(scene);
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
    const auto brush_schedule_model =
        splat::Trainer(brush_schedule).train(scene);
    require(
        brush_schedule_model.size() > scene.dense_cloud.points.size(),
        "ADC+ did not use brush's first refinement at iteration 200");

    options.input_is_dense = true;
    options.densification_strategy = splat::DensificationStrategy::adc_igs;
    options.evaluation_iterations = {2};
    options.evaluation_split_every = 2;
    std::size_t evaluations = 0;
    std::vector<std::size_t> trained_views;
    const auto dense_model = splat::Trainer(options).train(
        scene,
        [&](const splat::TrainingProgress& progress) {
            trained_views.push_back(progress.view_index);
            return true;
        },
        [&](const unsigned iteration, const splat::GaussianModel&) {
            require(iteration == 2, "GGGS evaluation callback used wrong iteration");
            ++evaluations;
        });
    require(
        dense_model.size() == scene.dense_cloud.points.size(),
        "dense point-cloud initialization incorrectly enabled densification");
    require(
        !dense_model.filter_3d.is_valid(),
        "appearance-only dense 3DGS unexpectedly enabled filter_3d");
    require(evaluations == 1, "GGGS evaluation callback was not invoked");
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

int main() {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "SKIP: no CUDA device\n";
            return 0;
        }
        test_mvs_camera_conversion();
        test_gggs_3d_filter();
        test_gggs_multi_view_geometry_and_ncc();
        test_geometry_stability_scheduler();
        test_forward_backward();
        test_normal_field_parameterization_and_occupancy();
        test_pam_smoke();
        test_sample_depth_batch_boundary();
        test_contribution_visibility_rejects_occluded_gaussians();
        test_alpha_parameter_gradients();
        test_adam_rejects_non_finite_gradients();
        test_fused_adam_parity();
        test_reduced_second_sh_adam();
        test_active_sh_prefix_adam();
        test_mask_loading();
        test_source_resolution_and_knn_initialization();
        test_colmap_text_loading();
        test_reality_capture_dataset_loading();
        test_openmvs_dataset_loading();
        test_mask_loss_modes();
        test_ssim_loss_and_scale_constraint();
        test_gggs_depth_normal_consistency();
        test_gggs_depth_normal_parameter_gradients();
        test_adc_plus_split_matches_brush();
        test_densification_strategies_and_dense_bypass();
        std::cout << "splat tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "splat test failed: " << error.what() << '\n';
        return 1;
    }
}
