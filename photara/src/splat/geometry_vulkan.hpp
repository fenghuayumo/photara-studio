#pragma once

#include "cuda_ops.hpp"

namespace photara::splat::detail {

// These passes use the shared TinyTensor Vulkan device. Only optional reduced
// logging terms cross to the host; images and parameter gradients stay resident.
void add_depth_normal_loss_vulkan(
    const RenderResult& rendered, const Camera& camera, float weight,
    LossGradients& gradients, bool collect_scalar_terms);
tinytensor::Tensor compute_3d_filter_vulkan(
    const tinytensor::Tensor& means, const std::vector<Camera>& cameras,
    float minimum_scale_factor, bool all_camera_euclidean);
tinytensor::Tensor unproject_depth_to_world_vulkan(
    const tinytensor::Tensor& depth, const Camera& camera);
void add_sample_depth_point_gradients_vulkan(
    const Camera& camera, const tinytensor::Tensor& points,
    LossGradients& gradients);
MultiViewLoss add_multi_view_loss_vulkan(
    const tinytensor::Tensor& points, const tinytensor::Tensor& inside,
    const RenderResult& rendered, const TrainingView& reference,
    const TrainingView& neighbour, const TrainingOptions& options,
    LossGradients& gradients, tinytensor::Tensor& point_gradient,
    bool collect_scalar_terms, tinytensor::Tensor* stability);
GeometryDistributionSummary summarize_geometry_distribution_vulkan(
    const GaussianModel& model);

}  // namespace photara::splat::detail
