#include "splat/rasterizer.hpp"
#include "rasterizer_vulkan.hpp"
#include <stdexcept>
#if defined(PHOTARA_HAS_CUDA_SPLAT)
#include "rasterizer_cuda.hpp"
#endif
namespace photara::splat {
RenderResult Rasterizer::forward(
    const GaussianModel& model,
    const Camera& camera,
    const RasterizeOptions& requested_options) const {
    if (model.means.device() == tinytensor::Device::Vulkan)
        return detail::vulkan_raster_forward(backend_, model, camera, requested_options);
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.forward(model, camera, requested_options);
#else
    throw std::runtime_error("forward: CUDA backend is not enabled for this operation");
#endif
}

RenderResult Rasterizer::forward_colors(
    const GaussianModel& model,
    RenderResult& source,
    const tinytensor::Tensor& colors_precomp) const {
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.forward_colors(model, source, colors_precomp);
#else
    throw std::runtime_error("forward_colors: CUDA backend is not enabled for this operation");
#endif
}

tinytensor::Tensor Rasterizer::pruning_scores(
    const GaussianModel& model,
    const Camera& camera,
    RasterizeOptions options, const bool gaussian_pop) const {
    if (model.means.device() == tinytensor::Device::Vulkan)
        return detail::vulkan_pruning_scores(backend_, model, camera, options, gaussian_pop);
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.pruning_scores(model, camera, options, gaussian_pop);
#else
    throw std::runtime_error("pruning_scores: CUDA backend is not enabled for this operation");
#endif
}

float Rasterizer::photometric_loss(
    const RenderResult& rendered,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const float ssim_weight,
    const float photometric_weight,
    const bool read_loss_value,
    tinytensor::Tensor* color_gradient) const {
    return detail::vulkan_photometric_loss(rendered, target, mask, mask_enabled, ssim_weight, photometric_weight, read_loss_value, color_gradient);
}

ModelGradients Rasterizer::backward(
    const GaussianModel& model,
    const RenderResult& rendered,
    const tinytensor::Tensor& grad_color,
    const tinytensor::Tensor& grad_alpha,
    const tinytensor::Tensor& grad_depth,
    const tinytensor::Tensor& grad_normal,
    const tinytensor::Tensor& densify_map,
    const SHAdamUpdate* sh_adam,
    const StructureAdamUpdate* structure_adam) const {
    if (model.means.device() == tinytensor::Device::Vulkan)
        return detail::vulkan_raster_backward(model, rendered, grad_color, grad_alpha, grad_depth, grad_normal, densify_map, sh_adam, structure_adam);
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.backward(model, rendered, grad_color, grad_alpha, grad_depth, grad_normal, densify_map, sh_adam, structure_adam);
#else
    throw std::runtime_error("backward: CUDA backend is not enabled for this operation");
#endif
}

void Rasterizer::materialize_visibility(RenderResult& rendered) const {
    if (rendered.visibility.is_valid()) return;
    if (rendered.context.backend_impl) { detail::vulkan_materialize_visibility(rendered); return; }
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    CudaRasterizer{}.materialize_visibility(rendered);
#else
    throw std::runtime_error("materialize_visibility: CUDA backend is not enabled for this operation");
#endif
}

const float* Rasterizer::projected_mean2d(const RenderResult& rendered) const {
    if (rendered.context.backend_impl) return nullptr;
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.projected_mean2d(rendered);
#else
    throw std::runtime_error("projected_mean2d: CUDA backend is not enabled for this operation");
#endif
}

DepthSampleResult Rasterizer::sample_depth(
    const GaussianModel& model,
    const tinytensor::Tensor& world_points,
    const Camera& camera,
    const RasterizeOptions& requested_options) const {
    if (model.means.device() == tinytensor::Device::Vulkan)
        return detail::vulkan_sample_depth(backend_, model, world_points, camera, requested_options);
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.sample_depth(model, world_points, camera, requested_options);
#else
    throw std::runtime_error("sample_depth: CUDA backend is not enabled for this operation");
#endif
}

DepthSampleGradients Rasterizer::sample_depth_backward(
    const GaussianModel& model,
    const DepthSampleResult& sampled,
    const tinytensor::Tensor& grad_camera_points) const {
    if (model.means.device() == tinytensor::Device::Vulkan)
        return detail::vulkan_sample_depth_backward(model, sampled, grad_camera_points);
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.sample_depth_backward(model, sampled, grad_camera_points);
#else
    throw std::runtime_error("sample_depth_backward: CUDA backend is not enabled for this operation");
#endif
}

OccupancyCache Rasterizer::prepare_occupancy_cache(
    const GaussianModel& model,
    const std::vector<Camera>& cameras,
    const RasterizeOptions& requested_options,
    std::size_t device_budget,
    std::size_t host_budget) const {
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.prepare_occupancy_cache(model, cameras, requested_options, device_budget, host_budget);
#else
    throw std::runtime_error("prepare_occupancy_cache: CUDA backend is not enabled for this operation");
#endif
}

OccupancyResult Rasterizer::evaluate_occupancy(
    OccupancyCache& cache,
    std::size_t camera_index,
    const tinytensor::Tensor& world_points) const {
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.evaluate_occupancy(cache, camera_index, world_points);
#else
    throw std::runtime_error("evaluate_occupancy: CUDA backend is not enabled for this operation");
#endif
}

OccupancyCacheStats Rasterizer::occupancy_cache_stats(const OccupancyCache& cache) const {
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.occupancy_cache_stats(cache);
#else
    throw std::runtime_error("occupancy_cache_stats: CUDA backend is not enabled for this operation");
#endif
}

OccupancyResult Rasterizer::evaluate_occupancy(
    const GaussianModel& model,
    const tinytensor::Tensor& world_points,
    const Camera& camera,
    const RasterizeOptions& requested_options) const {
    if (model.means.device() == tinytensor::Device::Vulkan)
        return detail::vulkan_occupancy(backend_, model, world_points, camera, requested_options);
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return CudaRasterizer{}.evaluate_occupancy(model, world_points, camera, requested_options);
#else
    throw std::runtime_error("evaluate_occupancy: CUDA backend is not enabled for this operation");
#endif
}

} // namespace photara::splat