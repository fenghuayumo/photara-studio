#include "splat/visualize.hpp"
#include "splat/rasterizer.hpp"
#include "visualize_diagnostics.hpp"
#include <stdexcept>
namespace photara::splat {
tinytensor::Tensor visualize(
    const GaussianModel& model,
    const Camera& camera,
    const VisualizeOptions& options) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model.means.device() == tinytensor::Device::Vulkan &&
        options.mode == VisualizationMode::points)
        return detail::visualize_points_vulkan(model, camera, options);
#endif
    const bool diagnostics =
        options.mode == VisualizationMode::splat && options.channel != VisualizationChannel::color;
    if (options.mode == VisualizationMode::splat ||
        (model.means.device() == tinytensor::Device::Vulkan &&
         options.mode == VisualizationMode::rings)) {
        Rasterizer rasterizer;
        RasterizeOptions raster;
        raster.active_sh_degree = options.active_sh_degree;
        raster.background = options.background;
        raster.kernel_size = options.kernel_size;
        raster.rasterization = options.rasterization;
        raster.scale_modifier = options.scale_modifier;
        raster.require_depth = diagnostics;
        raster.record_backward_state = false;
        raster.preview_rings = options.mode == VisualizationMode::rings;
        raster.preview_ring_scale = options.ring_scale;
        const auto rendered = rasterizer.forward(model, camera, raster);
        if (!diagnostics)
            return rendered.color;
#if defined(TINYTENSOR_HAS_VULKAN)
        if (rendered.color.device() == tinytensor::Device::Vulkan)
            return detail::visualize_diagnostics_vulkan(rendered, camera, options);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
        return detail::visualize_diagnostics_cuda(rendered, camera, options);
#else
        throw std::runtime_error("Visualization backend is not enabled");
#endif
    }
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return detail::visualize_overlay_cuda(model, camera, options);
#endif
    throw std::invalid_argument("This visualization mode requires the CUDA backend");
}

} // namespace photara::splat
