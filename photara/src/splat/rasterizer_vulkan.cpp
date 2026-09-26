#include "rasterizer_vulkan.hpp"

#include "vulkan/backend.hpp"
#include "splat_drender/vulkan_api.h"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>

namespace photara::splat::detail {
namespace {

using splat_drender::vulkan::SplatBufferView;

SplatBufferView buffer_view(const tinytensor::Tensor& tensor) {
    if (!tensor.is_valid()) return {};
    const auto view = tinytensor::vulkan::buffer_view(tensor);
    return {view.buffer, view.offset, view.bytes};
}

splat_drender::vulkan::SplatCamera camera_view(const Camera& camera) {
    splat_drender::vulkan::SplatCamera result;
    result.width = camera.width;
    result.height = camera.height;
    result.fx = camera.fx;
    result.fy = camera.fy;
    result.cx = camera.cx;
    result.cy = camera.cy;
    result.mode = static_cast<std::uint32_t>(camera.model);
    result.k1 = camera.k1;
    result.k2 = camera.k2;
    result.k3 = camera.k3;
    result.k4 = camera.k4;
    result.world_to_camera = camera.world_to_camera;
    result.center = camera.position;
    return result;
}

struct VulkanRasterBackend {
    splat_drender::vulkan::Context context;
    splat_drender::vulkan::SplatRasterizer rasterizer;

    VulkanRasterBackend()
        : context([] {
              if (!tinytensor::vulkan::available())
                  throw std::runtime_error(
                      "The Vulkan training backend is unavailable");
              const auto handles = tinytensor::vulkan::device_handles();
              splat_drender::vulkan::ContextOptions options;
              options.external_device.instance = handles.instance;
              options.external_device.physical_device = handles.physical_device;
              options.external_device.device = handles.device;
              options.external_device.queue = handles.queue;
              options.external_device.queue_family = handles.queue_family;
              return options;
          }()),
          rasterizer(context) {}
};

struct VulkanForwardContext {
    std::shared_ptr<VulkanRasterBackend> backend;
    tinytensor::Tensor zero_filter;
    splat_drender::vulkan::SplatDeviceFrame frame;
    splat_drender::vulkan::SplatDevicePhotometricOutput photometric;
    bool has_photometric{};
};

std::shared_ptr<VulkanRasterBackend> get_backend(std::shared_ptr<void>& value) {
    if (!value) value = std::make_shared<VulkanRasterBackend>();
    return std::static_pointer_cast<VulkanRasterBackend>(value);
}

std::shared_ptr<VulkanForwardContext> get_context(
    const RenderResult& rendered) {
    if (!rendered.context.backend_impl)
        throw std::invalid_argument(
            "Vulkan backward requires a live forward context");
    return std::static_pointer_cast<VulkanForwardContext>(
        rendered.context.backend_impl);
}

}  // namespace

RenderResult vulkan_raster_forward(
    std::shared_ptr<void>& backend_value, const GaussianModel& model,
    const Camera& camera, const RasterizeOptions& requested_options) {
    if (camera.width == 0 || camera.height == 0)
        throw std::invalid_argument("Splat camera dimensions must be positive");
    if (model.means.device() != tinytensor::Device::Vulkan)
        throw std::invalid_argument("Vulkan rasterization requires a Vulkan model");
    if (requested_options.colors_precomp.is_valid())
        throw std::invalid_argument(
            "Vulkan precomputed-color rasterization is not implemented yet");

    const auto backend = get_backend(backend_value);
    auto context = std::make_shared<VulkanForwardContext>();
    context->backend = backend;
    if (!model.filter_3d.is_valid())
        context->zero_filter = tinytensor::Tensor::zeros(
            {model.size()}, tinytensor::Device::Vulkan);
    splat_drender::vulkan::SplatDeviceGaussians gaussians;
    gaussians.means = buffer_view(model.means);
    gaussians.sh = buffer_view(model.sh);
    gaussians.log_scales = buffer_view(model.log_scales);
    gaussians.raw_rotations = buffer_view(model.quaternions);
    gaussians.opacity_logits = buffer_view(model.opacity_logits);
    gaussians.filter_3d = buffer_view(
        model.filter_3d.is_valid() ? model.filter_3d : context->zero_filter);
    gaussians.count = static_cast<std::uint32_t>(model.size());
    gaussians.sh_degree = std::min(
        requested_options.active_sh_degree, model.sh_degree);
    gaussians.sh_bases = static_cast<std::uint32_t>(model.sh.shape()[1]);

    splat_drender::vulkan::SplatSettings settings;
    std::copy(
        requested_options.background.begin(),
        requested_options.background.end(), settings.background);
    settings.kernel_size = requested_options.kernel_size;
    settings.scale_modifier = requested_options.scale_modifier;
    settings.need_depth = requested_options.require_depth;
    settings.pixel_snapshots = true;
    settings.point_depth_bracket = requested_options.point_depth_bracket;
    settings.point_depth_tolerance = requested_options.point_depth_tolerance;

    tinytensor::vulkan::synchronize();
    backend->rasterizer.bind_model_device(gaussians);
    context->frame = backend->rasterizer.render_device(
        camera_view(camera), settings);

    RenderResult result;
    result.color = tinytensor::Tensor::empty(
        {std::size_t{3}, camera.height, camera.width},
        tinytensor::Device::Vulkan);
    result.alpha = tinytensor::Tensor::empty(
        {camera.height, camera.width}, tinytensor::Device::Vulkan);
    if (settings.need_depth) {
        result.normal = tinytensor::Tensor::empty(
            {std::size_t{3}, camera.height, camera.width},
            tinytensor::Device::Vulkan);
        result.median_depth = tinytensor::Tensor::empty(
            {camera.height, camera.width}, tinytensor::Device::Vulkan);
    }
    result.radii = tinytensor::Tensor::empty(
        {model.size()}, tinytensor::Device::Vulkan,
        tinytensor::DataType::Int32);
    // splat_drender stores contribution visibility as uint32 flags.
    auto visibility_bits = tinytensor::Tensor::empty(
        {model.size()}, tinytensor::Device::Vulkan,
        tinytensor::DataType::Int32);

    splat_drender::vulkan::SplatDeviceFrame destination;
    destination.color = buffer_view(result.color);
    destination.alpha = buffer_view(result.alpha);
    destination.normal = buffer_view(result.normal);
    destination.median_depth = buffer_view(result.median_depth);
    destination.radii = buffer_view(result.radii);
    destination.visibility_bits = buffer_view(visibility_bits);
    destination.width = camera.width;
    destination.height = camera.height;
    backend->rasterizer.copy_frame_device(context->frame, destination);
    // TinyTensor's current Vulkan int32->float cast rounds these 0/1 flags to
    // zero. Convert the small per-Gaussian flag vector explicitly; render
    // attachments and all image-sized training data remain device-resident.
    const auto raw_visibility = visibility_bits.to_vector_int();
    std::vector<float> visibility(raw_visibility.size());
    std::transform(raw_visibility.begin(), raw_visibility.end(),
        visibility.begin(), [](const int value) { return value == 0 ? 0.F : 1.F; });
    result.visibility = tinytensor::Tensor::from_vector(
        visibility, {visibility.size()}, tinytensor::Device::Vulkan);
    result.rendered_instances = context->frame.instance_count;
    result.context.backend_impl = std::move(context);
    return result;
}

float vulkan_photometric_loss(
    const RenderResult& rendered, const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask, const bool mask_enabled,
    const float ssim_weight, const float photometric_weight) {
    auto context = get_context(rendered);
    if (target.device() != tinytensor::Device::Vulkan)
        throw std::invalid_argument(
            "Vulkan photometric target must be a Vulkan tensor");
    tinytensor::vulkan::synchronize();
    context->photometric = context->backend->rasterizer.fused_l1_ssim_device(
        context->frame.color, buffer_view(target), context->frame.width,
        context->frame.height, ssim_weight, photometric_weight,
        mask_enabled ? buffer_view(mask) : SplatBufferView{});
    context->has_photometric = true;
    return context->backend->rasterizer.read_photometric_loss(
        context->photometric);
}

ModelGradients vulkan_raster_backward(
    const GaussianModel& model, const RenderResult& rendered,
    const tinytensor::Tensor& grad_color,
    const tinytensor::Tensor& grad_alpha,
    const tinytensor::Tensor& grad_depth,
    const tinytensor::Tensor& grad_normal,
    const tinytensor::Tensor& densify_map,
    const SHAdamUpdate* sh_adam) {
    auto context = get_context(rendered);
    if (sh_adam != nullptr)
        throw std::invalid_argument(
            "Fused SH Adam is not supported by the Vulkan raster primitive");
    if (densify_map.is_valid() && densify_map.numel() != 0)
        throw std::invalid_argument(
            "Vulkan EMC densify-map backward is not implemented yet");
    const SplatBufferView color_gradient = grad_color.is_valid()
        ? buffer_view(grad_color)
        : context->has_photometric
            ? context->photometric.gradient
            : SplatBufferView{};
    if (color_gradient.buffer == VK_NULL_HANDLE)
        throw std::invalid_argument(
            "Vulkan backward requires a photometric color gradient");

    const std::size_t count = model.size();
    const std::size_t sh_values = model.sh.numel();
    auto packed = tinytensor::Tensor::zeros(
        {sh_values + count * 26U}, tinytensor::Device::Vulkan);
    tinytensor::vulkan::synchronize();
    context->backend->rasterizer.backward_device(
        color_gradient, buffer_view(grad_alpha), buffer_view(packed),
        buffer_view(grad_depth), buffer_view(grad_normal));

    std::size_t offset = 0;
    ModelGradients gradients;
    gradients.means = packed.slice(0, offset, offset + count * 3U)
        .reshape(tinytensor::TensorShape{count, std::size_t{3}});
    offset += count * 3U;
    gradients.sh = packed.slice(0, offset, offset + sh_values)
        .reshape(model.sh.shape());
    offset += sh_values;
    offset += count;       // activated opacity
    offset += count * 3U;  // activated scale
    offset += count * 4U;  // normalized rotation
    offset += count * 6U;  // covariance
    gradients.log_scales = packed.slice(
        0, offset, offset + count * 3U).reshape(
            tinytensor::TensorShape{count, std::size_t{3}});
    offset += count * 3U;
    gradients.quaternions = packed.slice(
        0, offset, offset + count * 4U).reshape(
            tinytensor::TensorShape{count, std::size_t{4}});
    offset += count * 4U;
    gradients.opacity_logits = packed.slice(
        0, offset, offset + count).reshape(
            tinytensor::TensorShape{count, std::size_t{1}});
    offset += count;
    gradients.refine_weight = packed.slice(0, offset, offset + count);
    return gradients;
}

}  // namespace photara::splat::detail
