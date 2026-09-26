#include "rasterizer_vulkan.hpp"

#include "vulkan/backend.hpp"
#include "splat_drender/vulkan_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace photara::splat::detail {
namespace {

using splat_drender::vulkan::SplatBufferView;

bool environment_flag(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr)
        return false;
    const bool enabled = value[0] == '1';
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
#endif
}

double elapsed_ms(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

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
    // Models without the optional 3D filter still need a valid device binding.
    // Keep one read-only zero buffer across frames instead of allocating and
    // clearing O(gaussians) storage on every training iteration.
    tinytensor::Tensor zero_filter;

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
              // TinyTensor creates the shared device and enables push
              // descriptors on it; the rasterizer records the same way when the
              // device really has them.
              options.external_device.push_descriptors =
                  tinytensor::vulkan::device_info().push_descriptors;
              return options;
          }()),
          rasterizer(context), profile_stages(
              environment_flag("SPLAT_VULKAN_PROFILE_STAGES")) {}

    void record_forward(const double value) {
        if (profile_stages) forward_ms += value;
    }
    void record_loss(const double value) {
        if (profile_stages) loss_ms += value;
    }
    void record_backward(const double value) {
        if (!profile_stages) return;
        backward_ms += value;
        if (++profile_samples < 100) return;
        const double inverse = 1.0 / static_cast<double>(profile_samples);
        std::fprintf(
            stderr,
            "splat_vulkan_stage_profile samples=%u forward_avg_ms=%.4f "
            "loss_avg_ms=%.4f backward_avg_ms=%.4f raster_total_avg_ms=%.4f\n",
            profile_samples, forward_ms * inverse, loss_ms * inverse,
            backward_ms * inverse,
            (forward_ms + loss_ms + backward_ms) * inverse);
        profile_samples = 0;
        forward_ms = loss_ms = backward_ms = 0.0;
    }

    bool profile_stages{};
    std::uint32_t profile_samples{};
    double forward_ms{};
    double loss_ms{};
    double backward_ms{};
};

struct VulkanForwardContext {
    std::shared_ptr<VulkanRasterBackend> backend;
    tinytensor::Tensor visibility_bits;
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
    if (!model.filter_3d.is_valid() &&
        (!backend->zero_filter.is_valid() ||
         backend->zero_filter.numel() < model.size()))
        backend->zero_filter = tinytensor::Tensor::zeros(
            {model.size()}, tinytensor::Device::Vulkan);
    splat_drender::vulkan::SplatDeviceGaussians gaussians;
    gaussians.means = buffer_view(model.means);
    gaussians.sh = buffer_view(model.sh);
    gaussians.log_scales = buffer_view(model.log_scales);
    gaussians.raw_rotations = buffer_view(model.quaternions);
    gaussians.opacity_logits = buffer_view(model.opacity_logits);
    gaussians.filter_3d = buffer_view(
        model.filter_3d.is_valid() ? model.filter_3d : backend->zero_filter);
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

    RenderResult result;
    if (requested_options.copy_attachments) {
        result.color = tinytensor::Tensor::empty(
            {std::size_t{3}, camera.height, camera.width},
            tinytensor::Device::Vulkan);
        result.alpha = tinytensor::Tensor::empty(
            {camera.height, camera.width}, tinytensor::Device::Vulkan);
    }
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
    context->visibility_bits = tinytensor::Tensor::empty(
        {model.size()}, tinytensor::Device::Vulkan,
        tinytensor::DataType::Int32);
    splat_drender::vulkan::SplatDeviceFrame destination;
    destination.color = buffer_view(result.color);
    destination.alpha = buffer_view(result.alpha);
    destination.normal = buffer_view(result.normal);
    destination.median_depth = buffer_view(result.median_depth);
    destination.radii = buffer_view(result.radii);
    destination.visibility_bits = buffer_view(context->visibility_bits);
    destination.width = camera.width;
    destination.height = camera.height;

    const auto profile_start = std::chrono::steady_clock::now();
    tinytensor::vulkan::synchronize();
    backend->rasterizer.bind_model_device(gaussians);
    context->frame = backend->rasterizer.render_device_copy(
        camera_view(camera), settings, destination);
    // Keep the per-Gaussian contribution flags device-resident. The 0/1
    // int32-to-float conversion is a single TinyTensor compute dispatch and
    // avoids a full device -> host -> device round trip for every frame.
    if (!requested_options.defer_visibility)
        result.visibility = context->visibility_bits.to(
            tinytensor::DataType::Float32);
    result.rendered_instances = context->frame.instance_count;
    backend->record_forward(elapsed_ms(profile_start));
    result.context.backend_impl = std::move(context);
    return result;
}

float vulkan_photometric_loss(
    const RenderResult& rendered, const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask, const bool mask_enabled,
    const float ssim_weight, const float photometric_weight,
    const bool read_loss_value) {
    auto context = get_context(rendered);
    const auto profile_start = std::chrono::steady_clock::now();
    if (target.device() != tinytensor::Device::Vulkan)
        throw std::invalid_argument(
            "Vulkan photometric target must be a Vulkan tensor");
    tinytensor::vulkan::synchronize();
    context->photometric = context->backend->rasterizer.fused_l1_ssim_device(
        context->frame.color, buffer_view(target), context->frame.width,
        context->frame.height, ssim_weight, photometric_weight,
        mask_enabled ? buffer_view(mask) : SplatBufferView{});
    context->has_photometric = true;
    float value = 0.0F;
    if (read_loss_value)
        value = context->backend->rasterizer.read_photometric_loss(
            context->photometric);
    context->backend->record_loss(elapsed_ms(profile_start));
    return value;
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
    const auto profile_start = std::chrono::steady_clock::now();
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
    // splat_project_backward writes every packed slot for every Gaussian,
    // including zeroing inactive/culled entries and unused SH coefficients.
    // Avoid a redundant ~60 MB device clear and its TinyTensor queue drain.
    auto packed = tinytensor::Tensor::empty(
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
    context->backend->record_backward(elapsed_ms(profile_start));
    return gradients;
}

void vulkan_materialize_visibility(RenderResult& rendered) {
    if (rendered.visibility.is_valid()) return;
    auto context = get_context(rendered);
    rendered.visibility = context->visibility_bits.to(
        tinytensor::DataType::Float32);
}

}  // namespace photara::splat::detail
