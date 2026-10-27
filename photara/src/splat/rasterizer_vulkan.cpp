#include "rasterizer_vulkan.hpp"

#include "vulkan/backend.hpp"
#include "splat_drender/vulkan_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <limits>
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
    result.mode = camera.orthographic ? 4u : static_cast<std::uint32_t>(camera.model);
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
    // Neighbour queries must never overwrite the reference frame's snapshots,
    // phase arena, or fused photometric gradient.
    std::shared_ptr<VulkanRasterBackend> sample_backend;
    std::weak_ptr<void> latest_sample;

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
            // Same for buffer float32 atomic adds: TinyTensor enables the
            // extension on the shared device when the hardware has it.
            options.external_device.buffer_float32_atomic_add =
                tinytensor::vulkan::device_info().buffer_atomic_f32;
            return options;
          }()),
          rasterizer(context), profile_stages(
              environment_flag("SPLAT_VULKAN_PROFILE_STAGES")) {}

    void record_forward(const double value, const double sync,
                        const double bind, const double render,
                        const double visibility) {
        if (!profile_stages) return;
        forward_ms += value;
        forward_sync_ms += sync;
        forward_bind_ms += bind;
        forward_render_ms += render;
        forward_visibility_ms += visibility;
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
            "loss_avg_ms=%.4f backward_avg_ms=%.4f raster_total_avg_ms=%.4f "
            "forward_sync_avg_ms=%.4f forward_bind_avg_ms=%.4f "
            "forward_render_avg_ms=%.4f forward_visibility_avg_ms=%.4f\n",
            profile_samples, forward_ms * inverse, loss_ms * inverse,
            backward_ms * inverse,
            (forward_ms + loss_ms + backward_ms) * inverse,
            forward_sync_ms * inverse, forward_bind_ms * inverse,
            forward_render_ms * inverse, forward_visibility_ms * inverse);
        std::fflush(stderr);
        profile_samples = 0;
        forward_ms = loss_ms = backward_ms = 0.0;
        forward_sync_ms = forward_bind_ms = forward_render_ms =
            forward_visibility_ms = 0.0;
    }

    bool profile_stages{};
    std::uint32_t profile_samples{};
    double forward_ms{};
    double forward_sync_ms{};
    double forward_bind_ms{};
    double forward_render_ms{};
    double forward_visibility_ms{};
    double loss_ms{};
    double backward_ms{};
};

struct VulkanForwardContext {
    std::shared_ptr<VulkanRasterBackend> backend;
    tinytensor::Tensor colors_precomp;
    tinytensor::Tensor visibility_bits;
    splat_drender::vulkan::SplatDeviceFrame frame;
    splat_drender::vulkan::SplatDevicePhotometricOutput photometric;
    bool has_photometric{};
};

struct VulkanSampleContext {
    std::shared_ptr<VulkanRasterBackend> backend;
    tinytensor::Tensor points;
    bool consumed = false;
};

thread_local std::weak_ptr<VulkanRasterBackend> active_training_backend;

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

splat_drender::vulkan::SplatDeviceGaussians device_gaussians(
    const std::shared_ptr<VulkanRasterBackend>& backend,
    const GaussianModel& model, const RasterizeOptions& requested_options) {
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
    if (requested_options.colors_precomp.is_valid()) {
        const auto& colors = requested_options.colors_precomp;
        if (colors.device() != tinytensor::Device::Vulkan ||
            colors.dtype() != tinytensor::DataType::Float32 ||
            !colors.is_contiguous() || colors.shape().rank() != 2 ||
            colors.shape()[0] != model.size() || colors.shape()[1] != 3)
            throw std::invalid_argument("Vulkan precomputed colors must be contiguous float32 [N,3]");
        gaussians.sh = buffer_view(colors);
        gaussians.colors_precomputed = true;
        gaussians.sh_degree = 0;
        gaussians.sh_bases = 1;
    }
    return gaussians;
}

splat_drender::vulkan::SplatSettings raster_settings(
    const GaussianModel& model, const RasterizeOptions& requested_options) {

    splat_drender::vulkan::SplatSettings settings;
    settings.primitive = model.render_method == RenderMethod::gaussian_3dgut
        ? splat_drender::vulkan::SplatPrimitive::gaussian_3dgut
        : splat_drender::vulkan::SplatPrimitive::gaussian_3dgs;
    std::copy(
        requested_options.background.begin(),
        requested_options.background.end(), settings.background);
    settings.kernel_size = requested_options.kernel_size;
    settings.alpha_clip = requested_options.rasterization.alpha_clip;
    settings.near_plane = requested_options.rasterization.near_plane;
    settings.pixel_center_offset = requested_options.rasterization.pixel_center_offset;
    settings.compensate_2d = requested_options.rasterization.compensate_2d;
    settings.scale_modifier = requested_options.scale_modifier;
    settings.need_depth = requested_options.require_depth;
    settings.pixel_snapshots = requested_options.record_backward_state;
    settings.preview_rings = requested_options.preview_rings;
    settings.preview_ring_scale = requested_options.preview_ring_scale;
    settings.point_depth_bracket = requested_options.point_depth_bracket;
    settings.point_depth_tolerance = requested_options.point_depth_tolerance;
    return settings;
}

}  // namespace

RenderResult vulkan_raster_forward(
    std::shared_ptr<void>& backend_value, const GaussianModel& model,
    const Camera& camera, const RasterizeOptions& requested_options) {
    if (camera.width == 0 || camera.height == 0)
        throw std::invalid_argument("Splat camera dimensions must be positive");
    if (model.means.device() != tinytensor::Device::Vulkan)
        throw std::invalid_argument("Vulkan rasterization requires a Vulkan model");

    const auto backend = get_backend(backend_value);
    active_training_backend = backend;
    auto context = std::make_shared<VulkanForwardContext>();
    context->backend = backend;
    context->colors_precomp = requested_options.colors_precomp;
    const auto gaussians = device_gaussians(backend, model, requested_options);
    const auto settings = raster_settings(model, requested_options);

    RenderResult result;
    if (requested_options.copy_attachments) {
        result.color = tinytensor::Tensor::empty(
            {std::size_t{3}, camera.height, camera.width},
            tinytensor::Device::Vulkan);
    }
    if ((requested_options.copy_attachments && !requested_options.copy_color_only) ||
        requested_options.copy_alpha)
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

    const auto profile_start = backend->profile_stages
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    tinytensor::vulkan::submit_async();
    const double sync_ms = backend->profile_stages
        ? elapsed_ms(profile_start) : 0.0;
    const bool reuse = requested_options.reuse_projection &&
        !environment_flag("SPLAT_VULKAN_DISABLE_NORMAL_REUSE");
    if (!reuse) backend->rasterizer.bind_model_device(gaussians);
    const double bind_ms = backend->profile_stages
        ? elapsed_ms(profile_start) - sync_ms : 0.0;
    context->frame = reuse
        ? backend->rasterizer.render_colors_device_copy(
            gaussians, camera_view(camera), settings, destination)
        : backend->rasterizer.render_device_copy(
            camera_view(camera), settings, destination);
    const double render_ms = backend->profile_stages
        ? elapsed_ms(profile_start) - sync_ms - bind_ms : 0.0;
    // Keep the per-Gaussian contribution flags device-resident. The 0/1
    // int32-to-float conversion is a single TinyTensor compute dispatch and
    // avoids a full device -> host -> device round trip for every frame.
    if (!requested_options.defer_visibility)
        result.visibility = context->visibility_bits.to(
            tinytensor::DataType::Float32);
    result.rendered_instances = context->frame.instance_count;
    if (backend->profile_stages) {
        const double total_ms = elapsed_ms(profile_start);
        backend->record_forward(total_ms, sync_ms, bind_ms, render_ms,
                                total_ms - sync_ms - bind_ms - render_ms);
    }
    result.context.backend_impl = std::move(context);
    return result;
}

tinytensor::Tensor vulkan_pruning_scores(
    std::shared_ptr<void>& backend_value, const GaussianModel& model,
    const Camera& camera, RasterizeOptions options) {
    options.require_depth = false;
    // Only final colour and contributor limits are needed; omit the large
    // per-bucket backward snapshots and their prefix scan during scoring.
    options.record_backward_state = false;
    options.colors_precomp = {};
    const auto backend = get_backend(backend_value);
    const auto gaussians = device_gaussians(backend, model, options);
    auto scores = tinytensor::Tensor::empty({model.size()}, tinytensor::Device::Vulkan);
    tinytensor::vulkan::submit_async();
    backend->rasterizer.bind_model_device(gaussians);
    // Keep all frame attachments internal; scoring never needs owned copies of
    // images, radii or visibility, nor a training context for backward.
    const auto frame = backend->rasterizer.render_device(camera_view(camera), raster_settings(model, options));
    (void)frame;
    backend->rasterizer.pruning_scores_device(buffer_view(scores));
    return scores;
}

bool vulkan_dispatch_color_correction(
    const std::span<const splat_drender::vulkan::SplatColorCorrectionCommand>
        commands) {
    const auto backend = active_training_backend.lock();
    if (!backend) return false;
    backend->rasterizer.color_correction_batch_device(commands);
    return true;
}

float vulkan_photometric_loss(
    const RenderResult& rendered, const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask, const bool mask_enabled,
    const float ssim_weight, const float photometric_weight,
    const bool read_loss_value,
    tinytensor::Tensor* color_gradient) {
    auto context = get_context(rendered);
    const auto profile_start = context->backend->profile_stages
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    if (target.device() != tinytensor::Device::Vulkan)
        throw std::invalid_argument(
            "Vulkan photometric target must be a Vulkan tensor");
    if (color_gradient != nullptr) {
        const tinytensor::TensorShape gradient_shape{
            std::size_t{3}, context->frame.height, context->frame.width};
        if (!color_gradient->is_valid() ||
            color_gradient->shape() != gradient_shape ||
            color_gradient->device() != tinytensor::Device::Vulkan)
            *color_gradient = tinytensor::Tensor::empty(
                gradient_shape, tinytensor::Device::Vulkan);
    }
    tinytensor::vulkan::submit_async();
    context->photometric = context->backend->rasterizer.fused_l1_ssim_device(
        rendered.color.is_valid() ? buffer_view(rendered.color)
                                  : context->frame.color,
        buffer_view(target), context->frame.width,
        context->frame.height, ssim_weight, photometric_weight,
        mask_enabled ? buffer_view(mask) : SplatBufferView{},
        color_gradient != nullptr ? buffer_view(*color_gradient)
                                  : SplatBufferView{});
    context->has_photometric = true;
    float value = 0.0F;
    if (read_loss_value)
        value = context->backend->rasterizer.read_photometric_loss(
            context->photometric);
    if (context->backend->profile_stages)
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
    const SHAdamUpdate* sh_adam,
    const StructureAdamUpdate* structure_adam) {
    auto context = get_context(rendered);
    const auto profile_start = context->backend->profile_stages
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    if (structure_adam && (!sh_adam || structure_adam->shape_erank_reg > 0.F || structure_adam->shape_erank_s3_reg > 0.F ||
        structure_adam->shape_quat_norm_reg > 0.F || structure_adam->shape_axis_ratio_reg > 0.F))
        throw std::invalid_argument("Vulkan fused structure Adam requires fused SH and no shape regularizers");
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
    const bool precomputed = context->colors_precomp.is_valid();
    if (precomputed && sh_adam)
        throw std::invalid_argument("Precomputed colors cannot use fused SH Adam");
    const std::size_t sh_values = sh_adam ? 0 : precomputed ? count * 3U : model.sh.numel();
    // splat_project_backward writes every packed slot for every Gaussian,
    // including zeroing inactive/culled entries and unused SH coefficients.
    // Avoid a redundant ~60 MB device clear and its TinyTensor queue drain.
    auto packed = tinytensor::Tensor::empty(
        {structure_adam ? count * (structure_adam->capture_opacity_gradient ? 2U : 1U)
            : context->backend->rasterizer.model_gradient_float_count() -
                (sh_adam ? model.sh.numel() : 0)}, tinytensor::Device::Vulkan);
    splat_drender::vulkan::SplatSHAdamUpdate update;
    if (sh_adam) {
        if (sh_adam->first.dtype() != tinytensor::DataType::Float16 ||
            sh_adam->packed.dtype() != tinytensor::DataType::UInt8 ||
            sh_adam->bounds.dtype() != tinytensor::DataType::Float32 ||
            sh_adam->first.numel() != model.sh.numel() ||
            sh_adam->packed.numel() != model.sh.numel() ||
            sh_adam->bounds.numel() != count * 4)
            throw std::invalid_argument("Vulkan fused SH Adam requires quantized state");
        update.parameter = buffer_view(model.sh);
        update.first = buffer_view(sh_adam->first);
        update.packed = buffer_view(sh_adam->packed);
        update.bounds = buffer_view(sh_adam->bounds);
        // TinyTensor allocations are word-padded; byte/halfword codecs access
        // complete words even when the last degree 0/2 row ends mid-word.
        update.first.bytes = (update.first.bytes + 3) & ~VkDeviceSize{3};
        update.packed.bytes = (update.packed.bytes + 3) & ~VkDeviceSize{3};
        const std::size_t stride = model.sh.shape()[1] * 3;
        const float regularization = stride > 3 && sh_adam->regularization_weight > 0.F
            ? 2.F * sh_adam->regularization_weight / static_cast<float>(count * (stride - 3))
            : 0.F;
        update.settings = {sh_adam->lr, sh_adam->rest_lr, sh_adam->beta1, sh_adam->beta2,
            sh_adam->correction1, sh_adam->correction2, sh_adam->epsilon, regularization};
    }
    splat_drender::vulkan::SplatStructureAdamUpdate structure_update;
    if (structure_adam) {
        const tinytensor::Tensor* moments[]{&structure_adam->means_first, &structure_adam->means_second,
            &structure_adam->scales_first, &structure_adam->scales_second,
            &structure_adam->rotations_first, &structure_adam->rotations_second,
            &structure_adam->opacity_first, &structure_adam->opacity_second};
        const std::size_t strides[]{3,3,3,3,4,4,1,1};
        for (int i = 0; i < 8; ++i) {
            if (moments[i]->dtype() != tinytensor::DataType::Float32 || moments[i]->numel() != count * strides[i])
                throw std::invalid_argument("Vulkan fused structure Adam requires full FP32 moments");
            structure_update.moments[i] = buffer_view(*moments[i]);
        }
        structure_update.settings = {structure_adam->means_lr, structure_adam->scales_lr,
            structure_adam->quaternions_lr, structure_adam->opacities_lr,
            structure_adam->beta1, structure_adam->beta2, structure_adam->correction1,
            structure_adam->correction2, structure_adam->epsilon,
            structure_adam->minimum_log_scale, structure_adam->maximum_log_scale,
            structure_adam->opacity_reg, structure_adam->scale_reg};
        structure_update.capture_opacity_gradient = structure_adam->capture_opacity_gradient;
    }
    tinytensor::vulkan::submit_async();
    context->backend->rasterizer.backward_device(
        color_gradient, buffer_view(grad_alpha), buffer_view(packed),
        buffer_view(grad_depth), buffer_view(grad_normal), sh_adam ? &update : nullptr,
        structure_adam ? &structure_update : nullptr);

    std::size_t offset = 0;
    ModelGradients gradients;
    if (structure_adam) {
        gradients.refine_weight = packed.slice(0, 0, count);
        if (structure_adam->capture_opacity_gradient)
            gradients.opacity_logits = packed.slice(0, count, count * 2U)
                .reshape(tinytensor::TensorShape{count, std::size_t{1}});
        if (context->backend->profile_stages)
            context->backend->record_backward(elapsed_ms(profile_start));
        return gradients;
    }
    gradients.means = packed.slice(0, offset, offset + count * 3U)
        .reshape(tinytensor::TensorShape{count, std::size_t{3}});
    offset += count * 3U;
    if (precomputed)
        gradients.colors_precomp = packed.slice(0, offset, offset + sh_values)
            .reshape(tinytensor::TensorShape{count, std::size_t{3}});
    else if (!sh_adam)
        gradients.sh = packed.slice(0, offset, offset + sh_values)
            .reshape(model.sh.shape());
    offset += sh_values;
    offset += count;       // activated opacity
    offset += count * 3U;  // activated scale
    offset += count * 4U;  // normalized rotation
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
    if (context->backend->profile_stages)
        context->backend->record_backward(elapsed_ms(profile_start));
    return gradients;
}

OccupancyResult vulkan_occupancy(
    std::shared_ptr<void>& backend_value, const GaussianModel& model,
    const tinytensor::Tensor& points, const Camera& camera,
    const RasterizeOptions& options, bool reuse_prepared_view) {
    if (!points.is_valid() || points.device()!=tinytensor::Device::Vulkan ||
        points.dtype()!=tinytensor::DataType::Float32 || !points.is_contiguous() ||
        points.shape().rank()!=2 || points.shape()[1]!=3 ||
        points.shape()[0]>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Vulkan occupancy requires float32 [P,3] points");
    OccupancyResult result;
    result.occupancy=tinytensor::Tensor::empty({points.shape()[0]},tinytensor::Device::Vulkan);
    auto inside=tinytensor::Tensor::empty({points.shape()[0]},tinytensor::Device::Vulkan,
        tinytensor::DataType::Int32);
    if(points.shape()[0]) {
        auto backend=get_backend(backend_value);
        tinytensor::vulkan::submit_async();
        if (!reuse_prepared_view)
            backend->rasterizer.bind_model_device(device_gaussians(backend,model,options));
        backend->rasterizer.occupancy_device(buffer_view(points),
            static_cast<std::uint32_t>(points.shape()[0]),camera_view(camera),raster_settings(model, options),
            buffer_view(result.occupancy),buffer_view(inside),reuse_prepared_view);
    }
    result.inside=inside.gt(0);
    return result;
}

DepthSampleResult vulkan_sample_depth(
    std::shared_ptr<void>& backend_value, const GaussianModel& model,
    const tinytensor::Tensor& points, const Camera& camera,
    const RasterizeOptions& options) {
    if (points.device() != tinytensor::Device::Vulkan ||
        points.dtype() != tinytensor::DataType::Float32 ||
        points.shape().rank() != 2 || points.shape()[1] != 3 || points.shape()[0] == 0 ||
        model.size() == 0)
        throw std::invalid_argument("Vulkan sample_depth requires float32 [P,3] points and Gaussians");
    auto owner = get_backend(backend_value);
    if (!owner->sample_backend) owner->sample_backend = std::make_shared<VulkanRasterBackend>();
    auto backend = owner->sample_backend;
    auto context = std::make_shared<VulkanSampleContext>();
    context->backend = backend;
    context->points = points;
    DepthSampleResult result;
    result.camera_points = tinytensor::Tensor::empty(points.shape(), tinytensor::Device::Vulkan);
    result.inside = tinytensor::Tensor::empty({points.shape()[0]}, tinytensor::Device::Vulkan,
        tinytensor::DataType::Int32);
    tinytensor::vulkan::submit_async();
    backend->rasterizer.bind_model_device(device_gaussians(backend, model, options));
    backend->rasterizer.sample_depth_device(buffer_view(points),
        static_cast<std::uint32_t>(points.shape()[0]), camera_view(camera), raster_settings(model, options),
        buffer_view(result.camera_points), buffer_view(result.inside));
    result.backend_impl = context;
    backend->latest_sample = context;
    return result;
}

DepthSampleGradients vulkan_sample_depth_backward(const GaussianModel& model,
    const DepthSampleResult& sampled, const tinytensor::Tensor& gradient) {
    auto context = std::static_pointer_cast<VulkanSampleContext>(sampled.backend_impl);
    if (!context || context->consumed || context->backend->latest_sample.lock().get() != context.get())
        throw std::invalid_argument("Vulkan sample backward requires the latest unconsumed query");
    if (gradient.device() != tinytensor::Device::Vulkan ||
        gradient.dtype() != tinytensor::DataType::Float32 || gradient.shape() != context->points.shape())
        throw std::invalid_argument("Vulkan sample gradient must have shape [P,3]");
    auto& rasterizer = context->backend->rasterizer;
    auto packed = tinytensor::Tensor::empty(
        {rasterizer.sample_gradient_float_count(true)}, tinytensor::Device::Vulkan);
    DepthSampleGradients result;
    result.points = tinytensor::Tensor::empty(context->points.shape(), tinytensor::Device::Vulkan);
    tinytensor::vulkan::submit_async();
    rasterizer.sample_depth_backward_device(buffer_view(gradient), buffer_view(packed), buffer_view(result.points), true);
    const auto count = model.size();
    std::size_t offset = 0;
    const auto take = [&](std::size_t size, const tinytensor::TensorShape& shape) {
        auto value = packed.slice(0, offset, offset + size).reshape(shape);
        offset += size;
        return value;
    };
    result.model.means = take(count * 3, model.means.shape());
    // Depth queries have no appearance derivative. Keep that field absent
    // instead of allocating/writing a full zero SH gradient for every row.
    offset += count * 8; // activated opacity, scales, and rotation
    result.model.log_scales = take(count * 3, model.log_scales.shape());
    result.model.quaternions = take(count * 4, model.quaternions.shape());
    result.model.opacity_logits = take(count, model.opacity_logits.shape());
    result.model.refine_weight = take(count, tinytensor::TensorShape{count});
    context->consumed = true;
    return result;
}

void vulkan_multi_view_loss(const RenderResult& rendered,
    const splat_drender::vulkan::SplatDeviceMultiViewInput& input,
    const splat_drender::vulkan::SplatDeviceMultiViewOutput& output) {
    if (rendered.context.backend_impl) {
        get_context(rendered)->backend->rasterizer.multi_view_loss_device(input, output);
    } else {
        // Synthetic loss tests may supply attachments without a raster context.
        static thread_local std::shared_ptr<void> fallback;
        get_backend(fallback)->rasterizer.multi_view_loss_device(input, output);
    }
}

void vulkan_materialize_visibility(RenderResult& rendered) {
    if (rendered.visibility.is_valid()) return;
    auto context = get_context(rendered);
    rendered.visibility = context->visibility_bits.to(
        tinytensor::DataType::Float32);
}

}  // namespace photara::splat::detail
