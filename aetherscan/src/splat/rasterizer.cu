#include "splat/rasterizer.hpp"

#include "cuda_ops.hpp"
#include "core/vram_profiler.hpp"

#include "splat_drender/api.h"

#include <cuda_runtime.h>

#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace aetherscan::splat {

struct RasterCameraConstants {
    tinytensor::Tensor storage;
    const float* view_matrix() const { return storage.ptr<float>(); }
    const float* position() const { return storage.ptr<float>() + 16; }
};

struct RasterContextImpl {
    detail::ActivatedParameters activated;
    RasterCameraConstants constants;
    tinytensor::Tensor gaussian_buffer;
    tinytensor::Tensor grad_buffer;
    tinytensor::Tensor instance_buffer;
    tinytensor::Tensor pixel_buffer;
    tinytensor::Tensor tile_buffer;
    tinytensor::Tensor colors_precomp;
    RasterizeOptions options;
    Camera camera;
    splat_drender::ForwardResult forward{};
};

struct DepthSampleContextImpl {
    detail::ActivatedParameters activated;
    tinytensor::Tensor world_points;
    RasterCameraConstants constants;
    tinytensor::Tensor gaussian_buffer;
    tinytensor::Tensor grad_buffer;
    tinytensor::Tensor instance_buffer;
    tinytensor::Tensor pixel_buffer;
    tinytensor::Tensor tile_buffer;
    tinytensor::Tensor point_buffer;
    tinytensor::Tensor n_contrib;
    tinytensor::Tensor median_depth;
    RasterizeOptions options;
    Camera camera;
    splat_drender::Rasterizer::SampleCounts counts{};
};

namespace {

struct HostCameraConstants { float values[19]; };

__global__ void write_camera_constants(
    const HostCameraConstants constants, float* output) {
    if (threadIdx.x < 19) output[threadIdx.x] = constants.values[threadIdx.x];
}

RasterCameraConstants prepare_camera_constants(const Camera& camera) {
    HostCameraConstants host;
    std::copy(camera.world_to_camera.begin(), camera.world_to_camera.end(),
              host.values);
    std::copy(camera.position.begin(), camera.position.end(), host.values + 16);
    RasterCameraConstants result{
        tinytensor::Tensor::empty({19}, tinytensor::Device::CUDA)};
    write_camera_constants<<<1, 32>>>(host, result.storage.ptr<float>());
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string("Prepare raster camera failed: ") + cudaGetErrorString(error));
    return result;
}

void require_cuda_float_contiguous(
    const tinytensor::Tensor& tensor, const char* name) {
    if (!tensor.is_valid() || tensor.device() != tinytensor::Device::CUDA ||
        tensor.dtype() != tinytensor::DataType::Float32 || !tensor.is_contiguous())
        throw std::invalid_argument(
            std::string(name) + " must be a contiguous CUDA float32 tensor");
}

std::function<char*(std::size_t)> resize_buffer(tinytensor::Tensor& buffer, const char* label) {
    return [&buffer, label](const std::size_t bytes) -> char* {
        tinytensor::VramScope scope(label);
        if (!buffer.is_valid() || buffer.numel() < bytes)
            buffer = tinytensor::Tensor::empty(
                {bytes}, tinytensor::Device::CUDA, tinytensor::DataType::UInt8);
        return reinterpret_cast<char*>(buffer.data_ptr());
    };
}

splat_drender::CameraMode camera_mode_of(const Camera& camera) {
    switch (camera.model) {
        case CameraModel::opencv_fisheye:
            return splat_drender::CameraMode::opencv_fisheye;
        case CameraModel::equirectangular:
            return splat_drender::CameraMode::equirectangular;
        default:
            return splat_drender::CameraMode::pinhole;
    }
}

splat_drender::CameraView camera_view_of(
    const Camera& camera, const RasterCameraConstants& constants) {
    splat_drender::CameraView view;
    view.width = static_cast<int>(camera.width);
    view.height = static_cast<int>(camera.height);
    view.fx = camera.fx;
    view.fy = camera.fy;
    view.cx = camera.cx;
    view.cy = camera.cy;
    view.mode = camera_mode_of(camera);
    view.k1 = camera.k1;
    view.k2 = camera.k2;
    view.k3 = camera.k3;
    view.k4 = camera.k4;
    view.world_to_camera = constants.view_matrix();
    view.center = constants.position();
    return view;
}

splat_drender::Gaussians gaussians_of(
    const GaussianModel& model, const detail::ActivatedParameters& activated,
    const RasterizeOptions& options, const tinytensor::Tensor& colors_precomp) {
    splat_drender::Gaussians g;
    g.count = static_cast<int>(model.size());
    g.means = model.means.ptr<float>();
    g.opacities = activated.opacities.ptr<float>();
    g.scales = activated.scales.ptr<float>();
    g.rotations = activated.quaternions.ptr<float>();
    g.sh_degree = static_cast<int>(options.active_sh_degree);
    g.sh_bases = model.sh.is_valid() && model.sh.shape().rank() == 3
        ? static_cast<int>(model.sh.shape()[1])
        : 0;
    if (colors_precomp.is_valid()) {
        g.colors = colors_precomp.ptr<float>();
        g.sh = nullptr;
        g.sh_degree = 0;
        g.sh_bases = 0;
    } else {
        g.sh = model.sh.ptr<float>();
        g.colors = nullptr;
    }
    return g;
}

splat_drender::RenderSettings settings_of(const RasterizeOptions& options) {
    splat_drender::RenderSettings s;
    s.background[0] = options.background[0];
    s.background[1] = options.background[1];
    s.background[2] = options.background[2];
    s.point_depth_bracket = options.point_depth_bracket;
    s.point_depth_tolerance = options.point_depth_tolerance;
    s.scale_modifier = options.scale_modifier;
    s.kernel_size = options.kernel_size;
    s.need_depth = options.require_depth;
    s.debug = options.debug;
    const char* fixed_points = std::getenv("AETHERSCAN_SPLAT_DEVICE_POINTS");
    s.device_point_lists = fixed_points && fixed_points[0] == '1';
    // Measurement override: keep the pre-optimization per-Gaussian workspace
    // (always allocate the ray-plane/normal scratch) so the footprint change
    // can be A/B'd inside one binary.
    const char* force_geometry =
        std::getenv("AETHERSCAN_SPLAT_FORCE_GEOMETRY_WORKSPACE");
    s.force_geometry_workspace = force_geometry && force_geometry[0] == '1';
    return s;
}

splat_drender::WorkspacePools pools_of(RasterContextImpl& context) {
    splat_drender::WorkspacePools pools;
    pools.gaussian = resize_buffer(context.gaussian_buffer, "workspace.gaussian");
    pools.grad = resize_buffer(context.grad_buffer, "workspace.grad");
    pools.instance = resize_buffer(context.instance_buffer, "workspace.instance");
    pools.pixel = resize_buffer(context.pixel_buffer, "workspace.pixel");
    pools.tile = resize_buffer(context.tile_buffer, "workspace.tile");
    return pools;
}

splat_drender::WorkspacePools pools_of(DepthSampleContextImpl& context) {
    splat_drender::WorkspacePools pools;
    pools.gaussian = resize_buffer(context.gaussian_buffer, "workspace.gaussian");
    pools.grad = resize_buffer(context.grad_buffer, "workspace.grad");
    pools.instance = resize_buffer(context.instance_buffer, "workspace.instance");
    pools.pixel = resize_buffer(context.pixel_buffer, "workspace.pixel");
    pools.tile = resize_buffer(context.tile_buffer, "workspace.tile");
    pools.point = resize_buffer(context.point_buffer, "workspace.point");
    return pools;
}

void validate_model(
    const GaussianModel& model, const RasterizeOptions& options) {
    require_cuda_float_contiguous(model.means, "model.means");
    require_cuda_float_contiguous(model.log_scales, "model.log_scales");
    require_cuda_float_contiguous(model.quaternions, "model.quaternions");
    require_cuda_float_contiguous(model.opacity_logits, "model.opacity_logits");
    require_cuda_float_contiguous(model.sh, "model.sh");
    if (model.filter_3d.is_valid())
        require_cuda_float_contiguous(model.filter_3d, "model.filter_3d");
    if (options.colors_precomp.is_valid())
        require_cuda_float_contiguous(
            options.colors_precomp, "colors_precomp");
    if (model.means.shape().rank() != 2 || model.means.shape()[1] != 3 ||
        model.log_scales.shape() != model.means.shape() ||
        model.quaternions.shape().rank() != 2 ||
        model.quaternions.shape()[0] != model.size() ||
        model.quaternions.shape()[1] != 4 ||
        model.opacity_logits.numel() != model.size() ||
        model.sh.shape().rank() != 3 || model.sh.shape()[0] != model.size() ||
        model.sh.shape()[2] != 3 ||
        (model.filter_3d.is_valid() &&
         (model.filter_3d.shape().rank() != 2 ||
          model.filter_3d.shape()[0] != model.size() ||
          model.filter_3d.shape()[1] != 1)))
        throw std::invalid_argument("Invalid splat model tensor shapes");
    if (options.colors_precomp.is_valid() &&
        (options.colors_precomp.shape().rank() != 2 ||
         options.colors_precomp.shape()[0] != model.size() ||
         options.colors_precomp.shape()[1] != 3))
        throw std::invalid_argument(
            "Precomputed colors must have shape [N,3]");
}

}  // namespace

RenderResult Rasterizer::forward(
    const GaussianModel& model, const Camera& camera,
    const RasterizeOptions& requested_options) const {
    validate_model(model, requested_options);
    if (camera.width == 0 || camera.height == 0)
        throw std::invalid_argument("Splat camera dimensions must be positive");

    RenderResult result;
    auto context = std::make_shared<RasterContextImpl>();
    context->camera = camera;
    context->options = requested_options;
    context->colors_precomp = requested_options.colors_precomp;
    context->options.active_sh_degree = std::min(
        requested_options.active_sh_degree, model.sh_degree);
    context->activated = detail::activate_parameters(model);
    context->constants = prepare_camera_constants(camera);

    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    result.color = tinytensor::Tensor::zeros(
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    result.alpha = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    // Color-only renders (RasterizeOptions::require_depth off) neither write
    // nor read these two channels; allocating them meant two zeroed images per
    // render. The renderer receives null pointers for them in that case.
    if (context->options.require_depth) {
        result.median_depth = tinytensor::Tensor::zeros(
            {camera.height, camera.width}, tinytensor::Device::CUDA);
        result.normal = tinytensor::Tensor::zeros(
            {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    }
    result.radii = tinytensor::Tensor::zeros(
        {model.size()}, tinytensor::Device::CUDA, tinytensor::DataType::Int32);
    result.visibility = tinytensor::Tensor::zeros(
        {model.size()}, tinytensor::Device::CUDA);

    if (model.size() != 0 && pixels != 0) {
        splat_drender::RenderOutputs out;
        out.color = result.color.ptr<float>();
        out.alpha = result.alpha.ptr<float>();
        out.median_depth = result.median_depth.ptr<float>();
        out.normal = result.normal.ptr<float>();
        out.visibility = result.visibility.ptr<float>();
        out.radii = result.radii.ptr<int>();
        context->forward = splat_drender::Rasterizer::forward(
            pools_of(*context),
            gaussians_of(model, context->activated, context->options,
                         context->colors_precomp),
            camera_view_of(camera, context->constants),
            settings_of(context->options), out);
    }
    result.rendered_instances = context->forward.instance_count;
    result.context.impl = std::move(context);
    return result;
}

ModelGradients Rasterizer::backward(
    const GaussianModel& model, const RenderResult& rendered,
    const tinytensor::Tensor& grad_color,
    const tinytensor::Tensor& grad_alpha,
    const tinytensor::Tensor& grad_depth,
    const tinytensor::Tensor& grad_normal,
    const tinytensor::Tensor& densify_map, const SHAdamUpdate* sh_adam) const {
    if (!rendered.context.impl)
        throw std::invalid_argument("Splat backward requires a live forward context");
    const auto& context = *rendered.context.impl;
    require_cuda_float_contiguous(grad_color, "grad_color");
    require_cuda_float_contiguous(grad_alpha, "grad_alpha");
    // The depth and normal channels only exist when the matching forward
    // rendered them; a color-only step passes empty tensors instead.
    if (context.options.require_depth) {
        require_cuda_float_contiguous(grad_depth, "grad_depth");
        require_cuda_float_contiguous(grad_normal, "grad_normal");
    }
    if (densify_map.is_valid() && densify_map.numel() != 0)
        require_cuda_float_contiguous(densify_map, "densify_map");
    const auto count = model.size();

    ModelGradients gradients;
    gradients.means = tinytensor::Tensor::zeros_like(model.means);
    if (sh_adam && (context.colors_precomp.is_valid() ||
        context.forward.instance_count <= 0 || model.sh.shape()[1] > 16 ||
        sh_adam->first.numel() != model.sh.numel() ||
        (sh_adam->second.numel() != model.size() &&
         sh_adam->second.numel() != model.sh.numel())))
        throw std::invalid_argument("Incompatible fused SH Adam context/state");
    if (!sh_adam) gradients.sh = tinytensor::Tensor::zeros_like(model.sh);
    auto grad_opacities = tinytensor::Tensor::zeros(
        {count, 1}, tinytensor::Device::CUDA);
    auto grad_scales = tinytensor::Tensor::zeros(
        {count, 3}, tinytensor::Device::CUDA);
    auto grad_quaternions = tinytensor::Tensor::zeros(
        {count, 4}, tinytensor::Device::CUDA);
    auto refine_weight = tinytensor::Tensor::zeros(
        {count}, tinytensor::Device::CUDA);
    tinytensor::Tensor densify_weight;
    tinytensor::Tensor densify_weight_den;
    const bool scatter_densify =
        densify_map.is_valid() && densify_map.numel() != 0;
    if (scatter_densify) {
        densify_weight = tinytensor::Tensor::zeros(
            {count}, tinytensor::Device::CUDA);
        densify_weight_den = tinytensor::Tensor::zeros(
            {count}, tinytensor::Device::CUDA);
    }
    tinytensor::Tensor grad_colors;
    if (context.colors_precomp.is_valid())
        grad_colors = tinytensor::Tensor::zeros(
            {count, 3}, tinytensor::Device::CUDA);

    if (count != 0 && context.forward.instance_count > 0) {
        splat_drender::ForwardOutputsView fwd_out;
        fwd_out.alpha = rendered.alpha.ptr<float>();
        fwd_out.median_depth = rendered.median_depth.ptr<float>();
        fwd_out.normal = rendered.normal.ptr<float>();
        fwd_out.radii = rendered.radii.ptr<int>();
        splat_drender::LossGradients dL;
        dL.color = grad_color.ptr<float>();
        dL.alpha = grad_alpha.ptr<float>();
        dL.median_depth = grad_depth.ptr<float>();
        dL.normal = grad_normal.ptr<float>();
        dL.densify_map = scatter_densify ? densify_map.ptr<float>() : nullptr;
        splat_drender::ModelGradients grads;
        if (sh_adam) {
            const auto stride = model.sh.shape()[1] * 3;
            // Tensor copies share storage; this opt-in backward owns the SH update.
            auto parameter = model.sh;
            auto first = sh_adam->first;
            auto second = sh_adam->second;
            grads.sh_adam = {parameter.ptr<float>(), first.ptr<float>(),
                second.ptr<float>(), sh_adam->second.numel() == count,
                sh_adam->lr, sh_adam->rest_lr, sh_adam->beta1, sh_adam->beta2,
                sh_adam->correction1, sh_adam->correction2, sh_adam->epsilon,
                stride > 3 && sh_adam->regularization_weight > 0.F
                    ? 2.F * sh_adam->regularization_weight / float(count * (stride - 3)) : 0.F};
        }
        grads.means = gradients.means.ptr<float>();
        grads.sh = context.colors_precomp.is_valid()
            ? nullptr
            : gradients.sh.ptr<float>();
        grads.colors = grad_colors.is_valid() ? grad_colors.ptr<float>() : nullptr;
        grads.opacities = grad_opacities.ptr<float>();
        grads.scales = grad_scales.ptr<float>();
        grads.rotations = grad_quaternions.ptr<float>();
        grads.refine_weight = refine_weight.ptr<float>();
        grads.densify_weight =
            scatter_densify ? densify_weight.ptr<float>() : nullptr;
        grads.densify_weight_den =
            scatter_densify ? densify_weight_den.ptr<float>() : nullptr;
        splat_drender::Rasterizer::backward(
            pools_of(*rendered.context.impl),
            gaussians_of(model, context.activated, context.options,
                         context.colors_precomp),
            camera_view_of(context.camera, context.constants),
            settings_of(context.options), context.forward, fwd_out, dL, grads);
    }
    detail::chain_parameter_gradients(
        model, context.activated, grad_scales, grad_quaternions,
        grad_opacities, gradients);
    gradients.refine_weight = std::move(refine_weight);
    gradients.densify_weight = std::move(densify_weight);
    gradients.densify_weight_den = std::move(densify_weight_den);
    if (context.colors_precomp.is_valid())
        gradients.colors_precomp = std::move(grad_colors);
    if (const char* prefix = std::getenv("AETHERSCAN_SPLAT_GRAD_DUMP")) {
        static int calls = 0; ++calls;
        if (calls <= 3 || calls == 10 || calls == 50 || calls == 100) {

            auto dump = [&](const char* name, const tinytensor::Tensor& tensor) {
                auto values = tensor.to_vector();
                std::ofstream file(std::string(prefix) + std::to_string(calls) + name, std::ios::binary);
                file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            };
            dump(".mean", gradients.means); dump(".scale", gradients.log_scales);
            dump(".quat", gradients.quaternions); dump(".opacity", gradients.opacity_logits);
            dump(".sh", gradients.sh); dump(".refine", gradients.refine_weight);
            dump(".rawscale", grad_scales); dump(".rawquat", grad_quaternions);
            dump(".inputmeans", model.means); dump(".inputscales", context.activated.scales);
            dump(".inputrot", context.activated.quaternions); dump(".inputopa", context.activated.opacities);
            dump(".inputsh", model.sh); dump(".inputcamera", context.constants.storage);
            dump(".gc", grad_color); dump(".ga", grad_alpha); dump(".gd", grad_depth); dump(".gn", grad_normal);
            std::ofstream camera_file(std::string(prefix) + std::to_string(calls) + ".hostcamera", std::ios::binary);
            const float host_camera[] = {context.camera.fx,context.camera.fy,context.camera.cx,context.camera.cy,
                context.options.background[0],context.options.background[1],context.options.background[2]};
            camera_file.write(reinterpret_cast<const char*>(host_camera),sizeof(host_camera));
        }
    }
    return gradients;
}

const float* Rasterizer::projected_mean2d(const RenderResult& rendered) const {
    if (!rendered.context.impl) return nullptr;
    const auto& buffer = rendered.context.impl->gaussian_buffer;
    const std::size_t count = rendered.radii.is_valid()
        ? rendered.radii.numel() : 0;
    if (!buffer.is_valid() || count == 0 ||
        buffer.numel() < count * sizeof(float) * 2)
        return nullptr;
    return reinterpret_cast<const float*>(buffer.data_ptr());
}

DepthSampleResult Rasterizer::sample_depth(
    const GaussianModel& model, const tinytensor::Tensor& world_points,
    const Camera& camera, const RasterizeOptions& requested_options) const {
    require_cuda_float_contiguous(world_points, "world_points");
    if (world_points.shape().rank() != 2 || world_points.shape()[1] != 3)
        throw std::invalid_argument("sample_depth points must have shape [P,3]");
    if (model.size() == 0 || world_points.shape()[0] == 0)
        throw std::invalid_argument("sample_depth requires Gaussians and points");
    validate_model(model, requested_options);
    auto context = std::make_shared<DepthSampleContextImpl>();
    context->camera = camera;
    context->options = requested_options;
    context->options.active_sh_degree = std::min(
        requested_options.active_sh_degree, model.sh_degree);
    context->world_points = world_points;
    context->activated = detail::activate_parameters(model);
    context->constants = prepare_camera_constants(camera);
    const std::size_t point_count = world_points.shape()[0];
    DepthSampleResult result;
    result.camera_points = tinytensor::Tensor::zeros(
        {point_count, std::size_t{3}}, tinytensor::Device::CUDA);
    result.inside = tinytensor::Tensor::zeros(
        {point_count}, tinytensor::Device::CUDA, tinytensor::DataType::Bool);
    context->n_contrib = tinytensor::Tensor::zeros(
        {point_count}, tinytensor::Device::CUDA, tinytensor::DataType::Int32);
    context->median_depth = tinytensor::Tensor::zeros(
        {point_count}, tinytensor::Device::CUDA);
    splat_drender::SampleOutputs out;
    out.ray_points = reinterpret_cast<float3*>(result.camera_points.ptr<float>());
    out.median_depth = context->median_depth.ptr<float>();
    out.n_contrib = reinterpret_cast<unsigned*>(context->n_contrib.ptr<int>());
    out.inside = result.inside.ptr<bool>();
    context->counts = splat_drender::Rasterizer::sample_depth(
        pools_of(*context),
        gaussians_of(model, context->activated, context->options, {}),
        camera_view_of(camera, context->constants),
        settings_of(context->options), world_points.ptr<float>(),
        static_cast<int>(point_count), out);
    result.context = std::move(context);
    return result;
}

DepthSampleGradients Rasterizer::sample_depth_backward(
    const GaussianModel& model, const DepthSampleResult& sampled,
    const tinytensor::Tensor& grad_camera_points) const {
    if (!sampled.context)
        throw std::invalid_argument(
            "sample_depth backward requires a live context");
    require_cuda_float_contiguous(grad_camera_points, "grad_camera_points");
    const auto& context = *sampled.context;
    const std::size_t point_count = context.world_points.shape()[0];
    const std::size_t count = model.size();
    DepthSampleGradients result;
    result.model.means = tinytensor::Tensor::zeros_like(model.means);
    result.model.sh = tinytensor::Tensor::zeros_like(model.sh);
    result.model.refine_weight = tinytensor::Tensor::zeros(
        {count}, tinytensor::Device::CUDA);
    result.points = tinytensor::Tensor::zeros_like(context.world_points);
    auto grad_opacities = tinytensor::Tensor::zeros(
        {count, std::size_t{1}}, tinytensor::Device::CUDA);
    auto grad_scales = tinytensor::Tensor::zeros(
        {count, std::size_t{3}}, tinytensor::Device::CUDA);
    auto grad_quaternions = tinytensor::Tensor::zeros(
        {count, std::size_t{4}}, tinytensor::Device::CUDA);
        splat_drender::SampleOutputsView fwd;
        fwd.median_depth = context.median_depth.ptr<float>();
        fwd.n_contrib = reinterpret_cast<const unsigned*>(context.n_contrib.ptr<int>());
        fwd.inside = sampled.inside.ptr<bool>();
    splat_drender::SampleGradients point_grads;
    point_grads.points = reinterpret_cast<float3*>(result.points.ptr<float>());
    splat_drender::ModelGradients grads;
    grads.means = result.model.means.ptr<float>();
    grads.sh = result.model.sh.ptr<float>();
    grads.opacities = grad_opacities.ptr<float>();
    grads.scales = grad_scales.ptr<float>();
    grads.rotations = grad_quaternions.ptr<float>();
    splat_drender::Rasterizer::sample_depth_backward(
        pools_of(*sampled.context),
        gaussians_of(model, context.activated, context.options, {}),
        camera_view_of(context.camera, context.constants),
        settings_of(context.options), context.world_points.ptr<float>(),
        static_cast<int>(point_count), context.counts, fwd,
        reinterpret_cast<const float3*>(grad_camera_points.ptr<float>()),
        point_grads, grads);
    detail::chain_parameter_gradients(
        model, context.activated, grad_scales, grad_quaternions,
        grad_opacities, result.model);
    return result;
}

OccupancyResult Rasterizer::evaluate_occupancy(
    const GaussianModel& model, const tinytensor::Tensor& world_points,
    const Camera& camera, const RasterizeOptions& requested_options) const {
    require_cuda_float_contiguous(world_points, "world_points");
    if (world_points.shape().rank() != 2 || world_points.shape()[1] != 3)
        throw std::invalid_argument("occupancy points must have shape [P,3]");
    if (model.size() == 0 || world_points.shape()[0] == 0)
        throw std::invalid_argument(
            "occupancy evaluation requires Gaussians and points");
    if (camera.width == 0 || camera.height == 0)
        throw std::invalid_argument(
            "occupancy camera dimensions must be positive");
    validate_model(model, requested_options);

    const detail::ActivatedParameters activated =
        detail::activate_parameters(model);
    const auto constants = prepare_camera_constants(camera);
    tinytensor::Tensor gaussian_buffer;
    tinytensor::Tensor instance_buffer;
    tinytensor::Tensor tile_buffer;
    tinytensor::Tensor point_buffer;
    const std::size_t point_count = world_points.shape()[0];
    OccupancyResult result;
    result.occupancy = tinytensor::Tensor::zeros(
        {point_count}, tinytensor::Device::CUDA);
    result.inside = tinytensor::Tensor::zeros(
        {point_count}, tinytensor::Device::CUDA, tinytensor::DataType::Bool);
    splat_drender::WorkspacePools pools;
    pools.gaussian = resize_buffer(gaussian_buffer, "workspace.gaussian");
    pools.instance = resize_buffer(instance_buffer, "workspace.instance");
    pools.tile = resize_buffer(tile_buffer, "workspace.tile");
    pools.point = resize_buffer(point_buffer, "workspace.point");
    splat_drender::OccupancyOutputs out;
    out.occupancy = result.occupancy.ptr<float>();
    out.inside = result.inside.ptr<bool>();
    RasterizeOptions options = requested_options;
    options.active_sh_degree = std::min(options.active_sh_degree, model.sh_degree);
    splat_drender::Rasterizer::evaluate_occupancy(
        pools, gaussians_of(model, activated, options, {}),
        camera_view_of(camera, constants), settings_of(options),
        world_points.ptr<float>(), static_cast<int>(point_count), out);
    return result;
}

}  // namespace aetherscan::splat



