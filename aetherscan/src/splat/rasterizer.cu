#include "splat/rasterizer.hpp"

#include "cuda_ops.hpp"
#include "rasterizer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace aetherscan::splat {

struct RasterCameraConstants {
    tinytensor::Tensor storage;
    const float* view_matrix() const { return storage.ptr<float>(); }
    const float* position() const { return storage.ptr<float>() + 16; }
    const float* background() const { return storage.ptr<float>() + 19; }
};

struct RasterContextImpl {
    detail::ActivatedParameters activated;
    RasterCameraConstants constants;
    tinytensor::Tensor geometry_buffer;
    tinytensor::Tensor binning_buffer;
    tinytensor::Tensor image_buffer;
    tinytensor::Tensor tile_buffer;
    tinytensor::Tensor colors_precomp;
    RasterizeOptions options;
    Camera camera;
    int rendered_instances{};
};

struct DepthSampleContextImpl {
    detail::ActivatedParameters activated;
    tinytensor::Tensor world_points;
    RasterCameraConstants constants;
    tinytensor::Tensor geometry_buffer;
    tinytensor::Tensor binning_buffer;
    tinytensor::Tensor point_buffer;
    tinytensor::Tensor point_binning_buffer;
    tinytensor::Tensor tile_buffer;
    tinytensor::Tensor duplicated_tile_buffer;
    RasterizeOptions options;
    Camera camera;
    int3 counts{};
};

namespace {

struct HostCameraConstants { float values[22]; };

__global__ void write_camera_constants(
    const HostCameraConstants constants, float* output) {
    if (threadIdx.x < 22) output[threadIdx.x] = constants.values[threadIdx.x];
}

RasterCameraConstants prepare_camera_constants(
    const Camera& camera, const std::array<float, 3>& background = {}) {
    HostCameraConstants host;
    std::copy(camera.world_to_camera.begin(), camera.world_to_camera.end(), host.values);
    std::copy(camera.position.begin(), camera.position.end(), host.values + 16);
    std::copy(background.begin(), background.end(), host.values + 19);
    RasterCameraConstants result{
        tinytensor::Tensor::empty({22}, tinytensor::Device::CUDA)};
    // CUDA captures launch arguments before returning. No temporary pinned
    // tensors or three blocking from_vector uploads on the training stream.
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

std::function<char*(std::size_t)> resize_buffer(tinytensor::Tensor& buffer) {
    return [&buffer](const std::size_t bytes) -> char* {
        if (!buffer.is_valid() || buffer.numel() < bytes)
            buffer = tinytensor::Tensor::empty(
                {bytes}, tinytensor::Device::CUDA, tinytensor::DataType::UInt8);
        return reinterpret_cast<char*>(buffer.data_ptr());
    };
}

std::function<char*(std::size_t)> resize_zeroed_buffer(
    tinytensor::Tensor& buffer) {
    return [&buffer](const std::size_t bytes) -> char* {
        // GGGS backward accumulates conic/opacity and other intermediates with
        // atomicAdd. FasterGS's torch wrapper uses resizeFunctional<true>,
        // which clears this entire chunk on every call. Recycled uninitialized
        // memory here corrupts all parameter gradients after the first step.
        buffer = tinytensor::Tensor::zeros(
            {bytes}, tinytensor::Device::CUDA,
            tinytensor::DataType::UInt8);
        return reinterpret_cast<char*>(buffer.data_ptr());
    };
}

}  // namespace

RenderResult Rasterizer::forward(
    const GaussianModel& model, const Camera& camera,
    const RasterizeOptions& requested_options) const {
    require_cuda_float_contiguous(model.means, "model.means");
    require_cuda_float_contiguous(model.log_scales, "model.log_scales");
    require_cuda_float_contiguous(model.quaternions, "model.quaternions");
    require_cuda_float_contiguous(model.opacity_logits, "model.opacity_logits");
    require_cuda_float_contiguous(model.sh, "model.sh");
    if (model.filter_3d.is_valid())
        require_cuda_float_contiguous(model.filter_3d, "model.filter_3d");
    if (requested_options.colors_precomp.is_valid())
        require_cuda_float_contiguous(
            requested_options.colors_precomp, "colors_precomp");
    if (camera.width == 0 || camera.height == 0)
        throw std::invalid_argument("GGGS camera dimensions must be positive");
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
        throw std::invalid_argument("Invalid GGGS model tensor shapes");
    if (requested_options.colors_precomp.is_valid() &&
        (requested_options.colors_precomp.shape().rank() != 2 ||
         requested_options.colors_precomp.shape()[0] != model.size() ||
         requested_options.colors_precomp.shape()[1] != 3))
        throw std::invalid_argument(
            "GGGS precomputed colors must have shape [N,3]");

    RenderResult result;
    auto context = std::make_shared<RasterContextImpl>();
    context->camera = camera;
    context->options = requested_options;
    context->colors_precomp = requested_options.colors_precomp;
    context->options.active_sh_degree = std::min(
        requested_options.active_sh_degree, model.sh_degree);
    context->activated = detail::activate_parameters(model);
    context->constants = prepare_camera_constants(
        camera, context->options.background);

    const std::size_t pixels = static_cast<std::size_t>(camera.width) * camera.height;
    result.color = tinytensor::Tensor::zeros(
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    result.alpha = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    result.median_depth = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    result.normal = tinytensor::Tensor::zeros(
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    result.radii = tinytensor::Tensor::zeros(
        {model.size()}, tinytensor::Device::CUDA, tinytensor::DataType::Int32);
    result.visibility = tinytensor::Tensor::zeros(
        {model.size()}, tinytensor::Device::CUDA);

    if (model.size() != 0 && pixels != 0) {
        const unsigned total_bases = static_cast<unsigned>(model.sh.shape()[1]);
        const unsigned requested_bases =
            (context->options.active_sh_degree + 1U) *
            (context->options.active_sh_degree + 1U);
        const int active_bases = static_cast<int>(std::min(requested_bases, total_bases));
        context->rendered_instances = CudaRasterizer::Rasterizer::forward(
            resize_buffer(context->geometry_buffer),
            resize_buffer(context->binning_buffer),
            resize_buffer(context->image_buffer),
            resize_buffer(context->tile_buffer),
            static_cast<int>(model.size()),
            static_cast<int>(context->options.active_sh_degree),
            static_cast<int>(total_bases), 0, 0,
            context->constants.background(),
            static_cast<int>(camera.width), static_cast<int>(camera.height),
            model.means.ptr<float>(),
            context->colors_precomp.is_valid()
                ? context->colors_precomp.ptr<float>()
                : nullptr,
            context->activated.opacities.ptr<float>(),
            context->activated.scales.ptr<float>(),
            context->activated.quaternions.ptr<float>(), nullptr,
            context->colors_precomp.is_valid() ? nullptr : model.sh.ptr<float>(),
            nullptr, nullptr, nullptr,
            context->options.scale_modifier,
            context->constants.view_matrix(),
            context->constants.position(), camera.fx, camera.fy,
            camera.cx, camera.cy, context->options.kernel_size, false,
            result.color.ptr<float>(), result.median_depth.ptr<float>(),
            result.alpha.ptr<float>(), result.normal.ptr<float>(),
            result.visibility.ptr<float>(), result.radii.ptr<int>(),
            context->options.require_depth,
            context->options.debug,
            static_cast<int>(camera.model), camera.k1, camera.k2,
            camera.k3, camera.k4);
        (void)active_bases; // The reference API derives active bases from SHD.
    }
    result.rendered_instances = context->rendered_instances;
    result.context.impl = std::move(context);
    return result;
}

ModelGradients Rasterizer::backward(
    const GaussianModel& model, const RenderResult& rendered,
    const tinytensor::Tensor& grad_color,
    const tinytensor::Tensor& grad_alpha,
    const tinytensor::Tensor& grad_depth,
    const tinytensor::Tensor& grad_normal) const {
    if (!rendered.context.impl)
        throw std::invalid_argument("GGGS backward requires a live forward context");
    require_cuda_float_contiguous(grad_color, "grad_color");
    require_cuda_float_contiguous(grad_alpha, "grad_alpha");
    require_cuda_float_contiguous(grad_depth, "grad_depth");
    require_cuda_float_contiguous(grad_normal, "grad_normal");
    const auto& context = *rendered.context.impl;
    const auto count = model.size();

    ModelGradients gradients;
    gradients.means = tinytensor::Tensor::zeros_like(model.means);
    gradients.sh = tinytensor::Tensor::zeros_like(model.sh);
    auto grad_means2d = tinytensor::Tensor::zeros({count, 3}, tinytensor::Device::CUDA);
    auto grad_colors = tinytensor::Tensor::zeros({count, 3}, tinytensor::Device::CUDA);
    auto grad_opacities = tinytensor::Tensor::zeros({count, 1}, tinytensor::Device::CUDA);
    auto grad_scales = tinytensor::Tensor::zeros({count, 3}, tinytensor::Device::CUDA);
    auto grad_quaternions = tinytensor::Tensor::zeros({count, 4}, tinytensor::Device::CUDA);
    auto grad_covariance = tinytensor::Tensor::zeros({count, 6}, tinytensor::Device::CUDA);
    auto refine_weight = tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA);
    tinytensor::Tensor scratch;

    if (count != 0) {
        CudaRasterizer::Rasterizer::backward(
            resize_zeroed_buffer(scratch), static_cast<int>(count),
            static_cast<int>(context.options.active_sh_degree),
            static_cast<int>(model.sh.shape()[1]), 0, 0,
            context.rendered_instances, context.constants.background(),
            static_cast<int>(context.camera.width),
            static_cast<int>(context.camera.height), model.means.ptr<float>(),
            context.colors_precomp.is_valid()
                ? context.colors_precomp.ptr<float>()
                : nullptr,
            context.activated.opacities.ptr<float>(),
            context.activated.scales.ptr<float>(),
            context.activated.quaternions.ptr<float>(), nullptr,
            context.colors_precomp.is_valid() ? nullptr : model.sh.ptr<float>(),
            nullptr, nullptr, nullptr,
            context.options.scale_modifier, context.constants.view_matrix(),
            context.constants.position(), context.camera.fx,
            context.camera.fy, context.camera.cx, context.camera.cy,
            context.options.kernel_size, rendered.radii.ptr<int>(),
            rendered.alpha.ptr<float>(), rendered.normal.ptr<float>(),
            rendered.median_depth.ptr<float>(),
            const_cast<char*>(reinterpret_cast<const char*>(
                context.geometry_buffer.data_ptr())),
            const_cast<char*>(reinterpret_cast<const char*>(
                context.binning_buffer.data_ptr())),
            const_cast<char*>(reinterpret_cast<const char*>(
                context.image_buffer.data_ptr())),
            const_cast<char*>(reinterpret_cast<const char*>(
                context.tile_buffer.data_ptr())),
            grad_color.ptr<float>(), grad_depth.ptr<float>(),
            grad_alpha.ptr<float>(), grad_normal.ptr<float>(),
            gradients.means.ptr<float>(), grad_means2d.ptr<float>(),
            grad_colors.ptr<float>(), grad_opacities.ptr<float>(),
            grad_scales.ptr<float>(), grad_quaternions.ptr<float>(),
            grad_covariance.ptr<float>(), gradients.sh.ptr<float>(),
            nullptr, nullptr, nullptr, refine_weight.ptr<float>(),
            context.options.require_depth, context.options.debug,
            static_cast<int>(context.camera.model), context.camera.k1,
            context.camera.k2, context.camera.k3, context.camera.k4);
    }
    detail::chain_parameter_gradients(
        model, context.activated, grad_scales, grad_quaternions,
        grad_opacities, gradients);
    gradients.refine_weight = std::move(refine_weight);
    if (context.colors_precomp.is_valid())
        gradients.colors_precomp = std::move(grad_colors);
    return gradients;
}

DepthSampleResult Rasterizer::sample_depth(
    const GaussianModel& model, const tinytensor::Tensor& world_points,
    const Camera& camera, const RasterizeOptions& requested_options) const {
    require_cuda_float_contiguous(world_points, "world_points");
    if (world_points.shape().rank() != 2 || world_points.shape()[1] != 3)
        throw std::invalid_argument("GGGS sample_depth points must have shape [P,3]");
    if (model.size() == 0 || world_points.shape()[0] == 0)
        throw std::invalid_argument("GGGS sample_depth requires Gaussians and points");
    auto context = std::make_shared<DepthSampleContextImpl>();
    context->camera = camera;
    context->options = requested_options;
    context->world_points = world_points;
    context->activated = detail::activate_parameters(model);
    context->constants = prepare_camera_constants(camera);
    const std::size_t point_count = world_points.shape()[0];
    DepthSampleResult result;
    result.camera_points = tinytensor::Tensor::zeros(
        {point_count, std::size_t{3}}, tinytensor::Device::CUDA);
    result.inside = tinytensor::Tensor::zeros(
        {point_count}, tinytensor::Device::CUDA, tinytensor::DataType::Bool);
    context->counts = CudaRasterizer::Rasterizer::sampleDepth(
        resize_buffer(context->geometry_buffer),
        resize_buffer(context->binning_buffer),
        resize_buffer(context->point_buffer),
        resize_buffer(context->point_binning_buffer),
        resize_buffer(context->tile_buffer),
        resize_buffer(context->duplicated_tile_buffer),
        static_cast<int>(point_count), static_cast<int>(model.size()),
        static_cast<int>(camera.width), static_cast<int>(camera.height),
        world_points.ptr<float>(), model.means.ptr<float>(),
        context->activated.opacities.ptr<float>(),
        context->activated.scales.ptr<float>(), requested_options.scale_modifier,
        context->activated.quaternions.ptr<float>(), nullptr,
        context->constants.view_matrix(), context->constants.position(),
        camera.fx, camera.fy, camera.cx, camera.cy,
        requested_options.kernel_size, false,
        result.camera_points.ptr<float>(), result.inside.ptr<bool>(),
        requested_options.debug,
        static_cast<int>(camera.model), camera.k1, camera.k2,
        camera.k3, camera.k4);
    result.context = std::move(context);
    return result;
}

DepthSampleGradients Rasterizer::sample_depth_backward(
    const GaussianModel& model, const DepthSampleResult& sampled,
    const tinytensor::Tensor& grad_camera_points) const {
    if (!sampled.context)
        throw std::invalid_argument("GGGS sample_depth backward requires a live context");
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
    auto grad_means2d = tinytensor::Tensor::zeros(
        {count, std::size_t{3}}, tinytensor::Device::CUDA);
    auto grad_points2d = tinytensor::Tensor::zeros(
        {point_count, std::size_t{2}}, tinytensor::Device::CUDA);
    auto grad_opacities = tinytensor::Tensor::zeros(
        {count, std::size_t{1}}, tinytensor::Device::CUDA);
    auto grad_scales = tinytensor::Tensor::zeros(
        {count, std::size_t{3}}, tinytensor::Device::CUDA);
    auto grad_quaternions = tinytensor::Tensor::zeros(
        {count, std::size_t{4}}, tinytensor::Device::CUDA);
    auto grad_covariance = tinytensor::Tensor::zeros(
        {count, std::size_t{6}}, tinytensor::Device::CUDA);
    tinytensor::Tensor scratch;
    CudaRasterizer::Rasterizer::sampleDepthBackward(
        resize_zeroed_buffer(scratch), static_cast<int>(point_count),
        static_cast<int>(count), context.counts.y, context.counts.x,
        context.counts.z, static_cast<int>(context.camera.width),
        static_cast<int>(context.camera.height),
        context.world_points.ptr<float>(), model.means.ptr<float>(),
        context.activated.opacities.ptr<float>(),
        context.activated.scales.ptr<float>(), context.options.scale_modifier,
        context.activated.quaternions.ptr<float>(), nullptr,
        context.constants.view_matrix(), context.constants.position(),
        context.camera.fx, context.camera.fy, context.camera.cx,
        context.camera.cy, context.options.kernel_size,
        const_cast<char*>(reinterpret_cast<const char*>(context.geometry_buffer.data_ptr())),
        const_cast<char*>(reinterpret_cast<const char*>(context.binning_buffer.data_ptr())),
        const_cast<char*>(reinterpret_cast<const char*>(context.point_buffer.data_ptr())),
        const_cast<char*>(reinterpret_cast<const char*>(context.point_binning_buffer.data_ptr())),
        const_cast<char*>(reinterpret_cast<const char*>(context.tile_buffer.data_ptr())),
        const_cast<char*>(reinterpret_cast<const char*>(context.duplicated_tile_buffer.data_ptr())),
        sampled.inside.ptr<bool>(), grad_camera_points.ptr<float>(),
        grad_means2d.ptr<float>(), grad_points2d.ptr<float>(),
        grad_opacities.ptr<float>(), result.model.means.ptr<float>(),
        grad_covariance.ptr<float>(), grad_scales.ptr<float>(),
        grad_quaternions.ptr<float>(), result.points.ptr<float>(),
        context.options.debug,
        static_cast<int>(context.camera.model), context.camera.k1,
        context.camera.k2, context.camera.k3, context.camera.k4);
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
        throw std::invalid_argument(
            "GGGS occupancy points must have shape [P,3]");
    if (model.size() == 0 || world_points.shape()[0] == 0)
        throw std::invalid_argument(
            "GGGS occupancy evaluation requires Gaussians and points");
    if (camera.width == 0 || camera.height == 0)
        throw std::invalid_argument(
            "GGGS occupancy camera dimensions must be positive");

    const detail::ActivatedParameters activated =
        detail::activate_parameters(model);
    const auto constants = prepare_camera_constants(camera);
    tinytensor::Tensor geometry_buffer;
    tinytensor::Tensor binning_buffer;
    tinytensor::Tensor point_buffer;
    tinytensor::Tensor point_binning_buffer;
    tinytensor::Tensor tile_buffer;
    tinytensor::Tensor duplicated_tile_buffer;
    const std::size_t point_count = world_points.shape()[0];
    auto transmittance = tinytensor::Tensor::zeros(
        {point_count}, tinytensor::Device::CUDA);
    OccupancyResult result;
    result.inside = tinytensor::Tensor::zeros(
        {point_count}, tinytensor::Device::CUDA,
        tinytensor::DataType::Bool);
    CudaRasterizer::Rasterizer::evaluateTransmittance(
        resize_buffer(geometry_buffer), resize_buffer(binning_buffer),
        resize_buffer(point_buffer), resize_buffer(point_binning_buffer),
        resize_buffer(tile_buffer), resize_buffer(duplicated_tile_buffer),
        static_cast<int>(point_count), static_cast<int>(model.size()),
        static_cast<int>(camera.width), static_cast<int>(camera.height),
        world_points.ptr<float>(), model.means.ptr<float>(),
        activated.opacities.ptr<float>(), activated.scales.ptr<float>(),
        requested_options.scale_modifier,
        activated.quaternions.ptr<float>(), nullptr,
        constants.view_matrix(), constants.position(), camera.fx,
        camera.fy, camera.cx, camera.cy, requested_options.kernel_size,
        false, transmittance.ptr<float>(), result.inside.ptr<bool>(),
        requested_options.debug,
        static_cast<int>(camera.model), camera.k1, camera.k2,
        camera.k3, camera.k4);
    result.occupancy =
        tinytensor::Tensor::ones_like(transmittance) - transmittance;
    return result;
}

}  // namespace aetherscan::splat
