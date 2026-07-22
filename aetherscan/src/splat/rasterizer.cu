#include "splat/rasterizer.hpp"

#include "cuda_ops.hpp"
#include "rasterizer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace aetherscan::splat {

struct RasterContextImpl {
    detail::ActivatedParameters activated;
    tinytensor::Tensor background;
    tinytensor::Tensor view_matrix;
    tinytensor::Tensor camera_position;
    tinytensor::Tensor geometry_buffer;
    tinytensor::Tensor binning_buffer;
    tinytensor::Tensor image_buffer;
    tinytensor::Tensor tile_buffer;
    RasterizeOptions options;
    Camera camera;
    int rendered_instances{};
};

namespace {

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
    if (camera.width == 0 || camera.height == 0)
        throw std::invalid_argument("GGGS camera dimensions must be positive");
    if (model.means.shape().rank() != 2 || model.means.shape()[1] != 3 ||
        model.log_scales.shape() != model.means.shape() ||
        model.quaternions.shape().rank() != 2 ||
        model.quaternions.shape()[0] != model.size() ||
        model.quaternions.shape()[1] != 4 ||
        model.opacity_logits.numel() != model.size() ||
        model.sh.shape().rank() != 3 || model.sh.shape()[0] != model.size() ||
        model.sh.shape()[2] != 3)
        throw std::invalid_argument("Invalid GGGS model tensor shapes");

    RenderResult result;
    auto context = std::make_shared<RasterContextImpl>();
    context->camera = camera;
    context->options = requested_options;
    context->options.active_sh_degree = std::min(
        requested_options.active_sh_degree, model.sh_degree);
    context->activated = detail::activate_parameters(model);
    context->background = tinytensor::Tensor::zeros(
        {3}, tinytensor::Device::CUDA);
    context->view_matrix = tinytensor::Tensor::from_vector(
        std::vector<float>(camera.world_to_camera.begin(), camera.world_to_camera.end()),
        {4, 4}, tinytensor::Device::CUDA);
    context->camera_position = tinytensor::Tensor::from_vector(
        std::vector<float>(camera.position.begin(), camera.position.end()),
        {3}, tinytensor::Device::CUDA);

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
            context->background.ptr<float>(),
            static_cast<int>(camera.width), static_cast<int>(camera.height),
            model.means.ptr<float>(), nullptr,
            context->activated.opacities.ptr<float>(),
            context->activated.scales.ptr<float>(),
            context->activated.quaternions.ptr<float>(), nullptr,
            model.sh.ptr<float>(), nullptr, nullptr, nullptr,
            context->options.scale_modifier,
            context->view_matrix.ptr<float>(),
            context->camera_position.ptr<float>(), camera.fx, camera.fy,
            camera.cx, camera.cy, context->options.kernel_size, false,
            result.color.ptr<float>(), result.median_depth.ptr<float>(),
            result.alpha.ptr<float>(), result.normal.ptr<float>(),
            result.radii.ptr<int>(), context->options.require_depth,
            context->options.debug);
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
            context.rendered_instances, context.background.ptr<float>(),
            static_cast<int>(context.camera.width),
            static_cast<int>(context.camera.height), model.means.ptr<float>(),
            nullptr, context.activated.opacities.ptr<float>(),
            context.activated.scales.ptr<float>(),
            context.activated.quaternions.ptr<float>(), nullptr,
            model.sh.ptr<float>(), nullptr, nullptr, nullptr,
            context.options.scale_modifier, context.view_matrix.ptr<float>(),
            context.camera_position.ptr<float>(), context.camera.fx,
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
            context.options.require_depth, context.options.debug);
    }
    detail::chain_parameter_gradients(
        model, context.activated, grad_scales, grad_quaternions,
        grad_opacities, gradients);
    gradients.refine_weight = std::move(refine_weight);
    return gradients;
}

}  // namespace aetherscan::splat
