#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace aetherscan::splat::detail {
namespace {

constexpr unsigned k_threads = 256;

void check_cuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
}

__device__ float sigmoid(const float value) {
    return 1.F / (1.F + expf(-value));
}

__global__ void activate_kernel(
    const float* log_scales, const float* raw_quaternions,
    const float* opacity_logits, float* scales, float* quaternions,
    float* opacities, const std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    scales[3 * index + 0] = expf(log_scales[3 * index + 0]);
    scales[3 * index + 1] = expf(log_scales[3 * index + 1]);
    scales[3 * index + 2] = expf(log_scales[3 * index + 2]);
    const float w = raw_quaternions[4 * index + 0];
    const float x = raw_quaternions[4 * index + 1];
    const float y = raw_quaternions[4 * index + 2];
    const float z = raw_quaternions[4 * index + 3];
    const float inverse_norm = rsqrtf(fmaxf(w * w + x * x + y * y + z * z, 1e-20F));
    quaternions[4 * index + 0] = w * inverse_norm;
    quaternions[4 * index + 1] = x * inverse_norm;
    quaternions[4 * index + 2] = y * inverse_norm;
    quaternions[4 * index + 3] = z * inverse_norm;
    opacities[index] = sigmoid(opacity_logits[index]);
}

__global__ void chain_gradient_kernel(
    const float* log_scales, const float* raw_quaternions,
    const float* opacities, const float* grad_scales,
    const float* grad_quaternions, const float* grad_opacities,
    float* grad_log_scales, float* grad_raw_quaternions,
    float* grad_opacity_logits, const std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    for (int axis = 0; axis < 3; ++axis) {
        const std::size_t offset = 3 * index + axis;
        grad_log_scales[offset] = grad_scales[offset] * expf(log_scales[offset]);
    }
    const float rw = raw_quaternions[4 * index + 0];
    const float rx = raw_quaternions[4 * index + 1];
    const float ry = raw_quaternions[4 * index + 2];
    const float rz = raw_quaternions[4 * index + 3];
    const float inverse_norm = rsqrtf(fmaxf(rw * rw + rx * rx + ry * ry + rz * rz, 1e-20F));
    const float q[4] = {rw * inverse_norm, rx * inverse_norm, ry * inverse_norm, rz * inverse_norm};
    const float dot = q[0] * grad_quaternions[4 * index + 0] +
                      q[1] * grad_quaternions[4 * index + 1] +
                      q[2] * grad_quaternions[4 * index + 2] +
                      q[3] * grad_quaternions[4 * index + 3];
    for (int component = 0; component < 4; ++component)
        grad_raw_quaternions[4 * index + component] = inverse_norm *
            (grad_quaternions[4 * index + component] - q[component] * dot);
    const float opacity = opacities[index];
    grad_opacity_logits[index] =
        grad_opacities[index] * opacity * (1.F - opacity);
}

__global__ void loss_kernel(
    const float* color, const float* alpha, const float* depth,
    const float* normal, const float* target_color,
    const float* target_depth, const float* target_normal,
    const float* mask, float* grad_color, float* grad_alpha,
    float* grad_depth, float* grad_normal, float* terms,
    const std::size_t pixels, const float photo_weight,
    const float depth_weight, const float normal_weight,
    const float alpha_weight, const float epsilon) {
    const std::size_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    const float valid = mask[pixel];
    const float inverse_pixels = 1.F / static_cast<float>(pixels);
    float rgb_loss = 0.F;
    for (int channel = 0; channel < 3; ++channel) {
        const std::size_t offset = static_cast<std::size_t>(channel) * pixels + pixel;
        const float difference = color[offset] - target_color[offset];
        const float robust = sqrtf(difference * difference + epsilon * epsilon);
        rgb_loss += robust;
        grad_color[offset] = photo_weight * valid * inverse_pixels / 3.F *
                             difference / robust;
    }
    if (terms)
        atomicAdd(
            terms + 0,
            photo_weight * valid * rgb_loss * inverse_pixels / 3.F);

    const bool has_depth = target_depth[pixel] > 0.F && depth[pixel] > 0.F;
    if (has_depth && valid > 0.F && depth_weight > 0.F) {
        const float scale = fmaxf(target_depth[pixel], 1e-4F);
        const float difference = (depth[pixel] - target_depth[pixel]) / scale;
        const float robust = sqrtf(difference * difference + epsilon * epsilon);
        grad_depth[pixel] = depth_weight * inverse_pixels * difference /
                            (robust * scale);
        if (terms)
            atomicAdd(terms + 1, depth_weight * robust * inverse_pixels);
    }

    const float tx = target_normal[pixel];
    const float ty = target_normal[pixels + pixel];
    const float tz = target_normal[2 * pixels + pixel];
    const float target_length2 = tx * tx + ty * ty + tz * tz;
    if (target_length2 > 0.25F && valid > 0.F && normal_weight > 0.F) {
        const float inverse_target_length = rsqrtf(target_length2);
        const float nx = normal[pixel];
        const float ny = normal[pixels + pixel];
        const float nz = normal[2 * pixels + pixel];
        const float dot = nx * tx * inverse_target_length +
                          ny * ty * inverse_target_length +
                          nz * tz * inverse_target_length;
        if (terms)
            atomicAdd(
                terms + 2,
                normal_weight * (1.F - dot) * inverse_pixels);
        grad_normal[pixel] = -normal_weight * tx * inverse_target_length * inverse_pixels;
        grad_normal[pixels + pixel] = -normal_weight * ty * inverse_target_length * inverse_pixels;
        grad_normal[2 * pixels + pixel] = -normal_weight * tz * inverse_target_length * inverse_pixels;
    }

    if (alpha_weight > 0.F) {
        const float prediction = fminf(fmaxf(alpha[pixel], 1e-6F), 1.F - 1e-6F);
        grad_alpha[pixel] = alpha_weight * inverse_pixels *
            (prediction - valid) / (prediction * (1.F - prediction));
        if (terms)
            atomicAdd(terms + 3, -alpha_weight * inverse_pixels *
                (valid * logf(prediction) +
                 (1.F - valid) * logf(1.F - prediction)));
    }
}

__global__ void adam_kernel(
    float* parameter, const float* gradient, float* first, float* second,
    const std::size_t count, const float learning_rate,
    const float secondary_learning_rate, const std::size_t group_stride,
    const float beta1, const float beta2, const float correction1,
    const float correction2, const float epsilon,
    const float clamp_min, const float clamp_max) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float previous = parameter[index];
    const float grad = gradient[index];
    // A single degenerate projected Gaussian can occasionally produce a
    // non-finite gradient in the imported rasterizer.  Do not allow it to
    // poison the parameter and both Adam moments permanently.
    if (!isfinite(previous) || !isfinite(grad)) {
        first[index] = 0.F;
        second[index] = 0.F;
        parameter[index] = isfinite(previous)
            ? fminf(fmaxf(previous, clamp_min), clamp_max)
            : fminf(fmaxf(0.F, clamp_min), clamp_max);
        return;
    }
    const float m = beta1 * first[index] + (1.F - beta1) * grad;
    const float v = beta2 * second[index] + (1.F - beta2) * grad * grad;
    if (!isfinite(m) || !isfinite(v)) {
        first[index] = 0.F;
        second[index] = 0.F;
        parameter[index] = fminf(fmaxf(previous, clamp_min), clamp_max);
        return;
    }
    first[index] = m;
    second[index] = v;
    const float lr = group_stride != 0 && index % group_stride >= 3
                         ? secondary_learning_rate
                         : learning_rate;
    const float candidate = previous - lr * (m / correction1) /
                          (sqrtf(v / correction2) + epsilon);
    const float updated = isfinite(candidate) ? candidate : previous;
    parameter[index] = fminf(fmaxf(updated, clamp_min), clamp_max);
}

}  // namespace

ActivatedParameters activate_parameters(const GaussianModel& model) {
    const std::size_t count = model.size();
    ActivatedParameters result{
        tinytensor::Tensor::empty({count, 3}, tinytensor::Device::CUDA),
        tinytensor::Tensor::empty({count, 4}, tinytensor::Device::CUDA),
        tinytensor::Tensor::empty({count, 1}, tinytensor::Device::CUDA)};
    if (count == 0) return result;
    activate_kernel<<<(count + k_threads - 1) / k_threads, k_threads>>>(
        model.log_scales.ptr<float>(), model.quaternions.ptr<float>(),
        model.opacity_logits.ptr<float>(), result.scales.ptr<float>(),
        result.quaternions.ptr<float>(), result.opacities.ptr<float>(), count);
    check_cuda(cudaGetLastError(), "activate GGGS parameters");
    return result;
}

void chain_parameter_gradients(
    const GaussianModel& model, const ActivatedParameters& activated,
    const tinytensor::Tensor& grad_scales,
    const tinytensor::Tensor& grad_quaternions,
    const tinytensor::Tensor& grad_opacities,
    ModelGradients& gradients) {
    const std::size_t count = model.size();
    gradients.log_scales = tinytensor::Tensor::zeros_like(model.log_scales);
    gradients.quaternions = tinytensor::Tensor::zeros_like(model.quaternions);
    gradients.opacity_logits = tinytensor::Tensor::zeros_like(model.opacity_logits);
    if (count == 0) return;
    chain_gradient_kernel<<<(count + k_threads - 1) / k_threads, k_threads>>>(
        model.log_scales.ptr<float>(), model.quaternions.ptr<float>(),
        activated.opacities.ptr<float>(), grad_scales.ptr<float>(),
        grad_quaternions.ptr<float>(), grad_opacities.ptr<float>(),
        gradients.log_scales.ptr<float>(), gradients.quaternions.ptr<float>(),
        gradients.opacity_logits.ptr<float>(), count);
    check_cuda(cudaGetLastError(), "chain GGGS parameter gradients");
}

LossGradients compute_training_loss(
    const RenderResult& rendered, const TrainingView& target,
    const TrainingOptions& options, const bool collect_scalar_terms) {
    const std::size_t pixels = static_cast<std::size_t>(target.camera.width) *
                               target.camera.height;
    LossGradients result{
        tinytensor::Tensor::zeros_like(rendered.color),
        tinytensor::Tensor::zeros_like(rendered.alpha),
        tinytensor::Tensor::zeros_like(rendered.median_depth),
        tinytensor::Tensor::zeros_like(rendered.normal)};
    tinytensor::Tensor terms;
    if (collect_scalar_terms)
        terms = tinytensor::Tensor::zeros({4}, tinytensor::Device::CUDA);
    loss_kernel<<<(pixels + k_threads - 1) / k_threads, k_threads>>>(
        rendered.color.ptr<float>(), rendered.alpha.ptr<float>(),
        rendered.median_depth.ptr<float>(), rendered.normal.ptr<float>(),
        target.rgb.ptr<float>(), target.depth.ptr<float>(),
        target.normal.ptr<float>(), target.mask.ptr<float>(),
        result.color.ptr<float>(), result.alpha.ptr<float>(),
        result.depth.ptr<float>(), result.normal.ptr<float>(),
        collect_scalar_terms ? terms.ptr<float>() : nullptr,
        pixels, options.photometric_weight,
        options.use_mvs_depth ? options.depth_weight : 0.F,
        options.use_mvs_normals ? options.normal_weight : 0.F,
        options.alpha_weight, options.charbonnier_epsilon);
    check_cuda(cudaGetLastError(), "compute GGGS training loss");
    if (collect_scalar_terms) {
        std::array<float, 4> host{};
        check_cuda(cudaMemcpy(
            host.data(), terms.ptr<float>(), sizeof(host), cudaMemcpyDeviceToHost),
            "download GGGS loss");
        result.rgb = host[0];
        result.depth_value = host[1];
        result.normal_value = host[2];
        result.total = host[0] + host[1] + host[2] + host[3];
    }
    return result;
}

AdamState make_adam_state(const tinytensor::Tensor& parameter) {
    return {tinytensor::Tensor::zeros_like(parameter),
            tinytensor::Tensor::zeros_like(parameter)};
}

void adam_step(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, const float learning_rate, const unsigned step,
    const TrainingOptions& options, const std::size_t group_stride,
    const float secondary_learning_rate, const float clamp_min,
    const float clamp_max) {
    const std::size_t count = parameter.numel();
    if (count == 0) return;
    const float correction1 = 1.F - std::pow(options.beta1, static_cast<float>(step));
    const float correction2 = 1.F - std::pow(options.beta2, static_cast<float>(step));
    adam_kernel<<<(count + k_threads - 1) / k_threads, k_threads>>>(
        parameter.ptr<float>(), gradient.ptr<float>(), state.first.ptr<float>(),
        state.second.ptr<float>(), count, learning_rate,
        secondary_learning_rate, group_stride, options.beta1, options.beta2,
        correction1, correction2, options.adam_epsilon, clamp_min, clamp_max);
    check_cuda(cudaGetLastError(), "GGGS Adam update");
}

}  // namespace aetherscan::splat::detail
