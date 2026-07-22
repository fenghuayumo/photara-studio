#include "cuda_ops.hpp"
#include "fused_ssim.hpp"

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
    const bool mask_enabled, const int alpha_mode,
    const float match_alpha_weight, const float l1_weight,
    const float geometry_epsilon) {
    const std::size_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    const float valid = mask_enabled ? mask[pixel] : 1.F;
    const float inverse_pixels = 1.F / static_cast<float>(pixels);
    float rgb_loss = 0.F;
    for (int channel = 0; channel < 3; ++channel) {
        const std::size_t offset = static_cast<std::size_t>(channel) * pixels + pixel;
        const float difference = color[offset] - target_color[offset];
        rgb_loss += fabsf(difference);
        grad_color[offset] = photo_weight * l1_weight * valid * inverse_pixels /
                             3.F * ((difference > 0.F) - (difference < 0.F));
    }
    if (terms)
        atomicAdd(
            terms + 0,
            photo_weight * l1_weight * valid * rgb_loss * inverse_pixels / 3.F);

    const bool has_depth = target_depth[pixel] > 0.F && depth[pixel] > 0.F;
    if (has_depth && valid > 0.F && depth_weight > 0.F) {
        const float scale = fmaxf(target_depth[pixel], 1e-4F);
        const float difference = (depth[pixel] - target_depth[pixel]) / scale;
        const float robust = sqrtf(
            difference * difference + geometry_epsilon * geometry_epsilon);
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

    if (mask_enabled && alpha_mode == 0) {
        // pygsplat alpha_mode="masked": discourage any opacity outside the
        // foreground without forcing the foreground itself to be opaque.
        grad_alpha[pixel] = (1.F - valid) * inverse_pixels;
        if (terms)
            atomicAdd(
                terms + 3, alpha[pixel] * (1.F - valid) * inverse_pixels);
    } else if (mask_enabled && alpha_mode == 1 && match_alpha_weight > 0.F) {
        // pygsplat alpha_mode="transparent": full-image BCE(alpha, mask).
        constexpr float clamp_epsilon = 1e-7F;
        const float raw_prediction = alpha[pixel];
        const float prediction = fminf(
            fmaxf(raw_prediction, clamp_epsilon), 1.F - clamp_epsilon);
        // torch.clamp, used by pygsplat before BCE, has zero derivative outside
        // its interval. Continuing to differentiate the clamped value produces
        // enormous gradients at saturated pixels and destabilizes opacity,
        // refine-weight accumulation, and pruning.
        const bool inside_clamp =
            raw_prediction > clamp_epsilon &&
            raw_prediction < 1.F - clamp_epsilon;
        grad_alpha[pixel] = inside_clamp
            ? match_alpha_weight * inverse_pixels *
                  (prediction - valid) /
                  (prediction * (1.F - prediction))
            : 0.F;
        if (terms)
            atomicAdd(terms + 3, -match_alpha_weight * inverse_pixels *
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

__global__ void constrain_scale_ratio_kernel(
    float* log_scales, const std::size_t count, const float maximum_log_ratio) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float* values = log_scales + 3 * index;
    const float minimum = fminf(values[0], fminf(values[1], values[2]));
    const float maximum = fmaxf(values[0], fmaxf(values[1], values[2]));
    if (maximum - minimum <= maximum_log_ratio) return;
    const float midpoint = 0.5F * (minimum + maximum);
    const float half_range = 0.5F * maximum_log_ratio;
    for (int axis = 0; axis < 3; ++axis)
        values[axis] = fminf(fmaxf(values[axis], midpoint - half_range),
                             midpoint + half_range);
}

__global__ void accumulate_densification_kernel(
    const float* refine_weight, const int* radii, float* gradient,
    float* count, float* max_screen_radius, float* priority,
    const std::size_t gaussian_count, const float inverse_resolution,
    const bool use_maximum) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= gaussian_count || radii[index] <= 0) return;
    const float weight = isfinite(refine_weight[index])
        ? fmaxf(refine_weight[index], 0.F)
        : 0.F;
    if (use_maximum)
        gradient[index] = fmaxf(gradient[index], weight);
    else
        gradient[index] += weight;
    count[index] += 1.F;
    const float screen = radii[index] * inverse_resolution;
    max_screen_radius[index] = fmaxf(max_screen_radius[index], screen);
    priority[index] += weight * (1.F + screen);
}

__device__ void rotate_quaternion(
    const float* raw, const float x, const float y, const float z,
    float& out_x, float& out_y, float& out_z) {
    const float inverse_norm = rsqrtf(fmaxf(
        raw[0] * raw[0] + raw[1] * raw[1] +
        raw[2] * raw[2] + raw[3] * raw[3], 1e-20F));
    const float w = raw[0] * inverse_norm;
    const float qx = raw[1] * inverse_norm;
    const float qy = raw[2] * inverse_norm;
    const float qz = raw[3] * inverse_norm;
    const float tx = 2.F * (qy * z - qz * y);
    const float ty = 2.F * (qz * x - qx * z);
    const float tz = 2.F * (qx * y - qy * x);
    out_x = x + w * tx + (qy * tz - qz * ty);
    out_y = y + w * ty + (qz * tx - qx * tz);
    out_z = z + w * tz + (qx * ty - qy * tx);
}

__global__ void split_gaussians_kernel(
    float* parent_means, float* parent_log_scales,
    float* parent_opacity_logits, const float* parent_quaternions,
    float* child_means, float* child_log_scales,
    float* child_opacity_logits, const int* parent_indices,
    const float* random_samples, const std::size_t split_count,
    const int mode, const float minimum_opacity) {
    const std::size_t child = blockIdx.x * blockDim.x + threadIdx.x;
    if (child >= split_count) return;
    const std::size_t parent = static_cast<std::size_t>(parent_indices[child]);
    const float* parent_quaternion = parent_quaternions + 4 * parent;
    float local[3]{};
    float log_scale_delta[3]{};
    if (mode == 4) {
        // Dense MVS already constrains the surface normal accurately. Split
        // only in the local tangent plane (local Z is initialized from the
        // fused-cloud normal) and preserve the normal-axis thickness.
        constexpr float tangent_offset = 0.5F;
        constexpr float tangent_scale = 0.7071067811865475F;
        local[0] = expf(parent_log_scales[3 * parent]) *
                   random_samples[3 * child] * tangent_offset;
        local[1] = expf(parent_log_scales[3 * parent + 1]) *
                   random_samples[3 * child + 1] * tangent_offset;
        local[2] = 0.F;
        log_scale_delta[0] = logf(tangent_scale);
        log_scale_delta[1] = logf(tangent_scale);
        log_scale_delta[2] = 0.F;
    } else if (mode == 3) {
        int largest = 0;
        if (parent_log_scales[3 * parent + 1] >
            parent_log_scales[3 * parent + largest]) largest = 1;
        if (parent_log_scales[3 * parent + 2] >
            parent_log_scales[3 * parent + largest]) largest = 2;
        for (int axis = 0; axis < 3; ++axis) {
            local[axis] = expf(parent_log_scales[3 * parent + axis]) *
                          random_samples[3 * child];
            log_scale_delta[axis] = axis == largest ? logf(0.5F) : 0.F;
        }
    } else {
        const float scale_factor = mode == 2 ? rsqrtf(2.F) : 1.F / 1.6F;
        const float sample_factor = mode == 2 ? rsqrtf(2.F) : 1.F;
        for (int axis = 0; axis < 3; ++axis) {
            local[axis] = expf(parent_log_scales[3 * parent + axis]) *
                          random_samples[3 * child + axis] * sample_factor;
            log_scale_delta[axis] = logf(scale_factor);
        }
    }
    float offset_x{}, offset_y{}, offset_z{};
    rotate_quaternion(
        parent_quaternion, local[0], local[1], local[2],
        offset_x, offset_y, offset_z);
    const float offsets[3]{offset_x, offset_y, offset_z};
    for (int axis = 0; axis < 3; ++axis) {
        const float center = parent_means[3 * parent + axis];
        parent_means[3 * parent + axis] = center - offsets[axis];
        child_means[3 * child + axis] = center + offsets[axis];
        parent_log_scales[3 * parent + axis] += log_scale_delta[axis];
        child_log_scales[3 * child + axis] += log_scale_delta[axis];
    }
    const float opacity = sigmoid(parent_opacity_logits[parent]);
    const float opacity_floor = mode == 1 ? 1e-8F : minimum_opacity;
    const float revised = fminf(fmaxf(
        1.F - sqrtf(fmaxf(1.F - opacity, 0.F)),
        opacity_floor), 1.F - opacity_floor);
    const float revised_logit = logf(revised / (1.F - revised));
    parent_opacity_logits[parent] = revised_logit;
    child_opacity_logits[child] = revised_logit;
}

__global__ void adc_decay_kernel(
    float* log_scales, float* opacity_logits, const std::size_t count,
    const float opacity_decay, const float log_scale_decay) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float opacity = fminf(fmaxf(
        sigmoid(opacity_logits[index]) - opacity_decay, 1e-12F),
        1.F - 1e-12F);
    opacity_logits[index] = logf(opacity / (1.F - opacity));
    for (int axis = 0; axis < 3; ++axis)
        log_scales[3 * index + axis] += log_scale_decay;
}

__device__ std::uint32_t hash_u32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__device__ float normal_sample(
    const std::uint32_t index, const std::uint32_t seed,
    const std::uint32_t axis) {
    const float u1 = (hash_u32(index * 3U + axis + seed * 17U) + 1.F) /
                     4294967297.F;
    const float u2 = (hash_u32(index * 7U + axis + seed * 29U) + 1.F) /
                     4294967297.F;
    return sqrtf(-2.F * logf(fmaxf(u1, 1e-12F))) *
           cosf(6.283185307179586F * u2);
}

__global__ void inject_adc_noise_kernel(
    float* means, const float* opacity_logits, const int* radii,
    const std::size_t count, const float standard_deviation,
    const float maximum_noise, const unsigned seed) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || radii[index] <= 0) return;
    const float inverse_opacity = 1.F - sigmoid(opacity_logits[index]);
    const float weight = powf(inverse_opacity, 150.F);
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        const float noise = fminf(fmaxf(
            normal_sample(static_cast<std::uint32_t>(index), seed, axis) *
                weight * standard_deviation,
            -maximum_noise), maximum_noise);
        means[3 * index + axis] += noise;
    }
}

__global__ void reset_opacity_kernel(
    float* opacity_logits, const std::size_t count,
    const float maximum_logit) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        opacity_logits[index] = fminf(opacity_logits[index], maximum_logit);
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
    const bool mask_enabled = options.use_mask && target.has_mask;
    const bool use_fused_photometric =
        target.camera.width > 10 && target.camera.height > 10;
    const float ssim_weight = std::clamp(options.ssim_weight, 0.F, 1.F);
    loss_kernel<<<(pixels + k_threads - 1) / k_threads, k_threads>>>(
        rendered.color.ptr<float>(), rendered.alpha.ptr<float>(),
        rendered.median_depth.ptr<float>(), rendered.normal.ptr<float>(),
        target.rgb.ptr<float>(), target.depth.ptr<float>(),
        target.normal.ptr<float>(), target.mask.ptr<float>(),
        result.color.ptr<float>(), result.alpha.ptr<float>(),
        result.depth.ptr<float>(), result.normal.ptr<float>(),
        collect_scalar_terms ? terms.ptr<float>() : nullptr,
        pixels, use_fused_photometric ? 0.F : options.photometric_weight,
        options.use_mvs_depth ? options.depth_weight : 0.F,
        options.use_mvs_normals ? options.normal_weight : 0.F,
        mask_enabled,
        options.alpha_mode == AlphaMode::masked ? 0 : 1,
        options.match_alpha_weight, 1.F,
        options.geometry_epsilon);
    check_cuda(cudaGetLastError(), "compute GGGS training loss");
    if (use_fused_photometric)
        fused_l1_ssim_loss(
            rendered.color, target.rgb, target.mask, mask_enabled,
            ssim_weight, options.photometric_weight, result.color,
            collect_scalar_terms ? terms.ptr<float>() : nullptr,
            target.camera.width, target.camera.height);
    if (collect_scalar_terms) {
        std::array<float, 4> host{};
        check_cuda(cudaMemcpy(
            host.data(), terms.ptr<float>(), sizeof(host), cudaMemcpyDeviceToHost),
            "download GGGS loss");
        result.rgb = host[0];
        result.alpha_value = host[3];
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

void constrain_scale_ratio(
    tinytensor::Tensor& log_scales, const float maximum_ratio) {
    if (maximum_ratio <= 1.F || log_scales.numel() == 0) return;
    const std::size_t count = log_scales.numel() / 3;
    constrain_scale_ratio_kernel<<<
        (count + k_threads - 1) / k_threads, k_threads>>>(
        log_scales.ptr<float>(), count, std::log(maximum_ratio));
    check_cuda(cudaGetLastError(), "constrain GGGS scale ratio");
}

DensificationStats make_densification_stats(const std::size_t count) {
    return {
        tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA),
        tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA),
        tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA),
        tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA)};
}

void accumulate_densification_stats(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& radii,
    DensificationStats& stats,
    const std::uint32_t width,
    const std::uint32_t height,
    const bool use_maximum) {
    const std::size_t count = refine_weight.numel();
    if (count == 0) return;
    const float inverse_resolution = 1.F /
        static_cast<float>(std::max<std::uint32_t>(1, std::min(width, height)));
    accumulate_densification_kernel<<<
        (count + k_threads - 1) / k_threads, k_threads>>>(
        refine_weight.ptr<float>(), radii.ptr<int>(),
        stats.gradient.ptr<float>(), stats.count.ptr<float>(),
        stats.max_screen_radius.ptr<float>(), stats.priority.ptr<float>(),
        count, inverse_resolution, use_maximum);
    check_cuda(cudaGetLastError(), "accumulate GGGS densification stats");
}

void split_gaussians(
    GaussianModel& parents,
    GaussianModel& children,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& random_samples,
    const int mode,
    const float minimum_opacity) {
    const std::size_t count = parent_indices.numel();
    if (count == 0) return;
    split_gaussians_kernel<<<
        (count + k_threads - 1) / k_threads, k_threads>>>(
        parents.means.ptr<float>(), parents.log_scales.ptr<float>(),
        parents.opacity_logits.ptr<float>(), parents.quaternions.ptr<float>(),
        children.means.ptr<float>(), children.log_scales.ptr<float>(),
        children.opacity_logits.ptr<float>(), parent_indices.ptr<int>(),
        random_samples.ptr<float>(), count, mode,
        std::clamp(minimum_opacity, 1e-8F, 0.49F));
    check_cuda(cudaGetLastError(), "split GGGS Gaussians");
}

void apply_adc_decay(
    GaussianModel& model, const float opacity_decay,
    const float scale_decay) {
    if (model.size() == 0) return;
    const float scale_factor = std::max(1.F - scale_decay, 1e-6F);
    adc_decay_kernel<<<
        (model.size() + k_threads - 1) / k_threads, k_threads>>>(
        model.log_scales.ptr<float>(), model.opacity_logits.ptr<float>(),
        model.size(), std::max(opacity_decay, 0.F), std::log(scale_factor));
    check_cuda(cudaGetLastError(), "decay ADC Gaussian parameters");
}

void inject_adc_noise(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    const float standard_deviation,
    const float maximum_noise,
    const unsigned seed) {
    if (model.size() == 0 || standard_deviation <= 0.F) return;
    inject_adc_noise_kernel<<<
        (model.size() + k_threads - 1) / k_threads, k_threads>>>(
        model.means.ptr<float>(), model.opacity_logits.ptr<float>(),
        radii.ptr<int>(), model.size(), standard_deviation,
        std::max(maximum_noise, 0.F), seed);
    check_cuda(cudaGetLastError(), "inject ADC exploration noise");
}

void reset_opacity(GaussianModel& model, const float maximum_opacity) {
    if (model.size() == 0) return;
    const float opacity = std::clamp(maximum_opacity, 1e-8F, 1.F - 1e-8F);
    reset_opacity_kernel<<<
        (model.size() + k_threads - 1) / k_threads, k_threads>>>(
        model.opacity_logits.ptr<float>(), model.size(),
        std::log(opacity / (1.F - opacity)));
    check_cuda(cudaGetLastError(), "reset GGGS opacity");
}

}  // namespace aetherscan::splat::detail
