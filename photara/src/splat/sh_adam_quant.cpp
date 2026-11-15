#include "cuda_backend.hpp"
#include "vulkan_backend.hpp"
#include "core/vram_profiler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
ShAdamQuant make_sh_adam_quant(
    const std::size_t rows,
    const int stride,
    const tinytensor::Device device) {

    if (stride <= 0 || stride > 64)
        throw std::invalid_argument("quantized SH Adam stride must be in [1, 64]");
    tinytensor::VramScope scope("optimizer.state");
    ShAdamQuant state;
    state.stride = stride;
    state.first = tinytensor::Tensor::zeros(
        {rows, static_cast<std::size_t>(stride)}, device, tinytensor::DataType::Float16);
    state.packed = tinytensor::Tensor::zeros(
        {rows, static_cast<std::size_t>(stride)}, device, tinytensor::DataType::UInt8);
    state.bounds = tinytensor::Tensor::zeros({rows, std::size_t{4}}, device);
    return state;
}

void sh_adam_quant_select_rows(ShAdamQuant& state, const tinytensor::Tensor& indices) {
    if (!state.active())
        return;
#if defined(TINYTENSOR_HAS_VULKAN)
    if (state.first.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::sh_adam_quant_select_rows(state, indices);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (state.first.device() == tinytensor::Device::CUDA)
        return cuda_backend::sh_adam_quant_select_rows(state, indices);
#endif
    throw std::runtime_error("sh_adam_quant_select_rows: requested backend is not enabled");
}

void sh_adam_quant_zero_rows(ShAdamQuant& state, const tinytensor::Tensor& indices) {
    if (!state.active() || !indices.is_valid() || indices.numel() == 0)
        return;
#if defined(TINYTENSOR_HAS_VULKAN)
    if (state.first.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::sh_adam_quant_zero_rows(state, indices);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (state.first.device() == tinytensor::Device::CUDA)
        return cuda_backend::sh_adam_quant_zero_rows(state, indices);
#endif
    throw std::runtime_error("sh_adam_quant_zero_rows: requested backend is not enabled");
}

void sh_adam_quant_append_zeros(ShAdamQuant& state, const std::size_t count) {

    if (!state.active() || count == 0)
        return;
    const auto device = state.first.device();
    state.first = tinytensor::Tensor::cat(
        {state.first, tinytensor::Tensor::zeros(
                          {count, static_cast<std::size_t>(state.stride)}, device,
                          tinytensor::DataType::Float16)},
        0);
    state.packed = tinytensor::Tensor::cat(
        {state.packed,
         tinytensor::Tensor::zeros(
             {count, static_cast<std::size_t>(state.stride)}, device, tinytensor::DataType::UInt8)},
        0);
    state.bounds = tinytensor::Tensor::cat(
        {state.bounds, tinytensor::Tensor::zeros({count, std::size_t{4}}, device)}, 0);
}

void sh_adam_quant_step(
    tinytensor::Tensor& parameter,
    const tinytensor::Tensor& gradient,
    ShAdamQuant& state,
    const int active_stride,
    const float learning_rate,
    const float rest_learning_rate,
    const float beta1,
    const float beta2,
    const float correction1,
    const float correction2,
    const float adam_epsilon,
    const float regularization_factor) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (parameter.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::sh_adam_quant_step(
            parameter, gradient, state, active_stride, learning_rate, rest_learning_rate, beta1,
            beta2, correction1, correction2, adam_epsilon, regularization_factor);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (parameter.device() == tinytensor::Device::CUDA)
        return cuda_backend::sh_adam_quant_step(
            parameter, gradient, state, active_stride, learning_rate, rest_learning_rate, beta1,
            beta2, correction1, correction2, adam_epsilon, regularization_factor);
#endif
    throw std::runtime_error("sh_adam_quant_step: requested backend is not enabled");
}

} // namespace photara::splat::detail
