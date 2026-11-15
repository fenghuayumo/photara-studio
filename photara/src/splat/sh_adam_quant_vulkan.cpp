#include "vulkan_backend.hpp"
#include "vulkan/backend.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <climits>
#include <stdexcept>

namespace photara::splat::detail {

void vulkan_backend::sh_adam_quant_select_rows(
    ShAdamQuant& state,
    const tinytensor::Tensor& indices) {

    if (!state.active())
        return;
    const auto device = state.first.device();
    if (!indices.is_valid() || indices.numel() == 0) {
        state.first = tinytensor::Tensor::zeros(
            {std::size_t{0}, static_cast<std::size_t>(state.stride)}, device,
            tinytensor::DataType::Float16);
        state.packed = tinytensor::Tensor::zeros(
            {std::size_t{0}, static_cast<std::size_t>(state.stride)}, device,
            tinytensor::DataType::UInt8);
        state.bounds = tinytensor::Tensor::zeros({std::size_t{0}, std::size_t{4}}, device);
        return;
    }
    tinytensor::Tensor index = indices.dtype() == tinytensor::DataType::Int32
                                   ? indices
                                   : indices.to(tinytensor::DataType::Int32);
    if (index.device() != device)
        index = index.to(device);

    state.first = state.first.index_select(0, index);
    state.packed = state.packed.index_select(0, index);
    state.bounds = state.bounds.index_select(0, index);
    return;
}

void vulkan_backend::sh_adam_quant_zero_rows(
    ShAdamQuant& state,
    const tinytensor::Tensor& indices) {

    if (!state.active() || !indices.is_valid() || indices.numel() == 0)
        return;
    tinytensor::Tensor index = indices.dtype() == tinytensor::DataType::Int32
                                   ? indices
                                   : indices.to(tinytensor::DataType::Int32);
    if (index.device() != state.first.device())
        index = index.to(state.first.device());

    state.first.index_fill_(0, index, 0.F);
    state.packed.index_fill_(0, index, 0.F);
    state.bounds.index_fill_(0, index, 0.F);
    return;
}

void vulkan_backend::sh_adam_quant_step(
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

    if (!state.active())
        throw std::invalid_argument("quantized SH Adam state is empty");
    if (active_stride <= 0 || active_stride > state.stride)
        throw std::invalid_argument("quantized SH Adam active stride is outside the row");
    const std::size_t count = parameter.numel();
    if (count == 0)
        return;
    if (state.stride <= 0 || count % static_cast<std::size_t>(state.stride) != 0)
        throw std::invalid_argument("quantized SH Adam parameter is not a whole number of rows");
    const int rows = static_cast<int>(count / static_cast<std::size_t>(state.stride));
    if (parameter.dtype() != tinytensor::DataType::Float32 ||
        gradient.dtype() != tinytensor::DataType::Float32 || gradient.numel() != count ||
        state.first.numel() != static_cast<std::size_t>(rows) * state.stride ||
        state.first.dtype() != tinytensor::DataType::Float16 ||
        state.packed.numel() != static_cast<std::size_t>(rows) * state.stride ||
        state.packed.dtype() != tinytensor::DataType::UInt8 ||
        state.bounds.dtype() != tinytensor::DataType::Float32 ||
        state.bounds.numel() != static_cast<std::size_t>(rows) * 4)
        throw std::invalid_argument("quantized SH Adam storage does not match the parameter");
    if (!parameter.is_contiguous() || !state.first.is_contiguous() ||
        !state.packed.is_contiguous() || !state.bounds.is_contiguous())
        throw std::invalid_argument("quantized SH Adam tensors must be contiguous");
    const auto device = parameter.device();
    if (gradient.device() != device || state.first.device() != device ||
        state.packed.device() != device || state.bounds.device() != device)
        throw std::invalid_argument("quantized SH Adam tensors must share a device");
    const tinytensor::Tensor gradient_storage =
        gradient.is_contiguous() ? gradient : gradient.contiguous();

    tinytensor::vulkan::AdamStepOptions options;
    options.group_stride = static_cast<std::uint32_t>(state.stride);
    options.active_row_stride = static_cast<std::uint32_t>(active_stride);
    options.learning_rate = learning_rate;
    options.secondary_learning_rate = rest_learning_rate;
    options.beta1 = beta1;
    options.beta2 = beta2;
    options.correction1 = correction1;
    options.correction2 = correction2;
    options.epsilon = adam_epsilon;
    options.grouped_rest_regularization = regularization_factor;
    tinytensor::vulkan::sh_adam_quant_step(
        parameter, gradient_storage, state.first, state.packed, state.bounds, options);
    return;
}

} // namespace photara::splat::detail
