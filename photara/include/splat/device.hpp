#pragma once

#include "splat/options.hpp"
#include "internal/tensor_impl.hpp"

namespace photara::splat {

[[nodiscard]] constexpr tinytensor::Device training_device(
    const TrainingBackend backend) noexcept {
    return backend == TrainingBackend::vulkan
        ? tinytensor::Device::Vulkan
        : tinytensor::Device::CUDA;
}

[[nodiscard]] constexpr tinytensor::Device training_device(
    const TrainingOptions& options) noexcept {
    return training_device(options.backend);
}

}  // namespace photara::splat
