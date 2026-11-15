#include "cuda_backend.hpp"
#include "vulkan_backend.hpp"
#include "core/vram_profiler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
GaussianModel sog_xt_training_forward(SogXtTrainingCodec& codec) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (codec.latent.means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::sog_xt_training_forward(codec);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (codec.latent.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::sog_xt_training_forward(codec);
#endif
    throw std::runtime_error("sog_xt_training_forward: requested backend is not enabled");
}
void sog_xt_training_backward(
    SogXtTrainingCodec& codec,
    const GaussianModel& decoded,
    ModelGradients& g,
    float tv,
    float qtv) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (codec.latent.means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::sog_xt_training_backward(codec, decoded, g, tv, qtv);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (codec.latent.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::sog_xt_training_backward(codec, decoded, g, tv, qtv);
#endif
    throw std::runtime_error("sog_xt_training_backward: requested backend is not enabled");
}
} // namespace photara::splat::detail
