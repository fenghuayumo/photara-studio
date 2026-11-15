#include "formats_sog_xt.hpp"
#include <stdexcept>
namespace photara::splat::detail {
GaussianModel sog_xt_training_forward(SogXtTrainingCodec& codec) {
    if (codec.latent.means.device() == tinytensor::Device::Vulkan)
        return sog_xt_training_forward_vulkan(codec);
    throw std::invalid_argument("SOG-XT CUDA backend is not enabled");
}
void sog_xt_training_backward(
    SogXtTrainingCodec& codec,
    const GaussianModel& decoded,
    ModelGradients& g,
    float tv,
    float qtv) {
    if (codec.latent.means.device() != tinytensor::Device::Vulkan)
        throw std::invalid_argument("SOG-XT CUDA backend is not enabled");
    sog_xt_training_backward_vulkan(codec, decoded, g, tv, qtv);
}
} // namespace photara::splat::detail
