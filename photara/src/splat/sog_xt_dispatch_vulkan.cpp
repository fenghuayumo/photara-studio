#include "vulkan_backend.hpp"
#include "formats_sog_xt.hpp"
#include <stdexcept>
namespace photara::splat::detail {

GaussianModel vulkan_backend::sog_xt_training_forward(SogXtTrainingCodec& codec) {

    return sog_xt_training_forward_vulkan(codec);
}

void vulkan_backend::sog_xt_training_backward(
    SogXtTrainingCodec& codec,
    const GaussianModel& decoded,
    ModelGradients& g,
    float tv,
    float qtv) {
    sog_xt_training_backward_vulkan(codec, decoded, g, tv, qtv);
}

} // namespace photara::splat::detail
