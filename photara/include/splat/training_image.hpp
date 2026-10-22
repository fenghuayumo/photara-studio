#pragma once

#include "io/image.hpp"
#include "mvs/types.hpp"
#include "splat/options.hpp"
#include "splat/types.hpp"

namespace photara::splat::training_data {

// CPU reference pixels in the rasterizer's projection. Shares the trainer's
// decode, undistortion, resize and mask sampling without creating GPU tensors.
struct TrainingImage {
    Camera camera;
    io::RgbImage rgb;
    // Empty when the loader supplies neither a validity nor a subject mask.
    io::GrayImage mask;
};

[[nodiscard]] Camera training_camera(
    const mvs::MvsView& view, const TrainingOptions& options,
    float resolution_scale = 1.F);

[[nodiscard]] TrainingImage load_training_image(
    const mvs::MvsView& view, const TrainingOptions& options);

}  // namespace photara::splat::training_data
