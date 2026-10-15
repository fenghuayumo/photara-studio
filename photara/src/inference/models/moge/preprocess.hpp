#pragma once
#include "io/image.hpp"
#include <vector>

namespace photara::inference::detail {
struct MogeImage {
    std::uint32_t rows = 0, columns = 0;
    std::vector<float> normalized_rgb; // CHW, ImageNet normalization.
};
// Native PyTorch bilinear antialias resize, align_corners=false. Grid uses
// Python's round-to-even convention, independent of the process FP rounding mode.
MogeImage preprocess_moge(const io::RgbImage& image, std::uint32_t num_tokens);
} // namespace photara::inference::detail
