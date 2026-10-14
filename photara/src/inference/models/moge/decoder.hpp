#pragma once
#include "encoder.hpp"

namespace photara::inference::detail {
struct DecoderResult : MogeRawGeometry {
    // Raw head outputs, row-major HWC. No output resize, point remapping,
    // normal normalization, sigmoid, metric exp or camera recovery yet.
    std::vector<EncoderStage> stages; // Optional token-major diagnostic maps.
};

// Fixed MoGe-2 ViT-B normal topology. The input aspect ratio belongs to the
// original image, which can differ from columns/rows after token-grid rounding.
DecoderResult decode_moge_vitb(MogeWeights& weights, GgmlRuntime& runtime,
    const EncoderResult& encoder, float input_aspect_ratio, bool capture_stages = false);
} // namespace photara::inference::detail
