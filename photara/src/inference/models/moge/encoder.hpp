#pragma once
#include "weights.hpp"
#include "runtime/ggml_runtime.hpp"
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace photara::inference::detail {
struct EncoderStage {
    std::string name;
    std::vector<float> values; // Token-major, embedding coordinate contiguous.
};
struct EncoderResult {
    std::uint32_t rows = 0, columns = 0;
    std::vector<float> features; // [rows*columns, 768], summed tap projections.
    std::vector<float> class_token; // [768], normalized block 11 class token.
    std::vector<EncoderStage> stages; // Optional diagnostics, no production cache.
};

// Input is already resized/normalized CHW RGB: [3, rows*14, columns*14].
// Image resizing/normalization and selection of a token grid belong to the
// model's future preprocess stage. Batch size is one; no register/mask tokens.
// Native DINOv2 positional interpolation (offset 0.1, no antialias) is included.
// Verified with F32 weights and strict F32 backend kernels. GGML_PREC_F32 does
// not disable Vulkan's implicit operand conversion; backend startup policy is
// owned by the caller, never changed here via process environment variables.
EncoderResult encode_moge_vitb(MogeWeights& weights, GgmlRuntime& runtime,
    std::uint32_t rows, std::uint32_t columns, std::span<const float> normalized_rgb,
    bool capture_stages = false);
} // namespace photara::inference::detail
