#pragma once

#include "weights.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace photara::inference::detail {

struct GroundingDinoHeadOutput {
    std::size_t queries = 0;
    std::size_t tokens = 0;
    std::vector<float> logits;
    std::vector<float> boxes;
};

class GroundingDinoModel {
public:
    GroundingDinoModel(
        GroundingDinoWeights& weights,
        GgmlRuntime& runtime);

    [[nodiscard]] GroundingDinoHeadOutput run(
        const std::vector<float>& chw_image,
        std::uint32_t height,
        std::uint32_t width,
        const std::vector<std::int32_t>& tokens);

private:
    struct SwinOutput {
        std::vector<float> maps[3];
        int heights[3]{};
        int widths[3]{};
    };

    [[nodiscard]] SwinOutput run_swin(
        const std::vector<float>& image,
        std::uint32_t height,
        std::uint32_t width);
    [[nodiscard]] std::vector<float> run_text(
        const std::vector<std::int32_t>& tokens,
        const std::vector<std::int32_t>& positions,
        std::vector<float>& phrase_mask,
        std::vector<float>& text_position);

    GroundingDinoWeights& weights_;
    GgmlRuntime& runtime_;
};

}  // namespace photara::inference::detail
