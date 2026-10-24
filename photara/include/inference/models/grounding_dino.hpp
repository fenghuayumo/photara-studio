#pragma once

#include "inference/model_info.hpp"
#include "inference/options.hpp"
#include "inference/runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace photara::inference {

struct GroundingDinoModelInfo {
    ModelInfo model;
    std::uint32_t encoder_layers = 0;
    std::uint32_t decoder_layers = 0;
    std::uint32_t text_layers = 0;
    std::uint32_t model_dimension = 0;
    std::uint32_t text_dimension = 0;
    std::uint32_t queries = 0;
    std::size_t tensor_count = 0;
    std::size_t tensor_bytes = 0;
};

struct GroundingDinoImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgb;  // HWC, top-left origin
};

struct GroundingDinoDetection {
    float x0 = 0;
    float y0 = 0;
    float x1 = 0;
    float y1 = 0;
    float score = 0;
    std::size_t phrase = 0;
};

struct GroundingDinoOptions {
    RuntimeOptions runtime{Backend::vulkan, 0, true, 0};
    float box_threshold = 0.35F;
    float text_threshold = 0.25F;
    float nms_threshold = 0.10F;
    std::uint32_t short_side = 800;
    std::uint32_t long_side_cap = 1333;
};

struct GroundingDinoDetectOptions {
    float box_threshold = 0.35F;
    float text_threshold = 0.25F;
    float nms_threshold = 0.10F;
};

// Text-to-box detector implemented directly on Photara's GGML runtime. The
// weights are the Apache-2.0 Grounding DINO safetensors checkpoints; no ONNX
// graph or external inference engine is involved.
class GroundingDinoDetector {
public:
    GroundingDinoDetector(
        const std::filesystem::path& weights,
        const std::filesystem::path& vocabulary,
        GroundingDinoOptions options);
    ~GroundingDinoDetector();
    GroundingDinoDetector(const GroundingDinoDetector&) = delete;
    GroundingDinoDetector& operator=(const GroundingDinoDetector&) = delete;

    [[nodiscard]] const GroundingDinoModelInfo& model_info() const noexcept;
    [[nodiscard]] const DeviceInfo& device_info() const noexcept;
    [[nodiscard]] std::vector<GroundingDinoDetection> detect(
        const GroundingDinoImage& image,
        const std::vector<std::string>& phrases,
        const GroundingDinoDetectOptions& options = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace photara::inference
