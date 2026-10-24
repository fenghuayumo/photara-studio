#pragma once

#include "inference/models/grounding_dino.hpp"
#include "../../runtime/ggml_runtime.hpp"
#include "ggml-backend.h"

#include <filesystem>
#include <map>
#include <memory>
#include <string>

namespace photara::inference::detail {

struct GroundingDinoConfig {
    std::uint32_t encoder_layers = 0;
    std::uint32_t decoder_layers = 0;
    std::uint32_t text_layers = 0;
    std::uint32_t model_dimension = 256;
    std::uint32_t text_dimension = 768;
    std::uint32_t queries = 900;
    std::uint32_t levels = 4;
    std::uint32_t sampling_points = 4;
    std::uint32_t encoder_heads = 8;
    std::uint32_t decoder_heads = 8;
    std::uint32_t swin_embedding = 96;
    std::uint32_t swin_depths[4] = {2, 2, 6, 2};
    std::uint32_t swin_heads[4] = {3, 6, 12, 24};
    std::uint32_t swin_window = 7;
};

class GroundingDinoWeights {
public:
    GroundingDinoWeights(
        const std::filesystem::path& path,
        ggml_backend_t backend,
        std::size_t maximum_tensor_bytes = 0);
    ~GroundingDinoWeights();
    GroundingDinoWeights(const GroundingDinoWeights&) = delete;
    GroundingDinoWeights& operator=(const GroundingDinoWeights&) = delete;

    [[nodiscard]] const GroundingDinoModelInfo& info() const noexcept;
    [[nodiscard]] const GroundingDinoConfig& config() const noexcept;
    [[nodiscard]] bool has(const std::string& name) const noexcept;
    [[nodiscard]] ggml_tensor* tensor(const std::string& name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace photara::inference::detail
