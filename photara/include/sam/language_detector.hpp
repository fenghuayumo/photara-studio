#pragma once

#include "io/image.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace photara::sam {

struct LanguageBox {
    float x0{};
    float y0{};
    float x1{};
    float y1{};
    float score{};
};

struct LanguageDetectOptions {
    float threshold{0.35F};
    float nms{0.1F};
    float min_area_fraction{0.001F};
    float max_area_fraction{1.F};
};

// Adapter from Photara's GGML Grounding DINO detector to the SAM mask
// pipeline. Boxes are normalized to the supplied image.
class LanguageDetector {
public:
    LanguageDetector(
        std::filesystem::path model_path,
        std::filesystem::path vocabulary_path,
        const std::string& backend);
    ~LanguageDetector();
    LanguageDetector(const LanguageDetector&) = delete;
    LanguageDetector& operator=(const LanguageDetector&) = delete;

    [[nodiscard]] std::vector<LanguageBox> detect(
        const io::RgbImage& image,
        const std::string& phrase,
        const LanguageDetectOptions& options) const;

private:
    class Impl;
    Impl* impl_{};
};

[[nodiscard]] bool language_detector_built();

}  // namespace photara::sam
