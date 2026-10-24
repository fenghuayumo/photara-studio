#include "sam/language_detector.hpp"

#if defined(PHOTARA_HAS_INFERENCE)
#include "inference/models/grounding_dino.hpp"
#endif

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace photara::sam {
namespace {

float box_area(const LanguageBox& box) {
    return std::max(0.0F, box.x1 - box.x0) *
           std::max(0.0F, box.y1 - box.y0);
}

}  // namespace

class LanguageDetector::Impl {
public:
#if defined(PHOTARA_HAS_INFERENCE)
    explicit Impl(
        std::filesystem::path model_path,
        std::filesystem::path vocabulary_path,
        const std::string& backend)
        : vocabulary(std::move(vocabulary_path)) {
        inference::GroundingDinoOptions options;
        options.runtime.backend = backend == "cpu"
            ? inference::Backend::cpu
            : inference::Backend::vulkan;
        if (backend != "auto" && backend != "cpu" && backend != "vulkan")
            throw std::invalid_argument("Grounding DINO supports auto, cpu, or vulkan");
        options.runtime.allow_cpu_fallback = backend == "auto";
        detector = std::make_unique<inference::GroundingDinoDetector>(
            std::move(model_path), vocabulary, options);
    }

    std::unique_ptr<inference::GroundingDinoDetector> detector;
    std::filesystem::path vocabulary;
#endif
};

#if defined(PHOTARA_HAS_INFERENCE)
LanguageDetector::LanguageDetector(
    std::filesystem::path model_path,
    std::filesystem::path vocabulary_path,
    const std::string& backend)
    : impl_(new Impl(
          std::move(model_path), std::move(vocabulary_path), backend)) {}
#else
LanguageDetector::LanguageDetector(
    std::filesystem::path,
    std::filesystem::path,
    const std::string&)
    : impl_(nullptr) {
    throw std::runtime_error("Language masks require the GGML inference build");
}
#endif

LanguageDetector::~LanguageDetector() {
    delete impl_;
}

bool language_detector_built() {
#if defined(PHOTARA_HAS_INFERENCE)
    return true;
#else
    return false;
#endif
}

std::vector<LanguageBox> LanguageDetector::detect(
    const io::RgbImage& image,
    const std::string& phrase,
    const LanguageDetectOptions& options) const {
#if !defined(PHOTARA_HAS_INFERENCE)
    (void)image;
    (void)phrase;
    (void)options;
    throw std::runtime_error("Language masks require the GGML inference build");
#else
    inference::GroundingDinoDetectOptions detector_options;
    detector_options.box_threshold = options.threshold;
    detector_options.nms_threshold = options.nms;
    inference::GroundingDinoImage detector_image;
    detector_image.width = image.width;
    detector_image.height = image.height;
    detector_image.rgb = image.pixels;
    const auto found = impl_->detector->detect(
        detector_image, std::vector<std::string>{phrase}, detector_options);
    std::vector<LanguageBox> boxes;
    boxes.reserve(found.size());
    for (const auto& detection : found) {
        LanguageBox box;
        box.x0 = detection.x0 / static_cast<float>(image.width);
        box.y0 = detection.y0 / static_cast<float>(image.height);
        box.x1 = detection.x1 / static_cast<float>(image.width);
        box.y1 = detection.y1 / static_cast<float>(image.height);
        box.score = detection.score;
        const float area = box_area(box);
        if (area < options.min_area_fraction ||
            area > options.max_area_fraction)
            continue;
        boxes.push_back(box);
    }
    return boxes;
#endif
}

}  // namespace photara::sam
