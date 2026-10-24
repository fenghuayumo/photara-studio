#include "inference/models/grounding_dino.hpp"

#include "model.hpp"
#include "tokenizer.hpp"
#include "weights.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace photara::inference {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error("Grounding DINO detector: " + message);
}

void target_size(
    const std::uint32_t height,
    const std::uint32_t width,
    const std::uint32_t short_side,
    const std::uint32_t long_side_cap,
    std::uint32_t& output_height,
    std::uint32_t& output_width) {
    const double minimum = std::min(height, width);
    const double maximum = std::max(height, width);
    double scale = short_side;
    if (long_side_cap > 0 && maximum / minimum * short_side > long_side_cap)
        scale = long_side_cap * minimum / maximum;
    if (height <= width) {
        output_height = std::max(1U, static_cast<std::uint32_t>(
            std::lround(scale)));
        output_width = std::max(1U, static_cast<std::uint32_t>(
            std::lround(scale * width / height)));
    } else {
        output_width = std::max(1U, static_cast<std::uint32_t>(
            std::lround(scale)));
        output_height = std::max(1U, static_cast<std::uint32_t>(
            std::lround(scale * height / width)));
    }
}

std::vector<float> preprocess(
    const GroundingDinoImage& image,
    const std::uint32_t height,
    const std::uint32_t width) {
    constexpr float mean[3] = {0.485F, 0.456F, 0.406F};
    constexpr float deviation[3] = {0.229F, 0.224F, 0.225F};
    std::vector<float> output(
        static_cast<std::size_t>(width) * height * 3U);
    const auto sample = [&](const std::uint32_t x, const std::uint32_t y,
                            const std::uint32_t channel) {
        const auto source_x = static_cast<std::uint32_t>(std::min<double>(
            image.width - 1U,
            static_cast<double>(x) * image.width / width));
        const auto source_y = static_cast<std::uint32_t>(std::min<double>(
            image.height - 1U,
            static_cast<double>(y) * image.height / height));
        return image.rgb[(static_cast<std::size_t>(source_y) *
                              image.width + source_x) * 3U + channel] /
               255.0F;
    };
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
            for (std::uint32_t channel = 0; channel < 3; ++channel)
                output[(static_cast<std::size_t>(channel) * height + y) * width + x] =
                    (sample(x, y, channel) - mean[channel]) /
                    deviation[channel];
    return output;
}

float sigmoid(const float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

float intersection_over_union(
    const GroundingDinoDetection& left,
    const GroundingDinoDetection& right) {
    const float x0 = std::max(left.x0, right.x0);
    const float y0 = std::max(left.y0, right.y0);
    const float x1 = std::min(left.x1, right.x1);
    const float y1 = std::min(left.y1, right.y1);
    const float intersection = std::max(0.0F, x1 - x0) *
                               std::max(0.0F, y1 - y0);
    const float left_area = std::max(0.0F, left.x1 - left.x0) *
                            std::max(0.0F, left.y1 - left.y0);
    const float right_area = std::max(0.0F, right.x1 - right.x0) *
                             std::max(0.0F, right.y1 - right.y0);
    const float union_area = left_area + right_area - intersection;
    return union_area <= 0.0F ? 0.0F : intersection / union_area;
}

}  // namespace

class GroundingDinoDetector::Impl {
public:
    Impl(
        const std::filesystem::path& weights_path,
        const std::filesystem::path& vocabulary_path,
        GroundingDinoOptions detector_options)
        : runtime(detector_options.runtime),
          short_side(detector_options.short_side),
          long_side_cap(detector_options.long_side_cap),
          weights(weights_path, runtime.backend()),
          model(weights, runtime) {
        tokenizer.load(vocabulary_path);
    }

    std::uint32_t short_side;
    std::uint32_t long_side_cap;
    detail::GgmlRuntime runtime;
    detail::GroundingDinoWeights weights;
    detail::GroundingDinoModel model;
    detail::GroundingDinoTokenizer tokenizer;
};

GroundingDinoDetector::GroundingDinoDetector(
    const std::filesystem::path& weights,
    const std::filesystem::path& vocabulary,
    GroundingDinoOptions options)
    : impl_(std::make_unique<Impl>(weights, vocabulary, std::move(options))) {}

GroundingDinoDetector::~GroundingDinoDetector() = default;

const GroundingDinoModelInfo& GroundingDinoDetector::model_info()
    const noexcept {
    return impl_->weights.info();
}

const DeviceInfo& GroundingDinoDetector::device_info() const noexcept {
    return impl_->runtime.device();
}

std::vector<GroundingDinoDetection> GroundingDinoDetector::detect(
    const GroundingDinoImage& image,
    const std::vector<std::string>& phrases,
    const GroundingDinoDetectOptions& options) {
    if (phrases.empty() || image.width == 0 || image.height == 0 ||
        image.rgb.size() <
            static_cast<std::size_t>(image.width) * image.height * 3U)
        return {};
    if (!(options.box_threshold >= 0.F && options.box_threshold <= 1.F))
        throw std::invalid_argument("Grounding DINO box threshold must be in [0, 1]");
    if (!(options.text_threshold >= 0.F && options.text_threshold <= 1.F))
        throw std::invalid_argument("Grounding DINO text threshold must be in [0, 1]");
    if (!(options.nms_threshold >= 0.F && options.nms_threshold <= 1.F))
        throw std::invalid_argument("Grounding DINO NMS threshold must be in [0, 1]");

    std::vector<std::int32_t> tokens{
        detail::GroundingDinoTokenizer::k_cls};
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    std::vector<std::size_t> phrase_indices;
    for (std::size_t phrase_index = 0; phrase_index < phrases.size();
         ++phrase_index) {
        std::string phrase = phrases[phrase_index];
        for (char& character : phrase)
            if (character == '.' || character == '?') character = ' ';
        const auto encoded = impl_->tokenizer.encode(phrase);
        if (encoded.size() <= 2U) continue;
        if (tokens.size() + encoded.size() > 255U) break;
        spans.emplace_back(
            tokens.size(), tokens.size() + encoded.size() - 2U);
        tokens.insert(
            tokens.end(), encoded.begin() + 1, encoded.end() - 1);
        tokens.push_back(detail::GroundingDinoTokenizer::k_period);
        phrase_indices.push_back(phrase_index);
    }
    tokens.push_back(detail::GroundingDinoTokenizer::k_sep);
    if (spans.empty()) return {};

    std::uint32_t height = 0;
    std::uint32_t width = 0;
    target_size(image.height, image.width, impl_->short_side,
        impl_->long_side_cap, height, width);
    const auto output = impl_->model.run(
        preprocess(image, height, width), height, width, tokens);
    std::vector<GroundingDinoDetection> detections;
    for (std::size_t query = 0; query < output.queries; ++query) {
        const float* logits =
            &output.logits[query * output.tokens];
        float score = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < output.tokens; ++token)
            score = std::max(score, sigmoid(logits[token]));
        if (score <= options.box_threshold) continue;
        float phrase_score = -std::numeric_limits<float>::infinity();
        std::size_t selected_span = 0;
        for (std::size_t span = 0; span < spans.size(); ++span) {
            for (std::size_t token = spans[span].first;
                 token < spans[span].second; ++token) {
                const float value = sigmoid(logits[token]);
                if (value > phrase_score) {
                    phrase_score = value;
                    selected_span = span;
                }
            }
        }
        if (phrase_score < options.text_threshold) continue;
        const float* box = &output.boxes[query * 4U];
        GroundingDinoDetection detection;
        detection.x0 = std::max(
            0.0F, (box[0] - 0.5F * box[2]) * image.width);
        detection.y0 = std::max(
            0.0F, (box[1] - 0.5F * box[3]) * image.height);
        detection.x1 = std::min(
            static_cast<float>(image.width),
            (box[0] + 0.5F * box[2]) * image.width);
        detection.y1 = std::min(
            static_cast<float>(image.height),
            (box[1] + 0.5F * box[3]) * image.height);
        detection.score = score;
        detection.phrase = phrase_indices[selected_span];
        if (detection.x1 > detection.x0 && detection.y1 > detection.y0)
            detections.push_back(detection);
    }
    std::sort(detections.begin(), detections.end(),
        [](const GroundingDinoDetection& left,
           const GroundingDinoDetection& right) {
            if (left.score != right.score) return left.score > right.score;
            return left.phrase < right.phrase;
        });
    std::vector<GroundingDinoDetection> kept;
    for (const auto& candidate : detections) {
        bool suppressed = false;
        for (const auto& existing : kept) {
            if (candidate.phrase == existing.phrase &&
                intersection_over_union(candidate, existing) >
                    options.nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
    }
    return kept;
}

}  // namespace photara::inference
