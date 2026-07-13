#include "features/features.hpp"
#include "features/onnx_session.hpp"
#include "features/registry.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aetherscan::features {
namespace {

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
std::size_t named_index(
    const std::vector<std::string>& names, const std::string& wanted) {
    const auto found = std::find(names.begin(), names.end(), wanted);
    if (found == names.end())
        throw std::runtime_error("ALIKED model is missing node: " + wanted);
    return static_cast<std::size_t>(found - names.begin());
}
#endif

}  // namespace

class AlikedExtractor::Impl {
public:
    explicit Impl(AlikedOptions value) : options(std::move(value)) {
        if (options.model_path.empty())
            throw std::invalid_argument("ALIKED model path is required");
        if (options.maximum_features == 0 ||
            options.maximum_features >
                static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()))
            throw std::invalid_argument("ALIKED maximum_features is invalid");
        if (options.keypoint_threshold < 0.F || options.keypoint_threshold > 1.F)
            throw std::invalid_argument(
                "ALIKED keypoint_threshold must be in [0, 1]");
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
        OnnxSessionConfig config;
        config.model_path = options.model_path;
        config.use_cuda = options.cuda;
        config.allow_cpu_fallback = options.allow_cpu_fallback;
        config.env_name = "AetherScan.ALIKED";
        session = std::make_unique<OnnxSession>(std::move(config));

        const auto& inputs = session->input_name_strings();
        const auto& outputs = session->output_name_strings();
        if (inputs.size() != 3 || outputs.size() != 3)
            throw std::runtime_error(
                "ALIKED model must have 3 inputs and 3 outputs");
        image_input = named_index(inputs, "image");
        maximum_input = named_index(inputs, "max_keypoints");
        score_input = named_index(inputs, "min_score");
        keypoints_output = named_index(outputs, "keypoints");
        descriptors_output = named_index(outputs, "descriptors");
        scores_output = named_index(outputs, "scores");
#else
        throw std::runtime_error(
            "ALIKED requires ONNX Runtime (AETHERSCAN_ENABLE_ONNX)");
#endif
    }

    AlikedOptions options;
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    std::unique_ptr<OnnxSession> session;
    std::size_t image_input{};
    std::size_t maximum_input{};
    std::size_t score_input{};
    std::size_t keypoints_output{};
    std::size_t descriptors_output{};
    std::size_t scores_output{};
#endif
};

AlikedExtractor::AlikedExtractor(AlikedOptions options)
    : impl_(std::make_shared<Impl>(std::move(options))) {}
AlikedExtractor::AlikedExtractor(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AlikedExtractor::~AlikedExtractor() = default;
AlikedExtractor::AlikedExtractor(AlikedExtractor&&) noexcept = default;
AlikedExtractor& AlikedExtractor::operator=(AlikedExtractor&&) noexcept =
    default;

bool AlikedExtractor::is_built() noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return true;
#else
    return false;
#endif
}

bool AlikedExtractor::is_available() const noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return impl_ != nullptr && impl_->session != nullptr;
#else
    return false;
#endif
}

ExtractorInfo AlikedExtractor::info() const {
    ExtractorInfo value;
    value.thread_safe = false;
    value.thread_affine = true;
    value.accepts_gray = false;
    value.accepts_rgb = true;
    value.metric = DescriptorMetric::inner_product;
    value.typical_descriptor_dimension = 128;
    return value;
}

std::unique_ptr<FeatureExtractor> AlikedExtractor::clone() const {
    return std::unique_ptr<FeatureExtractor>(new AlikedExtractor(impl_));
}

FeatureSet AlikedExtractor::extract_gray(
    std::span<const std::uint8_t> /*pixels*/, std::uint32_t /*width*/,
    std::uint32_t /*height*/, std::size_t /*row_stride*/) const {
    throw std::runtime_error("ALIKED extractor requires RGB input");
}

FeatureSet AlikedExtractor::extract_rgb(
    const std::span<const std::uint8_t> pixels, const std::uint32_t width,
    const std::uint32_t height, std::size_t row_stride) const {
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)pixels;
    (void)width;
    (void)height;
    (void)row_stride;
    throw std::runtime_error("ALIKED requires ONNX Runtime");
#else
    if (width == 0 || height == 0 || pixels.empty())
        throw std::invalid_argument("ALIKED requires a non-empty image");
    if (row_stride == 0) row_stride = static_cast<std::size_t>(width) * 3U;
    if (row_stride < static_cast<std::size_t>(width) * 3U ||
        pixels.size() < row_stride * height)
        throw std::invalid_argument("Invalid ALIKED RGB image view");

    constexpr std::uint32_t divisor = 32;
    const std::uint32_t padded_width =
        ((width + divisor - 1U) / divisor) * divisor;
    const std::uint32_t padded_height =
        ((height + divisor - 1U) / divisor) * divisor;
    const std::size_t plane =
        static_cast<std::size_t>(padded_width) * padded_height;
    std::vector<float> input(plane * 3U);
    for (std::uint32_t channel = 0; channel < 3; ++channel) {
        for (std::uint32_t y = 0; y < padded_height; ++y) {
            const std::uint32_t source_y = (std::min)(y, height - 1U);
            for (std::uint32_t x = 0; x < padded_width; ++x) {
                const std::uint32_t source_x = (std::min)(x, width - 1U);
                input[(static_cast<std::size_t>(channel) * padded_height + y) *
                          padded_width +
                      x] =
                    pixels[static_cast<std::size_t>(source_y) * row_stride +
                           static_cast<std::size_t>(source_x) * 3U + channel] /
                    255.F;
            }
        }
    }

    const std::array<std::int64_t, 4> image_shape{
        1, 3, static_cast<std::int64_t>(padded_height),
        static_cast<std::int64_t>(padded_width)};
    std::int64_t maximum =
        static_cast<std::int64_t>(impl_->options.maximum_features);
    float threshold = impl_->options.keypoint_threshold;
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    inputs.reserve(3);
    for (std::size_t index = 0; index < 3; ++index) {
        if (index == impl_->image_input) {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, input.data(), input.size(), image_shape.data(),
                image_shape.size()));
        } else if (index == impl_->maximum_input) {
            inputs.emplace_back(Ort::Value::CreateTensor<std::int64_t>(
                memory, &maximum, 1, nullptr, 0));
        } else if (index == impl_->score_input) {
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, &threshold, 1, nullptr, 0));
        } else {
            throw std::runtime_error("Unexpected ALIKED input ordering");
        }
    }

    std::vector<Ort::Value> outputs;
    {
        std::lock_guard lock(impl_->session->mutex());
        outputs = impl_->session->session().Run(
            Ort::RunOptions{nullptr}, impl_->session->input_names().data(),
            inputs.data(), inputs.size(),
            impl_->session->output_names().data(),
            impl_->session->output_names().size());
    }
    if (outputs.size() != 3)
        throw std::runtime_error("ALIKED returned unexpected outputs");

    const auto keypoint_shape = outputs[impl_->keypoints_output]
                                    .GetTensorTypeAndShapeInfo()
                                    .GetShape();
    const auto descriptor_shape = outputs[impl_->descriptors_output]
                                     .GetTensorTypeAndShapeInfo()
                                     .GetShape();
    const auto score_shape = outputs[impl_->scores_output]
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
    if (keypoint_shape.size() != 3 || keypoint_shape[0] != 1 ||
        keypoint_shape[2] != 2 || descriptor_shape.size() != 3 ||
        descriptor_shape[0] != 1 || score_shape.size() != 2 ||
        score_shape[0] != 1)
        throw std::runtime_error("Unexpected ALIKED output shapes");
    const std::size_t count = static_cast<std::size_t>(keypoint_shape[1]);
    const std::size_t dimension =
        static_cast<std::size_t>(descriptor_shape[2]);
    if (descriptor_shape[1] != static_cast<std::int64_t>(count) ||
        score_shape[1] != static_cast<std::int64_t>(count) || dimension == 0)
        throw std::runtime_error("Inconsistent ALIKED output shapes");

    const float* keypoints =
        outputs[impl_->keypoints_output].GetTensorData<float>();
    const float* descriptors =
        outputs[impl_->descriptors_output].GetTensorData<float>();
    const float* scores = outputs[impl_->scores_output].GetTensorData<float>();
    const float scale_x = 0.5F * static_cast<float>(padded_width - 1U);
    const float scale_y = 0.5F * static_cast<float>(padded_height - 1U);

    FeatureSet result;
    result.image_width = width;
    result.image_height = height;
    result.descriptor_dimension = dimension;
    result.storage = DescriptorStorage::float32;
    result.metric = DescriptorMetric::inner_product;
    result.extractor_name = "aliked";
    result.keypoints.reserve(count);
    result.descriptors.reserve(count * dimension);
    for (std::size_t index = 0; index < count; ++index) {
        if (scores[index] < threshold) continue;
        const float x = (keypoints[index * 2] + 1.F) * scale_x;
        const float y = (keypoints[index * 2 + 1] + 1.F) * scale_y;
        if (x < 0.F || x >= static_cast<float>(width) || y < 0.F ||
            y >= static_cast<float>(height))
            continue;
        result.keypoints.push_back({x, y, 1.F, 0.F, scores[index]});
        const float* row = descriptors + index * dimension;
        result.descriptors.insert(result.descriptors.end(), row, row + dimension);
    }
    return result;
#endif
}

void register_aliked_feature_backends() {
    register_extractor("aliked", [] {
        throw std::runtime_error(
            "aliked requires AlikedOptions.model_path; construct "
            "features::AlikedExtractor(options) from the frontend");
        return std::unique_ptr<FeatureExtractor>{};
    });
}

}  // namespace aetherscan::features
