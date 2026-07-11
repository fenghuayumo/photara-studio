#include "features/features.hpp"

#include "io/image.hpp"

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#endif

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aetherscan::features {

class LightGluePipeline::Impl {
public:
    explicit Impl(LightGlueOptions value) : options(std::move(value)) {
        if (options.model_path.empty() || options.input_width == 0 || options.input_height == 0)
            throw std::invalid_argument("LightGlue model path and input dimensions are required");
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (options.device == InferenceDevice::cuda) {
            try {
                OrtCUDAProviderOptions cuda{};
                session_options.AppendExecutionProvider_CUDA(cuda);
            } catch (const Ort::Exception&) {
                if (!options.allow_cpu_fallback) throw;
            }
        }
        session = std::make_unique<Ort::Session>(env, options.model_path.c_str(), session_options);
#else
        throw std::runtime_error(
            "LightGlue requires ONNX Runtime. Configure AETHERSCAN_ONNXRUNTIME_ROOT "
            "(OpenCV DNN backend was removed).");
#endif
    }
    LightGlueOptions options;
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "AetherScan.LightGlue"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
#endif
};

LightGluePipeline::LightGluePipeline(LightGlueOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
LightGluePipeline::~LightGluePipeline() = default;
LightGluePipeline::LightGluePipeline(LightGluePipeline&&) noexcept = default;
LightGluePipeline& LightGluePipeline::operator=(LightGluePipeline&&) noexcept = default;

ImagePairFeatures LightGluePipeline::match_files(
    const std::filesystem::path& first_path, const std::filesystem::path& second_path) {
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)first_path;
    (void)second_path;
    throw std::runtime_error("LightGlue requires ONNX Runtime");
#else
    const std::array<io::RgbImage, 2> original{io::load_rgb(first_path), io::load_rgb(second_path)};
    const std::uint32_t input_w = impl_->options.input_width;
    const std::uint32_t input_h = impl_->options.input_height;
    const bool gray = impl_->options.extractor == LightGlueExtractor::superpoint;
    const std::uint32_t channels = gray ? 1U : 3U;
    std::vector<float> blob(static_cast<std::size_t>(2) * channels * input_h * input_w);

    for (std::size_t image = 0; image < 2; ++image) {
        std::vector<std::uint8_t> resized(static_cast<std::size_t>(input_w) * input_h * 3);
        io::resize_bilinear(
            original[image].pixels.data(), original[image].width, original[image].height, 3,
            resized.data(), input_w, input_h);
        float* plane = blob.data() + image * channels * input_h * input_w;
        if (gray) {
            for (std::uint32_t y = 0; y < input_h; ++y)
                for (std::uint32_t x = 0; x < input_w; ++x) {
                    const std::uint8_t* rgb =
                        resized.data() + (static_cast<std::size_t>(y) * input_w + x) * 3;
                    const float value =
                        (0.299F * rgb[0] + 0.587F * rgb[1] + 0.114F * rgb[2]) / 255.0F;
                    plane[static_cast<std::size_t>(y) * input_w + x] = value;
                }
        } else {
            for (std::uint32_t c = 0; c < 3; ++c)
                for (std::uint32_t y = 0; y < input_h; ++y)
                    for (std::uint32_t x = 0; x < input_w; ++x) {
                        const std::uint8_t* rgb =
                            resized.data() + (static_cast<std::size_t>(y) * input_w + x) * 3;
                        plane[(static_cast<std::size_t>(c) * input_h + y) * input_w + x] =
                            rgb[c] / 255.0F;
                    }
        }
    }

    const std::array<std::int64_t, 4> input_shape{
        2, static_cast<std::int64_t>(channels), static_cast<std::int64_t>(input_h),
        static_cast<std::int64_t>(input_w)};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory, blob.data(), blob.size(), input_shape.data(), input_shape.size());
    const char* input_names[]{"images"};
    const char* output_names[]{"keypoints", "matches", "mscores"};
    auto values =
        impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 3);
    const auto keypoint_shape = values[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto match_shape = values[1].GetTensorTypeAndShapeInfo().GetShape();
    if (keypoint_shape.size() != 3 || keypoint_shape[0] != 2 || keypoint_shape[2] != 2 ||
        match_shape.size() != 2 || match_shape[1] != 3)
        throw std::runtime_error("Unexpected LightGlue ONNX Runtime output signature");
    const std::size_t keypoint_count = static_cast<std::size_t>(keypoint_shape[1]);
    const std::size_t match_count = static_cast<std::size_t>(match_shape[0]);
    const float* keypoints = values[0].GetTensorData<float>();
    const std::int64_t* matches = values[1].GetTensorData<std::int64_t>();
    const float* scores = values[2].GetTensorData<float>();

    ImagePairFeatures result;
    FeatureSet* sets[2]{&result.first, &result.second};
    for (std::size_t image = 0; image < 2; ++image) {
        auto& set = *sets[image];
        set.image_width = original[image].width;
        set.image_height = original[image].height;
        set.extractor_name = "lightglue";
        set.metric = DescriptorMetric::inner_product;
        set.keypoints.reserve(keypoint_count);
        const float sx = static_cast<float>(original[image].width) / input_w;
        const float sy = static_cast<float>(original[image].height) / input_h;
        for (std::size_t point = 0; point < keypoint_count; ++point) {
            const std::size_t offset = (image * keypoint_count + point) * 2;
            set.keypoints.push_back({keypoints[offset] * sx, keypoints[offset + 1] * sy});
        }
    }
    result.matches.matches.reserve(match_count);
    for (std::size_t i = 0; i < match_count; ++i) {
        const auto query = matches[i * 3 + 1], train = matches[i * 3 + 2];
        if (query < 0 || train < 0 || static_cast<std::size_t>(query) >= keypoint_count ||
            static_cast<std::size_t>(train) >= keypoint_count)
            throw std::runtime_error("LightGlue returned an invalid feature index");
        result.matches.matches.push_back(
            {static_cast<FeatureIndex>(query), static_cast<FeatureIndex>(train), scores[i]});
    }
    return result;
#endif
}

void register_lightglue_feature_backends() {
    // LightGlue needs a model path at construction time, so product code should
    // build LightGluePipeline(options) directly. The name "lightglue" is reserved
    // for documentation; bind a factory with a captured model when desired:
    //   register_pair_pipeline("lightglue", [opts]{ return std::make_unique<LightGluePipeline>(opts); });
}

}  // namespace aetherscan::features
