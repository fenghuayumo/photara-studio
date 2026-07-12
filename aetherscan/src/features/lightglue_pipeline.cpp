#include "features/features.hpp"
#include "features/registry.hpp"

#include "io/image.hpp"

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aetherscan::features {

class LightGluePipeline::Impl {
public:
    explicit Impl(LightGlueOptions value) : options(std::move(value)) {
        if (options.model_path.empty() || options.input_width == 0 ||
            options.input_height == 0)
            throw std::invalid_argument(
                "LightGlue model path and input dimensions are required");
        if (options.min_score < 0.F || options.min_score > 1.F)
            throw std::invalid_argument(
                "LightGlue min_score must be in [0, 1]");
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (options.device == InferenceDevice::cuda) {
            try {
                OrtCUDAProviderOptions cuda{};
                session_options.AppendExecutionProvider_CUDA(cuda);
            } catch (const Ort::Exception&) {
                if (!options.allow_cpu_fallback) throw;
            }
        }
#if defined(_WIN32)
        session = std::make_unique<Ort::Session>(
            env, options.model_path.wstring().c_str(), session_options);
#else
        session = std::make_unique<Ort::Session>(
            env, options.model_path.string().c_str(), session_options);
#endif
#else
        throw std::runtime_error(
            "LightGlue requires ONNX Runtime. Configure "
            "AETHERSCAN_ONNXRUNTIME_ROOT "
            "(OpenCV DNN backend was removed).");
#endif
    }

    LightGlueOptions options;
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "AetherScan.LightGlue"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    mutable std::mutex mutex;

    struct PreparedImage {
        std::uint32_t width{};
        std::uint32_t height{};
        std::vector<float> network_input;  // C*H*W
    };

    PreparedImage prepare_rgb(const io::RgbImage& image) const {
        const std::uint32_t input_w = options.input_width;
        const std::uint32_t input_h = options.input_height;
        const bool gray = options.extractor == LightGlueExtractor::superpoint;
        const std::uint32_t channels = gray ? 1U : 3U;

        PreparedImage prepared;
        prepared.width = image.width;
        prepared.height = image.height;
        prepared.network_input.resize(
            static_cast<std::size_t>(channels) * input_h * input_w);

        std::vector<std::uint8_t> resized(
            static_cast<std::size_t>(input_w) * input_h * 3U);
        io::resize_bilinear(
            image.pixels.data(), image.width, image.height, 3, resized.data(),
            input_w, input_h);

        float* plane = prepared.network_input.data();
        if (gray) {
            for (std::uint32_t y = 0; y < input_h; ++y) {
                for (std::uint32_t x = 0; x < input_w; ++x) {
                    const std::uint8_t* rgb =
                        resized.data() +
                        (static_cast<std::size_t>(y) * input_w + x) * 3U;
                    plane[static_cast<std::size_t>(y) * input_w + x] =
                        (0.299F * rgb[0] + 0.587F * rgb[1] + 0.114F * rgb[2]) /
                        255.0F;
                }
            }
        } else {
            for (std::uint32_t c = 0; c < 3; ++c) {
                for (std::uint32_t y = 0; y < input_h; ++y) {
                    for (std::uint32_t x = 0; x < input_w; ++x) {
                        const std::uint8_t* rgb =
                            resized.data() +
                            (static_cast<std::size_t>(y) * input_w + x) * 3U;
                        plane[(static_cast<std::size_t>(c) * input_h + y) *
                                  input_w +
                              x] = rgb[c] / 255.0F;
                    }
                }
            }
        }
        return prepared;
    }

    ImagePairFeatures match_prepared(
        const PreparedImage& first, const PreparedImage& second) const {
        const std::uint32_t input_w = options.input_width;
        const std::uint32_t input_h = options.input_height;
        const bool gray = options.extractor == LightGlueExtractor::superpoint;
        const std::uint32_t channels = gray ? 1U : 3U;
        const std::size_t plane_size =
            static_cast<std::size_t>(channels) * input_h * input_w;

        std::vector<float> blob(plane_size * 2U);
        std::copy(
            first.network_input.begin(), first.network_input.end(), blob.begin());
        std::copy(
            second.network_input.begin(),
            second.network_input.end(),
            blob.begin() + static_cast<std::ptrdiff_t>(plane_size));

        const std::array<std::int64_t, 4> input_shape{
            2,
            static_cast<std::int64_t>(channels),
            static_cast<std::int64_t>(input_h),
            static_cast<std::int64_t>(input_w)};
        Ort::MemoryInfo memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory, blob.data(), blob.size(), input_shape.data(),
            input_shape.size());
        const char* input_names[]{"images"};
        const char* output_names[]{"keypoints", "matches", "mscores"};

        std::vector<Ort::Value> values;
        {
            std::lock_guard<std::mutex> lock(mutex);
            values = session->Run(
                Ort::RunOptions{nullptr}, input_names, &input, 1, output_names,
                3);
        }
        if (values.size() != 3)
            throw std::runtime_error("LightGlue returned unexpected outputs");

        const auto keypoint_shape =
            values[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto match_shape =
            values[1].GetTensorTypeAndShapeInfo().GetShape();
        if (keypoint_shape.size() != 3 || keypoint_shape[0] != 2 ||
            keypoint_shape[2] != 2 || match_shape.size() != 2 ||
            match_shape[1] != 3)
            throw std::runtime_error(
                "Unexpected LightGlue ONNX Runtime output signature");

        const std::size_t keypoint_count =
            static_cast<std::size_t>(keypoint_shape[1]);
        const std::size_t match_count =
            static_cast<std::size_t>(match_shape[0]);
        const float* keypoints = values[0].GetTensorData<float>();
        const auto match_type =
            values[1].GetTensorTypeAndShapeInfo().GetElementType();
        const float* scores = values[2].GetTensorData<float>();

        ImagePairFeatures result;
        FeatureSet* sets[2]{&result.first, &result.second};
        const PreparedImage* sources[2]{&first, &second};
        for (std::size_t image = 0; image < 2; ++image) {
            auto& set = *sets[image];
            set.image_width = sources[image]->width;
            set.image_height = sources[image]->height;
            set.extractor_name = "lightglue";
            set.metric = DescriptorMetric::inner_product;
            set.keypoints.reserve(keypoint_count);
            const float sx =
                static_cast<float>(sources[image]->width) /
                static_cast<float>(input_w);
            const float sy =
                static_cast<float>(sources[image]->height) /
                static_cast<float>(input_h);
            for (std::size_t point = 0; point < keypoint_count; ++point) {
                const std::size_t offset =
                    (image * keypoint_count + point) * 2U;
                set.keypoints.push_back(
                    {keypoints[offset] * sx, keypoints[offset + 1] * sy});
            }
        }

        result.matches.matches.reserve(match_count);
        auto push_match = [&](std::int64_t query, std::int64_t train,
                              float score) {
            if (score < options.min_score) return;
            if (query < 0 || train < 0 ||
                static_cast<std::size_t>(query) >= keypoint_count ||
                static_cast<std::size_t>(train) >= keypoint_count)
                throw std::runtime_error(
                    "LightGlue returned an invalid feature index");
            result.matches.matches.push_back(
                {static_cast<FeatureIndex>(query),
                 static_cast<FeatureIndex>(train),
                 score});
        };

        if (match_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            const std::int64_t* matches = values[1].GetTensorData<std::int64_t>();
            for (std::size_t i = 0; i < match_count; ++i)
                push_match(matches[i * 3 + 1], matches[i * 3 + 2], scores[i]);
        } else if (match_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* matches = values[1].GetTensorData<float>();
            for (std::size_t i = 0; i < match_count; ++i)
                push_match(
                    static_cast<std::int64_t>(matches[i * 3 + 1]),
                    static_cast<std::int64_t>(matches[i * 3 + 2]),
                    scores[i]);
        } else {
            throw std::runtime_error(
                "LightGlue returned an unexpected matches element type");
        }
        return result;
    }
#endif
};

LightGluePipeline::LightGluePipeline(LightGlueOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
LightGluePipeline::~LightGluePipeline() = default;
LightGluePipeline::LightGluePipeline(LightGluePipeline&&) noexcept = default;
LightGluePipeline& LightGluePipeline::operator=(LightGluePipeline&&) noexcept =
    default;

bool LightGluePipeline::is_built() noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return true;
#else
    return false;
#endif
}

bool LightGluePipeline::is_available() const noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return impl_ != nullptr && impl_->session != nullptr;
#else
    return false;
#endif
}

const LightGlueOptions& LightGluePipeline::options() const {
    return impl_->options;
}

ImagePairFeatures LightGluePipeline::match_files(
    const std::filesystem::path& first_path,
    const std::filesystem::path& second_path) {
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)first_path;
    (void)second_path;
    throw std::runtime_error("LightGlue requires ONNX Runtime");
#else
    return match_rgb(io::load_rgb(first_path), io::load_rgb(second_path));
#endif
}

ImagePairFeatures LightGluePipeline::match_rgb(
    const io::RgbImage& first, const io::RgbImage& second) {
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)first;
    (void)second;
    throw std::runtime_error("LightGlue requires ONNX Runtime");
#else
    if (first.width == 0 || first.height == 0 || second.width == 0 ||
        second.height == 0)
        throw std::invalid_argument(
            "LightGlue requires non-empty RGB images");
    const auto prepared0 = impl_->prepare_rgb(first);
    const auto prepared1 = impl_->prepare_rgb(second);
    return impl_->match_prepared(prepared0, prepared1);
#endif
}

void register_lightglue_feature_backends() {
    // LightGlue needs a model path at construction time, so product code
    // builds LightGluePipeline(options) directly (see run_frontend). Optional
    // factory binding:
    //   register_pair_pipeline("lightglue", [opts] {
    //       return std::make_unique<LightGluePipeline>(opts);
    //   });
}

}  // namespace aetherscan::features
