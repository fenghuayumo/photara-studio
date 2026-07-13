#include "features/features.hpp"
#include "features/compat.hpp"
#include "features/registry.hpp"

#include "io/image.hpp"

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aetherscan::features {

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
namespace {

enum class LightGlueModelLayout {
    // Legacy batch: images[2,C,H,W] → keypoints, matches[M,3], mscores
    batched_images,
    // fabio-sim / Kornia end2end:
    // image0/image1[1,1,H,W] → kpts0/1, matches0/1, mscores0/1
    dual_image_end2end,
};

}  // namespace
#endif

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
        detect_layout();
#else
        throw std::runtime_error(
            "LightGlue requires ONNX Runtime. Configure "
            "AETHERSCAN_ENABLE_ONNX / AETHERSCAN_ONNXRUNTIME_ROOT.");
#endif
    }

    LightGlueOptions options;
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "AetherScan.LightGlue"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    mutable std::mutex mutex;
    LightGlueModelLayout layout{LightGlueModelLayout::batched_images};
    std::vector<std::string> input_names_owned;
    std::vector<std::string> output_names_owned;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;

    void detect_layout() {
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t num_inputs = session->GetInputCount();
        const std::size_t num_outputs = session->GetOutputCount();
        input_names_owned.reserve(num_inputs);
        input_name_ptrs.reserve(num_inputs);
        for (std::size_t i = 0; i < num_inputs; ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_names_owned.emplace_back(name.get());
            input_name_ptrs.push_back(input_names_owned.back().c_str());
        }
        output_names_owned.reserve(num_outputs);
        output_name_ptrs.reserve(num_outputs);
        for (std::size_t i = 0; i < num_outputs; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names_owned.emplace_back(name.get());
            output_name_ptrs.push_back(output_names_owned.back().c_str());
        }

        const bool has_images =
            std::find(
                input_names_owned.begin(), input_names_owned.end(), "images") !=
            input_names_owned.end();
        const bool has_image0 =
            std::find(
                input_names_owned.begin(), input_names_owned.end(),
                "image0") != input_names_owned.end();
        const bool has_image1 =
            std::find(
                input_names_owned.begin(), input_names_owned.end(),
                "image1") != input_names_owned.end();

        if (has_image0 && has_image1) {
            layout = LightGlueModelLayout::dual_image_end2end;
            if (options.extractor != LightGlueExtractor::superpoint) {
                // End2end SuperPoint models expect single-channel input.
                options.extractor = LightGlueExtractor::superpoint;
            }
        } else if (has_images) {
            layout = LightGlueModelLayout::batched_images;
        } else {
            throw std::runtime_error(
                "Unsupported LightGlue ONNX inputs; expected \"images\" or "
                "\"image0\"/\"image1\"");
        }
    }

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

    ImagePairFeatures match_batched(
        const PreparedImage& first, const PreparedImage& second) const {
        const std::uint32_t input_w = options.input_width;
        const std::uint32_t input_h = options.input_height;
        const bool gray = options.extractor == LightGlueExtractor::superpoint;
        const std::uint32_t channels = gray ? 1U : 3U;
        const std::size_t plane_size =
            static_cast<std::size_t>(channels) * input_h * input_w;

        std::vector<float> blob(plane_size * 2U);
        std::copy(
            first.network_input.begin(), first.network_input.end(),
            blob.begin());
        std::copy(
            second.network_input.begin(), second.network_input.end(),
            blob.begin() + static_cast<std::ptrdiff_t>(plane_size));

        const std::array<std::int64_t, 4> input_shape{
            2, static_cast<std::int64_t>(channels),
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
                "Unexpected batched LightGlue ONNX output signature");

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
            const float sx = static_cast<float>(sources[image]->width) /
                static_cast<float>(input_w);
            const float sy = static_cast<float>(sources[image]->height) /
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
                 static_cast<FeatureIndex>(train), score});
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
                    static_cast<std::int64_t>(matches[i * 3 + 2]), scores[i]);
        } else {
            throw std::runtime_error(
                "LightGlue returned an unexpected matches element type");
        }
        return result;
    }

    ImagePairFeatures match_dual_end2end(
        const PreparedImage& first, const PreparedImage& second) const {
        const std::uint32_t input_w = options.input_width;
        const std::uint32_t input_h = options.input_height;
        // SuperPoint end2end: NCHW gray [1,1,H,W]
        const std::array<std::int64_t, 4> shape0{
            1, 1, static_cast<std::int64_t>(input_h),
            static_cast<std::int64_t>(input_w)};
        const std::array<std::int64_t, 4> shape1 = shape0;

        Ort::MemoryInfo memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<float> input0 = first.network_input;
        std::vector<float> input1 = second.network_input;
        std::vector<Ort::Value> inputs;
        inputs.reserve(2);
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, input0.data(), input0.size(), shape0.data(),
            shape0.size()));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            memory, input1.data(), input1.size(), shape1.data(),
            shape1.size()));

        // Bind by known names regardless of session order.
        const char* run_inputs[]{"image0", "image1"};
        const char* run_outputs[]{
            "kpts0", "kpts1", "matches0", "matches1", "mscores0", "mscores1"};

        std::vector<Ort::Value> values;
        {
            std::lock_guard<std::mutex> lock(mutex);
            values = session->Run(
                Ort::RunOptions{nullptr}, run_inputs, inputs.data(), 2,
                run_outputs, 6);
        }
        if (values.size() != 6)
            throw std::runtime_error(
                "LightGlue end2end returned unexpected outputs");

        auto read_kpts = [&](Ort::Value& tensor, FeatureSet& set,
                             const PreparedImage& source) {
            const auto info = tensor.GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            if (shape.size() != 3 || shape[0] != 1 || shape[2] != 2)
                throw std::runtime_error(
                    "Unexpected LightGlue kpts output shape");
            const std::size_t count = static_cast<std::size_t>(shape[1]);
            set.image_width = source.width;
            set.image_height = source.height;
            set.extractor_name = "lightglue";
            set.metric = DescriptorMetric::inner_product;
            set.keypoints.clear();
            set.keypoints.reserve(count);
            const float sx =
                static_cast<float>(source.width) / static_cast<float>(input_w);
            const float sy = static_cast<float>(source.height) /
                static_cast<float>(input_h);
            if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                const std::int64_t* data = tensor.GetTensorData<std::int64_t>();
                for (std::size_t i = 0; i < count; ++i)
                    set.keypoints.push_back(
                        {static_cast<float>(data[i * 2]) * sx,
                         static_cast<float>(data[i * 2 + 1]) * sy});
            } else if (
                info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                const float* data = tensor.GetTensorData<float>();
                for (std::size_t i = 0; i < count; ++i)
                    set.keypoints.push_back(
                        {data[i * 2] * sx, data[i * 2 + 1] * sy});
            } else {
                throw std::runtime_error(
                    "Unexpected LightGlue kpts element type");
            }
            return count;
        };

        ImagePairFeatures result;
        const std::size_t count0 =
            read_kpts(values[0], result.first, first);
        const std::size_t count1 =
            read_kpts(values[1], result.second, second);

        const auto matches0_info = values[2].GetTensorTypeAndShapeInfo();
        const auto matches0_shape = matches0_info.GetShape();
        std::size_t match_len = 0;
        if (matches0_shape.size() == 1)
            match_len = static_cast<std::size_t>(matches0_shape[0]);
        else if (matches0_shape.size() == 2 && matches0_shape[0] == 1)
            match_len = static_cast<std::size_t>(matches0_shape[1]);
        else
            throw std::runtime_error(
                "Unexpected LightGlue matches0 output shape");
        if (match_len != count0)
            throw std::runtime_error(
                "LightGlue matches0 length does not match kpts0");

        const float* scores = values[4].GetTensorData<float>();
        result.matches.matches.reserve(match_len / 4U + 8U);

        auto push = [&](std::int64_t q, std::int64_t t, float score) {
            if (t < 0 || score < options.min_score) return;
            if (static_cast<std::size_t>(q) >= count0 ||
                static_cast<std::size_t>(t) >= count1)
                throw std::runtime_error(
                    "LightGlue end2end match index out of range");
            result.matches.matches.push_back(
                {static_cast<FeatureIndex>(q), static_cast<FeatureIndex>(t),
                 score});
        };

        if (matches0_info.GetElementType() ==
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            const std::int64_t* matches0 =
                values[2].GetTensorData<std::int64_t>();
            for (std::size_t i = 0; i < match_len; ++i)
                push(static_cast<std::int64_t>(i), matches0[i], scores[i]);
        } else if (
            matches0_info.GetElementType() ==
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* matches0 = values[2].GetTensorData<float>();
            for (std::size_t i = 0; i < match_len; ++i)
                push(
                    static_cast<std::int64_t>(i),
                    static_cast<std::int64_t>(matches0[i]), scores[i]);
        } else {
            throw std::runtime_error(
                "Unexpected LightGlue matches0 element type");
        }
        return result;
    }

    ImagePairFeatures match_prepared(
        const PreparedImage& first, const PreparedImage& second) const {
        if (layout == LightGlueModelLayout::dual_image_end2end)
            return match_dual_end2end(first, second);
        return match_batched(first, second);
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
    // Model path is required at construction; frontend builds
    // LightGluePipeline(options) directly. Register the recipe name so
    // list_pair_pipelines() / has_pair_pipeline() advertise it.
    register_pair_pipeline(
        std::string(kLightGlueEnd2EndPipeline),
        []() -> std::unique_ptr<PairFeaturePipeline> {
            throw std::runtime_error(
                "lightglue_end2end requires LightGlueOptions.model_path; "
                "construct features::LightGluePipeline(options) from the "
                "frontend (--pipeline lightglue_end2end)");
        });
}

}  // namespace aetherscan::features
