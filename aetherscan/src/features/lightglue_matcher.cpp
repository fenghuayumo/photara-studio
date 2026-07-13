#include "features/features.hpp"
#include "features/onnx_session.hpp"
#include "features/registry.hpp"

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
namespace {

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
enum class LightGlueInputLayout { normalized_four, pixel_six, sift_ten };

bool contains_name(
    const std::vector<std::string>& names, const std::string& wanted) {
    return std::find(names.begin(), names.end(), wanted) != names.end();
}

std::size_t named_index(
    const std::vector<std::string>& names, const std::string& wanted) {
    const auto found = std::find(names.begin(), names.end(), wanted);
    if (found == names.end())
        throw std::runtime_error("LightGlue model is missing node: " + wanted);
    return static_cast<std::size_t>(found - names.begin());
}

struct PreparedFeatures {
    std::vector<float> keypoints;
    std::vector<float> descriptors;
    std::vector<float> scales;
    std::vector<float> orientations;
    std::array<float, 2> image_size{};
};

struct PreparedFeatureSlot {
    std::uint64_t descriptor_identity{};
    std::uint64_t descriptor_generation{};
    std::uint32_t image_width{};
    std::uint32_t image_height{};
    std::size_t keypoint_count{};
    std::size_t descriptor_dimension{};
    DescriptorMetric metric{DescriptorMetric::l2};
    PreparedFeatures prepared;

    [[nodiscard]] bool matches(const FeatureSet& features) const noexcept {
        return descriptor_identity != 0 &&
               descriptor_identity == features.descriptor_identity &&
               descriptor_generation == features.descriptor_generation &&
               image_width == features.image_width &&
               image_height == features.image_height &&
               keypoint_count == features.keypoints.size() &&
               descriptor_dimension == features.descriptor_dimension &&
               metric == features.metric;
    }
};

void root_normalize_rows(
    std::vector<float>& descriptors, const std::size_t count,
    const std::size_t dimension) {
    for (std::size_t row = 0; row < count; ++row) {
        float* values = descriptors.data() + row * dimension;
        double l1 = 0.0;
        for (std::size_t column = 0; column < dimension; ++column)
            l1 += (std::max)(0.F, values[column]);
        const float inverse =
            static_cast<float>(1.0 / (std::max)(l1, 1e-12));
        for (std::size_t column = 0; column < dimension; ++column)
            values[column] =
                std::sqrt((std::max)(0.F, values[column]) * inverse);
    }
}

PreparedFeatures prepare_features(
    const FeatureSet& features, const LightGlueInputLayout layout,
    const std::size_t maximum_features) {
    const std::size_t count = maximum_features == 0
        ? features.keypoints.size()
        : std::min(features.keypoints.size(), maximum_features);
    const std::size_t dimension = features.descriptor_dimension;
    if (features.storage != DescriptorStorage::float32 ||
        features.descriptors.size() !=
            features.keypoints.size() * dimension)
        throw std::invalid_argument(
            "LightGlue requires float32 descriptor storage");

    PreparedFeatures prepared;
    prepared.keypoints.resize(count * 2U);
    if (layout == LightGlueInputLayout::normalized_four) {
        const float inv_w = features.image_width > 0
            ? 2.F / static_cast<float>(features.image_width)
            : 0.F;
        const float inv_h = features.image_height > 0
            ? 2.F / static_cast<float>(features.image_height)
            : 0.F;
        for (std::size_t index = 0; index < count; ++index) {
            prepared.keypoints[index * 2] =
                features.keypoints[index].x * inv_w - 1.F;
            prepared.keypoints[index * 2 + 1] =
                features.keypoints[index].y * inv_h - 1.F;
        }
    } else {
        for (std::size_t index = 0; index < count; ++index) {
            prepared.keypoints[index * 2] = features.keypoints[index].x;
            prepared.keypoints[index * 2 + 1] = features.keypoints[index].y;
        }
    }

    prepared.descriptors.assign(
        features.descriptors.begin(),
        features.descriptors.begin() + count * dimension);
    if (layout == LightGlueInputLayout::sift_ten &&
        features.metric != DescriptorMetric::l2_root)
        root_normalize_rows(prepared.descriptors, count, dimension);

    prepared.image_size = {
        static_cast<float>(features.image_width),
        static_cast<float>(features.image_height)};
    if (layout == LightGlueInputLayout::sift_ten) {
        prepared.scales.resize(count);
        prepared.orientations.resize(count);
        for (std::size_t index = 0; index < count; ++index) {
            prepared.scales[index] = features.keypoints[index].scale;
            prepared.orientations[index] =
                features.keypoints[index].orientation;
        }
    }
    return prepared;
}

void prepare_feature_slot(
    PreparedFeatureSlot& slot, const FeatureSet& features,
    const LightGlueInputLayout layout, const std::size_t maximum_features) {
    PreparedFeatures prepared =
        prepare_features(features, layout, maximum_features);
    slot.prepared = std::move(prepared);
    slot.descriptor_identity = features.descriptor_identity;
    slot.descriptor_generation = features.descriptor_generation;
    slot.image_width = features.image_width;
    slot.image_height = features.image_height;
    slot.keypoint_count = features.keypoints.size();
    slot.descriptor_dimension = features.descriptor_dimension;
    slot.metric = features.metric;
}
#endif

}  // namespace

class LightGlueMatcher::Impl {
public:
    explicit Impl(LightGlueMatcherOptions value) : options(std::move(value)) {
        if (options.model_path.empty())
            throw std::invalid_argument(
                "LightGlue matcher model path is required");
        if (options.min_score < 0.F || options.min_score > 1.F)
            throw std::invalid_argument(
                "LightGlue matcher min_score must be in [0, 1]");
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
        OnnxSessionConfig config;
        config.model_path = options.model_path;
        config.use_cuda = options.device == InferenceDevice::cuda;
        config.allow_cpu_fallback = options.allow_cpu_fallback;
        config.env_name = "AetherScan.LightGlueMatcher";
        session = std::make_unique<OnnxSession>(std::move(config));

        const auto& inputs = session->input_name_strings();
        for (const char* required : {"kpts0", "kpts1", "desc0", "desc1"})
            if (!contains_name(inputs, required))
                throw std::runtime_error(
                    std::string("LightGlue model is missing input: ") + required);
        const bool has_image_sizes =
            contains_name(inputs, "image_size0") &&
            contains_name(inputs, "image_size1");
        const bool has_scale_orientation =
            contains_name(inputs, "scales0") &&
            contains_name(inputs, "scales1") &&
            contains_name(inputs, "oris0") && contains_name(inputs, "oris1");
        if (inputs.size() == 4 && !has_image_sizes && !has_scale_orientation) {
            layout = LightGlueInputLayout::normalized_four;
        } else if (inputs.size() == 6 && has_image_sizes &&
                   !has_scale_orientation) {
            layout = LightGlueInputLayout::pixel_six;
        } else if (inputs.size() == 10 && has_image_sizes &&
                   has_scale_orientation) {
            layout = LightGlueInputLayout::sift_ten;
        } else {
            throw std::runtime_error(
                "Unsupported LightGlue inputs; expected 4-input SuperPoint/DISK, "
                "6-input ALIKED, or 10-input SIFT layout");
        }

        const auto& outputs = session->output_name_strings();
        matches_output = named_index(outputs, "matches0");
        scores_output = named_index(outputs, "mscores0");
        if (outputs.size() != 2)
            throw std::runtime_error(
                "LightGlue model must have matches0 and mscores0 outputs");

        const std::size_t desc_index = named_index(inputs, "desc0");
        const auto desc_shape = session->session()
                                    .GetInputTypeInfo(desc_index)
                                    .GetTensorTypeAndShapeInfo()
                                    .GetShape();
        if (desc_shape.size() == 3 && desc_shape[2] > 0)
            model_descriptor_dimension =
                static_cast<std::size_t>(desc_shape[2]);
#else
        throw std::runtime_error(
            "LightGlue matcher requires ONNX Runtime "
            "(AETHERSCAN_ENABLE_ONNX)");
#endif
    }

    LightGlueMatcherOptions options;
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    std::unique_ptr<OnnxSession> session;
    LightGlueInputLayout layout{LightGlueInputLayout::normalized_four};
    std::size_t matches_output{};
    std::size_t scores_output{};
    std::size_t model_descriptor_dimension{};
    PreparedFeatureSlot first_cache;
    PreparedFeatureSlot second_cache;
#endif
};

LightGlueMatcher::LightGlueMatcher(LightGlueMatcherOptions options)
    : impl_(std::make_shared<Impl>(std::move(options))) {}
LightGlueMatcher::LightGlueMatcher(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
LightGlueMatcher::~LightGlueMatcher() = default;
LightGlueMatcher::LightGlueMatcher(LightGlueMatcher&&) noexcept = default;
LightGlueMatcher& LightGlueMatcher::operator=(LightGlueMatcher&&) noexcept =
    default;

bool LightGlueMatcher::is_built() noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return true;
#else
    return false;
#endif
}

bool LightGlueMatcher::is_available() const noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return impl_ != nullptr && impl_->session != nullptr;
#else
    return false;
#endif
}

const LightGlueMatcherOptions& LightGlueMatcher::options() const {
    return impl_->options;
}

std::unique_ptr<FeatureMatcher> LightGlueMatcher::clone() const {
    return std::unique_ptr<FeatureMatcher>(new LightGlueMatcher(impl_));
}

MatchSet LightGlueMatcher::match(
    const FeatureSet& query, const FeatureSet& train) const {
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)query;
    (void)train;
    throw std::runtime_error("LightGlue matcher requires ONNX Runtime");
#else
    if (query.keypoints.empty() || train.keypoints.empty()) return {};
    if (query.descriptor_dimension == 0 ||
        query.descriptor_dimension != train.descriptor_dimension)
        throw std::invalid_argument(
            "LightGlue requires matching descriptor dimensions");
    const std::size_t expected_dimension = impl_->options.descriptor_dimension != 0
        ? impl_->options.descriptor_dimension
        : impl_->model_descriptor_dimension;
    if (expected_dimension != 0 &&
        query.descriptor_dimension != expected_dimension)
        throw std::invalid_argument(
            "LightGlue descriptor dimension does not match the model");

    // The cached vectors back the Ort input tensors, so hold the session lock
    // from cache preparation through Run(). Pair ordering groups the same
    // query image together, making the first slot hot for many consecutive
    // matches while the second slot rotates through train images.
    std::unique_lock session_lock(impl_->session->mutex());
    if (!impl_->first_cache.matches(query)) {
        if (impl_->second_cache.matches(query))
            std::swap(impl_->first_cache, impl_->second_cache);
        else
            prepare_feature_slot(
                impl_->first_cache, query, impl_->layout,
                impl_->options.maximum_features);
    }
    PreparedFeatures& first = impl_->first_cache.prepared;

    PreparedFeatures* second_ptr = nullptr;
    if (impl_->first_cache.matches(train)) {
        second_ptr = &impl_->first_cache.prepared;
    } else {
        if (!impl_->second_cache.matches(train))
            prepare_feature_slot(
                impl_->second_cache, train, impl_->layout,
                impl_->options.maximum_features);
        second_ptr = &impl_->second_cache.prepared;
    }
    PreparedFeatures& second = *second_ptr;
    const std::size_t first_count = first.keypoints.size() / 2U;
    const std::size_t second_count = second.keypoints.size() / 2U;
    const std::array<std::int64_t, 3> k0_shape{
        1, static_cast<std::int64_t>(first_count), 2};
    const std::array<std::int64_t, 3> k1_shape{
        1, static_cast<std::int64_t>(second_count), 2};
    const std::array<std::int64_t, 3> d0_shape{
        1, static_cast<std::int64_t>(first_count),
        static_cast<std::int64_t>(query.descriptor_dimension)};
    const std::array<std::int64_t, 3> d1_shape{
        1, static_cast<std::int64_t>(second_count),
        static_cast<std::int64_t>(train.descriptor_dimension)};
    const std::array<std::int64_t, 2> image_size_shape{1, 2};
    const std::array<std::int64_t, 2> s0_shape{
        1, static_cast<std::int64_t>(first_count)};
    const std::array<std::int64_t, 2> s1_shape{
        1, static_cast<std::int64_t>(second_count)};

    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputs;
    inputs.reserve(impl_->session->input_names().size());
    for (const std::string& name : impl_->session->input_name_strings()) {
        if (name == "kpts0")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, first.keypoints.data(), first.keypoints.size(),
                k0_shape.data(), k0_shape.size()));
        else if (name == "kpts1")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, second.keypoints.data(), second.keypoints.size(),
                k1_shape.data(), k1_shape.size()));
        else if (name == "desc0")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, first.descriptors.data(), first.descriptors.size(),
                d0_shape.data(), d0_shape.size()));
        else if (name == "desc1")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, second.descriptors.data(), second.descriptors.size(),
                d1_shape.data(), d1_shape.size()));
        else if (name == "image_size0")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, first.image_size.data(), first.image_size.size(),
                image_size_shape.data(), image_size_shape.size()));
        else if (name == "image_size1")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, second.image_size.data(), second.image_size.size(),
                image_size_shape.data(), image_size_shape.size()));
        else if (name == "scales0")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, first.scales.data(), first.scales.size(),
                s0_shape.data(), s0_shape.size()));
        else if (name == "scales1")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, second.scales.data(), second.scales.size(),
                s1_shape.data(), s1_shape.size()));
        else if (name == "oris0")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, first.orientations.data(), first.orientations.size(),
                s0_shape.data(), s0_shape.size()));
        else if (name == "oris1")
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                memory, second.orientations.data(), second.orientations.size(),
                s1_shape.data(), s1_shape.size()));
        else
            throw std::runtime_error("Unsupported LightGlue input: " + name);
    }

    std::vector<Ort::Value> outputs = impl_->session->session().Run(
        Ort::RunOptions{nullptr}, impl_->session->input_names().data(),
        inputs.data(), inputs.size(), impl_->session->output_names().data(),
        impl_->session->output_names().size());
    session_lock.unlock();
    if (outputs.size() != 2)
        throw std::runtime_error("LightGlue returned unexpected outputs");

    const Ort::Value& match_value = outputs[impl_->matches_output];
    const Ort::Value& score_value = outputs[impl_->scores_output];
    const auto match_info = match_value.GetTensorTypeAndShapeInfo();
    const auto match_shape = match_info.GetShape();
    const std::size_t score_count =
        score_value.GetTensorTypeAndShapeInfo().GetElementCount();
    const float* scores = score_value.GetTensorData<float>();

    MatchSet result;
    if (match_shape.size() == 2 && match_shape[1] == 2) {
        const std::size_t count = static_cast<std::size_t>(match_shape[0]);
        if (score_count != count ||
            match_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
            throw std::runtime_error("Unexpected LightGlue pair-list outputs");
        const auto* matches = match_value.GetTensorData<std::int64_t>();
        result.matches.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto q = matches[index * 2];
            const auto t = matches[index * 2 + 1];
            if (scores[index] < impl_->options.min_score || q < 0 || t < 0 ||
                static_cast<std::size_t>(q) >= query.keypoints.size() ||
                static_cast<std::size_t>(t) >= train.keypoints.size())
                continue;
            result.matches.push_back(
                {static_cast<FeatureIndex>(q), static_cast<FeatureIndex>(t),
                 scores[index]});
        }
        return result;
    }

    const std::size_t count = match_info.GetElementCount();
    if ((match_shape.size() != 1 &&
         !(match_shape.size() == 2 && match_shape[0] == 1)) ||
        count != first_count || score_count != count)
        throw std::runtime_error("Unexpected LightGlue match-vector outputs");
    result.matches.reserve(count);
    const auto append = [&](const std::size_t query_index,
                            const std::int64_t train_index) {
        if (train_index < 0 ||
            static_cast<std::size_t>(train_index) >= train.keypoints.size() ||
            scores[query_index] < impl_->options.min_score)
            return;
        result.matches.push_back(
            {static_cast<FeatureIndex>(query_index),
             static_cast<FeatureIndex>(train_index), scores[query_index]});
    };
    if (match_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const auto* matches = match_value.GetTensorData<std::int64_t>();
        for (std::size_t index = 0; index < count; ++index)
            append(index, matches[index]);
    } else if (
        match_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        const auto* matches = match_value.GetTensorData<float>();
        for (std::size_t index = 0; index < count; ++index)
            append(index, static_cast<std::int64_t>(matches[index]));
    } else {
        throw std::runtime_error("Unsupported LightGlue matches0 data type");
    }
    return result;
#endif
}

void register_lightglue_matcher_backend() {
    register_matcher("lightglue", [] {
        throw std::runtime_error(
            "lightglue matcher requires LightGlueMatcherOptions.model_path; "
            "construct features::LightGlueMatcher(options) from the frontend");
        return std::unique_ptr<FeatureMatcher>{};
    });
}

}  // namespace aetherscan::features
