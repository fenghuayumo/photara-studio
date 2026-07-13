#include "features/features.hpp"
#include "features/onnx_session.hpp"
#include "features/registry.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aetherscan::features {

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
        OnnxSessionConfig cfg;
        cfg.model_path = options.model_path;
        cfg.use_cuda = options.device == InferenceDevice::cuda;
        cfg.allow_cpu_fallback = options.allow_cpu_fallback;
        cfg.env_name = "AetherScan.LightGlueMatcher";
        session = std::make_unique<OnnxSession>(std::move(cfg));
#else
        throw std::runtime_error(
            "LightGlue matcher requires ONNX Runtime "
            "(AETHERSCAN_ENABLE_ONNX)");
#endif
    }

    LightGlueMatcherOptions options;
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    std::unique_ptr<OnnxSession> session;

    static void normalize_keypoints(
        const FeatureSet& features, std::vector<float>& out_xy) {
        out_xy.resize(features.keypoints.size() * 2U);
        const float inv_w =
            features.image_width > 0
                ? 2.F / static_cast<float>(features.image_width)
                : 0.F;
        const float inv_h =
            features.image_height > 0
                ? 2.F / static_cast<float>(features.image_height)
                : 0.F;
        for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
            out_xy[i * 2] = features.keypoints[i].x * inv_w - 1.F;
            out_xy[i * 2 + 1] = features.keypoints[i].y * inv_h - 1.F;
        }
    }
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
            "LightGlue matcher requires matching descriptor dimensions");
    if (impl_->options.descriptor_dimension != 0 &&
        query.descriptor_dimension != impl_->options.descriptor_dimension)
        throw std::invalid_argument(
            "LightGlue matcher descriptor dimension mismatch with model");
    if (query.descriptors.size() !=
            query.keypoints.size() * query.descriptor_dimension ||
        train.descriptors.size() !=
            train.keypoints.size() * train.descriptor_dimension)
        throw std::invalid_argument(
            "LightGlue matcher requires float32 descriptor storage");

    std::vector<float> kpts0;
    std::vector<float> kpts1;
    Impl::normalize_keypoints(query, kpts0);
    Impl::normalize_keypoints(train, kpts1);

    // Non-const buffers required by CreateTensor.
    std::vector<float> desc0 = query.descriptors;
    std::vector<float> desc1 = train.descriptors;

    const std::array<std::int64_t, 3> k0_shape{
        1, static_cast<std::int64_t>(query.keypoints.size()), 2};
    const std::array<std::int64_t, 3> k1_shape{
        1, static_cast<std::int64_t>(train.keypoints.size()), 2};
    const std::array<std::int64_t, 3> d0_shape{
        1, static_cast<std::int64_t>(query.keypoints.size()),
        static_cast<std::int64_t>(query.descriptor_dimension)};
    const std::array<std::int64_t, 3> d1_shape{
        1, static_cast<std::int64_t>(train.keypoints.size()),
        static_cast<std::int64_t>(train.descriptor_dimension)};

    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<Ort::Value, 4> inputs{
        Ort::Value::CreateTensor<float>(
            memory, kpts0.data(), kpts0.size(), k0_shape.data(),
            k0_shape.size()),
        Ort::Value::CreateTensor<float>(
            memory, kpts1.data(), kpts1.size(), k1_shape.data(),
            k1_shape.size()),
        Ort::Value::CreateTensor<float>(
            memory, desc0.data(), desc0.size(), d0_shape.data(),
            d0_shape.size()),
        Ort::Value::CreateTensor<float>(
            memory, desc1.data(), desc1.size(), d1_shape.data(),
            d1_shape.size())};

    // Bind by name order expected by fabio-sim models.
    const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
    const char* output_names[] = {"matches0", "mscores0"};

    std::vector<Ort::Value> outputs;
    {
        std::lock_guard lock(impl_->session->mutex());
        outputs = impl_->session->session().Run(
            Ort::RunOptions{nullptr}, input_names, inputs.data(), 4,
            output_names, 2);
    }
    if (outputs.size() != 2)
        throw std::runtime_error("LightGlue matcher returned unexpected outputs");

    const auto match_shape =
        outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (match_shape.size() != 2 || match_shape[1] != 2)
        throw std::runtime_error("Unexpected LightGlue matches0 shape");
    const std::size_t match_count = static_cast<std::size_t>(match_shape[0]);
    const auto* matches = outputs[0].GetTensorData<std::int64_t>();
    const float* scores = outputs[1].GetTensorData<float>();

    MatchSet result;
    result.matches.reserve(match_count);
    for (std::size_t i = 0; i < match_count; ++i) {
        const float score = scores[i];
        if (score < impl_->options.min_score) continue;
        const auto q = matches[i * 2];
        const auto t = matches[i * 2 + 1];
        if (q < 0 || t < 0 ||
            static_cast<std::size_t>(q) >= query.keypoints.size() ||
            static_cast<std::size_t>(t) >= train.keypoints.size())
            continue;
        result.matches.push_back(
            {static_cast<FeatureIndex>(q), static_cast<FeatureIndex>(t),
             score});
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
