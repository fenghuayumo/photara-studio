#include "features/features.hpp"
#include "features/onnx_session.hpp"
#include "features/registry.hpp"

#include "io/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aetherscan::features {
namespace {

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
void select_top_keypoints(
    std::vector<float>& scores,
    std::vector<float>& kpts_xy,  // interleaved x,y in network pixels
    std::vector<float>& descriptors,
    const std::size_t descriptor_dim,
    const std::size_t maximum_features,
    const float keypoint_threshold) {
    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(), order.end(),
        [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });

    std::size_t keep = order.size();
    if (maximum_features > 0) keep = std::min(keep, maximum_features);
    if (keypoint_threshold > 0.F) {
        std::size_t above = 0;
        for (; above < keep; ++above) {
            if (scores[order[above]] < keypoint_threshold) break;
        }
        keep = above;
    }

    std::vector<float> new_scores(keep);
    std::vector<float> new_kpts(keep * 2U);
    std::vector<float> new_desc(keep * descriptor_dim);
    for (std::size_t i = 0; i < keep; ++i) {
        const std::size_t src = order[i];
        new_scores[i] = scores[src];
        new_kpts[i * 2] = kpts_xy[src * 2];
        new_kpts[i * 2 + 1] = kpts_xy[src * 2 + 1];
        std::copy_n(
            descriptors.data() + src * descriptor_dim, descriptor_dim,
            new_desc.data() + i * descriptor_dim);
    }
    scores = std::move(new_scores);
    kpts_xy = std::move(new_kpts);
    descriptors = std::move(new_desc);
}

FeatureSet build_feature_set(
    std::vector<float> kpts_xy,
    std::vector<float> scores,
    std::vector<float> descriptors,
    const std::size_t descriptor_dim,
    const std::uint32_t orig_w,
    const std::uint32_t orig_h,
    const std::uint32_t net_w,
    const std::uint32_t net_h,
    const std::string& extractor_name) {
    FeatureSet set;
    set.image_width = orig_w;
    set.image_height = orig_h;
    set.descriptor_dimension = descriptor_dim;
    set.metric = DescriptorMetric::inner_product;
    set.extractor_name = extractor_name;
    set.storage = DescriptorStorage::float32;
    const float sx = static_cast<float>(orig_w) / static_cast<float>(net_w);
    const float sy = static_cast<float>(orig_h) / static_cast<float>(net_h);
    const std::size_t count = scores.size();
    set.keypoints.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        set.keypoints[i].x = kpts_xy[i * 2] * sx;
        set.keypoints[i].y = kpts_xy[i * 2 + 1] * sy;
        set.keypoints[i].response = scores[i];
        set.keypoints[i].scale = 1.F;
    }
    set.descriptors = std::move(descriptors);
    return set;
}
#endif

}  // namespace

class SuperPointExtractor::Impl {
public:
    explicit Impl(SuperPointOptions value) : options(std::move(value)) {
        if (options.model_path.empty() || options.input_width == 0 ||
            options.input_height == 0)
            throw std::invalid_argument(
                "SuperPoint model path and input size are required");
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
        OnnxSessionConfig cfg;
        cfg.model_path = options.model_path;
        cfg.use_cuda = options.cuda;
        cfg.allow_cpu_fallback = options.allow_cpu_fallback;
        cfg.env_name = "AetherScan.SuperPoint";
        session = std::make_unique<OnnxSession>(std::move(cfg));
#else
        throw std::runtime_error(
            "SuperPoint requires ONNX Runtime (AETHERSCAN_ENABLE_ONNX)");
#endif
    }

    SuperPointOptions options;
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    std::unique_ptr<OnnxSession> session;
#endif
};

SuperPointExtractor::SuperPointExtractor(SuperPointOptions options)
    : impl_(std::make_shared<Impl>(std::move(options))) {}
SuperPointExtractor::SuperPointExtractor(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SuperPointExtractor::~SuperPointExtractor() = default;
SuperPointExtractor::SuperPointExtractor(SuperPointExtractor&&) noexcept =
    default;
SuperPointExtractor& SuperPointExtractor::operator=(
    SuperPointExtractor&&) noexcept = default;

bool SuperPointExtractor::is_built() noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return true;
#else
    return false;
#endif
}

bool SuperPointExtractor::is_available() const noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return impl_ != nullptr && impl_->session != nullptr;
#else
    return false;
#endif
}

ExtractorInfo SuperPointExtractor::info() const {
    ExtractorInfo value;
    value.thread_safe = false;
    value.thread_affine = true;
    value.accepts_gray = true;
    value.accepts_rgb = false;
    value.metric = DescriptorMetric::inner_product;
    value.typical_descriptor_dimension = 256;
    return value;
}

std::unique_ptr<FeatureExtractor> SuperPointExtractor::clone() const {
    return std::unique_ptr<FeatureExtractor>(new SuperPointExtractor(impl_));
}

FeatureSet SuperPointExtractor::extract_gray(
    std::span<const std::uint8_t> pixels, std::uint32_t width,
    std::uint32_t height, std::size_t row_stride) const {
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)pixels;
    (void)width;
    (void)height;
    (void)row_stride;
    throw std::runtime_error("SuperPoint requires ONNX Runtime");
#else
    if (width == 0 || height == 0 || pixels.empty())
        throw std::invalid_argument("SuperPoint requires a non-empty image");
    if (row_stride == 0) row_stride = width;

    const std::uint32_t net_w = impl_->options.input_width;
    const std::uint32_t net_h = impl_->options.input_height;
    std::vector<std::uint8_t> rgb(
        static_cast<std::size_t>(width) * height * 3U);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint8_t g =
                pixels[static_cast<std::size_t>(y) * row_stride + x];
            const std::size_t o =
                (static_cast<std::size_t>(y) * width + x) * 3U;
            rgb[o] = g;
            rgb[o + 1] = g;
            rgb[o + 2] = g;
        }
    }
    std::vector<std::uint8_t> resized(
        static_cast<std::size_t>(net_w) * net_h * 3U);
    io::resize_bilinear(
        rgb.data(), width, height, 3, resized.data(), net_w, net_h);

    std::vector<float> input(
        static_cast<std::size_t>(net_w) * net_h);
    for (std::uint32_t y = 0; y < net_h; ++y) {
        for (std::uint32_t x = 0; x < net_w; ++x) {
            const std::uint8_t* px =
                resized.data() +
                (static_cast<std::size_t>(y) * net_w + x) * 3U;
            input[static_cast<std::size_t>(y) * net_w + x] =
                (0.299F * px[0] + 0.587F * px[1] + 0.114F * px[2]) / 255.F;
        }
    }

    const std::array<std::int64_t, 4> shape{
        1, 1, static_cast<std::int64_t>(net_h),
        static_cast<std::int64_t>(net_w)};
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory, input.data(), input.size(), shape.data(), shape.size());

    std::vector<Ort::Value> outputs;
    {
        std::lock_guard lock(impl_->session->mutex());
        outputs = impl_->session->session().Run(
            Ort::RunOptions{nullptr}, impl_->session->input_names().data(),
            &tensor, 1, impl_->session->output_names().data(),
            impl_->session->output_names().size());
    }
    if (outputs.size() < 3)
        throw std::runtime_error("SuperPoint returned unexpected outputs");

    const auto k_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (k_shape.size() != 3 || k_shape[0] != 1 || k_shape[2] != 2)
        throw std::runtime_error("Unexpected SuperPoint keypoint shape");
    const std::size_t count = static_cast<std::size_t>(k_shape[1]);
    const auto* k_i64 = outputs[0].GetTensorData<std::int64_t>();
    const float* scores_ptr = outputs[1].GetTensorData<float>();
    const float* desc_ptr = outputs[2].GetTensorData<float>();
    constexpr std::size_t kDim = 256;

    std::vector<float> kpts(count * 2U);
    std::vector<float> scores(count);
    std::vector<float> descriptors(count * kDim);
    for (std::size_t i = 0; i < count; ++i) {
        kpts[i * 2] = static_cast<float>(k_i64[i * 2]);
        kpts[i * 2 + 1] = static_cast<float>(k_i64[i * 2 + 1]);
        scores[i] = scores_ptr[i];
        std::copy_n(desc_ptr + i * kDim, kDim, descriptors.data() + i * kDim);
    }
    select_top_keypoints(
        scores, kpts, descriptors, kDim, impl_->options.maximum_features,
        impl_->options.keypoint_threshold);
    return build_feature_set(
        std::move(kpts), std::move(scores), std::move(descriptors), kDim,
        width, height, net_w, net_h, "superpoint");
#endif
}

void register_superpoint_feature_backends() {
    register_extractor("superpoint", [] {
        throw std::runtime_error(
            "superpoint requires SuperPointOptions.model_path; construct "
            "features::SuperPointExtractor(options) from the frontend");
        return std::unique_ptr<FeatureExtractor>{};
    });
}

// ---- DISK -----------------------------------------------------------------------

class DiskExtractor::Impl {
public:
    explicit Impl(DiskOptions value) : options(std::move(value)) {
        if (options.model_path.empty() || options.input_width == 0 ||
            options.input_height == 0)
            throw std::invalid_argument(
                "DISK model path and input size are required");
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
        OnnxSessionConfig cfg;
        cfg.model_path = options.model_path;
        cfg.use_cuda = options.cuda;
        cfg.allow_cpu_fallback = options.allow_cpu_fallback;
        cfg.env_name = "AetherScan.DISK";
        session = std::make_unique<OnnxSession>(std::move(cfg));
#else
        throw std::runtime_error(
            "DISK requires ONNX Runtime (AETHERSCAN_ENABLE_ONNX)");
#endif
    }

    DiskOptions options;
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    std::unique_ptr<OnnxSession> session;
#endif
};

DiskExtractor::DiskExtractor(DiskOptions options)
    : impl_(std::make_shared<Impl>(std::move(options))) {}
DiskExtractor::DiskExtractor(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DiskExtractor::~DiskExtractor() = default;
DiskExtractor::DiskExtractor(DiskExtractor&&) noexcept = default;
DiskExtractor& DiskExtractor::operator=(DiskExtractor&&) noexcept = default;

bool DiskExtractor::is_built() noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return true;
#else
    return false;
#endif
}

bool DiskExtractor::is_available() const noexcept {
#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
    return impl_ != nullptr && impl_->session != nullptr;
#else
    return false;
#endif
}

ExtractorInfo DiskExtractor::info() const {
    ExtractorInfo value;
    value.thread_safe = false;
    value.thread_affine = true;
    value.accepts_gray = false;
    value.accepts_rgb = true;
    value.metric = DescriptorMetric::inner_product;
    value.typical_descriptor_dimension = 128;
    return value;
}

std::unique_ptr<FeatureExtractor> DiskExtractor::clone() const {
    return std::unique_ptr<FeatureExtractor>(new DiskExtractor(impl_));
}

FeatureSet DiskExtractor::extract_gray(
    std::span<const std::uint8_t> /*pixels*/, std::uint32_t /*width*/,
    std::uint32_t /*height*/, std::size_t /*row_stride*/) const {
    throw std::runtime_error("DISK extractor requires RGB input");
}

FeatureSet DiskExtractor::extract_rgb(
    std::span<const std::uint8_t> pixels, std::uint32_t width,
    std::uint32_t height, std::size_t row_stride) const {
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)pixels;
    (void)width;
    (void)height;
    (void)row_stride;
    throw std::runtime_error("DISK requires ONNX Runtime");
#else
    if (width == 0 || height == 0 || pixels.empty())
        throw std::invalid_argument("DISK requires a non-empty image");
    if (row_stride == 0) row_stride = static_cast<std::size_t>(width) * 3U;

    const std::uint32_t net_w = impl_->options.input_width;
    const std::uint32_t net_h = impl_->options.input_height;
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>(width) * height * 3U);
    for (std::uint32_t y = 0; y < height; ++y) {
        std::copy_n(
            pixels.data() + static_cast<std::size_t>(y) * row_stride,
            static_cast<std::size_t>(width) * 3U,
            packed.data() + static_cast<std::size_t>(y) * width * 3U);
    }
    std::vector<std::uint8_t> resized(
        static_cast<std::size_t>(net_w) * net_h * 3U);
    io::resize_bilinear(
        packed.data(), width, height, 3, resized.data(), net_w, net_h);

    std::vector<float> input(
        static_cast<std::size_t>(3U) * net_h * net_w);
    for (std::uint32_t c = 0; c < 3; ++c) {
        for (std::uint32_t y = 0; y < net_h; ++y) {
            for (std::uint32_t x = 0; x < net_w; ++x) {
                const std::uint8_t* rgb =
                    resized.data() +
                    (static_cast<std::size_t>(y) * net_w + x) * 3U;
                input[(static_cast<std::size_t>(c) * net_h + y) * net_w + x] =
                    rgb[c] / 255.F;
            }
        }
    }

    const std::array<std::int64_t, 4> shape{
        1, 3, static_cast<std::int64_t>(net_h),
        static_cast<std::int64_t>(net_w)};
    Ort::MemoryInfo memory =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory, input.data(), input.size(), shape.data(), shape.size());

    std::vector<Ort::Value> outputs;
    {
        std::lock_guard lock(impl_->session->mutex());
        outputs = impl_->session->session().Run(
            Ort::RunOptions{nullptr}, impl_->session->input_names().data(),
            &tensor, 1, impl_->session->output_names().data(),
            impl_->session->output_names().size());
    }
    if (outputs.size() < 3)
        throw std::runtime_error("DISK returned unexpected outputs");

    const auto k_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (k_shape.size() != 3 || k_shape[0] != 1 || k_shape[2] != 2)
        throw std::runtime_error("Unexpected DISK keypoint shape");
    const std::size_t count = static_cast<std::size_t>(k_shape[1]);
    const auto* k_i64 = outputs[0].GetTensorData<std::int64_t>();
    const float* scores_ptr = outputs[1].GetTensorData<float>();
    const float* desc_ptr = outputs[2].GetTensorData<float>();
    constexpr std::size_t kDim = 128;

    std::vector<float> kpts(count * 2U);
    std::vector<float> scores(count);
    std::vector<float> descriptors(count * kDim);
    for (std::size_t i = 0; i < count; ++i) {
        kpts[i * 2] = static_cast<float>(k_i64[i * 2]);
        kpts[i * 2 + 1] = static_cast<float>(k_i64[i * 2 + 1]);
        scores[i] = scores_ptr[i];
        std::copy_n(desc_ptr + i * kDim, kDim, descriptors.data() + i * kDim);
    }
    select_top_keypoints(
        scores, kpts, descriptors, kDim, impl_->options.maximum_features,
        impl_->options.keypoint_threshold);
    return build_feature_set(
        std::move(kpts), std::move(scores), std::move(descriptors), kDim,
        width, height, net_w, net_h, "disk");
#endif
}

void register_disk_feature_backends() {
    register_extractor("disk", [] {
        throw std::runtime_error(
            "disk requires DiskOptions.model_path; construct "
            "features::DiskExtractor(options) from the frontend");
        return std::unique_ptr<FeatureExtractor>{};
    });
}

}  // namespace aetherscan::features
