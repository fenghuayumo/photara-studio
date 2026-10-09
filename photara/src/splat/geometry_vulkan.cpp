#include "geometry_vulkan.hpp"
#include "rasterizer_vulkan.hpp"
#include "splat_geometry.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include "splat_drender/vulkan_api.h"
#include "vulkan/backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace photara::splat::detail {
namespace {
using tinytensor::Tensor;
constexpr auto device = tinytensor::Device::Vulkan;
struct Push {
    std::uint32_t mode{}, count{}, width{}, height{};
    float fx{}, fy{}, cx{}, cy{}, weight{}, focal{};
    std::uint32_t camera_count{}, euclidean{};
};
static_assert(sizeof(Push) == 48);

struct Backend {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::EncoderRing ring;
    std::array<std::unique_ptr<std::array<Tensor, 6>>, 4> retained;
    Backend() {
        const auto handles = tinytensor::vulkan::device_handles();
        const auto info = tinytensor::vulkan::device_info();
        photara::vk::ExternalDevice external;
        external.instance = handles.instance;
        external.physical = handles.physical_device;
        external.device = handles.device;
        external.queue = handles.queue;
        external.queue_family = handles.queue_family;
        external.enabled.push_descriptors = info.push_descriptors;
        external.enabled.buffer_atomic_f32 = info.buffer_atomic_f32;
        device = photara::vk::Device::adopt(external);
        pipeline = device.create_compute(std::as_bytes(std::span{splat_geometry_hlsl_spv}), 6, sizeof(Push));
        ring = photara::vk::EncoderRing(device, 4, photara::vk::BarrierPolicy::after_compute);
    }
    ~Backend() {
        // TinyTensor's buffer pool does not own these external submissions.
        // Keep their allocations alive until the corresponding fence retires.
        for (std::uint32_t i = 0; i < ring.size(); ++i) ring.at(i).wait();
    }
};
Backend& backend() { static thread_local Backend value; return value; }

void launch(Push push, const std::array<Tensor, 6>& tensors) {
    if (push.count == 0) return;
    tinytensor::vulkan::submit_async();
    auto& vk = backend();
    std::array<photara::vk::BufferBinding, 6> bindings;
    const auto fallback = std::find_if(tensors.begin(), tensors.end(),
        [](const auto& t) { return t.is_valid() && t.numel() > 0; });
    if (fallback == tensors.end()) throw std::invalid_argument("Empty Vulkan geometry dispatch");
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i].is_valid() && tensors[i].numel() > 0 ? tensors[i] : *fallback;
        const auto view = tinytensor::vulkan::buffer_view(t);
        bindings[i] = {view.buffer, view.offset, view.bytes};
    }
    const auto groups = (push.count + 255u) / 256u;
    auto& encoder = vk.ring.acquire();
    // Construct fresh Tensor handles: assigning into an existing Tensor view
    // would copy into that view instead of replacing its owner.
    vk.retained[vk.ring.index()] = std::make_unique<std::array<Tensor, 6>>(tensors);
    encoder.dispatch(vk.pipeline, bindings, &push, sizeof(push),
        std::min(groups, 65535u), (groups + 65534u) / 65535u);
    vk.ring.submit();
}
Push camera_push(std::uint32_t mode, const Camera& camera) {
    if (camera.model != CameraModel::pinhole || camera.width == 0 || camera.height == 0 ||
        !(camera.fx > 0.F) || !(camera.fy > 0.F))
        throw std::invalid_argument("Vulkan geometry losses require a valid pinhole camera");
    Push push;
    push.mode = mode;
    push.width = camera.width; push.height = camera.height;
    push.count = camera.width * camera.height;
    push.fx = camera.fx; push.fy = camera.fy; push.cx = camera.cx; push.cy = camera.cy;
    return push;
}
Tensor matrix(const Camera& camera) {
    return Tensor::from_vector(std::vector<float>(camera.world_to_camera.begin(), camera.world_to_camera.end()),
        {std::size_t{16}}, device);
}
void geometry_gradients(const RenderResult& rendered, LossGradients& gradients) {
    if (!gradients.depth.is_valid()) gradients.depth = Tensor::zeros_like(rendered.median_depth);
    if (!gradients.normal.is_valid()) gradients.normal = Tensor::zeros_like(rendered.normal);
}
splat_drender::vulkan::SplatBufferView buffer(const Tensor& tensor) {
    if (!tensor.is_valid()) return {};
    const auto view = tinytensor::vulkan::buffer_view(tensor);
    return {view.buffer, view.offset, view.bytes};
}
splat_drender::vulkan::SplatCamera camera_view(const Camera& camera) {
    splat_drender::vulkan::SplatCamera result;
    result.width = camera.width; result.height = camera.height;
    result.fx = camera.fx; result.fy = camera.fy; result.cx = camera.cx; result.cy = camera.cy;
    result.mode = static_cast<std::uint32_t>(camera.model);
    result.world_to_camera = camera.world_to_camera; result.center = camera.position;
    return result;
}
}  // namespace

void add_depth_normal_loss_vulkan(const RenderResult& rendered, const Camera& camera,
    float weight, LossGradients& gradients, bool collect_scalar_terms) {
    if (!rendered.median_depth.is_valid() || !rendered.normal.is_valid()) {
        if (weight > 0.F) throw std::invalid_argument("Vulkan depth-normal loss requires geometry attachments");
        return;
    }
    geometry_gradients(rendered, gradients);
    if (!(weight > 0.F)) return;
    auto push = camera_push(0, camera);
    push.weight = weight;
    auto terms = Tensor::zeros({1}, device);
    launch(push, {rendered.median_depth, rendered.normal, {}, gradients.depth, gradients.normal, terms});
    if (collect_scalar_terms) {
        const float value = terms.to_vector()[0];
        gradients.normal_value += value;
        gradients.total += value;
    }
}

Tensor compute_3d_filter_vulkan(const Tensor& means, const std::vector<Camera>& cameras,
    float minimum_scale_factor, bool all_camera_euclidean) {
    if (means.dtype() != tinytensor::DataType::Float32 || means.shape().rank() != 2 ||
        means.shape()[1] != 3 || cameras.empty())
        throw std::invalid_argument("Vulkan 3D filter requires float32 [N,3] means and cameras");
    auto result = Tensor::empty({means.shape()[0], std::size_t{1}}, device);
    if (means.shape()[0] == 0) return result;
    std::vector<float> packed(cameras.size() * 27);
    float focal = 0.F;
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        const auto& c = cameras[i]; const auto b = 27 * i;
        std::copy(c.world_to_camera.begin(), c.world_to_camera.end(), packed.begin() + b);
        packed[b+16] = c.fx; packed[b+17] = c.fy;
        packed[b+18] = static_cast<float>(c.width); packed[b+19] = static_cast<float>(c.height);
        packed[b+20] = static_cast<float>(c.model); packed[b+21] = c.cx; packed[b+22] = c.cy;
        packed[b+23] = c.k1; packed[b+24] = c.k2; packed[b+25] = c.k3; packed[b+26] = c.k4;
        const bool panorama = c.model == CameraModel::equirectangular;
        all_camera_euclidean = all_camera_euclidean || panorama;
        focal = std::max(focal, panorama ? c.width / (2.F * 3.14159265358979323846F) : c.fx);
    }
    auto camera_tensor = Tensor::from_vector(packed, {packed.size()}, device);
    auto maximum = Tensor::zeros({1}, device);
    Push push;
    push.mode = 1; push.count = static_cast<std::uint32_t>(means.shape()[0]);
    push.camera_count = static_cast<std::uint32_t>(cameras.size());
    push.euclidean = all_camera_euclidean; push.focal = std::max(focal, 1e-6F);
    push.weight = std::sqrt(std::max(minimum_scale_factor, 0.F));
    launch(push, {means, {}, camera_tensor, result, {}, maximum});
    push.mode = 2;
    launch(push, {means, {}, camera_tensor, result, {}, maximum});
    return result;
}

Tensor unproject_depth_to_world_vulkan(const Tensor& depth, const Camera& camera) {
    const auto push = camera_push(3, camera);
    if (depth.numel() != push.count) throw std::invalid_argument("Vulkan depth unprojection shape mismatch");
    auto points = Tensor::empty({std::size_t{push.count}, std::size_t{3}}, device);
    launch(push, {depth, {}, matrix(camera), points, {}, {}});
    return points;
}

void add_sample_depth_point_gradients_vulkan(const Camera& camera, const Tensor& points,
    LossGradients& gradients) {
    const auto push = camera_push(4, camera);
    if (points.numel() != 3ull * push.count || gradients.depth.numel() != push.count)
        throw std::invalid_argument("Vulkan depth pullback shape mismatch");
    launch(push, {points, {}, matrix(camera), gradients.depth, {}, {}});
}

MultiViewLoss add_multi_view_loss_vulkan(const Tensor& points, const Tensor& inside,
    const RenderResult& rendered, const TrainingView& reference, const TrainingView& neighbour,
    const TrainingOptions& options, LossGradients& gradients, Tensor& point_gradient,
    bool collect_scalar_terms, Tensor* stability) {
    geometry_gradients(rendered, gradients);
    auto depth = Tensor::empty(rendered.median_depth.shape(), device);
    auto normal = Tensor::empty(rendered.normal.shape(), device);
    point_gradient = Tensor::empty(points.shape(), device);
    auto terms = Tensor::empty({5}, device);
    splat_drender::vulkan::SplatDeviceMultiViewInput input;
    input.reference_camera = camera_view(reference.camera); input.neighbour_camera = camera_view(neighbour.camera);
    input.reference_depth = buffer(rendered.median_depth); input.reference_normal = buffer(rendered.normal);
    input.reference_gray = buffer(reference.gray); input.neighbour_gray = buffer(neighbour.gray);
    input.sampled_neighbour_points = buffer(points); input.sampled_inside = buffer(inside);
    // sample_depth's Vulkan mask is a uint32 array, unlike CUDA's byte bools.
    if (inside.dtype() != tinytensor::DataType::Int32)
        throw std::invalid_argument("Vulkan multi-view inside flags must be int32");
    input.reference_mask = reference.has_mask ? buffer(reference.mask) : splat_drender::vulkan::SplatBufferView{};
    input.neighbour_mask = neighbour.has_mask ? buffer(neighbour.mask) : splat_drender::vulkan::SplatBufferView{};
    input.geometry_weight = options.multi_view_geo_weight; input.ncc_weight = options.multi_view_ncc_weight;
    input.pixel_noise_threshold = options.multi_view_pixel_noise_threshold;
    input.robust_ncc = options.multi_view_robust_ncc;
    input.ncc_lambda_reference = options.multi_view_ncc_lambda_reference;
    input.ncc_sharpness = options.multi_view_ncc_sharpness; input.ncc_min_weight = options.multi_view_ncc_min_weight;
    tinytensor::vulkan::submit_async();
    vulkan_multi_view_loss(rendered, input, {buffer(depth), buffer(normal), buffer(point_gradient), buffer(terms)});
    gradients.depth.copy_from(gradients.depth.add(depth));
    gradients.normal.copy_from(gradients.normal.add(normal));
    if (stability) {
        if (stability->device() != device || stability->numel() != 3 ||
            stability->dtype() != tinytensor::DataType::Float32)
            throw std::invalid_argument("Vulkan stability accumulator must be float32[3]");
        Push push; push.mode = 5; push.count = 1;
        launch(push, {terms, {}, {}, *stability, {}, {}});
    }
    MultiViewLoss result;
    if (collect_scalar_terms) {
        const auto values = terms.to_vector();
        result.geometry_pixels = static_cast<std::size_t>(values[1]);
        result.geometry_candidates = static_cast<std::size_t>(values[4]);
        result.ncc_pixels = static_cast<std::size_t>(values[3]);
        result.geometry = values[1] > 0.F ? values[0] / values[1] : 0.F;
        result.ncc = values[3] > 0.F ? values[2] / values[3] : 0.F;
    }
    return result;
}

GeometryDistributionSummary summarize_geometry_distribution_vulkan(const GaussianModel& model) {
    GeometryDistributionSummary result;
    if (model.size() == 0) return result;
    auto terms = Tensor::zeros({6}, device);
    Push push; push.mode = 6; push.count = static_cast<std::uint32_t>(model.size());
    launch(push, {model.log_scales, model.opacity_logits, {}, {}, {}, terms});
    const auto values = terms.to_vector();
    const auto moment = [&](unsigned i) {
        const float mean = values[i] / model.size();
        return std::array{mean, std::sqrt(std::max(values[i+1] / model.size() - mean*mean, 0.F))};
    };
    const auto o = moment(0), s = moment(2), a = moment(4);
    result.opacity_mean = o[0]; result.opacity_stddev = o[1];
    result.log_scale_mean = s[0]; result.log_scale_stddev = s[1];
    result.log_anisotropy_mean = a[0]; result.log_anisotropy_stddev = a[1];
    return result;
}
}  // namespace photara::splat::detail
