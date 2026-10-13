#include "pam_vulkan.hpp"
#include "pam_field.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include "vulkan/backend.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
namespace photara::splat::detail {
namespace {
using tinytensor::Tensor;
constexpr auto device = tinytensor::Device::Vulkan;
struct Push {
    int root;
    std::uint32_t count, neighbors, refine;
    float iso, min_norm, step;
    std::uint32_t padding{};
};
static_assert(sizeof(Push) == 32);
struct Backend {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::EncoderRing ring;
    Backend() {
        auto handles = tinytensor::vulkan::device_handles();
        photara::vk::ExternalDevice e;
        e.instance = handles.instance;
        e.physical = handles.physical_device;
        e.device = handles.device;
        e.queue = handles.queue;
        e.queue_family = handles.queue_family;
        e.enabled.push_descriptors = tinytensor::vulkan::device_info().push_descriptors;
        device = photara::vk::Device::adopt(e);
        pipeline =
            device.create_compute(std::as_bytes(std::span{pam_field_hlsl_spv}), 6, sizeof(Push));
        ring = photara::vk::EncoderRing(device, 1, photara::vk::BarrierPolicy::after_compute);
    }
};
void validate(const PamGpuField& field, const Tensor& points, unsigned neighbors) {
    const auto valid = [](const Tensor& t, tinytensor::DataType type) {
        return t.is_valid() && t.device() == device && t.dtype() == type && t.is_contiguous();
    };
    if (!valid(points, tinytensor::DataType::Float32) || points.shape().rank() != 2 ||
        points.shape()[1] != 3 || points.shape()[0] > std::numeric_limits<std::uint32_t>::max() ||
        !valid(field.nodes, tinytensor::DataType::Int32) ||
        !valid(field.values, tinytensor::DataType::Float32) ||
        !valid(field.bounds, tinytensor::DataType::Float32) || field.root < 0 || neighbors == 0 ||
        neighbors > pam_gpu_max_neighbors)
        throw std::invalid_argument("Invalid PAM Vulkan field query");
}
void launch(Push push, const std::array<Tensor, 6>& tensors) {
    if (!push.count)
        return;
    tinytensor::vulkan::submit_async();
    static thread_local Backend backend;
    std::array<photara::vk::BufferBinding, 6> bindings;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto v = tinytensor::vulkan::buffer_view(tensors[i]);
        bindings[i] = {v.buffer, v.offset, v.bytes};
    }
    const auto groups = (push.count + 63u) / 64u;
    auto& encoder = backend.ring.acquire();
    encoder.dispatch(backend.pipeline, bindings, &push, sizeof(push), std::min(groups, 65535u),
                     (groups + 65534u) / 65535u);
    backend.ring.submit();
    // Retire external submissions before Tensor owners can return buffers to
    // TinyTensor's pool. Points themselves remain on the Vulkan device.
    encoder.wait();
}
} // namespace
Tensor pam_field_gradients_vulkan(const PamGpuField& field, const Tensor& points,
                                  unsigned neighbors) {
    validate(field, points, neighbors);
    auto output = Tensor::empty(points.shape(), device);
    Push push{field.root,
              static_cast<std::uint32_t>(points.shape()[0]),
              static_cast<std::uint32_t>(std::min<std::size_t>(neighbors, field.values.shape()[0])),
              0,
              0,
              0,
              0};
    launch(push, {field.nodes, field.values, field.bounds, points, field.values, output});
    return output;
}
void pam_refine_points_vulkan(const PamGpuField& field, Tensor& points, const Tensor& occupancy,
                              unsigned neighbors, float iso, float minimum, float step) {
    validate(field, points, neighbors);
    if (!occupancy.is_valid() || occupancy.device() != device || !occupancy.is_contiguous() ||
        occupancy.dtype() != tinytensor::DataType::Float32 || occupancy.shape().rank() != 1 ||
        occupancy.shape()[0] != points.shape()[0])
        throw std::invalid_argument("Invalid PAM Vulkan occupancy");
    Push push{field.root,
              static_cast<std::uint32_t>(points.shape()[0]),
              static_cast<std::uint32_t>(std::min<std::size_t>(neighbors, field.values.shape()[0])),
              1,
              iso,
              minimum,
              step};
    launch(push, {field.nodes, field.values, field.bounds, points, occupancy, points});
}
} // namespace photara::splat::detail
