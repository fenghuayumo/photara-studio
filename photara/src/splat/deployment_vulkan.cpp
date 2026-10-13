#include "deployment_vulkan.hpp"
#include "deployment.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include "vulkan/backend.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
namespace photara::splat::detail {
namespace {
struct Push {
    unsigned mode, count, first, nx, ny, nz;
    float ox, oy, oz, size, threshold;
    unsigned padding{};
};
static_assert(sizeof(Push) == 48);
struct Backend {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::EncoderRing ring;
    Backend() {
        if (!tinytensor::vulkan::available())
            throw std::runtime_error("Vulkan compute unavailable");
        const auto h = tinytensor::vulkan::device_handles();
        photara::vk::ExternalDevice e;
        e.instance = h.instance;
        e.physical = h.physical_device;
        e.device = h.device;
        e.queue = h.queue;
        e.queue_family = h.queue_family;
        e.enabled.push_descriptors = tinytensor::vulkan::device_info().push_descriptors;
        device = photara::vk::Device::adopt(e);
        pipeline =
            device.create_compute(std::as_bytes(std::span{deployment_hlsl_spv}), 4, sizeof(Push));
        ring = photara::vk::EncoderRing(device, 1, photara::vk::BarrierPolicy::after_compute);
    }
};
std::vector<float> dispatch(const tinytensor::Tensor &v, const tinytensor::Tensor &x,
                            const tinytensor::Tensor &y, Push push) {
    if (!push.count)
        return {};
    constexpr auto gpu = tinytensor::Device::Vulkan;
    auto result = tinytensor::Tensor::empty({push.count}, gpu);
    tinytensor::vulkan::submit_async();
    static thread_local Backend backend;
    const std::array tensors{v, x, y, result};
    std::array<photara::vk::BufferBinding, 4> bindings;
    for (unsigned i = 0; i < 4; ++i) {
        const auto view = tinytensor::vulkan::buffer_view(tensors[i]);
        bindings[i] = {view.buffer, view.offset, view.bytes};
    }
    auto &encoder = backend.ring.acquire();
    const unsigned groups = (push.count + 63) / 64;
    encoder.dispatch(backend.pipeline, bindings, &push, sizeof(push), std::min(groups, 65535u),
                     (groups + 65534) / 65535);
    backend.ring.submit();
    encoder.wait();
    return result.to_vector();
}
tinytensor::Tensor upload(const std::vector<std::int32_t> &values) {
    return tinytensor::Tensor::from_vector(values.empty() ? std::vector<std::int32_t>{0} : values,
                                           {std::max<std::size_t>(1, values.size())},
                                           tinytensor::Device::Vulkan);
}
} // namespace
std::vector<float> lod_costs_vulkan(const std::vector<float> &values,
                                    const std::vector<std::int32_t> &pairs) {
    auto v = tinytensor::Tensor::from_vector(values, {values.size()}, tinytensor::Device::Vulkan);
    return dispatch(v, upload(pairs), upload({}), Push{0, static_cast<unsigned>(pairs.size() / 2)});
}
std::function<std::vector<float>(unsigned, unsigned)> prepare_voxel_field_vulkan(
    const std::vector<float> &values, const std::vector<std::int32_t> &offsets,
    const std::vector<std::int32_t> &indices, const std::array<unsigned, 3> &dims,
    const std::array<float, 3> &origin, float size, float threshold) {
    auto v = tinytensor::Tensor::from_vector(values, {values.size()}, tinytensor::Device::Vulkan);
    auto x = upload(offsets), y = upload(indices);
    const auto total = std::size_t(dims[0]) * dims[1] * dims[2];
    return [v, x, y, dims, origin, size, threshold, total](unsigned first, unsigned count) {
        if (first > total || count > total - first)
            throw std::invalid_argument("Invalid voxel batch");
        return dispatch(v, x, y,
                        Push{1, count, first, dims[0], dims[1], dims[2], origin[0], origin[1],
                             origin[2], size, threshold});
    };
}
} // namespace photara::splat::detail
