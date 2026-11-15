#include "visualize_diagnostics.hpp"
#include "visualize_points.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include "vulkan/backend.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace photara::splat::detail {
namespace {
struct Push {
    unsigned count, width, height, bases, degree, stage, model;
    float size, near_clip, br, bg, bb;
    std::array<unsigned, 5> offsets{};
};
struct Backend {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::CommandEncoder encoder;
    Backend() {
        auto h = tinytensor::vulkan::device_handles();
        photara::vk::ExternalDevice e;
        e.instance = h.instance;
        e.physical = h.physical_device;
        e.device = h.device;
        e.queue = h.queue;
        e.queue_family = h.queue_family;
        e.enabled.push_descriptors = tinytensor::vulkan::device_info().push_descriptors;
        device = photara::vk::Device::adopt(e);
        pipeline = device.create_compute(
            std::as_bytes(std::span{visualize_points_hlsl_spv}), 5, sizeof(Push));
        encoder = device.encoder(photara::vk::BarrierPolicy::after_compute);
    }
};
} // namespace
tinytensor::Tensor visualize_points_vulkan(
    const GaussianModel& model,
    const Camera& c,
    const VisualizeOptions& o) {
    using tinytensor::Tensor;
    constexpr auto vk = tinytensor::Device::Vulkan;
    const std::uint64_t pixels = std::uint64_t(c.width) * c.height;
    if (!pixels || pixels > std::numeric_limits<unsigned>::max() / 4 ||
        model.size() > std::numeric_limits<unsigned>::max() / 48 ||
        !std::isfinite(o.point_size_px) || o.point_size_px <= 0 ||
        o.point_size_px > float(std::numeric_limits<int>::max() / 4) || model.sh_degree > 3 ||
        !std::isfinite(o.rasterization.near_plane) || o.rasterization.near_plane < 0)
        throw std::invalid_argument("Invalid Vulkan points camera, size, or SH degree");
    for (const auto* t : {&model.means, &model.sh})
        if (!t->is_valid() || t->device() != vk || t->dtype() != tinytensor::DataType::Float32 ||
            !t->is_contiguous())
            throw std::invalid_argument("Vulkan points requires contiguous float32 tensors");
    if (model.means.numel() != model.size() * 3 || model.sh.shape().rank() != 3 ||
        model.sh.shape()[0] != model.size() || model.sh.shape()[2] != 3 ||
        model.sh.shape()[1] < (model.sh_degree + 1) * (model.sh_degree + 1))
        throw std::invalid_argument("Invalid Vulkan points model shape");
    auto output = Tensor::empty({3, c.height, c.width}, vk);
    if (!model.size())
        return Tensor::zeros_like(output);
    std::vector<float> values(c.world_to_camera.begin(), c.world_to_camera.end());
    values.insert(values.end(), c.position.begin(), c.position.end());
    values.insert(values.end(), {c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.k3, c.k4});
    auto camera = Tensor::from_vector(values, {values.size()}, vk);
    auto accum = Tensor::zeros({std::size_t(pixels) * 4}, vk);
    Push p{
        unsigned(model.size()),
        c.width,
        c.height,
        unsigned(model.sh.shape()[1]),
        std::min(o.active_sh_degree, model.sh_degree),
        0,
        c.orthographic ? 4u : unsigned(c.model),
        o.point_size_px,
        o.rasterization.near_plane,
        o.background[0],
        o.background[1],
        o.background[2]};
    std::array<photara::vk::BufferBinding, 5> bindings;
    const std::array tensors{model.means, model.sh, camera, accum, output};
    for (unsigned i = 0; i < 5; ++i) {
        auto v = tinytensor::vulkan::buffer_view(tensors[i]);
        p.offsets[i] = unsigned(v.offset / 4);
        bindings[i] = {v.buffer, 0, v.offset + v.bytes};
    }
    static thread_local Backend b;
    tinytensor::vulkan::synchronize();
    auto dispatch = [&](unsigned n) {
        auto groups = (n + 255) / 256;
        b.encoder.dispatch(
            b.pipeline, bindings, &p, sizeof(p), std::min(groups, 65535u),
            (groups + 65534) / 65535);
    };
    dispatch(p.count);
    p.stage = 1;
    dispatch(unsigned(pixels));
    b.encoder.submit();
    b.encoder.wait();
    return output;
}
} // namespace photara::splat::detail
