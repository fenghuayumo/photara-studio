#include "focus_mask.hpp"

#include "focus_mask.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include "vulkan/backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace photara::splat::detail {
namespace {

struct FocusMaskPush {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t model = 0;
    float fx = 1.F;
    float fy = 1.F;
    float cx = 0.F;
    float cy = 0.F;
    float k1 = 0.F;
    float k2 = 0.F;
    float k3 = 0.F;
    float k4 = 0.F;
    float origin_x = 0.F;
    float origin_y = 0.F;
    float origin_z = 0.F;
    float m00 = 0.F;
    float m01 = 0.F;
    float m02 = 0.F;
    float m10 = 0.F;
    float m11 = 0.F;
    float m12 = 0.F;
    float m20 = 0.F;
    float m21 = 0.F;
    float m22 = 0.F;
    float half_x = 0.F;
    float half_y = 0.F;
    float half_z = 0.F;
};

static_assert(sizeof(FocusMaskPush) == 104);

FocusMaskPush make_push(const FocusMaskFrame& frame) {
    FocusMaskPush push;
    push.width = frame.width;
    push.height = frame.height;
    push.model = frame.model;
    push.fx = frame.fx;
    push.fy = frame.fy;
    push.cx = frame.cx;
    push.cy = frame.cy;
    push.k1 = frame.k1;
    push.k2 = frame.k2;
    push.k3 = frame.k3;
    push.k4 = frame.k4;
    push.origin_x = frame.origin_x;
    push.origin_y = frame.origin_y;
    push.origin_z = frame.origin_z;
    push.m00 = frame.camera_to_box[0];
    push.m01 = frame.camera_to_box[1];
    push.m02 = frame.camera_to_box[2];
    push.m10 = frame.camera_to_box[3];
    push.m11 = frame.camera_to_box[4];
    push.m12 = frame.camera_to_box[5];
    push.m20 = frame.camera_to_box[6];
    push.m21 = frame.camera_to_box[7];
    push.m22 = frame.camera_to_box[8];
    push.half_x = frame.half_x;
    push.half_y = frame.half_y;
    push.half_z = frame.half_z;
    return push;
}

struct FocusMaskVulkan {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::CommandEncoder encoder;

    FocusMaskVulkan() {
        if (!tinytensor::vulkan::available())
            throw std::runtime_error("Vulkan focus mask needs a compute device");
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
        const auto spirv = std::as_bytes(std::span{focus_mask_hlsl_spv});
        pipeline = device.create_compute(spirv, 1, sizeof(FocusMaskPush));
        encoder = device.encoder(photara::vk::BarrierPolicy::after_compute);
    }
};

FocusMaskVulkan& backend() {
    static thread_local FocusMaskVulkan instance;
    return instance;
}

}  // namespace

tinytensor::Tensor focus_view_mask_vulkan(const FocusMaskFrame& frame) {
    const std::size_t pixels =
        static_cast<std::size_t>(frame.width) * frame.height;
    auto mask = tinytensor::Tensor::empty(
        {frame.height, frame.width}, tinytensor::Device::Vulkan);
    if (pixels == 0) return mask;
    tinytensor::vulkan::submit_async();
    auto& vulkan = backend();
    const auto view = tinytensor::vulkan::buffer_view(mask);
    const photara::vk::BufferBinding binding{
        view.buffer, view.offset, view.bytes};
    const FocusMaskPush push = make_push(frame);
    const std::uint32_t groups = static_cast<std::uint32_t>(
        (pixels + 255) / 256);
    const std::array bindings{binding};
    vulkan.encoder.dispatch(
        vulkan.pipeline, bindings, &push, sizeof(push), groups);
    vulkan.encoder.submit();
    return mask;
}

}  // namespace photara::splat::detail
