#include "photara_vk/photara_vk.hpp"
#include "visualize_diagnostics.hlsl.embedded.hpp"
#include "visualize_diagnostics.hpp"
#include "vulkan/backend.hpp"
#include <span>
#include <vector>

namespace photara::splat::detail {
namespace {
struct DiagnosticsVulkan {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::CommandEncoder encoder;
    DiagnosticsVulkan() {
        const auto handles = tinytensor::vulkan::device_handles();
        photara::vk::ExternalDevice external;
        external.instance = handles.instance;
        external.physical = handles.physical_device;
        external.device = handles.device;
        external.queue = handles.queue;
        external.queue_family = handles.queue_family;
        external.enabled.push_descriptors = tinytensor::vulkan::device_info().push_descriptors;
        device = photara::vk::Device::adopt(external);
        pipeline = device.create_compute(std::as_bytes(std::span{visualize_diagnostics_hlsl_spv}),
                                         5, sizeof(DiagnosticPush));
        encoder = device.encoder(photara::vk::BarrierPolicy::after_compute);
    }
};
} // namespace
tinytensor::Tensor visualize_diagnostics_vulkan(const RenderResult &rendered, const Camera &camera,
                                                const VisualizeOptions &options) {
    static thread_local DiagnosticsVulkan backend;
    auto color =
        tinytensor::Tensor::empty({3, camera.height, camera.width}, tinytensor::Device::Vulkan);
    std::vector<float> initial(260, 0.F);
    // Stats are uint bit patterns stored in an ordinary float tensor buffer.
    initial[1] = std::numeric_limits<float>::max();
    initial[2] = options.depth_near;
    initial[3] = options.depth_far;
    auto stats = tinytensor::Tensor::from_vector(initial, {260}, tinytensor::Device::Vulkan);
    const auto binding = [](const tinytensor::Tensor &tensor) {
        const auto view = tinytensor::vulkan::buffer_view(tensor);
        return photara::vk::BufferBinding{view.buffer, view.offset, view.bytes};
    };
    const std::array bindings{binding(rendered.median_depth), binding(rendered.alpha),
                              binding(rendered.normal), binding(color), binding(stats)};
    auto push = diagnostic_push(camera, options);
    tinytensor::vulkan::submit_async();
    const unsigned groups = (push.pixels + 255) / 256;
    if (options.channel == VisualizationChannel::depth && options.automatic_depth) {
        for (unsigned stage : {1u, 2u, 3u}) {
            push.stage = stage;
            backend.encoder.dispatch(backend.pipeline, bindings, &push, sizeof(push),
                                     stage == 3 ? 1 : groups);
        }
    }
    push.stage = 4;
    backend.encoder.dispatch(backend.pipeline, bindings, &push, sizeof(push), groups);
    backend.encoder.submit();
    return color;
}
} // namespace photara::splat::detail
