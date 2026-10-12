#include "adc_backends.hpp"
#include "adc_stats.hlsl.embedded.hpp"
#include "adc_noise.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include "vulkan/backend.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace photara::splat::detail {
namespace {
using tinytensor::Tensor;
constexpr auto device = tinytensor::Device::Vulkan;
struct StatsPush {
    std::uint32_t count, flags;
    float inverse_resolution, power, screen_threshold;
    std::array<std::uint32_t, 12> offsets{};
};
struct NoisePush {
    std::uint32_t count, seed;
    float standard_deviation, maximum_noise;
    std::array<std::uint32_t, 3> offsets{};
};
static_assert(sizeof(StatsPush) == 68 && sizeof(NoisePush) == 28);

template<std::size_t N> struct Backend {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::EncoderRing ring;
    std::array<std::unique_ptr<std::array<Tensor, N>>, 4> retained;
    Backend(std::span<const std::byte> shader, std::uint32_t push_bytes) {
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
        pipeline = device.create_compute(shader, N, push_bytes);
        ring = photara::vk::EncoderRing(device, 4, photara::vk::BarrierPolicy::after_compute);
    }
    ~Backend() {
        for (std::uint32_t i = 0; i < ring.size(); ++i) ring.at(i).wait();
    }
    template<class Push> void launch(const Push& push, const std::array<Tensor, N>& tensors) {
        tinytensor::vulkan::submit_async();
        std::array<photara::vk::BufferBinding, N> bindings;
        auto adjusted = push;
        for (std::size_t i = 0; i < N; ++i) {
            const auto view = tinytensor::vulkan::buffer_view(
                tensors[i].is_valid() ? tensors[i] : tensors[0]);
            // Tensor slices need not satisfy descriptor-offset alignment.
            // Bind from zero and carry their element offsets in push constants.
            if (view.offset / 4 > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("Vulkan ADC tensor offset exceeds shader range");
            adjusted.offsets[i] = static_cast<std::uint32_t>(view.offset / 4);
            bindings[i] = {view.buffer, 0, view.offset + view.bytes};
        }
        auto& encoder = ring.acquire();
        // Acquire retires this slot's fence before releasing previous owners.
        // Retain fresh handles, avoiding assignment into an existing Tensor view.
        retained[ring.index()] = std::make_unique<std::array<Tensor, N>>(tensors);
        const auto groups = (static_cast<std::uint64_t>(push.count) + 255u) / 256u;
        encoder.dispatch(pipeline, bindings, &adjusted, sizeof(adjusted),
            static_cast<std::uint32_t>(std::min<std::uint64_t>(groups, 65535u)),
            static_cast<std::uint32_t>((groups + 65534u) / 65535u));
        ring.submit();
    }
};

void validate(const Tensor& t, std::size_t size, tinytensor::DataType dtype) {
    if (!t.is_valid() || t.device() != device || t.dtype() != dtype ||
        !t.is_contiguous() || t.numel() != size)
        throw std::invalid_argument("Vulkan ADC requires matching contiguous tensors");
}
std::uint32_t checked_count(std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("Vulkan ADC Gaussian count exceeds shader range");
    return static_cast<std::uint32_t>(count);
}
}  // namespace

void accumulate_densification_stats_vulkan(
    const Tensor& refine_weight, const Tensor& visibility, const Tensor& radii,
    DensificationStats& stats, const AdcStatsOptions& options,
    const Tensor& geometry_gradient,
    const Tensor& image_error, const Tensor& normalized_size) {
    const auto n = options.count;
    validate(radii, n, tinytensor::DataType::Int32);
    if (options.require_contribution_visibility) validate(visibility, n, tinytensor::DataType::Float32);
    for (const auto* t : std::array<const Tensor*, 7>{&refine_weight, &stats.gradient,
             &stats.count, &stats.max_screen_radius, &stats.priority,
             &stats.geometry_gradient, &stats.image_error})
        validate(*t, n, tinytensor::DataType::Float32);
    for (const auto* t : {&geometry_gradient, &image_error, &normalized_size})
        if (t->is_valid()) validate(*t, n, tinytensor::DataType::Float32);
    const std::uint32_t flags = (options.use_maximum ? 1u : 0u) |
        (options.require_contribution_visibility ? 2u : 0u) |
        (geometry_gradient.is_valid() ? 4u : 0u) |
        (image_error.is_valid() ? 8u : 0u) | (normalized_size.is_valid() ? 16u : 0u);
    const StatsPush push{checked_count(n), flags, options.inverse_resolution,
        options.step_score_power, options.oversize_screen_threshold};
    static thread_local Backend<12> vk(std::as_bytes(std::span{adc_stats_hlsl_spv}), sizeof(StatsPush));
    vk.launch(push, {refine_weight, visibility, radii, geometry_gradient,
        image_error, normalized_size, stats.gradient, stats.count,
        stats.max_screen_radius, stats.priority, stats.geometry_gradient, stats.image_error});
}

void inject_adc_noise_vulkan(GaussianModel& model, const Tensor& visibility,
    const AdcNoiseOptions& options) {
    const auto n = model.size();
    validate(model.means, n * 3, tinytensor::DataType::Float32);
    validate(model.opacity_logits, n, tinytensor::DataType::Float32);
    validate(visibility, n, tinytensor::DataType::Float32);
    checked_count(n * 3);
    const NoisePush push{checked_count(n), options.seed, options.standard_deviation, options.maximum_noise};
    static thread_local Backend<3> vk(std::as_bytes(std::span{adc_noise_hlsl_spv}), sizeof(NoisePush));
    vk.launch(push, {model.means, model.opacity_logits, visibility});
}
}  // namespace photara::splat::detail
