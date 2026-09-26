#include "splat_drender/vulkan_api.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "context_internal.hpp"

namespace splat_drender::vulkan {
namespace {

constexpr std::uint32_t k_tile = 16;
constexpr std::uint32_t k_gauss_slots = 8;
constexpr std::uint32_t k_streams = 6;
constexpr std::uint32_t k_single_sort_limit = 131072;
constexpr std::uint32_t k_push_uints = 20;
constexpr std::uint32_t k_flag_sh = 1;
constexpr std::uint32_t k_flag_scales = 2;
constexpr std::uint32_t k_flag_geometry = 4;
constexpr std::uint32_t k_flag_snapshot = 8;
constexpr std::uint32_t k_flag_stats = 16;
constexpr std::uint32_t k_flag_raw_parameters = 32;

struct Push {
    std::uint32_t u[k_push_uints]{};
};
static_assert(sizeof(Push) == 80);

void set_float(Push& push, int index, float value) {
    std::memcpy(&push.u[index], &value, sizeof(float));
}

std::uint32_t div_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::uint32_t scan_scratch_uints(std::uint32_t count) {
    return 4 * div_up(count, 256) + 64;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

VkDescriptorBufferInfo descriptor(const Buffer& buffer) {
    return {buffer.handle, 0, buffer.size};
}

VkDescriptorBufferInfo descriptor(
    const SplatBufferView& view, const std::uint64_t required_bytes) {
    return {view.buffer, view.offset, required_bytes};
}

void clear_buffer(Buffer& buffer) {
    if (!buffer.host_visible()) return;
    std::memset(buffer.mapped, 0, static_cast<std::size_t>(buffer.size));
}

template <typename T>
std::vector<T> download_vector(
    Context::Impl& context, const Buffer& buffer, std::size_t count, std::size_t offset_bytes = 0) {
    std::vector<T> values(count);
    if (count != 0) context.read_buffer(buffer, values.data(), count * sizeof(T), offset_bytes);
    return values;
}

void zero_buffer(Context::Impl& context, Buffer& buffer, const std::size_t bytes) {
    if (bytes == 0) return;
    context.fill_buffer(buffer, 0, bytes);
}

bool environment_flag(const char* name) {
#ifdef _WIN32
    char* raw_value = nullptr;
    std::size_t value_size = 0;
    if (_dupenv_s(&raw_value, &value_size, name) != 0 || raw_value == nullptr)
        return false;
    const bool enabled = raw_value[0] == '1';
    std::free(raw_value);
    return enabled;
#else
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
#endif
}

std::uint32_t environment_interval(
    const char* name, const std::uint32_t default_value) {
#ifdef _WIN32
    char* raw_value = nullptr;
    std::size_t value_size = 0;
    if (_dupenv_s(&raw_value, &value_size, name) != 0 || raw_value == nullptr)
        return default_value;
    const int parsed = std::atoi(raw_value);
    std::free(raw_value);
#else
    const char* value = std::getenv(name);
    const int parsed = value == nullptr ? 0 : std::atoi(value);
#endif
    return parsed > 0 ? static_cast<std::uint32_t>(parsed) : default_value;
}

} // namespace

class SplatRasterizer::Impl {
public:
    explicit Impl(Context::Impl& context)
        : context_(context),
          dummy_(context.create_buffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)),
          preprocess_(context.create_pipeline("splat_preprocess.hlsl.spv", 12, sizeof(Push))),
          scan_(context.create_pipeline("splat_scan.hlsl.spv", 1, sizeof(Push))),
          emit_(context.create_pipeline("splat_emit.hlsl.spv", 7, sizeof(Push))),
          hist_(context.create_pipeline("splat_radix_hist.hlsl.spv", 3, sizeof(Push))),
          scatter_(context.create_pipeline("splat_radix_scatter.hlsl.spv", 8, sizeof(Push))),
          ranges_(context.create_pipeline("splat_ranges.hlsl.spv", 3, sizeof(Push))),
          blend_(context.create_pipeline("splat_blend.hlsl.spv", 7, sizeof(Push))),
          median_backward_(context.create_pipeline("splat_median_backward.hlsl.spv", 7, sizeof(Push))),
          blend_backward_(context.create_pipeline("splat_blend_backward.hlsl.spv", 12, sizeof(Push))),
          blend_backward_no_geometry_(
              context.create_pipeline(
                  "splat_blend_backward_no_geometry.hlsl.spv", 12, sizeof(Push))),
          blend_backward_no_geometry_subgroup_(
              context.create_pipeline(
                  "splat_blend_backward_no_geometry_subgroup.hlsl.spv",
                  12, sizeof(Push))),
          sample_depth_(context.create_pipeline("splat_sample_depth.hlsl.spv", 7, sizeof(Push))),
          sample_depth_backward_(context.create_pipeline("splat_sample_depth_backward.hlsl.spv", 10, sizeof(Push))),
          multi_view_(context.create_pipeline("splat_multi_view.hlsl.spv", 10, sizeof(Push))),
          ssim_(context.create_pipeline("splat_ssim.hlsl.spv", 12, sizeof(Push))),
          loss_reduce_(context.create_pipeline("splat_loss_reduce.hlsl.spv", 2, sizeof(Push))),
          project_backward_(context.create_pipeline("splat_project_backward.hlsl.spv", 15, sizeof(Push))),
          clear_(context.create_pipeline("splat_clear.hlsl.spv", 1, sizeof(Push))),
          pack_(context.create_pipeline("splat_pack_rgba.hlsl.spv", 2, sizeof(Push))) {
        clear_buffer(dummy_);
        create_backward_timestamp_profiler();
        create_forward_timestamp_profiler();
        query_subgroup_properties();
    }

    ~Impl() {
        if (recording_ && command_ != VK_NULL_HANDLE) {
            vkEndCommandBuffer(command_);
            recording_ = false;
        }
        if (!pending_sets_.empty()) {
            vkFreeDescriptorSets(
                context_.device, context_.descriptor_pool,
                static_cast<std::uint32_t>(pending_sets_.size()), pending_sets_.data());
        }
        if (command_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(context_.device, context_.command_pool, 1, &command_);
        }
        if (backward_timestamp_pool_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(context_.device, backward_timestamp_pool_, nullptr);
        }
    }

    void upload_model(const SplatGaussians& gaussians) {
        const auto count = static_cast<std::uint32_t>(gaussians.means.size() / 3);
        require(count > 0 && gaussians.means.size() == static_cast<std::size_t>(count) * 3, "means must have shape [N, 3]");
        require(gaussians.opacities.size() == count, "opacities must have shape [N]");
        const bool has_sh = !gaussians.sh.empty();
        const bool has_colors = !gaussians.colors.empty();
        require(has_sh != has_colors, "exactly one of sh or colors is required");
        const bool has_scales = !gaussians.scales.empty();
        const bool has_cov = !gaussians.covariances.empty();
        require(has_scales != has_cov, "exactly one of scales or covariances is required");
        if (has_sh) {
            require(gaussians.sh_degree <= 3 && gaussians.sh_bases >= (gaussians.sh_degree + 1) * (gaussians.sh_degree + 1),
                    "sh_bases does not cover sh_degree");
            require(gaussians.sh.size() == static_cast<std::size_t>(count) * gaussians.sh_bases * 3,
                    "sh must have shape [N, sh_bases, 3]");
        } else {
            require(gaussians.colors.size() == static_cast<std::size_t>(count) * 3, "colors must have shape [N, 3]");
        }
        if (has_scales) {
            require(gaussians.scales.size() == static_cast<std::size_t>(count) * 3, "scales must have shape [N, 3]");
            require(gaussians.rotations.size() == static_cast<std::size_t>(count) * 4, "rotations must have shape [N, 4]");
        } else {
            require(gaussians.covariances.size() == static_cast<std::size_t>(count) * 6, "covariances must have shape [N, 6]");
        }
        const bool any_raw = !gaussians.log_scales.empty() || !gaussians.raw_rotations.empty() ||
                             !gaussians.opacity_logits.empty();
        const bool raw_chain = !gaussians.log_scales.empty() && !gaussians.raw_rotations.empty() &&
                               !gaussians.opacity_logits.empty();
        require(!any_raw || raw_chain,
                "log_scales, raw_rotations, and opacity_logits must be supplied together");
        require(!raw_chain || has_scales, "the raw activation chain requires scales and rotations");
        if (raw_chain) {
            require(gaussians.log_scales.size() == static_cast<std::size_t>(count) * 3,
                    "log_scales must have shape [N, 3]");
            require(gaussians.raw_rotations.size() == static_cast<std::size_t>(count) * 4,
                    "raw_rotations must have shape [N, 4]");
            require(gaussians.opacity_logits.size() == count,
                    "opacity_logits must have shape [N]");
            require(gaussians.filter_3d.empty() || gaussians.filter_3d.size() == count,
                    "filter_3d must be empty or have shape [N]");
        } else {
            require(gaussians.filter_3d.empty(),
                    "filter_3d requires the raw activation parameters");
        }

        const auto& color_src = has_sh ? gaussians.sh : gaussians.colors;
        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& means = grow(means_, gaussians.means.size() * sizeof(float));
        Buffer& opacities = grow(opacities_, gaussians.opacities.size() * sizeof(float));
        Buffer& scales = grow(scales_, (has_scales ? gaussians.scales.size() : 1) * sizeof(float));
        Buffer& rotations = grow(rotations_, (has_scales ? gaussians.rotations.size() : 1) * sizeof(float));
        Buffer& covariances = grow(covariances_, (has_cov ? gaussians.covariances.size() : 1) * sizeof(float));
        Buffer& colors = grow(colors_, color_src.size() * sizeof(float));
        Buffer& raw_log_scales = grow(raw_log_scales_, (raw_chain ? gaussians.log_scales.size() : 1) * sizeof(float));
        Buffer& raw_rotations = grow(raw_rotations_, (raw_chain ? gaussians.raw_rotations.size() : 1) * sizeof(float));
        Buffer& opacity_logits = grow(opacity_logits_, (raw_chain ? gaussians.opacity_logits.size() : 1) * sizeof(float));
        Buffer& filter_3d = grow(filter_3d_, (raw_chain ? count : 1) * sizeof(float));
        context_.write_buffer(means, gaussians.means.data(), gaussians.means.size() * sizeof(float));
        context_.write_buffer(opacities, gaussians.opacities.data(), gaussians.opacities.size() * sizeof(float));
        if (has_scales) {
            context_.write_buffer(scales, gaussians.scales.data(), gaussians.scales.size() * sizeof(float));
            context_.write_buffer(rotations, gaussians.rotations.data(), gaussians.rotations.size() * sizeof(float));
        }
        if (has_cov) {
            context_.write_buffer(covariances, gaussians.covariances.data(), gaussians.covariances.size() * sizeof(float));
        }
        context_.write_buffer(colors, color_src.data(), color_src.size() * sizeof(float));
        if (raw_chain) {
            context_.write_buffer(raw_log_scales, gaussians.log_scales.data(), gaussians.log_scales.size_bytes());
            context_.write_buffer(raw_rotations, gaussians.raw_rotations.data(), gaussians.raw_rotations.size_bytes());
            context_.write_buffer(opacity_logits, gaussians.opacity_logits.data(), gaussians.opacity_logits.size_bytes());
            if (gaussians.filter_3d.empty()) {
                zero_buffer(context_, filter_3d, static_cast<std::size_t>(count) * sizeof(float));
            } else {
                context_.write_buffer(filter_3d, gaussians.filter_3d.data(), gaussians.filter_3d.size_bytes());
            }
        }
        has_sh_ = has_sh;
        has_scales_ = has_scales;
        raw_chain_ = raw_chain;
        count_ = count;
        sh_degree_ = gaussians.sh_degree;
        sh_bases_ = gaussians.sh_bases;
        device_model_bound_ = false;
        model_ready_ = true;
        last_frame_has_snapshots_ = false;
    }

    void bind_model_device(const SplatDeviceGaussians& gaussians) {
        require(gaussians.count > 0, "device Gaussian count must be positive");
        require(gaussians.sh_degree <= 3 &&
                    gaussians.sh_bases >=
                        (gaussians.sh_degree + 1) * (gaussians.sh_degree + 1),
                "device sh_bases does not cover sh_degree");
        const std::uint64_t count = gaussians.count;
        const auto valid = [](const SplatBufferView& view,
                              const std::uint64_t bytes) {
            return view.buffer != VK_NULL_HANDLE && view.bytes >= bytes;
        };
        require(valid(gaussians.means, count * 3 * sizeof(float)),
                "device means must have shape [N,3]");
        require(valid(gaussians.sh,
                      count * gaussians.sh_bases * 3 * sizeof(float)),
                "device SH must have shape [N,B,3]");
        require(valid(gaussians.log_scales, count * 3 * sizeof(float)),
                "device log scales must have shape [N,3]");
        require(valid(gaussians.raw_rotations, count * 4 * sizeof(float)),
                "device rotations must have shape [N,4]");
        require(valid(gaussians.opacity_logits, count * sizeof(float)),
                "device opacity logits must have shape [N]");
        require(valid(gaussians.filter_3d, count * sizeof(float)),
                "device filter_3d must have shape [N]");
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context_.physical_device, &properties);
        const std::uint64_t alignment = std::max<std::uint64_t>(
            4, properties.limits.minStorageBufferOffsetAlignment);
        require(gaussians.means.offset % alignment == 0 &&
                    gaussians.sh.offset % alignment == 0 &&
                    gaussians.log_scales.offset % alignment == 0 &&
                    gaussians.raw_rotations.offset % alignment == 0 &&
                    gaussians.opacity_logits.offset % alignment == 0 &&
                    gaussians.filter_3d.offset % alignment == 0,
                "device model offsets do not satisfy minStorageBufferOffsetAlignment");
        device_model_ = gaussians;
        device_model_bound_ = true;
        has_sh_ = true;
        has_scales_ = true;
        raw_chain_ = true;
        count_ = gaussians.count;
        sh_degree_ = gaussians.sh_degree;
        sh_bases_ = gaussians.sh_bases;
        model_ready_ = true;
        last_frame_has_snapshots_ = false;
        sample_live_ = false;
    }

    void update_means_and_opacities(std::span<const float> means, std::span<const float> opacities) {
        require(model_ready_, "no Gaussian model is loaded");
        require(!device_model_bound_,
                "host model updates are invalid while a device model is bound");
        require(means.size() == static_cast<std::size_t>(count_) * 3, "means must have shape [N, 3]");
        require(opacities.size() == count_, "opacities must have shape [N]");
        const std::scoped_lock lock(context_.dispatch_mutex);
        context_.write_buffer(means_, means.data(), means.size() * sizeof(float));
        context_.write_buffer(opacities_, opacities.data(), opacities.size() * sizeof(float));
        last_frame_has_snapshots_ = false;
    }

    void clear_model() {
        model_ready_ = false;
        device_model_bound_ = false;
        device_model_ = {};
        count_ = 0;
        last_frame_has_snapshots_ = false;
        sample_live_ = false;
    }
    bool has_model() const noexcept { return model_ready_; }

    VkDescriptorBufferInfo model_means() const {
        return device_model_bound_
            ? descriptor(device_model_.means,
                         static_cast<std::uint64_t>(count_) * 3 * sizeof(float))
            : descriptor(means_);
    }
    VkDescriptorBufferInfo model_opacities() const {
        return device_model_bound_
            ? descriptor(device_model_.opacity_logits,
                         static_cast<std::uint64_t>(count_) * sizeof(float))
            : descriptor(opacities_);
    }
    VkDescriptorBufferInfo model_scales() const {
        return device_model_bound_
            ? descriptor(device_model_.log_scales,
                         static_cast<std::uint64_t>(count_) * 3 * sizeof(float))
            : descriptor(scales_);
    }
    VkDescriptorBufferInfo model_rotations() const {
        return device_model_bound_
            ? descriptor(device_model_.raw_rotations,
                         static_cast<std::uint64_t>(count_) * 4 * sizeof(float))
            : descriptor(rotations_);
    }
    VkDescriptorBufferInfo model_covariances() const {
        return device_model_bound_
            ? descriptor(device_model_.filter_3d,
                         static_cast<std::uint64_t>(count_) * sizeof(float))
            : descriptor(covariances_);
    }
    VkDescriptorBufferInfo model_colors() const {
        return device_model_bound_
            ? descriptor(device_model_.sh,
                         static_cast<std::uint64_t>(count_) * sh_bases_ * 3 *
                             sizeof(float))
            : descriptor(colors_);
    }
    VkDescriptorBufferInfo model_log_scales() const {
        return device_model_bound_
            ? descriptor(device_model_.log_scales,
                         static_cast<std::uint64_t>(count_) * 3 * sizeof(float))
            : descriptor(raw_log_scales_);
    }
    VkDescriptorBufferInfo model_raw_rotations() const {
        return device_model_bound_
            ? descriptor(device_model_.raw_rotations,
                         static_cast<std::uint64_t>(count_) * 4 * sizeof(float))
            : descriptor(raw_rotations_);
    }
    VkDescriptorBufferInfo model_opacity_logits() const {
        return device_model_bound_
            ? descriptor(device_model_.opacity_logits,
                         static_cast<std::uint64_t>(count_) * sizeof(float))
            : descriptor(opacity_logits_);
    }
    VkDescriptorBufferInfo model_filter_3d() const {
        return device_model_bound_
            ? descriptor(device_model_.filter_3d,
                         static_cast<std::uint64_t>(count_) * sizeof(float))
            : descriptor(filter_3d_);
    }

    struct FrameCounts {
        std::uint32_t instances = 0;
        std::uint32_t visible = 0;
    };

    // Everything the frame computes, up to the blended image. `pack_rgba` adds
    // the 8-bit conversion to the same command batch, which is what the editor
    // preview hands to its own image.
    FrameCounts record_frame(
        const SplatCamera& camera, const SplatSettings& settings,
        const bool pack_rgba,
        const SplatDeviceFrame* destination = nullptr) {
        sample_live_ = false;
        require(model_ready_, "no Gaussian model is loaded");
        require(camera.width > 0 && camera.height > 0, "camera width and height must be positive");
        require(camera.mode == 0 || camera.mode == 1 || camera.mode == 3 || camera.mode == 4,
                "camera mode must be pinhole, fisheye, equirectangular, or orthographic");
        require(camera.world_to_camera.size() == 16 && camera.center.size() == 3, "camera transform must be 16 floats and center 3 floats");
        const std::scoped_lock lock(context_.dispatch_mutex);
        const std::uint32_t count = count_;
        const bool has_sh = has_sh_;
        const bool has_scales = has_scales_;
        // 76 bytes of per-frame constants: the one buffer worth keeping mapped.
        Buffer& camera_buffer = grow(camera_, 19 * sizeof(float), BufferMemory::host_visible);
        const std::uint32_t grid_x = div_up(camera.width, k_tile);
        const std::uint32_t grid_y = div_up(camera.height, k_tile);
        const std::uint32_t tiles = grid_x * grid_y;
        const std::uint32_t tile_bits = tiles > 1 ? 32u - static_cast<std::uint32_t>(std::countl_zero(tiles - 1)) : 0u;
        const auto pixels = static_cast<std::uint32_t>(static_cast<std::uint64_t>(camera.width) * camera.height);
        const int wrap_width = camera.mode == 3 ? static_cast<int>(camera.width) : 0;
        float camera_pack[19]{};
        std::memcpy(camera_pack, camera.world_to_camera.data(), 16 * sizeof(float));
        std::memcpy(camera_pack + 16, camera.center.data(), 3 * sizeof(float));
        camera_buffer.upload(camera_pack, sizeof(camera_pack));

        const auto gauss_u_count = k_streams * count + scan_scratch_uints(count);
        Buffer& gauss_f = grow(gauss_f_, static_cast<std::uint64_t>(count) * k_gauss_slots * sizeof(float) * 4);
        Buffer& gauss_u = grow(gauss_u_, static_cast<std::uint64_t>(gauss_u_count) * sizeof(std::uint32_t));
        write_forward_timestamp(0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

        Push prep{};
        prep.u[0] = count;
        prep.u[1] = sh_degree_;
        prep.u[2] = sh_bases_;
        prep.u[3] = (has_sh ? k_flag_sh : 0) |
                    (has_scales ? k_flag_scales : 0) |
                    (settings.need_depth ? k_flag_geometry : 0) |
                    (device_model_bound_ ? k_flag_raw_parameters : 0);
        prep.u[4] = camera.width;
        prep.u[5] = camera.height;
        prep.u[6] = grid_x;
        prep.u[7] = grid_y;
        set_float(prep, 8, camera.fx);
        set_float(prep, 9, camera.fy);
        set_float(prep, 10, camera.cx);
        set_float(prep, 11, camera.cy);
        set_float(prep, 12, camera.k1);
        set_float(prep, 13, camera.k2);
        set_float(prep, 14, camera.k3);
        set_float(prep, 15, camera.k4);
        prep.u[16] = camera.mode;
        set_float(prep, 17, settings.kernel_size);
        set_float(prep, 18, settings.scale_modifier);
        prep.u[19] = static_cast<std::uint32_t>(wrap_width);
        dispatch_infos(
            preprocess_,
            {model_means(), model_opacities(), model_scales(),
             model_rotations(), model_covariances(), model_colors(),
             descriptor(camera_buffer), descriptor(gauss_f), descriptor(gauss_u),
             descriptor(dummy_), descriptor(dummy_), descriptor(dummy_)},
            prep, div_up(count, 256));

        inclusive_scan(gauss_u, 0, 4 * count, count, k_streams * count);
        inclusive_scan(gauss_u, count, 5 * count, count, k_streams * count);
        Buffer& frame_counts = grow(
            frame_counts_, 2 * sizeof(std::uint32_t), BufferMemory::host_visible);
        Push counts_push = emit_constants(1, 4, grid_x, grid_y, wrap_width, count);
        dispatch(
            emit_, {&gauss_f, &gauss_u, &dummy_, &dummy_, &frame_counts,
                    &dummy_, &dummy_},
            counts_push, 1);
        write_forward_timestamp(1, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        flush_batch();
        std::uint32_t frame_count_values[2]{};
        frame_counts.download(frame_count_values, sizeof(frame_count_values));
        const std::uint32_t instances = frame_count_values[0];
        const std::uint32_t visible = frame_count_values[1];

        Buffer* instance_values = &dummy_;
        Buffer* range_lo = &dummy_;
        Buffer* range_hi = &dummy_;
        Buffer& tile_ranges = grow(tile_ranges_, static_cast<std::uint64_t>(tiles) * 2 * sizeof(std::uint32_t));
        zero_buffer(
            context_, tile_ranges,
            static_cast<std::size_t>(tiles) * 2 * sizeof(std::uint32_t));
        const bool single_sort = instances <= k_single_sort_limit;
        if (instances > 0) {
            if (visible == 0) throw std::runtime_error("Vulkan splat preprocessing produced instances without visible Gaussians");
            const auto key_bytes = static_cast<std::uint64_t>(instances) * sizeof(std::uint32_t);
            Buffer& lo0 = grow(lo0_, key_bytes);
            Buffer& lo1 = grow(lo1_, key_bytes);
            Buffer& hi0 = grow(hi0_, key_bytes);
            Buffer& hi1 = grow(hi1_, key_bytes);
            Buffer& val0 = grow(val0_, key_bytes);
            Buffer& val1 = grow(val1_, key_bytes);
            instance_values = &val0;
            range_lo = &lo0;
            range_hi = &hi0;
            const std::uint32_t hist_blocks = div_up(instances, 1024);
            const std::uint32_t hist_n = 256 * hist_blocks;
            Buffer& hist = grow(hist_space_, static_cast<std::uint64_t>(2 * hist_n + scan_scratch_uints(hist_n)) * sizeof(std::uint32_t));
            if (single_sort) {
                Push emit_push = emit_constants(count, 0, grid_x, grid_y, wrap_width, count);
                dispatch(emit_, {&gauss_f, &gauss_u, &lo0, &hi0, &val0, &dummy_, &dummy_}, emit_push, div_up(count, 256));
                radix_sort(true, instances, lo0, hi0, val0, lo1, hi1, val1, hist, hist_n, 32u + tile_bits);
            } else {
                Push depth_push = emit_constants(count, 1, grid_x, grid_y, wrap_width, count);
                dispatch(emit_, {&gauss_f, &gauss_u, &lo0, &hi0, &val0, &dummy_, &dummy_}, depth_push, div_up(count, 256));
                radix_sort(false, visible, lo0, hi0, val0, lo1, hi1, val1, hist, 256 * div_up(visible, 1024));
                Buffer& compact = grow(compact_, static_cast<std::uint64_t>(visible + scan_scratch_uints(visible)) * sizeof(std::uint32_t));
                Push gather_push = emit_constants(visible, 2, grid_x, grid_y, wrap_width, count);
                dispatch(emit_, {&gauss_f, &gauss_u, &dummy_, &dummy_, &compact, &val0, &dummy_},
                         gather_push, div_up(visible, 256));
                inclusive_scan(compact, 0, 0, visible, visible);
                Push tile_push = emit_constants(visible, 3, grid_x, grid_y, wrap_width, count);
                dispatch(emit_, {&gauss_f, &gauss_u, &lo1, &hi1, &val1, &val0, &compact},
                         tile_push, div_up(visible, 256));
                radix_sort(
                    false, instances, lo1, hi1, val1, lo0, hi0, val0,
                    hist, hist_n, tile_bits);
                instance_values = &val1;
                range_lo = &lo1;
                range_hi = &hi1;
            }
            Push range_push{};
            range_push.u[0] = instances;
            range_push.u[1] = single_sort ? 1u : 0u;
            dispatch(ranges_, {range_lo, range_hi, &tile_ranges}, range_push, div_up(instances, 256));
        }

        const std::uint32_t bucket_limit = instances == 0
            ? 0u
            : static_cast<std::uint32_t>((static_cast<std::uint64_t>(instances) + 31ull * tiles) / 32ull + 1ull);
        Buffer* bucket_offsets = &dummy_;
        if (settings.pixel_snapshots) {
            // Backward snapshots index their variable number of per-tile
            // buckets through this inclusive prefix. Generate and scan it on
            // the GPU so a training forward does not synchronize with the CPU.
            bucket_offsets = &grow(
                bucket_offsets_,
                static_cast<std::uint64_t>(tiles + scan_scratch_uints(tiles)) * sizeof(std::uint32_t));
            Push bucket_push = emit_constants(tiles, 5, grid_x, grid_y, wrap_width, count);
            dispatch(
                emit_, {&dummy_, &dummy_, &tile_ranges, &dummy_, bucket_offsets, &dummy_, &dummy_},
                bucket_push, div_up(tiles, 256));
            inclusive_scan(*bucket_offsets, 0, 0, tiles, tiles);
        }
        write_forward_timestamp(2, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        const std::uint32_t snap_buckets = std::max(bucket_limit, 1u);
        Buffer& out_f = grow(
            out_f_, static_cast<std::uint64_t>(pixels) * (pack_rgba ? 4u : 11u) * sizeof(float));
        Buffer* out_u = pack_rgba
            ? &dummy_
            : &grow(
                  out_u_, static_cast<std::uint64_t>(count + pixels + tiles +
                                                     (settings.pixel_snapshots ? snap_buckets : 1u)) *
                              sizeof(std::uint32_t));
        // The per-pixel bucket snapshots only exist for the backward pass: a
        // forward frame neither writes nor reads them (at a million instances
        // the buffer is hundreds of megabytes of pure overhead).
        Buffer& snap = settings.pixel_snapshots
            ? grow(snap_, static_cast<std::uint64_t>(snap_buckets) * 256ull * 2ull * sizeof(float) * 4)
            : dummy_;
        if (!pack_rgba) {
            const std::size_t clear_uints = settings.pixel_snapshots
                ? static_cast<std::size_t>(count) + pixels + tiles + snap_buckets
                : count;
            zero_buffer(
                context_, *out_u,
                clear_uints * sizeof(std::uint32_t));
        }

        Push blend_push{};
        blend_push.u[0] = camera.width;
        blend_push.u[1] = camera.height;
        blend_push.u[2] = grid_x;
        blend_push.u[3] = (settings.need_depth ? k_flag_geometry : 0u) |
                          (settings.pixel_snapshots ? k_flag_snapshot : 0u) |
                          (!pack_rgba ? k_flag_stats : 0u);
        blend_push.u[4] = camera.mode;
        blend_push.u[5] = static_cast<std::uint32_t>(wrap_width);
        set_float(blend_push, 6, camera.fx);
        set_float(blend_push, 7, camera.fy);
        set_float(blend_push, 8, camera.cx);
        set_float(blend_push, 9, camera.cy);
        set_float(blend_push, 10, camera.k1);
        set_float(blend_push, 11, camera.k2);
        set_float(blend_push, 12, camera.k3);
        set_float(blend_push, 13, camera.k4);
        set_float(blend_push, 14, settings.background[0]);
        set_float(blend_push, 15, settings.background[1]);
        set_float(blend_push, 16, settings.background[2]);
        blend_push.u[17] = count;
        blend_push.u[18] = pixels;
        blend_push.u[19] = snap_buckets;
        dispatch(blend_, {&tile_ranges, instance_values, &gauss_f, &out_f, out_u, bucket_offsets, &snap},
                 blend_push, grid_x, grid_y);
        if (pack_rgba) {
            Buffer& rgba = grow(
                rgba_, static_cast<std::uint64_t>(pixels) * 4, BufferMemory::device_local,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            Push pack_push{};
            pack_push.u[0] = pixels;
            dispatch(pack_, {&out_f, &rgba}, pack_push, div_up(pixels, 256));
        }
        if (destination != nullptr) {
            const std::uint64_t plane_bytes = pixels * sizeof(float);
            SplatDeviceFrame source;
            source.color = {out_f.handle, 0, 3 * plane_bytes};
            source.alpha = {out_f.handle, 3 * plane_bytes, plane_bytes};
            if (settings.need_depth) {
                source.normal = {out_f.handle, 4 * plane_bytes, 3 * plane_bytes};
                source.median_depth = {out_f.handle, 7 * plane_bytes, plane_bytes};
            }
            source.radii = {
                gauss_u.handle,
                static_cast<std::uint64_t>(3) * count * sizeof(std::uint32_t),
                static_cast<std::uint64_t>(count) * sizeof(std::uint32_t)};
            source.visibility_bits = {
                out_u->handle, 0,
                static_cast<std::uint64_t>(count) * sizeof(std::uint32_t)};
            source.width = camera.width;
            source.height = camera.height;
            record_copy_frame_device(source, *destination);
        }
        write_forward_timestamp(3, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        flush_batch();
        collect_forward_timestamps();
        last_frame_has_snapshots_ = settings.pixel_snapshots && !pack_rgba;
        last_frame_has_geometry_ = settings.need_depth && !pack_rgba;
        last_instance_values_ = instance_values;
        last_width_ = camera.width;
        last_height_ = camera.height;
        last_grid_x_ = grid_x;
        last_tiles_ = tiles;
        last_pixels_ = pixels;
        last_bucket_limit_ = snap_buckets;
        last_instances_ = instances;
        last_mode_ = camera.mode;
        last_wrap_width_ = wrap_width;
        last_fx_ = camera.fx;
        last_fy_ = camera.fy;
        last_cx_ = camera.cx;
        last_cy_ = camera.cy;
        last_k1_ = camera.k1;
        last_k2_ = camera.k2;
        last_k3_ = camera.k3;
        last_k4_ = camera.k4;
        last_kernel_size_ = settings.kernel_size;
        last_scale_modifier_ = settings.scale_modifier;
        last_background_[0] = settings.background[0];
        last_background_[1] = settings.background[1];
        last_background_[2] = settings.background[2];
        return {instances, visible};
    }

    SplatForwardOutput collect(
        const SplatCamera& camera, const SplatSettings& settings, const FrameCounts counts,
        Buffer& gauss_u, Buffer& out_f, Buffer& out_u) {
        const std::scoped_lock lock(context_.dispatch_mutex);
        const std::uint32_t count = count_;
        const std::uint32_t pixels =
            static_cast<std::uint32_t>(static_cast<std::uint64_t>(camera.width) * camera.height);
        const std::uint32_t instances = counts.instances;
        const std::uint32_t visible = counts.visible;
        SplatForwardOutput output;
        output.width = camera.width;
        output.height = camera.height;
        output.instance_count = static_cast<int>(instances);
        output.visible_count = static_cast<int>(visible);
        output.color = download_vector<float>(context_, out_f, static_cast<std::size_t>(pixels) * 3);
        output.alpha = download_vector<float>(context_, out_f, pixels, static_cast<std::size_t>(pixels) * 3 * sizeof(float));
        if (settings.need_depth) {
            output.normal = download_vector<float>(context_, out_f, static_cast<std::size_t>(pixels) * 3,
                                                   static_cast<std::size_t>(pixels) * 4 * sizeof(float));
            output.median_depth = download_vector<float>(context_, out_f, pixels, static_cast<std::size_t>(pixels) * 7 * sizeof(float));
        }
        auto visibility_bits = download_vector<std::uint32_t>(context_, out_u, count);
        output.visibility.resize(count);
        for (std::uint32_t i = 0; i < count; ++i) output.visibility[i] = visibility_bits[i] == 0 ? 0.0F : 1.0F;
        auto radius_bits = download_vector<std::uint32_t>(
            context_, gauss_u, count, static_cast<std::size_t>(3 * count) * sizeof(std::uint32_t));
        output.radii.resize(count);
        for (std::uint32_t i = 0; i < count; ++i) output.radii[i] = static_cast<int>(radius_bits[i]);
        return output;
    }

public:
    SplatForwardOutput render(const SplatCamera& camera, const SplatSettings& settings) {
        const FrameCounts counts = record_frame(camera, settings, false);
        return collect(camera, settings, counts, gauss_u_, out_f_, out_u_);
    }

    SplatDeviceFrame render_device(
        const SplatCamera& camera, const SplatSettings& settings) {
        const FrameCounts counts = record_frame(camera, settings, false);
        return make_device_frame(camera, settings, counts);
    }

    SplatDeviceFrame render_device_copy(
        const SplatCamera& camera, const SplatSettings& settings,
        const SplatDeviceFrame& destination) {
        const FrameCounts counts =
            record_frame(camera, settings, false, &destination);
        return make_device_frame(camera, settings, counts);
    }

    [[nodiscard]] SplatDeviceFrame make_device_frame(
        const SplatCamera& camera, const SplatSettings& settings,
        const FrameCounts counts) const {
        const std::uint64_t pixels =
            static_cast<std::uint64_t>(camera.width) * camera.height;
        const std::uint64_t plane_bytes = pixels * sizeof(float);
        SplatDeviceFrame frame;
        frame.color = {out_f_.handle, 0, 3 * plane_bytes};
        frame.alpha = {out_f_.handle, 3 * plane_bytes, plane_bytes};
        if (settings.need_depth) {
            frame.normal = {out_f_.handle, 4 * plane_bytes, 3 * plane_bytes};
            frame.median_depth = {out_f_.handle, 7 * plane_bytes, plane_bytes};
        }
        frame.radii = {
            gauss_u_.handle,
            static_cast<std::uint64_t>(3) * count_ * sizeof(std::uint32_t),
            static_cast<std::uint64_t>(count_) * sizeof(std::uint32_t)};
        frame.visibility_bits = {
            out_u_.handle, 0,
            static_cast<std::uint64_t>(count_) * sizeof(std::uint32_t)};
        frame.width = camera.width;
        frame.height = camera.height;
        frame.instance_count = static_cast<int>(counts.instances);
        frame.visible_count = static_cast<int>(counts.visible);
        return frame;
    }

    void copy_frame_device(
        const SplatDeviceFrame& source,
        const SplatDeviceFrame& destination) {
        require(source.width == destination.width &&
                    source.height == destination.height,
                "copy_frame_device requires matching image extents");
        const std::scoped_lock lock(context_.dispatch_mutex);
        begin_batch();
        record_copy_frame_device(source, destination);
        flush_batch();
    }

    void dispatch_median_backward_info(
        const VkDescriptorBufferInfo& loss_depth, Buffer& median_state) {
        Push push{};
        push.u[0] = last_width_;
        push.u[1] = last_height_;
        push.u[2] = last_grid_x_;
        push.u[3] = last_mode_;
        push.u[4] = static_cast<std::uint32_t>(last_wrap_width_);
        push.u[5] = count_;
        push.u[6] = last_pixels_;
        set_float(push, 7, last_fx_);
        set_float(push, 8, last_fy_);
        set_float(push, 9, last_cx_);
        set_float(push, 10, last_cy_);
        set_float(push, 11, last_k1_);
        set_float(push, 12, last_k2_);
        set_float(push, 13, last_k3_);
        set_float(push, 14, last_k4_);
        dispatch_infos(
            median_backward_,
            {descriptor(tile_ranges_), descriptor(*last_instance_values_),
             descriptor(gauss_f_), descriptor(out_f_), descriptor(out_u_),
             loss_depth, descriptor(median_state)},
            push, last_grid_x_, div_up(last_height_, k_tile));
    }

    void dispatch_median_backward(Buffer& loss_depth, Buffer& median_state) {
        dispatch_median_backward_info(descriptor(loss_depth), median_state);
    }

    SplatBlendGradients backward_blend(
        const std::span<const float> dL_color, const std::span<const float> dL_alpha,
        const std::span<const float> dL_depth, const std::span<const float> dL_normal) {
        require(last_frame_has_snapshots_,
                "backward_blend requires the latest render to use pixel_snapshots=true");
        require(dL_color.size() == static_cast<std::size_t>(last_pixels_) * 3,
                "dL_color must have shape [3,H,W]");
        require(dL_alpha.size() == last_pixels_, "dL_alpha must have shape [H,W]");
        require(dL_depth.empty() || (last_frame_has_geometry_ && dL_depth.size() == last_pixels_),
                "dL_median_depth must be empty or have shape [H,W] after a geometry forward");
        require(dL_normal.empty() ||
                    (last_frame_has_geometry_ && dL_normal.size() == static_cast<std::size_t>(last_pixels_) * 3),
                "dL_normal must be empty or have shape [3,H,W] after a geometry forward");
        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& loss_color = grow(loss_color_, dL_color.size_bytes());
        Buffer& loss_alpha = grow(loss_alpha_, dL_alpha.size_bytes());
        Buffer& loss_depth = grow(loss_depth_, static_cast<std::size_t>(last_pixels_) * sizeof(float));
        Buffer& loss_normal = grow(loss_normal_, static_cast<std::size_t>(last_pixels_) * 3 * sizeof(float));
        Buffer& median_state = grow(median_state_, static_cast<std::size_t>(last_pixels_) * 2 * sizeof(float));
        const std::size_t grad_count = static_cast<std::size_t>(count_) * 18;
        Buffer& grad = grow(blend_grad_, grad_count * sizeof(float));
        context_.write_buffer(loss_color, dL_color.data(), dL_color.size_bytes());
        context_.write_buffer(loss_alpha, dL_alpha.data(), dL_alpha.size_bytes());
        if (dL_depth.empty()) zero_buffer(context_, loss_depth, static_cast<std::size_t>(last_pixels_) * sizeof(float));
        else context_.write_buffer(loss_depth, dL_depth.data(), dL_depth.size_bytes());
        if (dL_normal.empty()) zero_buffer(context_, loss_normal, static_cast<std::size_t>(last_pixels_) * 3 * sizeof(float));
        else context_.write_buffer(loss_normal, dL_normal.data(), dL_normal.size_bytes());
        zero_buffer(context_, median_state, static_cast<std::size_t>(last_pixels_) * 2 * sizeof(float));
        zero_buffer(context_, grad, grad_count * sizeof(float));

        if (last_frame_has_geometry_) dispatch_median_backward(loss_depth, median_state);

        Push push{};
        push.u[0] = last_width_;
        push.u[1] = last_height_;
        push.u[2] = last_grid_x_;
        push.u[3] = last_mode_;
        push.u[4] = static_cast<std::uint32_t>(last_wrap_width_);
        push.u[5] = count_;
        push.u[6] = last_pixels_;
        push.u[7] = last_tiles_;
        push.u[8] = last_bucket_limit_;
        set_float(push, 9, last_background_[0]);
        set_float(push, 10, last_background_[1]);
        set_float(push, 11, last_background_[2]);
        push.u[12] = last_frame_has_geometry_ ? 1u : 0u;
        push.u[13] = !dL_alpha.empty() ? 1u : 0u;
        dispatch(
            blend_backward_,
            {&tile_ranges_, last_instance_values_, &gauss_f_, &out_f_, &out_u_,
             &bucket_offsets_, &snap_, &loss_color, &loss_alpha, &loss_normal,
             &median_state, &grad},
            push, div_up(last_bucket_limit_, 8));
        flush_batch();

        const auto packed = download_vector<float>(context_, grad, grad_count);
        SplatBlendGradients output;
        output.mean2d.assign(packed.begin(), packed.begin() + static_cast<std::ptrdiff_t>(count_) * 3);
        output.conic_opacity.assign(
            packed.begin() + static_cast<std::ptrdiff_t>(count_) * 3,
            packed.begin() + static_cast<std::ptrdiff_t>(count_) * 7);
        output.colors.assign(
            packed.begin() + static_cast<std::ptrdiff_t>(count_) * 7,
            packed.begin() + static_cast<std::ptrdiff_t>(count_) * 10);
        output.ray_plane.assign(
            packed.begin() + static_cast<std::ptrdiff_t>(count_) * 10,
            packed.begin() + static_cast<std::ptrdiff_t>(count_) * 14);
        output.normal.assign(
            packed.begin() + static_cast<std::ptrdiff_t>(count_) * 14,
            packed.begin() + static_cast<std::ptrdiff_t>(count_) * 17);
        return output;
    }

    SplatModelGradients backward(
        const std::span<const float> dL_color, const std::span<const float> dL_alpha,
        const std::span<const float> dL_depth, const std::span<const float> dL_normal) {
        // This also leaves the packed blend gradients resident for the model
        // pass below. Keeping this path simple makes the public CPU API useful
        // for parity tests; the device-resident training API can fuse the two
        // submissions without changing either shader.
        (void)backward_blend(dL_color, dL_alpha, dL_depth, dL_normal);
        const std::scoped_lock lock(context_.dispatch_mutex);
        const std::size_t feature_count = has_sh_
            ? static_cast<std::size_t>(count_) * sh_bases_ * 3
            : static_cast<std::size_t>(count_) * 3;
        const std::size_t total_count = feature_count + static_cast<std::size_t>(count_) * 26;
        Buffer& gradient = grow(model_grad_, total_count * sizeof(float));

        Push push{};
        push.u[0] = count_;
        push.u[1] = (has_sh_ ? 1u : 0u) | (has_scales_ ? 2u : 0u) |
                    (raw_chain_ ? 4u : 0u);
        push.u[2] = last_mode_;
        push.u[3] = last_width_;
        push.u[4] = last_height_;
        push.u[5] = sh_degree_;
        push.u[6] = sh_bases_;
        set_float(push, 7, last_fx_);
        set_float(push, 8, last_fy_);
        set_float(push, 9, last_cx_);
        set_float(push, 10, last_cy_);
        set_float(push, 11, last_k1_);
        set_float(push, 12, last_k2_);
        set_float(push, 13, last_k3_);
        set_float(push, 14, last_k4_);
        set_float(push, 15, last_kernel_size_);
        set_float(push, 16, last_scale_modifier_);
        dispatch_infos(
            project_backward_,
            {model_means(), model_opacities(), model_scales(),
             model_rotations(), model_covariances(), model_colors(),
             descriptor(camera_), descriptor(gauss_f_), descriptor(gauss_u_),
             descriptor(blend_grad_), model_log_scales(),
             model_raw_rotations(), model_opacity_logits(), model_filter_3d(),
             descriptor(gradient)},
            push, div_up(count_, 256));
        flush_batch();

        const auto packed = download_vector<float>(context_, gradient, total_count);
        SplatModelGradients output;
        std::size_t offset = 0;
        const auto take = [&](std::vector<float>& destination, const std::size_t size) {
            destination.assign(packed.begin() + static_cast<std::ptrdiff_t>(offset),
                               packed.begin() + static_cast<std::ptrdiff_t>(offset + size));
            offset += size;
        };
        take(output.means, static_cast<std::size_t>(count_) * 3);
        if (has_sh_) take(output.sh, feature_count);
        else take(output.colors, feature_count);
        take(output.opacities, count_);
        if (has_scales_) {
            take(output.scales, static_cast<std::size_t>(count_) * 3);
            take(output.rotations, static_cast<std::size_t>(count_) * 4);
            offset += static_cast<std::size_t>(count_) * 6;
        } else {
            offset += static_cast<std::size_t>(count_) * 7;
            take(output.covariances, static_cast<std::size_t>(count_) * 6);
        }
        if (raw_chain_) {
            take(output.log_scales, static_cast<std::size_t>(count_) * 3);
            take(output.raw_rotations, static_cast<std::size_t>(count_) * 4);
            take(output.opacity_logits, count_);
        } else {
            offset += static_cast<std::size_t>(count_) * 8;
        }
        take(output.refine_weight, count_);
        return output;
    }

    void backward_blend_device(
        const SplatBufferView& dL_color, const SplatBufferView& dL_alpha,
        const SplatBufferView& packed_gradients,
        const SplatBufferView& dL_depth, const SplatBufferView& dL_normal) {
        require(last_frame_has_snapshots_,
                "backward_blend_device requires the latest render to use pixel_snapshots=true");
        const std::uint64_t color_bytes = static_cast<std::uint64_t>(last_pixels_) * 3 * sizeof(float);
        const std::uint64_t alpha_bytes = static_cast<std::uint64_t>(last_pixels_) * sizeof(float);
        const std::uint64_t depth_bytes = static_cast<std::uint64_t>(last_pixels_) * sizeof(float);
        const std::uint64_t normal_bytes = static_cast<std::uint64_t>(last_pixels_) * 3 * sizeof(float);
        const std::uint64_t gradient_bytes = static_cast<std::uint64_t>(count_) * 18 * sizeof(float);
        require(dL_color.buffer != VK_NULL_HANDLE && dL_color.bytes >= color_bytes,
                "device dL_color is too small");
        require(dL_alpha.buffer == VK_NULL_HANDLE || dL_alpha.bytes >= alpha_bytes,
                "device dL_alpha is too small");
        require(packed_gradients.buffer != VK_NULL_HANDLE && packed_gradients.bytes >= gradient_bytes,
                "device packed_gradients is too small");
        require(dL_depth.buffer == VK_NULL_HANDLE ||
                    (last_frame_has_geometry_ && dL_depth.bytes >= depth_bytes),
                "device dL_median_depth is too small or the forward had no geometry");
        require(dL_normal.buffer == VK_NULL_HANDLE ||
                    (last_frame_has_geometry_ && dL_normal.bytes >= normal_bytes),
                "device dL_normal is too small or the forward had no geometry");
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context_.physical_device, &properties);
        const std::uint64_t alignment = std::max<std::uint64_t>(
            4, properties.limits.minStorageBufferOffsetAlignment);
        require(dL_color.offset % alignment == 0 && dL_alpha.offset % alignment == 0 &&
                    packed_gradients.offset % alignment == 0 &&
                    (dL_depth.buffer == VK_NULL_HANDLE || dL_depth.offset % alignment == 0) &&
                    (dL_normal.buffer == VK_NULL_HANDLE || dL_normal.offset % alignment == 0),
                "device buffer offsets do not satisfy minStorageBufferOffsetAlignment");

        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& zero_depth = grow(loss_depth_, depth_bytes);
        Buffer& zero_normal = grow(loss_normal_, normal_bytes);
        Buffer& median_state = grow(median_state_, depth_bytes * 2);
        if (dL_depth.buffer == VK_NULL_HANDLE) zero_buffer(context_, zero_depth, depth_bytes);
        if (dL_normal.buffer == VK_NULL_HANDLE) zero_buffer(context_, zero_normal, normal_bytes);
        zero_buffer(context_, median_state, depth_bytes * 2);
        const VkDescriptorBufferInfo depth_info = dL_depth.buffer == VK_NULL_HANDLE
            ? descriptor(zero_depth)
            : VkDescriptorBufferInfo{dL_depth.buffer, dL_depth.offset, depth_bytes};
        const VkDescriptorBufferInfo normal_info = dL_normal.buffer == VK_NULL_HANDLE
            ? descriptor(zero_normal)
            : VkDescriptorBufferInfo{dL_normal.buffer, dL_normal.offset, normal_bytes};
        if (last_frame_has_geometry_) dispatch_median_backward_info(depth_info, median_state);
        Push clear_push{};
        clear_push.u[0] = static_cast<std::uint32_t>(gradient_bytes / sizeof(float));
        dispatch_infos(
            clear_, {{packed_gradients.buffer, packed_gradients.offset, gradient_bytes}},
            clear_push, div_up(clear_push.u[0], 256));

        Push push{};
        push.u[0] = last_width_;
        push.u[1] = last_height_;
        push.u[2] = last_grid_x_;
        push.u[3] = last_mode_;
        push.u[4] = static_cast<std::uint32_t>(last_wrap_width_);
        push.u[5] = count_;
        push.u[6] = last_pixels_;
        push.u[7] = last_tiles_;
        push.u[8] = last_bucket_limit_;
        set_float(push, 9, last_background_[0]);
        set_float(push, 10, last_background_[1]);
        set_float(push, 11, last_background_[2]);
        push.u[12] = last_frame_has_geometry_ ? 1u : 0u;
        push.u[13] = dL_alpha.buffer != VK_NULL_HANDLE ? 1u : 0u;
        std::vector<VkDescriptorBufferInfo> infos{
            descriptor(tile_ranges_), descriptor(*last_instance_values_), descriptor(gauss_f_),
            descriptor(out_f_), descriptor(out_u_), descriptor(bucket_offsets_), descriptor(snap_),
            {dL_color.buffer, dL_color.offset, color_bytes},
            {dL_alpha.buffer, dL_alpha.offset, alpha_bytes},
            normal_info, descriptor(median_state),
            {packed_gradients.buffer, packed_gradients.offset, gradient_bytes}};
        dispatch_infos(
            last_frame_has_geometry_ ? blend_backward_
                                     : (subgroup_backward_supported_ && subgroup_size_ == 32u
                                            ? blend_backward_no_geometry_subgroup_
                                            : blend_backward_no_geometry_),
            infos, push, div_up(last_bucket_limit_, 8));
        flush_batch();
    }

    std::uint64_t blend_gradient_float_count() const noexcept {
        return static_cast<std::uint64_t>(count_) * 18;
    }

    std::uint64_t model_gradient_float_count() const noexcept {
        const std::uint64_t feature_count = has_sh_
            ? static_cast<std::uint64_t>(count_) * sh_bases_ * 3
            : static_cast<std::uint64_t>(count_) * 3;
        return feature_count + static_cast<std::uint64_t>(count_) * 26;
    }

    SplatDepthSamples sample_depth(
        const std::span<const float> world_points, const SplatCamera& camera,
        SplatSettings settings) {
        require(!world_points.empty() && world_points.size() % 3 == 0,
                "sample_depth world_points must have shape [P,3]");
        settings.need_depth = true;
        settings.pixel_snapshots = false;
        (void)record_frame(camera, settings, false);
        const auto point_count = static_cast<std::uint32_t>(world_points.size() / 3);
        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& points = grow(sample_points_, world_points.size_bytes());
        Buffer& output_f = grow(
            sample_f_, static_cast<std::uint64_t>(point_count) * 4 * sizeof(float));
        Buffer& output_u = grow(
            sample_u_, static_cast<std::uint64_t>(point_count) * 2 * sizeof(std::uint32_t));
        context_.write_buffer(points, world_points.data(), world_points.size_bytes());
        zero_buffer(context_, output_f, static_cast<std::size_t>(point_count) * 4 * sizeof(float));
        zero_buffer(context_, output_u, static_cast<std::size_t>(point_count) * 2 * sizeof(std::uint32_t));
        Push push{};
        push.u[0] = point_count;
        push.u[1] = last_width_;
        push.u[2] = last_height_;
        push.u[3] = last_grid_x_;
        push.u[4] = last_mode_;
        push.u[5] = count_;
        set_float(push, 6, last_fx_); set_float(push, 7, last_fy_);
        set_float(push, 8, last_cx_); set_float(push, 9, last_cy_);
        set_float(push, 10, last_k1_); set_float(push, 11, last_k2_);
        set_float(push, 12, last_k3_); set_float(push, 13, last_k4_);
        set_float(push, 14, settings.point_depth_bracket);
        set_float(push, 15, settings.point_depth_tolerance);
        dispatch(
            sample_depth_,
            {&tile_ranges_, last_instance_values_, &gauss_f_, &camera_, &points,
             &output_f, &output_u},
            push, div_up(point_count, 256));
        flush_batch();
        const auto packed_f = download_vector<float>(
            context_, output_f, static_cast<std::size_t>(point_count) * 4);
        const auto packed_u = download_vector<std::uint32_t>(
            context_, output_u, static_cast<std::size_t>(point_count) * 2);
        SplatDepthSamples output;
        output.camera_points.assign(
            packed_f.begin(), packed_f.begin() + static_cast<std::ptrdiff_t>(point_count) * 3);
        output.median_depth.assign(
            packed_f.begin() + static_cast<std::ptrdiff_t>(point_count) * 3, packed_f.end());
        output.n_contrib.assign(
            packed_u.begin(), packed_u.begin() + static_cast<std::ptrdiff_t>(point_count));
        output.inside.assign(
            packed_u.begin() + static_cast<std::ptrdiff_t>(point_count), packed_u.end());
        sample_point_count_ = point_count;
        sample_live_ = true;
        return output;
    }

    SplatDepthSampleGradients sample_depth_backward(
        const std::span<const float> dL_camera_points) {
        require(sample_live_, "sample_depth_backward requires a live sample_depth result");
        require(dL_camera_points.size() == static_cast<std::size_t>(sample_point_count_) * 3,
                "dL_camera_points must have shape [P,3]");
        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& loss = grow(sample_loss_, dL_camera_points.size_bytes());
        Buffer& point_gradient = grow(
            sample_point_grad_, static_cast<std::uint64_t>(sample_point_count_) * 3 * sizeof(float));
        Buffer& blend_gradient = grow(
            blend_grad_, blend_gradient_float_count() * sizeof(float));
        context_.write_buffer(loss, dL_camera_points.data(), dL_camera_points.size_bytes());
        zero_buffer(context_, point_gradient,
                    static_cast<std::size_t>(sample_point_count_) * 3 * sizeof(float));
        zero_buffer(context_, blend_gradient,
                    static_cast<std::size_t>(blend_gradient_float_count()) * sizeof(float));
        Push sample_push{};
        sample_push.u[0] = sample_point_count_;
        sample_push.u[1] = last_width_; sample_push.u[2] = last_height_;
        sample_push.u[3] = last_grid_x_; sample_push.u[4] = last_mode_;
        sample_push.u[5] = count_;
        set_float(sample_push, 6, last_fx_); set_float(sample_push, 7, last_fy_);
        set_float(sample_push, 8, last_cx_); set_float(sample_push, 9, last_cy_);
        set_float(sample_push, 10, last_k1_); set_float(sample_push, 11, last_k2_);
        set_float(sample_push, 12, last_k3_); set_float(sample_push, 13, last_k4_);
        dispatch(
            sample_depth_backward_,
            {&tile_ranges_, last_instance_values_, &gauss_f_, &camera_, &sample_points_,
             &sample_f_, &sample_u_, &loss, &blend_gradient, &point_gradient},
            sample_push, div_up(sample_point_count_, 256));

        const std::size_t total_count = static_cast<std::size_t>(model_gradient_float_count());
        Buffer& gradient = grow(model_grad_, total_count * sizeof(float));
        Push project_push{};
        project_push.u[0] = count_;
        project_push.u[1] = (has_sh_ ? 1u : 0u) | (has_scales_ ? 2u : 0u) |
                            (raw_chain_ ? 4u : 0u);
        project_push.u[2] = last_mode_; project_push.u[3] = last_width_;
        project_push.u[4] = last_height_; project_push.u[5] = sh_degree_;
        project_push.u[6] = sh_bases_;
        set_float(project_push, 7, last_fx_); set_float(project_push, 8, last_fy_);
        set_float(project_push, 9, last_cx_); set_float(project_push, 10, last_cy_);
        set_float(project_push, 11, last_k1_); set_float(project_push, 12, last_k2_);
        set_float(project_push, 13, last_k3_); set_float(project_push, 14, last_k4_);
        set_float(project_push, 15, last_kernel_size_);
        set_float(project_push, 16, last_scale_modifier_);
        dispatch_infos(
            project_backward_,
            {model_means(), model_opacities(), model_scales(),
             model_rotations(), model_covariances(), model_colors(),
             descriptor(camera_), descriptor(gauss_f_), descriptor(gauss_u_),
             descriptor(blend_gradient), model_log_scales(),
             model_raw_rotations(), model_opacity_logits(), model_filter_3d(),
             descriptor(gradient)},
            project_push, div_up(count_, 256));
        flush_batch();

        const auto packed = download_vector<float>(context_, gradient, total_count);
        SplatDepthSampleGradients result;
        result.points = download_vector<float>(
            context_, point_gradient, static_cast<std::size_t>(sample_point_count_) * 3);
        std::size_t offset = 0;
        const auto take = [&](std::vector<float>& destination, const std::size_t size) {
            destination.assign(packed.begin() + static_cast<std::ptrdiff_t>(offset),
                               packed.begin() + static_cast<std::ptrdiff_t>(offset + size));
            offset += size;
        };
        take(result.model.means, static_cast<std::size_t>(count_) * 3);
        const std::size_t feature_count = has_sh_
            ? static_cast<std::size_t>(count_) * sh_bases_ * 3
            : static_cast<std::size_t>(count_) * 3;
        if (has_sh_) take(result.model.sh, feature_count);
        else take(result.model.colors, feature_count);
        take(result.model.opacities, count_);
        if (has_scales_) {
            take(result.model.scales, static_cast<std::size_t>(count_) * 3);
            take(result.model.rotations, static_cast<std::size_t>(count_) * 4);
            offset += static_cast<std::size_t>(count_) * 6;
        } else {
            offset += static_cast<std::size_t>(count_) * 7;
            take(result.model.covariances, static_cast<std::size_t>(count_) * 6);
        }
        if (raw_chain_) {
            take(result.model.log_scales, static_cast<std::size_t>(count_) * 3);
            take(result.model.raw_rotations, static_cast<std::size_t>(count_) * 4);
            take(result.model.opacity_logits, count_);
        }
        sample_live_ = false;
        return result;
    }

    SplatMultiViewOutput multi_view_loss(const SplatMultiViewInput& input) {
        const auto& reference = input.reference_camera;
        const auto& neighbour = input.neighbour_camera;
        require(reference.mode == 0 && neighbour.mode == 0,
                "multi_view_loss matches the CUDA pinhole-only training path");
        require(reference.width > 0 && reference.height > 0 &&
                    neighbour.width > 0 && neighbour.height > 0,
                "multi_view_loss camera dimensions must be positive");
        require(reference.world_to_camera.size() == 16 &&
                    neighbour.world_to_camera.size() == 16,
                "multi_view_loss camera transforms must contain 16 floats");
        const std::size_t pixels =
            static_cast<std::size_t>(reference.width) * reference.height;
        const std::size_t neighbour_pixels =
            static_cast<std::size_t>(neighbour.width) * neighbour.height;
        require(input.reference_depth.size() == pixels,
                "multi_view_loss reference_depth must have shape [H,W]");
        require(input.reference_normal.size() == 3 * pixels,
                "multi_view_loss reference_normal must have shape [3,H,W]");
        require(input.sampled_neighbour_points.size() == 3 * pixels,
                "multi_view_loss sampled_neighbour_points must have shape [H,W,3]");
        require(input.sampled_inside.size() == pixels,
                "multi_view_loss sampled_inside must have shape [H,W]");
        const bool enable_ncc = input.ncc_weight > 0.0F;
        require(!enable_ncc || (input.reference_gray.size() == pixels &&
                    input.neighbour_gray.size() == neighbour_pixels),
                "multi_view_loss NCC images have the wrong shape");
        require(input.reference_mask.empty() || input.reference_mask.size() == pixels,
                "multi_view_loss reference_mask has the wrong shape");
        require(input.neighbour_mask.empty() ||
                    input.neighbour_mask.size() == neighbour_pixels,
                "multi_view_loss neighbour_mask has the wrong shape");
        require(input.geometry_weight >= 0.0F && input.ncc_weight >= 0.0F,
                "multi_view_loss weights must be non-negative");

        float rr[9]{};
        float rn[9]{};
        float tr[3]{};
        float tn[3]{};
        for (int row = 0; row < 3; ++row) {
            tr[row] = reference.world_to_camera[12 + row];
            tn[row] = neighbour.world_to_camera[12 + row];
            for (int column = 0; column < 3; ++column) {
                rr[3 * row + column] = reference.world_to_camera[4 * column + row];
                rn[3 * row + column] = neighbour.world_to_camera[4 * column + row];
            }
        }
        float transform[12]{};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                for (int k = 0; k < 3; ++k)
                    transform[3 * row + column] +=
                        rn[3 * row + k] * rr[3 * column + k];
            }
            transform[9 + row] = tn[row];
            for (int column = 0; column < 3; ++column)
                transform[9 + row] -=
                    transform[3 * row + column] * tr[column];
        }

        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& depth = grow(multi_depth_, pixels * sizeof(float));
        Buffer& normal = grow(multi_normal_, 3 * pixels * sizeof(float));
        Buffer& reference_gray = grow(multi_reference_gray_, pixels * sizeof(float));
        Buffer& sampled = grow(multi_sampled_, 3 * pixels * sizeof(float));
        Buffer& inside = grow(multi_inside_, pixels * sizeof(std::uint32_t));
        Buffer& neighbour_gray = grow(
            multi_neighbour_gray_, neighbour_pixels * sizeof(float));
        Buffer& transform_buffer = grow(multi_transform_, sizeof(transform));
        Buffer& reference_mask = grow(multi_reference_mask_, pixels * sizeof(float));
        Buffer& neighbour_mask = grow(
            multi_neighbour_mask_, neighbour_pixels * sizeof(float));
        const std::size_t output_count = 7 * pixels + 5;
        Buffer& output = grow(multi_output_, output_count * sizeof(float));
        context_.write_buffer(depth, input.reference_depth.data(), input.reference_depth.size_bytes());
        context_.write_buffer(normal, input.reference_normal.data(), input.reference_normal.size_bytes());
        context_.write_buffer(sampled, input.sampled_neighbour_points.data(),
                              input.sampled_neighbour_points.size_bytes());
        context_.write_buffer(inside, input.sampled_inside.data(), input.sampled_inside.size_bytes());
        context_.write_buffer(transform_buffer, transform, sizeof(transform));
        if (enable_ncc) {
            context_.write_buffer(reference_gray, input.reference_gray.data(),
                                  input.reference_gray.size_bytes());
            context_.write_buffer(neighbour_gray, input.neighbour_gray.data(),
                                  input.neighbour_gray.size_bytes());
        }
        if (!input.reference_mask.empty())
            context_.write_buffer(reference_mask, input.reference_mask.data(),
                                  input.reference_mask.size_bytes());
        if (!input.neighbour_mask.empty())
            context_.write_buffer(neighbour_mask, input.neighbour_mask.data(),
                                  input.neighbour_mask.size_bytes());
        zero_buffer(context_, output, output_count * sizeof(float));

        Push push{};
        push.u[0] = reference.width;
        push.u[1] = reference.height;
        push.u[2] = neighbour.width;
        push.u[3] = neighbour.height;
        set_float(push, 4, reference.fx); set_float(push, 5, reference.fy);
        set_float(push, 6, reference.cx); set_float(push, 7, reference.cy);
        set_float(push, 8, neighbour.fx); set_float(push, 9, neighbour.fy);
        set_float(push, 10, neighbour.cx); set_float(push, 11, neighbour.cy);
        set_float(push, 12, input.pixel_noise_threshold);
        push.u[13] = (!input.reference_mask.empty() ? 1u : 0u) |
                     (!input.neighbour_mask.empty() ? 2u : 0u) |
                     (input.robust_ncc ? 4u : 0u) |
                     (enable_ncc ? 8u : 0u);
        set_float(push, 14, input.geometry_weight);
        set_float(push, 15, input.ncc_weight);
        set_float(push, 16, input.ncc_lambda_reference);
        set_float(push, 17, input.ncc_sharpness);
        set_float(push, 18, std::clamp(input.ncc_min_weight, 0.0F, 1.0F));
        const std::vector<Buffer*> buffers{
            &depth, &normal, &reference_gray, &sampled, &inside,
            &neighbour_gray, &transform_buffer, &reference_mask,
            &neighbour_mask, &output};
        dispatch(multi_view_, buffers, push, div_up(reference.width, 16),
                 div_up(reference.height, 16));
        push.u[13] |= 0x80000000u;
        dispatch(multi_view_, buffers, push, div_up(reference.width, 16),
                 div_up(reference.height, 16));
        flush_batch();

        const auto packed = download_vector<float>(context_, output, output_count);
        SplatMultiViewOutput result;
        result.reference_depth_gradient.assign(
            packed.begin(), packed.begin() + static_cast<std::ptrdiff_t>(pixels));
        result.reference_normal_gradient.assign(
            packed.begin() + static_cast<std::ptrdiff_t>(pixels),
            packed.begin() + static_cast<std::ptrdiff_t>(4 * pixels));
        result.sampled_point_gradient.assign(
            packed.begin() + static_cast<std::ptrdiff_t>(4 * pixels),
            packed.begin() + static_cast<std::ptrdiff_t>(7 * pixels));
        const float* terms = packed.data() + 7 * pixels;
        result.geometry_pixels = static_cast<std::uint64_t>(terms[1]);
        result.geometry_candidates = static_cast<std::uint64_t>(terms[4]);
        result.ncc_pixels = static_cast<std::uint64_t>(terms[3]);
        result.geometry = result.geometry_pixels != 0
            ? terms[0] / terms[1] : 0.0F;
        result.ncc = result.ncc_pixels != 0 ? terms[2] / terms[3] : 0.0F;
        return result;
    }

    SplatDevicePhotometricOutput dispatch_photometric_locked(
        const VkDescriptorBufferInfo& prediction,
        const VkDescriptorBufferInfo& target,
        const VkDescriptorBufferInfo& mask,
        const bool mask_enabled,
        const std::uint32_t width, const std::uint32_t height,
        const float ssim_weight, const float photometric_weight) {
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        const std::size_t count = 3 * pixels;
        Buffer& work0 = grow(ssim_work0_, count * sizeof(float));
        Buffer& work1 = grow(ssim_work1_, count * sizeof(float));
        Buffer& work2 = grow(ssim_work2_, count * sizeof(float));
        Buffer& work3 = grow(ssim_work3_, count * sizeof(float));
        Buffer& work4 = grow(ssim_work4_, count * sizeof(float));
        Buffer& dmu = grow(ssim_dmu_, count * sizeof(float));
        Buffer& dvariance = grow(ssim_dvariance_, count * sizeof(float));
        Buffer& dcovariance = grow(ssim_dcovariance_, count * sizeof(float));
        Buffer& output = grow(ssim_output_, 2 * count * sizeof(float));
        const std::uint32_t reduction_groups =
            div_up(static_cast<std::uint32_t>(count), 256);
        Buffer& reduction = grow(
            ssim_reduction_, static_cast<std::size_t>(reduction_groups) * sizeof(float));
        // The loss output is consumed by the following compute dispatches, so
        // record its clear in their batch instead of paying a standalone
        // transfer submission and queue wait every training step.
        begin_batch();
        const VkDeviceSize output_bytes = 2 * count * sizeof(float);
        vkCmdFillBuffer(command_, output.handle, 0, output_bytes, 0u);
        VkBufferMemoryBarrier output_fill_barrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        output_fill_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        output_fill_barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        output_fill_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_fill_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        output_fill_barrier.buffer = output.handle;
        output_fill_barrier.offset = 0;
        output_fill_barrier.size = output_bytes;
        vkCmdPipelineBarrier(
            command_, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
            &output_fill_barrier, 0, nullptr);

        Push push{};
        push.u[0] = width;
        push.u[1] = height;
        set_float(push, 2, ssim_weight);
        push.u[3] = mask_enabled ? 1u : 0u;
        const float normalization = photometric_weight /
            static_cast<float>(3U * (width - 10U) * (height - 10U));
        set_float(push, 4, normalization);
        push.u[5] = 0u;
        const std::vector<VkDescriptorBufferInfo> buffers{
            prediction, target, mask,
            descriptor(work0), descriptor(work1), descriptor(work2),
            descriptor(work3), descriptor(work4), descriptor(dmu),
            descriptor(dvariance), descriptor(dcovariance), descriptor(output)};
        dispatch_infos(
            ssim_, buffers, push, div_up(width, 16), div_up(height, 16), 3);
        push.u[5] = 1u;
        dispatch_infos(
            ssim_, buffers, push, div_up(width, 16), div_up(height, 16), 3);
        push.u[5] = 2u;
        dispatch_infos(
            ssim_, buffers, push, div_up(width, 16), div_up(height, 16), 3);
        push.u[5] = 3u;
        dispatch_infos(
            ssim_, buffers, push, div_up(width, 16), div_up(height, 16), 3);

        Push reduce_push{};
        reduce_push.u[0] = width;
        reduce_push.u[1] = height;
        reduce_push.u[2] = 0u;
        reduce_push.u[3] = reduction_groups;
        set_float(reduce_push, 4, normalization);
        const std::vector<VkDescriptorBufferInfo> reduce_buffers{
            descriptor(output), descriptor(reduction)};
        dispatch_infos(loss_reduce_, reduce_buffers, reduce_push, reduction_groups);
        reduce_push.u[2] = 1u;
        dispatch_infos(loss_reduce_, reduce_buffers, reduce_push, 1);

        return {
            {output.handle, 0, count * sizeof(float)},
            {output.handle, count * sizeof(float), count * sizeof(float)},
            {reduction.handle, 0, sizeof(float)}};
    }

    SplatPhotometricOutput photometric_loss(
        const std::span<const float> prediction,
        const std::span<const float> target,
        const std::uint32_t width, const std::uint32_t height,
        const float ssim_weight, const float photometric_weight,
        const std::span<const float> mask) {
        require(width > 10 && height > 10,
                "photometric_loss requires width and height > 10");
        const std::size_t pixels = static_cast<std::size_t>(width) * height;
        const std::size_t count = 3 * pixels;
        require(prediction.size() == count && target.size() == count,
                "photometric_loss images must have shape [3,H,W]");
        require(mask.empty() || mask.size() == pixels,
                "photometric_loss mask must have shape [H,W]");
        require(ssim_weight >= 0.0F && ssim_weight <= 1.0F &&
                    photometric_weight >= 0.0F,
                "photometric_loss weights are out of range");
        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& prediction_buffer = grow(ssim_prediction_, count * sizeof(float));
        Buffer& target_buffer = grow(ssim_target_, count * sizeof(float));
        Buffer& mask_buffer = grow(ssim_mask_, pixels * sizeof(float));
        context_.write_buffer(prediction_buffer, prediction.data(), prediction.size_bytes());
        context_.write_buffer(target_buffer, target.data(), target.size_bytes());
        if (!mask.empty())
            context_.write_buffer(mask_buffer, mask.data(), mask.size_bytes());
        (void)dispatch_photometric_locked(
            descriptor(prediction_buffer), descriptor(target_buffer),
            mask.empty() ? descriptor(dummy_) : descriptor(mask_buffer),
            !mask.empty(), width, height, ssim_weight, photometric_weight);
        flush_batch();
        SplatPhotometricOutput result;
        result.gradient = download_vector<float>(context_, ssim_output_, count);
        context_.read_buffer(ssim_reduction_, &result.loss, sizeof(float));
        return result;
    }

    SplatDevicePhotometricOutput fused_l1_ssim_device(
        const SplatBufferView& prediction, const SplatBufferView& target,
        const std::uint32_t width, const std::uint32_t height,
        const float ssim_weight, const float photometric_weight,
        const SplatBufferView& mask) {
        require(width > 10 && height > 10,
                "fused_l1_ssim_device requires width and height > 10");
        const std::uint64_t pixels = static_cast<std::uint64_t>(width) * height;
        const std::uint64_t image_bytes = 3 * pixels * sizeof(float);
        const std::uint64_t mask_bytes = pixels * sizeof(float);
        require(prediction.buffer != VK_NULL_HANDLE && prediction.bytes >= image_bytes,
                "device prediction is too small");
        require(target.buffer != VK_NULL_HANDLE && target.bytes >= image_bytes,
                "device target is too small");
        require(mask.buffer == VK_NULL_HANDLE || mask.bytes >= mask_bytes,
                "device mask is too small");
        require(ssim_weight >= 0.0F && ssim_weight <= 1.0F &&
                    photometric_weight >= 0.0F,
                "photometric_loss weights are out of range");
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context_.physical_device, &properties);
        const std::uint64_t alignment = std::max<std::uint64_t>(
            4, properties.limits.minStorageBufferOffsetAlignment);
        require(prediction.offset % alignment == 0 && target.offset % alignment == 0 &&
                    (mask.buffer == VK_NULL_HANDLE || mask.offset % alignment == 0),
                "device buffer offsets do not satisfy minStorageBufferOffsetAlignment");
        const std::scoped_lock lock(context_.dispatch_mutex);
        const auto result = dispatch_photometric_locked(
            {prediction.buffer, prediction.offset, image_bytes},
            {target.buffer, target.offset, image_bytes},
            mask.buffer == VK_NULL_HANDLE
                ? descriptor(dummy_)
                : VkDescriptorBufferInfo{mask.buffer, mask.offset, mask_bytes},
            mask.buffer != VK_NULL_HANDLE, width, height,
            ssim_weight, photometric_weight);
        flush_batch();
        return result;
    }

    float read_photometric_loss(const SplatDevicePhotometricOutput& output) {
        require(output.loss_scalar.buffer == ssim_reduction_.handle &&
                    output.loss_scalar.offset == 0 &&
                    output.loss_scalar.bytes >= sizeof(float),
                "photometric output is not owned by this rasterizer");
        const std::scoped_lock lock(context_.dispatch_mutex);
        float value = 0.0F;
        context_.read_buffer(ssim_reduction_, &value, sizeof(float));
        return value;
    }

    void backward_device(
        const SplatBufferView& dL_color, const SplatBufferView& dL_alpha,
        const SplatBufferView& packed_model_gradients,
        const SplatBufferView& dL_depth, const SplatBufferView& dL_normal) {
        require(last_frame_has_snapshots_,
                "backward_device requires the latest render to use pixel_snapshots=true");
        const std::uint64_t color_bytes = static_cast<std::uint64_t>(last_pixels_) * 3 * sizeof(float);
        const std::uint64_t alpha_bytes = static_cast<std::uint64_t>(last_pixels_) * sizeof(float);
        const std::uint64_t depth_bytes = alpha_bytes;
        const std::uint64_t normal_bytes = color_bytes;
        const std::uint64_t blend_bytes = blend_gradient_float_count() * sizeof(float);
        const std::uint64_t model_bytes = model_gradient_float_count() * sizeof(float);
        require(dL_color.buffer != VK_NULL_HANDLE && dL_color.bytes >= color_bytes,
                "device dL_color is too small");
        require(dL_alpha.buffer == VK_NULL_HANDLE || dL_alpha.bytes >= alpha_bytes,
                "device dL_alpha is too small");
        require(packed_model_gradients.buffer != VK_NULL_HANDLE &&
                    packed_model_gradients.bytes >= model_bytes,
                "device packed_model_gradients is too small");
        require(dL_depth.buffer == VK_NULL_HANDLE ||
                    (last_frame_has_geometry_ && dL_depth.bytes >= depth_bytes),
                "device dL_median_depth is too small or the forward had no geometry");
        require(dL_normal.buffer == VK_NULL_HANDLE ||
                    (last_frame_has_geometry_ && dL_normal.bytes >= normal_bytes),
                "device dL_normal is too small or the forward had no geometry");
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context_.physical_device, &properties);
        const std::uint64_t alignment = std::max<std::uint64_t>(
            4, properties.limits.minStorageBufferOffsetAlignment);
        require(dL_color.offset % alignment == 0 &&
                    (dL_alpha.buffer == VK_NULL_HANDLE ||
                     dL_alpha.offset % alignment == 0) &&
                    packed_model_gradients.offset % alignment == 0 &&
                    (dL_depth.buffer == VK_NULL_HANDLE || dL_depth.offset % alignment == 0) &&
                    (dL_normal.buffer == VK_NULL_HANDLE || dL_normal.offset % alignment == 0),
                "device buffer offsets do not satisfy minStorageBufferOffsetAlignment");

        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& blend_gradient = grow(blend_grad_, blend_bytes);
        // All backward shaders use pc.u13 to avoid touching the alpha-gradient
        // descriptor when no alpha objective is active. Binding the tiny dummy
        // buffer removes an image-sized clear and its queue drain.
        Buffer& zero_depth = last_frame_has_geometry_
            ? grow(loss_depth_, depth_bytes)
            : dummy_;
        Buffer& zero_normal = last_frame_has_geometry_
            ? grow(loss_normal_, normal_bytes)
            : dummy_;
        Buffer& median_state = last_frame_has_geometry_
            ? grow(median_state_, depth_bytes * 2)
            : dummy_;
        if (last_frame_has_geometry_) {
            if (dL_depth.buffer == VK_NULL_HANDLE)
                zero_buffer(context_, zero_depth, depth_bytes);
            if (dL_normal.buffer == VK_NULL_HANDLE)
                zero_buffer(context_, zero_normal, normal_bytes);
            zero_buffer(context_, median_state, depth_bytes * 2);
        }
        const VkDescriptorBufferInfo alpha_info = dL_alpha.buffer == VK_NULL_HANDLE
            ? descriptor(dummy_)
            : VkDescriptorBufferInfo{dL_alpha.buffer, dL_alpha.offset, alpha_bytes};
        const VkDescriptorBufferInfo depth_info = dL_depth.buffer == VK_NULL_HANDLE
            ? descriptor(zero_depth)
            : VkDescriptorBufferInfo{dL_depth.buffer, dL_depth.offset, depth_bytes};
        const VkDescriptorBufferInfo normal_info = dL_normal.buffer == VK_NULL_HANDLE
            ? descriptor(zero_normal)
            : VkDescriptorBufferInfo{dL_normal.buffer, dL_normal.offset, normal_bytes};
        if (last_frame_has_geometry_) dispatch_median_backward_info(depth_info, median_state);
        begin_batch();
        write_backward_timestamp(0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        vkCmdFillBuffer(command_, blend_gradient.handle, 0, blend_bytes, 0u);
        VkBufferMemoryBarrier fill_barrier{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        fill_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        fill_barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        fill_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fill_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        fill_barrier.buffer = blend_gradient.handle;
        fill_barrier.offset = 0;
        fill_barrier.size = blend_bytes;
        vkCmdPipelineBarrier(
            command_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &fill_barrier, 0, nullptr);
        write_backward_timestamp(1, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        Push blend_push{};
        blend_push.u[0] = last_width_;
        blend_push.u[1] = last_height_;
        blend_push.u[2] = last_grid_x_;
        blend_push.u[3] = last_mode_;
        blend_push.u[4] = static_cast<std::uint32_t>(last_wrap_width_);
        blend_push.u[5] = count_;
        blend_push.u[6] = last_pixels_;
        blend_push.u[7] = last_tiles_;
        blend_push.u[8] = last_bucket_limit_;
        set_float(blend_push, 9, last_background_[0]);
        set_float(blend_push, 10, last_background_[1]);
        set_float(blend_push, 11, last_background_[2]);
        blend_push.u[12] = last_frame_has_geometry_ ? 1u : 0u;
        blend_push.u[13] = dL_alpha.buffer != VK_NULL_HANDLE ? 1u : 0u;
        std::vector<VkDescriptorBufferInfo> blend_infos{
            descriptor(tile_ranges_), descriptor(*last_instance_values_), descriptor(gauss_f_),
            descriptor(out_f_), descriptor(out_u_), descriptor(bucket_offsets_), descriptor(snap_),
            {dL_color.buffer, dL_color.offset, color_bytes},
            alpha_info, normal_info,
            descriptor(median_state), descriptor(blend_gradient)};
        const ComputePipeline& blend_pipeline = last_frame_has_geometry_
            ? blend_backward_
            : (subgroup_backward_supported_ && subgroup_size_ == 32u
                   ? blend_backward_no_geometry_subgroup_
                   : blend_backward_no_geometry_);
        dispatch_infos(
            blend_pipeline, blend_infos, blend_push,
            div_up(last_bucket_limit_, 8));
        write_backward_timestamp(2, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        Push project_push{};
        project_push.u[0] = count_;
        project_push.u[1] = (has_sh_ ? 1u : 0u) | (has_scales_ ? 2u : 0u) |
                            (raw_chain_ ? 4u : 0u);
        project_push.u[2] = last_mode_;
        project_push.u[3] = last_width_;
        project_push.u[4] = last_height_;
        project_push.u[5] = sh_degree_;
        project_push.u[6] = sh_bases_;
        set_float(project_push, 7, last_fx_);
        set_float(project_push, 8, last_fy_);
        set_float(project_push, 9, last_cx_);
        set_float(project_push, 10, last_cy_);
        set_float(project_push, 11, last_k1_);
        set_float(project_push, 12, last_k2_);
        set_float(project_push, 13, last_k3_);
        set_float(project_push, 14, last_k4_);
        set_float(project_push, 15, last_kernel_size_);
        set_float(project_push, 16, last_scale_modifier_);
        std::vector<VkDescriptorBufferInfo> project_infos{
            model_means(), model_opacities(), model_scales(), model_rotations(),
            model_covariances(), model_colors(), descriptor(camera_), descriptor(gauss_f_),
            descriptor(gauss_u_), descriptor(blend_gradient), model_log_scales(),
            model_raw_rotations(), model_opacity_logits(), model_filter_3d(),
            {packed_model_gradients.buffer, packed_model_gradients.offset, model_bytes}};
        dispatch_infos(
            project_backward_, project_infos, project_push, div_up(count_, 256));
        write_backward_timestamp(3, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        flush_batch();
        collect_backward_timestamps();
    }

    SplatRgbaImage render_rgba_device(const SplatCamera& camera, SplatSettings settings) {
        settings.need_depth = false;
        settings.pixel_snapshots = false;
        record_frame(camera, settings, true);
        SplatRgbaImage image;
        image.buffer = rgba_.handle;
        image.bytes = static_cast<std::uint64_t>(camera.width) * camera.height * 4U;
        image.width = camera.width;
        image.height = camera.height;
        return image;
    }

    void render_rgba(const SplatCamera& camera, const SplatSettings& settings, std::vector<std::uint8_t>& rgba) {
        const SplatRgbaImage image = render_rgba_device(camera, settings);
        rgba.resize(static_cast<std::size_t>(image.bytes));
        const std::scoped_lock lock(context_.dispatch_mutex);
        context_.read_buffer(rgba_, rgba.data(), rgba.size());
    }

private:
    void record_copy_frame_device(
        const SplatDeviceFrame& source,
        const SplatDeviceFrame& destination) {
        require(recording_, "frame copies require an active command batch");
        require(source.width == destination.width &&
                    source.height == destination.height,
                "copy_frame_device requires matching image extents");
        struct CopyPair {
            SplatBufferView source;
            SplatBufferView destination;
        };
        const std::array<CopyPair, 6> pairs{{
            {source.color, destination.color},
            {source.alpha, destination.alpha},
            {source.normal, destination.normal},
            {source.median_depth, destination.median_depth},
            {source.radii, destination.radii},
            {source.visibility_bits, destination.visibility_bits},
        }};

        // The final blend dispatch ends with a compute-to-compute barrier.
        // Buffer copies additionally need the three source buffers made
        // available to transfer. Scope those dependencies to the copied ranges
        // rather than flushing every compute write in the frame batch.
        const auto source_barrier = [](const SplatBufferView& source) {
            VkBufferMemoryBarrier barrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = source.buffer;
            barrier.offset = source.offset;
            barrier.size = source.bytes;
            return barrier;
        };
        std::array<VkBufferMemoryBarrier, 3> barriers{
            VkBufferMemoryBarrier{}, VkBufferMemoryBarrier{},
            VkBufferMemoryBarrier{}};
        std::uint32_t barrier_count = 0;
        SplatBufferView color_dependency = source.color;
        color_dependency.bytes = 0;
        const auto extend_color = [&](const SplatBufferView& value) {
            if (value.buffer == VK_NULL_HANDLE) return;
            color_dependency.bytes = std::max(
                color_dependency.bytes,
                value.offset + value.bytes - color_dependency.offset);
        };
        if (destination.color.buffer != VK_NULL_HANDLE)
            extend_color(source.color);
        if (destination.alpha.buffer != VK_NULL_HANDLE)
            extend_color(source.alpha);
        if (destination.normal.buffer != VK_NULL_HANDLE)
            extend_color(source.normal);
        if (destination.median_depth.buffer != VK_NULL_HANDLE)
            extend_color(source.median_depth);
        if (color_dependency.bytes != 0)
            barriers[barrier_count++] = source_barrier(color_dependency);
        if (destination.radii.buffer != VK_NULL_HANDLE)
            barriers[barrier_count++] = source_barrier(source.radii);
        if (destination.visibility_bits.buffer != VK_NULL_HANDLE)
            barriers[barrier_count++] = source_barrier(source.visibility_bits);
        vkCmdPipelineBarrier(
            command_,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, barrier_count, barriers.data(), 0, nullptr);
        for (const CopyPair& pair : pairs) {
            if (pair.destination.buffer == VK_NULL_HANDLE) continue;
            require(pair.source.buffer != VK_NULL_HANDLE &&
                        pair.destination.bytes >= pair.source.bytes,
                    "copy_frame_device destination is too small");
            VkBufferCopy region{};
            region.srcOffset = pair.source.offset;
            region.dstOffset = pair.destination.offset;
            region.size = pair.source.bytes;
            vkCmdCopyBuffer(
                command_, pair.source.buffer, pair.destination.buffer,
                1, &region);
        }
    }

    void query_subgroup_properties() {
        VkPhysicalDeviceSubgroupProperties subgroup{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup};
        vkGetPhysicalDeviceProperties2(context_.physical_device, &properties);
        subgroup_size_ = subgroup.subgroupSize;
        subgroup_backward_supported_ =
            (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0 &&
            (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
        subgroup_backward_supported_ &=
            !environment_flag("SPLAT_DRENDER_DISABLE_SUBGROUP_BACKWARD");
    }

    void create_backward_timestamp_profiler() {
        backward_profile_enabled_ = environment_flag("SPLAT_DRENDER_PROFILE_BACKWARD");
        backward_profile_interval_ = environment_interval(
            "SPLAT_DRENDER_PROFILE_BACKWARD_INTERVAL", 100);
        if (!backward_profile_enabled_) return;
        if (!create_timestamp_query_pool(
                backward_timestamp_pool_,
                "splat_vulkan_backward_profile"))
            backward_profile_enabled_ = false;
    }

    // The forward profiler follows the same contract as the backward one: the
    // timestamps split the frame into the count pass (preprocess + per-Gaussian
    // scans + the instance tally), the tile sort, and the blend.
    void create_forward_timestamp_profiler() {
        forward_profile_enabled_ = environment_flag("SPLAT_DRENDER_PROFILE_FORWARD");
        forward_profile_interval_ = environment_interval(
            "SPLAT_DRENDER_PROFILE_FORWARD_INTERVAL", 100);
        if (!forward_profile_enabled_) return;
        if (!create_timestamp_query_pool(
                forward_timestamp_pool_, "splat_vulkan_forward_profile"))
            forward_profile_enabled_ = false;
    }

    bool create_timestamp_query_pool(
        VkQueryPool& pool, const char* label) {
        if (!timestamp_properties_queried_) {
            timestamp_properties_queried_ = true;
            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(
                context_.physical_device, &family_count, nullptr);
            std::vector<VkQueueFamilyProperties> families(family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(
                context_.physical_device, &family_count, families.data());
            timestamp_valid_bits_ =
                context_.queue_family_index < families.size()
                    ? families[context_.queue_family_index].timestampValidBits
                    : 0;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context_.physical_device, &properties);
            timestamp_period_ = properties.limits.timestampPeriod;
        }
        if (timestamp_valid_bits_ == 0) {
            std::fprintf(
                stderr,
                "%s disabled: queue has timestampValidBits=0\n", label);
            return false;
        }
        if (pool != VK_NULL_HANDLE) return true;
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = 4;
        if (vkCreateQueryPool(context_.device, &info, nullptr, &pool) != VK_SUCCESS)
            throw std::runtime_error("vkCreateQueryPool failed for stage profiling");
        return true;
    }

    void write_forward_timestamp(
        const std::uint32_t index, const VkPipelineStageFlagBits stage) {
        if (!forward_profile_enabled_) return;
        begin_batch();
        if (index == 0) {
            vkCmdResetQueryPool(command_, forward_timestamp_pool_, 0, 4);
        }
        vkCmdWriteTimestamp(command_, stage, forward_timestamp_pool_, index);
    }

    void collect_forward_timestamps() {
        if (!forward_profile_enabled_) return;
        std::array<std::uint64_t, 4> timestamps{};
        if (vkGetQueryPoolResults(
                context_.device, forward_timestamp_pool_, 0, 4,
                sizeof(timestamps), timestamps.data(), sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            return;
        const double ticks_to_ms =
            static_cast<double>(timestamp_period_) / 1.0e6;
        const double counts_ms = static_cast<double>(timestamp_delta(
            timestamps[0], timestamps[1], timestamp_valid_bits_)) * ticks_to_ms;
        const double sort_ms = static_cast<double>(timestamp_delta(
            timestamps[1], timestamps[2], timestamp_valid_bits_)) * ticks_to_ms;
        const double blend_ms = static_cast<double>(timestamp_delta(
            timestamps[2], timestamps[3], timestamp_valid_bits_)) * ticks_to_ms;
        forward_counts_total_ms_ += counts_ms;
        forward_sort_total_ms_ += sort_ms;
        forward_blend_total_ms_ += blend_ms;
        ++forward_profile_samples_;
        if (forward_profile_samples_ < forward_profile_interval_) return;
        const double samples = static_cast<double>(forward_profile_samples_);
        std::fprintf(
            stderr,
            "splat_vulkan_forward_profile samples=%u gaussians=%u instances=%u "
            "counts_avg_ms=%.4f sort_avg_ms=%.4f blend_avg_ms=%.4f "
            "total_avg_ms=%.4f\n",
            forward_profile_samples_, count_, last_instances_,
            forward_counts_total_ms_ / samples, forward_sort_total_ms_ / samples,
            forward_blend_total_ms_ / samples,
            (forward_counts_total_ms_ + forward_sort_total_ms_ +
             forward_blend_total_ms_) / samples);
        forward_profile_samples_ = 0;
        forward_counts_total_ms_ = 0.0;
        forward_sort_total_ms_ = 0.0;
        forward_blend_total_ms_ = 0.0;
    }

    void write_backward_timestamp(
        const std::uint32_t index, const VkPipelineStageFlagBits stage) {
        if (!backward_profile_enabled_) return;
        if (index == 0) {
            vkCmdResetQueryPool(
                command_, backward_timestamp_pool_, 0, 4);
        }
        vkCmdWriteTimestamp(command_, stage, backward_timestamp_pool_, index);
    }

    [[nodiscard]] static std::uint64_t timestamp_delta(
        const std::uint64_t begin, const std::uint64_t end,
        const std::uint32_t valid_bits) {
        if (valid_bits >= 64) return end - begin;
        const std::uint64_t mask = (std::uint64_t{1} << valid_bits) - 1;
        const std::uint64_t masked_begin = begin & mask;
        const std::uint64_t masked_end = end & mask;
        return masked_end >= masked_begin
            ? masked_end - masked_begin
            : ((mask - masked_begin + 1) + masked_end) & mask;
    }

    void collect_backward_timestamps() {
        if (!backward_profile_enabled_) return;
        std::array<std::uint64_t, 4> timestamps{};
        const VkResult result = vkGetQueryPoolResults(
            context_.device, backward_timestamp_pool_, 0, 4,
            sizeof(timestamps), timestamps.data(), sizeof(std::uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS) {
            std::fprintf(
                stderr,
                "splat_vulkan_backward_profile query result failed: %d\n",
                static_cast<int>(result));
            return;
        }

        const double ticks_to_ms =
            static_cast<double>(timestamp_period_) / 1.0e6;
        const double fill_ms = static_cast<double>(timestamp_delta(
            timestamps[0], timestamps[1], timestamp_valid_bits_)) *
            ticks_to_ms;
        const double blend_ms = static_cast<double>(timestamp_delta(
            timestamps[1], timestamps[2], timestamp_valid_bits_)) *
            ticks_to_ms;
        const double project_ms = static_cast<double>(timestamp_delta(
            timestamps[2], timestamps[3], timestamp_valid_bits_)) *
            ticks_to_ms;
        backward_fill_total_ms_ += fill_ms;
        backward_blend_total_ms_ += blend_ms;
        backward_project_total_ms_ += project_ms;
        ++backward_profile_samples_;

        if (backward_profile_samples_ < backward_profile_interval_) return;
        const double samples = static_cast<double>(backward_profile_samples_);
        std::fprintf(
            stderr,
            "splat_vulkan_backward_profile samples=%u gaussians=%u buckets=%u "
            "pixels=%u fill_avg_ms=%.4f blend_avg_ms=%.4f project_avg_ms=%.4f "
            "total_avg_ms=%.4f\n",
            backward_profile_samples_, count_, last_bucket_limit_, last_pixels_,
            backward_fill_total_ms_ / samples,
            backward_blend_total_ms_ / samples,
            backward_project_total_ms_ / samples,
            (backward_fill_total_ms_ + backward_blend_total_ms_ +
             backward_project_total_ms_) / samples);
        backward_profile_samples_ = 0;
        backward_fill_total_ms_ = 0.0;
        backward_blend_total_ms_ = 0.0;
        backward_project_total_ms_ = 0.0;
    }

    // Every buffer the shaders touch lives in device memory: an unmapped buffer
    // keeps the GPU at VRAM bandwidth instead of running the whole forward over
    // PCIe. Only the tiny per-frame control blocks stay host visible.
    Buffer alloc(
        VkDeviceSize bytes, const BufferMemory memory = BufferMemory::device_local,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) const {
        // Device-local buffers are uploaded/read/cleared through the transfer
        // path as well as bound as storage. Declaring both capabilities keeps
        // those operations valid under Vulkan validation.
        if (memory == BufferMemory::device_local) {
            usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        }
        return context_.create_buffer(std::max<VkDeviceSize>(bytes, 4), usage, memory);
    }

    Buffer& grow(
        Buffer& buffer, VkDeviceSize bytes, const BufferMemory memory = BufferMemory::device_local,
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        if (buffer.device == VK_NULL_HANDLE || buffer.size < bytes) buffer = alloc(bytes, memory, usage);
        return buffer;
    }

    void begin_batch() {
        if (recording_) return;
        if (command_ == VK_NULL_HANDLE) {
            VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            info.commandPool = context_.command_pool;
            info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            info.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(context_.device, &info, &command_) != VK_SUCCESS) {
                throw std::runtime_error("vkAllocateCommandBuffers failed");
            }
        }
        if (vkResetCommandBuffer(command_, 0) != VK_SUCCESS) {
            throw std::runtime_error("vkResetCommandBuffer failed");
        }
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(command_, &begin) != VK_SUCCESS) {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }
        recording_ = true;
    }

    void flush_batch() {
        if (!recording_) return;
        if (vkEndCommandBuffer(command_) != VK_SUCCESS) {
            throw std::runtime_error("vkEndCommandBuffer failed");
        }
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        if (vkQueueSubmit(context_.queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS ||
            vkQueueWaitIdle(context_.queue) != VK_SUCCESS) {
            throw std::runtime_error("Vulkan splat submit failed");
        }
        if (!pending_sets_.empty()) {
            vkFreeDescriptorSets(
                context_.device, context_.descriptor_pool,
                static_cast<std::uint32_t>(pending_sets_.size()), pending_sets_.data());
            pending_sets_.clear();
        }
        recording_ = false;
    }

    void dispatch(const ComputePipeline& pipeline, const std::vector<Buffer*>& buffers, const Push& push,
                  std::uint32_t groups_x, std::uint32_t groups_y = 1,
                  std::uint32_t groups_z = 1) {
        std::vector<VkDescriptorBufferInfo> infos;
        infos.reserve(buffers.size());
        for (const Buffer* buffer : buffers) infos.push_back(descriptor(*buffer));
        dispatch_infos(pipeline, infos, push, groups_x, groups_y, groups_z);
    }

    void dispatch_infos(const ComputePipeline& pipeline, const std::vector<VkDescriptorBufferInfo>& infos,
                        const Push& push, std::uint32_t groups_x, std::uint32_t groups_y = 1,
                        std::uint32_t groups_z = 1) {
        if (groups_x == 0 || groups_y == 0 || groups_z == 0) return;
        begin_batch();
        std::vector<VkWriteDescriptorSet> writes(infos.size());
        for (std::uint32_t i = 0; i < infos.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        if (context_.push_descriptors) {
            // The bindings live in the command buffer, so a later dispatch in
            // the same batch cannot rewrite what an earlier one reads.
            context_.cmd_push_descriptor(
                command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout, 0,
                static_cast<std::uint32_t>(writes.size()), writes.data());
        } else {
            VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            set_info.descriptorPool = context_.descriptor_pool;
            set_info.descriptorSetCount = 1;
            set_info.pSetLayouts = &pipeline.descriptor_set_layout;
            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(context_.device, &set_info, &descriptor_set) != VK_SUCCESS) {
                throw std::runtime_error("vkAllocateDescriptorSets failed");
            }
            pending_sets_.push_back(descriptor_set);
            for (VkWriteDescriptorSet& write : writes) write.dstSet = descriptor_set;
            vkUpdateDescriptorSets(
                context_.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                nullptr);
            vkCmdBindDescriptorSets(
                command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout, 0, 1,
                &descriptor_set, 0, nullptr);
        }
        vkCmdPushConstants(
            command_, pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        vkCmdDispatch(command_, groups_x, groups_y, groups_z);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(
            command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    void inclusive_scan(Buffer& buffer, std::uint32_t src, std::uint32_t dst, std::uint32_t count,
                        std::uint32_t scratch) {
        if (count == 0) return;
        const std::uint32_t blocks = div_up(count, 256);
        Push push{};
        push.u[0] = count;
        push.u[1] = 0;
        push.u[2] = src;
        push.u[3] = dst;
        push.u[4] = scratch;
        dispatch(scan_, {&buffer}, push, blocks);
        if (blocks == 1) return;
        inclusive_scan(buffer, scratch, scratch + blocks, blocks, scratch + 2 * blocks);
        push.u[1] = 1;
        push.u[4] = scratch + blocks;
        dispatch(scan_, {&buffer}, push, blocks);
    }

    static Push emit_constants(std::uint32_t count, std::uint32_t mode, std::uint32_t grid_x, std::uint32_t grid_y,
                               int wrap_width, std::uint32_t gaussian_count) {
        Push push{};
        push.u[0] = count;
        push.u[1] = mode;
        push.u[2] = grid_x;
        push.u[3] = grid_y;
        push.u[4] = static_cast<std::uint32_t>(wrap_width);
        push.u[5] = gaussian_count;
        return push;
    }

    // `key_bits` is the number of meaningful bits; the packed 64-bit key only
    // has 32 depth bits plus the tile bits, and every bit above them is zero,
    // so those passes sort nothing. The count must stay even: the sort has to
    // end back in buffer 0.
    //
    // A 32-bit key follows the same rule. The two-phase instance sort keys by
    // tile id alone and relies on radix stability to keep the depth order that
    // the emitter produced, so a 13-bit tile field needs two 8-bit passes
    // instead of four; the two all-zero digit passes it used to run were pure
    // histogram, scan and scatter traffic over every instance.
    void radix_sort(bool key_is_64, std::uint32_t count, Buffer& lo0, Buffer& hi0, Buffer& val0, Buffer& lo1, Buffer& hi1,
                    Buffer& val1, Buffer& hist, std::uint32_t hist_n, std::uint32_t key_bits = 32) {
        if (count == 0) return;
        const std::uint32_t blocks = div_up(count, 1024);
        const std::uint32_t needed = (key_bits + 7u) / 8u;
        const std::uint32_t passes = std::min(8u, (needed + 1u) & ~1u);
        bool source_is_zero = true;
        for (std::uint32_t pass = 0; pass < passes; ++pass) {
            Buffer& in_lo = source_is_zero ? lo0 : lo1;
            Buffer& in_hi = source_is_zero ? hi0 : hi1;
            Buffer& in_val = source_is_zero ? val0 : val1;
            Buffer& out_lo = source_is_zero ? lo1 : lo0;
            Buffer& out_hi = source_is_zero ? hi1 : hi0;
            Buffer& out_val = source_is_zero ? val1 : val0;
            Push push{};
            push.u[0] = count;
            push.u[1] = blocks;
            push.u[2] = pass * 8;
            push.u[3] = key_is_64 ? 1u : 0u;
            dispatch(hist_, {&in_lo, &in_hi, &hist}, push, blocks);
            inclusive_scan(hist, 0, hist_n, hist_n, 2 * hist_n);
            push.u[4] = hist_n;
            dispatch(scatter_, {&in_lo, &in_hi, &in_val, &out_lo, &out_hi, &out_val, &hist, &hist}, push, blocks);
            source_is_zero = !source_is_zero;
        }
        if (!source_is_zero) {
            throw std::logic_error("radix sort pass count must be even so the result stays in buffer 0");
        }
    }

    Context::Impl& context_;
    Buffer dummy_;
    ComputePipeline preprocess_;
    ComputePipeline scan_;
    ComputePipeline emit_;
    ComputePipeline hist_;
    ComputePipeline scatter_;
    ComputePipeline ranges_;
    ComputePipeline blend_;
    ComputePipeline median_backward_;
    ComputePipeline blend_backward_;
    ComputePipeline blend_backward_no_geometry_;
    ComputePipeline blend_backward_no_geometry_subgroup_;
    ComputePipeline sample_depth_;
    ComputePipeline sample_depth_backward_;
    ComputePipeline multi_view_;
    ComputePipeline ssim_;
    ComputePipeline loss_reduce_;
    ComputePipeline project_backward_;
    ComputePipeline clear_;
    ComputePipeline pack_;
    VkCommandBuffer command_{};
    VkQueryPool backward_timestamp_pool_{};
    bool backward_profile_enabled_{};
    bool subgroup_backward_supported_{};
    std::uint32_t subgroup_size_{};
    std::uint32_t backward_profile_interval_{100};
    std::uint32_t backward_profile_samples_{};
    // Queue timestamp support, queried once for whichever stage profiler is on.
    bool timestamp_properties_queried_{};
    std::uint32_t timestamp_valid_bits_{};
    float timestamp_period_{};
    VkQueryPool forward_timestamp_pool_{VK_NULL_HANDLE};
    bool forward_profile_enabled_{};
    std::uint32_t forward_profile_interval_{100};
    std::uint32_t forward_profile_samples_{};
    double forward_counts_total_ms_{};
    double forward_sort_total_ms_{};
    double forward_blend_total_ms_{};
    double backward_fill_total_ms_{};
    double backward_blend_total_ms_{};
    double backward_project_total_ms_{};
    std::vector<VkDescriptorSet> pending_sets_;
    bool recording_{};
    bool model_ready_{};
    bool has_sh_{};
    bool has_scales_{true};
    bool raw_chain_{};
    std::uint32_t count_{};
    std::uint32_t sh_degree_{};
    std::uint32_t sh_bases_{};
    Buffer means_;
    Buffer opacities_;
    Buffer scales_;
    Buffer rotations_;
    Buffer covariances_;
    Buffer colors_;
    Buffer raw_log_scales_;
    Buffer raw_rotations_;
    Buffer opacity_logits_;
    Buffer filter_3d_;
    SplatDeviceGaussians device_model_{};
    bool device_model_bound_ = false;
    Buffer camera_;
    Buffer gauss_f_;
    Buffer gauss_u_;
    Buffer lo0_;
    Buffer lo1_;
    Buffer hi0_;
    Buffer hi1_;
    Buffer val0_;
    Buffer val1_;
    Buffer frame_counts_;
    Buffer hist_space_;
    Buffer compact_;
    Buffer tile_ranges_;
    Buffer bucket_offsets_;
    Buffer out_f_;
    Buffer out_u_;
    Buffer snap_;
    Buffer rgba_;
    Buffer ssim_prediction_;
    Buffer ssim_target_;
    Buffer ssim_mask_;
    Buffer ssim_work0_;
    Buffer ssim_work1_;
    Buffer ssim_work2_;
    Buffer ssim_work3_;
    Buffer ssim_work4_;
    Buffer ssim_dmu_;
    Buffer ssim_dvariance_;
    Buffer ssim_dcovariance_;
    Buffer ssim_output_;
    Buffer ssim_reduction_;
    Buffer loss_color_;
    Buffer loss_alpha_;
    Buffer loss_depth_;
    Buffer loss_normal_;
    Buffer median_state_;
    Buffer blend_grad_;
    Buffer model_grad_;
    Buffer sample_points_;
    Buffer sample_f_;
    Buffer sample_u_;
    Buffer sample_loss_;
    Buffer sample_point_grad_;
    Buffer multi_depth_;
    Buffer multi_normal_;
    Buffer multi_reference_gray_;
    Buffer multi_sampled_;
    Buffer multi_inside_;
    Buffer multi_neighbour_gray_;
    Buffer multi_transform_;
    Buffer multi_reference_mask_;
    Buffer multi_neighbour_mask_;
    Buffer multi_output_;
    bool last_frame_has_snapshots_{};
    bool last_frame_has_geometry_{};
    bool sample_live_{};
    std::uint32_t sample_point_count_{};
    Buffer* last_instance_values_ = &dummy_;
    std::uint32_t last_width_{};
    std::uint32_t last_height_{};
    std::uint32_t last_grid_x_{};
    std::uint32_t last_tiles_{};
    std::uint32_t last_pixels_{};
    std::uint32_t last_bucket_limit_{};
    std::uint32_t last_instances_{};
    std::uint32_t last_mode_{};
    int last_wrap_width_{};
    float last_background_[3]{};
    float last_fx_{};
    float last_fy_{};
    float last_cx_{};
    float last_cy_{};
    float last_k1_{};
    float last_k2_{};
    float last_k3_{};
    float last_k4_{};
    float last_kernel_size_{};
    float last_scale_modifier_{1.0F};
};

SplatRasterizer::SplatRasterizer(Context& context) : impl_(std::make_unique<Impl>(*context.impl_)) {}
SplatRasterizer::~SplatRasterizer() = default;
SplatRasterizer::SplatRasterizer(SplatRasterizer&&) noexcept = default;
SplatRasterizer& SplatRasterizer::operator=(SplatRasterizer&&) noexcept = default;

SplatForwardOutput SplatRasterizer::forward(const SplatGaussians& gaussians, const SplatCamera& camera,
                                            const SplatSettings& settings) {
    impl_->upload_model(gaussians);
    return impl_->render(camera, settings);
}

void SplatRasterizer::upload_model(const SplatGaussians& gaussians) { impl_->upload_model(gaussians); }

void SplatRasterizer::bind_model_device(const SplatDeviceGaussians& gaussians) {
    impl_->bind_model_device(gaussians);
}

void SplatRasterizer::update_means_and_opacities(
    const std::span<const float> means, const std::span<const float> opacities) {
    impl_->update_means_and_opacities(means, opacities);
}

void SplatRasterizer::clear_model() { impl_->clear_model(); }
bool SplatRasterizer::has_model() const noexcept { return impl_->has_model(); }

SplatForwardOutput SplatRasterizer::render(const SplatCamera& camera, const SplatSettings& settings) {
    return impl_->render(camera, settings);
}

SplatBlendGradients SplatRasterizer::backward_blend(
    const std::span<const float> dL_color, const std::span<const float> dL_alpha,
    const std::span<const float> dL_median_depth,
    const std::span<const float> dL_normal) {
    return impl_->backward_blend(dL_color, dL_alpha, dL_median_depth, dL_normal);
}

SplatModelGradients SplatRasterizer::backward(
    const std::span<const float> dL_color, const std::span<const float> dL_alpha,
    const std::span<const float> dL_median_depth,
    const std::span<const float> dL_normal) {
    return impl_->backward(dL_color, dL_alpha, dL_median_depth, dL_normal);
}

SplatDeviceFrame SplatRasterizer::render_device(
    const SplatCamera& camera, const SplatSettings& settings) {
    return impl_->render_device(camera, settings);
}

SplatDeviceFrame SplatRasterizer::render_device_copy(
    const SplatCamera& camera, const SplatSettings& settings,
    const SplatDeviceFrame& destination) {
    return impl_->render_device_copy(camera, settings, destination);
}

void SplatRasterizer::copy_frame_device(
    const SplatDeviceFrame& source,
    const SplatDeviceFrame& destination) {
    impl_->copy_frame_device(source, destination);
}

void SplatRasterizer::backward_blend_device(
    const SplatBufferView& dL_color, const SplatBufferView& dL_alpha,
    const SplatBufferView& packed_gradients,
    const SplatBufferView& dL_median_depth, const SplatBufferView& dL_normal) {
    impl_->backward_blend_device(
        dL_color, dL_alpha, packed_gradients, dL_median_depth, dL_normal);
}

std::uint64_t SplatRasterizer::blend_gradient_float_count() const noexcept {
    return impl_->blend_gradient_float_count();
}

void SplatRasterizer::backward_device(
    const SplatBufferView& dL_color, const SplatBufferView& dL_alpha,
    const SplatBufferView& packed_model_gradients,
    const SplatBufferView& dL_median_depth, const SplatBufferView& dL_normal) {
    impl_->backward_device(
        dL_color, dL_alpha, packed_model_gradients, dL_median_depth, dL_normal);
}

std::uint64_t SplatRasterizer::model_gradient_float_count() const noexcept {
    return impl_->model_gradient_float_count();
}

SplatDepthSamples SplatRasterizer::sample_depth(
    const std::span<const float> world_points, const SplatCamera& camera,
    const SplatSettings& settings) {
    return impl_->sample_depth(world_points, camera, settings);
}

SplatDepthSampleGradients SplatRasterizer::sample_depth_backward(
    const std::span<const float> dL_camera_points) {
    return impl_->sample_depth_backward(dL_camera_points);
}

SplatMultiViewOutput SplatRasterizer::multi_view_loss(
    const SplatMultiViewInput& input) {
    return impl_->multi_view_loss(input);
}

SplatPhotometricOutput SplatRasterizer::photometric_loss(
    const std::span<const float> prediction,
    const std::span<const float> target,
    const std::uint32_t width, const std::uint32_t height,
    const float ssim_weight, const float photometric_weight,
    const std::span<const float> mask) {
    return impl_->photometric_loss(
        prediction, target, width, height, ssim_weight,
        photometric_weight, mask);
}

SplatDevicePhotometricOutput SplatRasterizer::fused_l1_ssim_device(
    const SplatBufferView& prediction, const SplatBufferView& target,
    const std::uint32_t width, const std::uint32_t height,
    const float ssim_weight, const float photometric_weight,
    const SplatBufferView& mask) {
    return impl_->fused_l1_ssim_device(
        prediction, target, width, height, ssim_weight,
        photometric_weight, mask);
}

float SplatRasterizer::read_photometric_loss(
    const SplatDevicePhotometricOutput& output) {
    return impl_->read_photometric_loss(output);
}

void SplatRasterizer::render_rgba(
    const SplatCamera& camera, const SplatSettings& settings, std::vector<std::uint8_t>& rgba) {
    impl_->render_rgba(camera, settings, rgba);
}

SplatRgbaImage SplatRasterizer::render_rgba_device(const SplatCamera& camera, const SplatSettings& settings) {
    return impl_->render_rgba_device(camera, settings);
}

} // namespace splat_drender::vulkan
