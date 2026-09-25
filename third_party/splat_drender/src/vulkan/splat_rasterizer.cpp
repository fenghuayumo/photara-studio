#include "splat_drender/vulkan_api.h"

#include <algorithm>
#include <bit>
#include <cmath>
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
          pack_(context.create_pipeline("splat_pack_rgba.hlsl.spv", 2, sizeof(Push))) {
        clear_buffer(dummy_);
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

        const auto& color_src = has_sh ? gaussians.sh : gaussians.colors;
        const std::scoped_lock lock(context_.dispatch_mutex);
        Buffer& means = grow(means_, gaussians.means.size() * sizeof(float));
        Buffer& opacities = grow(opacities_, gaussians.opacities.size() * sizeof(float));
        Buffer& scales = grow(scales_, (has_scales ? gaussians.scales.size() : 1) * sizeof(float));
        Buffer& rotations = grow(rotations_, (has_scales ? gaussians.rotations.size() : 1) * sizeof(float));
        Buffer& covariances = grow(covariances_, (has_cov ? gaussians.covariances.size() : 1) * sizeof(float));
        Buffer& colors = grow(colors_, color_src.size() * sizeof(float));
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
        has_sh_ = has_sh;
        has_scales_ = has_scales;
        count_ = count;
        sh_degree_ = gaussians.sh_degree;
        sh_bases_ = gaussians.sh_bases;
        model_ready_ = true;
    }

    void update_means_and_opacities(std::span<const float> means, std::span<const float> opacities) {
        require(model_ready_, "no Gaussian model is loaded");
        require(means.size() == static_cast<std::size_t>(count_) * 3, "means must have shape [N, 3]");
        require(opacities.size() == count_, "opacities must have shape [N]");
        const std::scoped_lock lock(context_.dispatch_mutex);
        context_.write_buffer(means_, means.data(), means.size() * sizeof(float));
        context_.write_buffer(opacities_, opacities.data(), opacities.size() * sizeof(float));
    }

    void clear_model() { model_ready_ = false; count_ = 0; }
    bool has_model() const noexcept { return model_ready_; }

    struct FrameCounts {
        std::uint32_t instances = 0;
        std::uint32_t visible = 0;
    };

    // Everything the frame computes, up to the blended image. `pack_rgba` adds
    // the 8-bit conversion to the same command batch, which is what the editor
    // preview hands to its own image.
    FrameCounts record_frame(const SplatCamera& camera, const SplatSettings& settings, const bool pack_rgba) {
        require(model_ready_, "no Gaussian model is loaded");
        require(camera.width > 0 && camera.height > 0, "camera width and height must be positive");
        require(camera.mode == 0 || camera.mode == 1 || camera.mode == 3 || camera.mode == 4,
                "camera mode must be pinhole, fisheye, equirectangular, or orthographic");
        require(camera.world_to_camera.size() == 16 && camera.center.size() == 3, "camera transform must be 16 floats and center 3 floats");
        const std::scoped_lock lock(context_.dispatch_mutex);
        const std::uint32_t count = count_;
        const bool has_sh = has_sh_;
        const bool has_scales = has_scales_;
        Buffer& means = means_;
        Buffer& opacities = opacities_;
        Buffer& scales = scales_;
        Buffer& rotations = rotations_;
        Buffer& covariances = covariances_;
        Buffer& colors = colors_;
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

        Push prep{};
        prep.u[0] = count;
        prep.u[1] = sh_degree_;
        prep.u[2] = sh_bases_;
        prep.u[3] = (has_sh ? k_flag_sh : 0) | (has_scales ? k_flag_scales : 0) | (settings.need_depth ? k_flag_geometry : 0);
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
        dispatch(preprocess_, {&means, &opacities, &scales, &rotations, &covariances, &colors, &camera_buffer,
                               &gauss_f, &gauss_u, &dummy_, &dummy_, &dummy_},
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
        flush_batch();
        std::uint32_t frame_count_values[2]{};
        frame_counts.download(frame_count_values, sizeof(frame_count_values));
        const std::uint32_t instances = frame_count_values[0];
        const std::uint32_t visible = frame_count_values[1];

        Buffer* instance_values = &dummy_;
        Buffer* range_lo = &dummy_;
        Buffer* range_hi = &dummy_;
        Buffer& tile_ranges = grow(tile_ranges_, static_cast<std::uint64_t>(tiles) * 2 * sizeof(std::uint32_t));
        zero_buffer(context_, tile_ranges, static_cast<std::size_t>(tiles) * 2 * sizeof(std::uint32_t));
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
        if (!pack_rgba)
            zero_buffer(context_, *out_u, static_cast<std::size_t>(count) * sizeof(std::uint32_t));

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
        flush_batch();
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
    // Every buffer the shaders touch lives in device memory: an unmapped buffer
    // keeps the GPU at VRAM bandwidth instead of running the whole forward over
    // PCIe. Only the tiny per-frame control blocks stay host visible.
    Buffer alloc(
        VkDeviceSize bytes, const BufferMemory memory = BufferMemory::device_local,
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) const {
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
                  std::uint32_t groups_x, std::uint32_t groups_y = 1) {
        if (groups_x == 0 || groups_y == 0) return;
        begin_batch();
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = context_.descriptor_pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &pipeline.descriptor_set_layout;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(context_.device, &set_info, &descriptor_set) != VK_SUCCESS) {
            throw std::runtime_error("vkAllocateDescriptorSets failed");
        }
        pending_sets_.push_back(descriptor_set);
        std::vector<VkDescriptorBufferInfo> infos;
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        infos.reserve(buffers.size());
        for (const Buffer* buffer : buffers) infos.push_back(descriptor(*buffer));
        for (std::uint32_t i = 0; i < buffers.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(
            context_.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
        vkCmdBindDescriptorSets(
            command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout, 0, 1,
            &descriptor_set, 0, nullptr);
        vkCmdPushConstants(
            command_, pipeline.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
        vkCmdDispatch(command_, groups_x, groups_y, 1);
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
    void radix_sort(bool key_is_64, std::uint32_t count, Buffer& lo0, Buffer& hi0, Buffer& val0, Buffer& lo1, Buffer& hi1,
                    Buffer& val1, Buffer& hist, std::uint32_t hist_n, std::uint32_t key_bits = 32) {
        if (count == 0) return;
        const std::uint32_t blocks = div_up(count, 1024);
        const std::uint32_t needed = key_is_64 ? (key_bits + 7u) / 8u : 4u;
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
    ComputePipeline pack_;
    VkCommandBuffer command_{};
    std::vector<VkDescriptorSet> pending_sets_;
    bool recording_{};
    bool model_ready_{};
    bool has_sh_{};
    bool has_scales_{true};
    std::uint32_t count_{};
    std::uint32_t sh_degree_{};
    std::uint32_t sh_bases_{};
    Buffer means_;
    Buffer opacities_;
    Buffer scales_;
    Buffer rotations_;
    Buffer covariances_;
    Buffer colors_;
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

void SplatRasterizer::update_means_and_opacities(
    const std::span<const float> means, const std::span<const float> opacities) {
    impl_->update_means_and_opacities(means, opacities);
}

void SplatRasterizer::clear_model() { impl_->clear_model(); }
bool SplatRasterizer::has_model() const noexcept { return impl_->has_model(); }

SplatForwardOutput SplatRasterizer::render(const SplatCamera& camera, const SplatSettings& settings) {
    return impl_->render(camera, settings);
}

void SplatRasterizer::render_rgba(
    const SplatCamera& camera, const SplatSettings& settings, std::vector<std::uint8_t>& rgba) {
    impl_->render_rgba(camera, settings, rgba);
}

SplatRgbaImage SplatRasterizer::render_rgba_device(const SplatCamera& camera, const SplatSettings& settings) {
    return impl_->render_rgba_device(camera, settings);
}

} // namespace splat_drender::vulkan
