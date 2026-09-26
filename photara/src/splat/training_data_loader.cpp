#include "training_data_loader.hpp"

#include "core/camera_projection.hpp"
#include "io/image.hpp"
#include "cuda_ops.hpp"
#include "core/logging.hpp"
#include "splat/trainer.hpp"

#include <cuda_runtime_api.h>
#include "internal/cuda_stream_context.hpp"
#include "core/vram_profiler.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <filesystem>
#include <fstream>
#endif

namespace photara::splat {
namespace {

constexpr std::size_t k_mib = std::size_t{1024} * 1024;
constexpr std::size_t k_gib = k_mib * 1024;
// Device cache budget feedback. The window is long enough that one decision
// costs nothing and short enough to react inside a single epoch.
constexpr std::size_t k_device_budget_window = 512;
constexpr std::size_t k_min_device_budget = std::size_t{64} * k_mib;
constexpr std::size_t k_min_device_growth = std::size_t{256} * k_mib;
constexpr double k_device_hit_rate_target = 0.5;
constexpr double k_device_cache_max_share = 0.4;
constexpr std::size_t k_max_pending_device_uploads = 2;
// A larger cache has to recover at least this much iteration time during the
// following observation window. Smaller changes are indistinguishable from
// normal training noise and do not justify retaining the extra VRAM.
constexpr double k_device_step_improvement = 0.01;

std::size_t saturate_add(
    const std::size_t left, const std::size_t right) noexcept {
    return left > std::numeric_limits<std::size_t>::max() - right
        ? std::numeric_limits<std::size_t>::max()
        : left + right;
}

std::size_t saturate_multiply(
    const std::size_t left, const std::size_t right) noexcept {
    if (left == 0 || right == 0) return 0;
    return left > std::numeric_limits<std::size_t>::max() / right
        ? std::numeric_limits<std::size_t>::max()
        : left * right;
}

std::size_t available_system_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return static_cast<std::size_t>(status.ullAvailPhys);
    return 0;
#elif defined(__APPLE__)
    uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0)
        return static_cast<std::size_t>(bytes / 4);
    return 0;
#else
    std::error_code error;
    const auto line = std::filesystem::path("/proc/meminfo");
    if (std::filesystem::exists(line, error)) {
        std::ifstream stream(line);
        std::string key;
        std::size_t value_kb = 0;
        while (stream >> key >> value_kb) {
            if (key == "MemAvailable:")
                return saturate_multiply(value_kb, std::size_t{1024});
        }
    }
    return 0;
#endif
}

void finalize_equirect_intrinsics(Camera& camera) {
    camera.fx = camera.fy =
        static_cast<float>(camera.width) / (2.F * 3.14159265358979323846F);
    camera.cx = 0.5F * static_cast<float>(camera.width);
    camera.cy = 0.5F * static_cast<float>(camera.height);
    camera.k1 = camera.k2 = camera.k3 = camera.k4 = 0.F;
}

Camera make_camera_impl(
    const mvs::MvsView& view, const bool native_source) {
    Camera camera;
    // Force evaluation: translation() returns a temporary, so retaining the
    // lazy Eigen cast expression with `auto` would leave a dangling operand.
    const Eigen::Vector3f translation =
        view.pose.translation().cast<float>();
    const Eigen::Matrix3f rotation = view.pose.R.cast<float>();
    // Transpose a conventional row-major W2C into the contiguous layout used
    // by the reference CUDA kernels (the same conversion as torch .t()).
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            float value = 0.F;
            if (row < 3 && column < 3) value = rotation(row, column);
            else if (row < 3 && column == 3) value = translation(row);
            else if (row == 3 && column == 3) value = 1.F;
            camera.world_to_camera[static_cast<std::size_t>(column) * 4 + row] = value;
        }
    }
    const Eigen::Vector3f position = view.pose.C.cast<float>();
    camera.position = {position.x(), position.y(), position.z()};
    if (native_source) {
        camera.model = view.source_model;
        camera.fx = view.src_fx;
        camera.fy = view.src_fy;
        camera.cx = view.src_cx;
        camera.cy = view.src_cy;
        camera.width = view.src_width != 0 ? view.src_width : view.width;
        camera.height = view.src_height != 0 ? view.src_height : view.height;
        camera.k1 = view.k1;
        camera.k2 = view.k2;
        camera.k3 = view.p1;
        camera.k4 = view.p2;
        if (camera.model == CameraModel::equirectangular)
            finalize_equirect_intrinsics(camera);
    } else {
        camera.model = CameraModel::pinhole;
        camera.fx = view.fx;
        camera.fy = view.fy;
        camera.cx = view.cx;
        camera.cy = view.cy;
        camera.width = view.width;
        camera.height = view.height;
    }
    return camera;
}

}
 
namespace training_data {

bool uses_native_training_camera(
    const mvs::MvsView& view, const TrainingOptions& options) {
    if (view.source_model == CameraModel::equirectangular)
        return true;
    return uses_native_splat_projection(view.source_model) &&
           !options.undistort_to_pinhole;
}

Camera training_camera(
    const mvs::MvsView& view, const TrainingOptions& options,
    const float resolution_scale) {
    if (options.undistort_to_pinhole &&
        view.source_model == CameraModel::equirectangular)
        throw std::invalid_argument(
            "Equirectangular cameras cannot be undistorted to pinhole; "
            "train them natively");
    Camera camera = make_camera_impl(
        view, uses_native_training_camera(view, options));
    if (!uses_native_splat_projection(camera.model) &&
        options.use_source_resolution && view.src_width != 0 &&
        view.src_height != 0) {
        camera.fx = view.src_fx;
        camera.fy = view.src_fy;
        camera.cx = view.src_cx;
        camera.cy = view.src_cy;
        camera.width = view.src_width;
        camera.height = view.src_height;
    }
    const std::uint32_t largest =
        std::max(camera.width, camera.height);
    if (options.max_image_dimension != 0 &&
        largest > options.max_image_dimension) {
        const float scale = static_cast<float>(
            options.max_image_dimension) / static_cast<float>(largest);
        const std::uint32_t old_width = camera.width;
        const std::uint32_t old_height = camera.height;
        camera.width = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::lround(
                   old_width * scale)));
        camera.height = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::lround(
                   old_height * scale)));
        const float scale_x = static_cast<float>(camera.width) / old_width;
        const float scale_y = static_cast<float>(camera.height) / old_height;
        camera.fx *= scale_x;
        camera.fy *= scale_y;
        camera.cx = (camera.cx + 0.5F) * scale_x - 0.5F;
        camera.cy = (camera.cy + 0.5F) * scale_y - 0.5F;
        if (camera.model == CameraModel::equirectangular)
            finalize_equirect_intrinsics(camera);
    }
    const float level_scale = std::clamp(resolution_scale, 1e-3F, 1.F);
    if (level_scale < 1.F) {
        const std::uint32_t old_width = camera.width;
        const std::uint32_t old_height = camera.height;
        camera.width = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::lround(
                   old_width * level_scale)));
        camera.height = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::lround(
                   old_height * level_scale)));
        const float scale_x = static_cast<float>(camera.width) / old_width;
        const float scale_y = static_cast<float>(camera.height) / old_height;
        camera.fx *= scale_x;
        camera.fy *= scale_y;
        camera.cx = (camera.cx + 0.5F) * scale_x - 0.5F;
        camera.cy = (camera.cy + 0.5F) * scale_y - 0.5F;
        if (camera.model == CameraModel::equirectangular)
            finalize_equirect_intrinsics(camera);
    }
    return camera;
}

float progressive_resolution_scale(
    const unsigned iteration, const TrainingOptions& options) {
    if (!options.progressive_resolution ||
        options.progressive_resolution_interval == 0)
        return 1.F;
    const unsigned level =
        (std::max(iteration, 1U) - 1U) /
        options.progressive_resolution_interval;
    return std::min(
        1.F, std::clamp(options.progressive_initial_scale, 1e-3F, 1.F) *
                 std::pow(2.F, static_cast<float>(level)));
}

std::vector<std::vector<std::size_t>> compute_multi_view_neighbours(
    const std::vector<Camera>& cameras,
    const std::vector<std::size_t>& active_indices,
    const TrainingOptions& options) {
    std::vector<std::vector<std::size_t>> result(cameras.size());
    for (const std::size_t reference : active_indices) {
        std::vector<std::tuple<float, float, std::size_t>> candidates;
        const auto& camera = cameras[reference];
        const Eigen::Vector3f center(
            camera.position[0], camera.position[1], camera.position[2]);
        Eigen::Vector3f forward(
            camera.world_to_camera[2], camera.world_to_camera[6],
            camera.world_to_camera[10]);
        forward.normalize();
        for (const std::size_t index : active_indices) {
            if (index == reference) continue;
            const auto& other = cameras[index];
            const Eigen::Vector3f other_center(
                other.position[0], other.position[1], other.position[2]);
            const float distance = (center - other_center).norm();
            Eigen::Vector3f other_forward(
                other.world_to_camera[2], other.world_to_camera[6],
                other.world_to_camera[10]);
            other_forward.normalize();
            const float angle = std::acos(std::clamp(
                forward.dot(other_forward), -1.F, 1.F)) *
                57.29577951308232F;
            if (angle < options.multi_view_max_angle &&
                distance > options.multi_view_min_distance &&
                distance < options.multi_view_max_distance)
                candidates.emplace_back(distance, angle, index);
        }
        std::sort(candidates.begin(), candidates.end());
        const std::size_t count = std::min<std::size_t>(
            options.multi_view_num, candidates.size());
        result[reference].reserve(count);
        for (std::size_t slot = 0; slot < count; ++slot)
            result[reference].push_back(std::get<2>(candidates[slot]));
    }
    return result;
}
}  // namespace training_data

namespace {

void sample_rgb(
    const io::RgbImage& image, const float x, const float y,
    std::uint8_t& red, std::uint8_t& green, std::uint8_t& blue) {
    if (x < 0.F || y < 0.F || x > static_cast<float>(image.width - 1) ||
        y > static_cast<float>(image.height - 1)) {
        red = green = blue = 0;
        return;
    }
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, static_cast<int>(image.width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(image.height) - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float w00 = (1.F - tx) * (1.F - ty);
    const float w10 = tx * (1.F - ty);
    const float w01 = (1.F - tx) * ty;
    const float w11 = tx * ty;
    const auto at = [&](const int px, const int py, const int channel) {
        return static_cast<float>(image.pixels[
            (static_cast<std::size_t>(py) * image.width +
             static_cast<std::size_t>(px)) *
                3 +
            static_cast<std::size_t>(channel)]);
    };
    const auto quantize = [](const float value) {
        return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.F, 255.F)));
    };
    red = quantize(
        at(x0, y0, 0) * w00 + at(x1, y0, 0) * w10 + at(x0, y1, 0) * w01 +
        at(x1, y1, 0) * w11);
    green = quantize(
        at(x0, y0, 1) * w00 + at(x1, y0, 1) * w10 + at(x0, y1, 1) * w01 +
        at(x1, y1, 1) * w11);
    blue = quantize(
        at(x0, y0, 2) * w00 + at(x1, y0, 2) * w10 + at(x0, y1, 2) * w01 +
        at(x1, y1, 2) * w11);
}

std::pair<float, float> source_coordinate(
    const mvs::MvsView& view, const std::uint32_t x, const std::uint32_t y,
    const Camera& output_camera, const io::RgbImage& source) {
    if (uses_native_splat_projection(output_camera.model)) {
        const float scale_x =
            static_cast<float>(output_camera.width) / source.width;
        const float scale_y =
            static_cast<float>(output_camera.height) / source.height;
        return {(static_cast<float>(x) + 0.5F) / scale_x - 0.5F,
                (static_cast<float>(y) + 0.5F) / scale_y - 0.5F};
    }
    const bool distorted = view.source_model == CameraModel::opencv_fisheye || view.k1 != 0.F || view.k2 != 0.F ||
                           view.p1 != 0.F || view.p2 != 0.F;
    if (!distorted) {
        const float scale_x =
            static_cast<float>(output_camera.width) / source.width;
        const float scale_y =
            static_cast<float>(output_camera.height) / source.height;
        return {(static_cast<float>(x) + 0.5F) / scale_x - 0.5F,
                (static_cast<float>(y) + 0.5F) / scale_y - 0.5F};
    }
    const double xn =
        (static_cast<double>(x) - output_camera.cx) / output_camera.fx;
    const double yn =
        (static_cast<double>(y) - output_camera.cy) / output_camera.fy;
    const auto pixel = project_camera_plane(view.source_model, xn, yn,
        view.k1, view.k2, view.p1, view.p2);
    float source_x = static_cast<float>(
        view.src_fx * pixel.x + view.src_cx);
    float source_y = static_cast<float>(
        view.src_fy * pixel.y + view.src_cy);
    // JPEG DCT scaling can reduce the decoded source relative to the camera
    // metadata. Distortion is evaluated in normalized coordinates, then the
    // resulting metadata-space pixel is mapped into the decoded bitmap.
    const std::uint32_t metadata_width =
        view.src_width != 0 ? view.src_width : view.width;
    const std::uint32_t metadata_height =
        view.src_height != 0 ? view.src_height : view.height;
    if (metadata_width != 0 && metadata_height != 0 &&
        (source.width != metadata_width ||
         source.height != metadata_height)) {
        source_x = (source_x + 0.5F) *
            static_cast<float>(source.width) / metadata_width - 0.5F;
        source_y = (source_y + 0.5F) *
            static_cast<float>(source.height) / metadata_height - 0.5F;
    }
    return {source_x, source_y};
}
std::filesystem::path find_mask_path(
    const std::filesystem::path& directory,
    const std::filesystem::path& image_path) {
    if (directory.empty() || !std::filesystem::is_directory(directory))
        return {};
    const auto exact = directory / image_path.filename();
    if (std::filesystem::is_regular_file(exact)) return exact;
    static constexpr std::array<const char*, 6> extensions{
        ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"};
    for (const char* extension : extensions) {
        const auto candidate = directory / (image_path.stem().string() + extension);
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    return {};
}

std::filesystem::path resolve_mask_path(
    const mvs::MvsView& view, const TrainingOptions& options) {
    if (!options.mask_dir.empty())
        return find_mask_path(options.mask_dir, view.path);
    const auto nested = find_mask_path(view.path.parent_path() / "masks", view.path);
    if (!nested.empty()) return nested;
    return find_mask_path(
        view.path.parent_path().parent_path() / "masks", view.path);
}

float sample_mask_coverage(
    const io::GrayImage& mask, const io::RgbImage& source,
    const float source_x, const float source_y) {
    const float x = (source_x + 0.5F) * mask.width / source.width - 0.5F;
    const float y = (source_y + 0.5F) * mask.height / source.height - 0.5F;
    if (x < -0.5F || y < -0.5F ||
        x > static_cast<float>(mask.width) - 0.5F ||
        y > static_cast<float>(mask.height) - 0.5F)
        return 0.F;
    const int x0 = std::clamp(
        static_cast<int>(std::floor(x)), 0,
        static_cast<int>(mask.width) - 1);
    const int y0 = std::clamp(
        static_cast<int>(std::floor(y)), 0,
        static_cast<int>(mask.height) - 1);
    const int x1 = std::min(x0 + 1, static_cast<int>(mask.width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(mask.height) - 1);
    const float tx = std::clamp(x - static_cast<float>(x0), 0.F, 1.F);
    const float ty = std::clamp(y - static_cast<float>(y0), 0.F, 1.F);
    const auto at = [&](const int px, const int py) {
        return static_cast<float>(
                   mask.pixels[
                       static_cast<std::size_t>(py) * mask.width + px]) /
            255.F;
    };
    return std::clamp(
        (at(x0, y0) * (1.F - tx) + at(x1, y0) * tx) * (1.F - ty) +
            (at(x0, y1) * (1.F - tx) + at(x1, y1) * tx) * ty,
        0.F, 1.F);
}

float sample_projected_foreground_coverage(
    const mvs::MvsView& view, const Camera& output_camera,
    const std::uint32_t x, const std::uint32_t y,
    const float mask_denominator) {
    const std::size_t working_pixels =
        static_cast<std::size_t>(view.width) * view.height;
    if (view.foreground_mask.size() != working_pixels ||
        view.width == 0 || view.height == 0 ||
        uses_native_splat_projection(output_camera.model))
        return 1.F;

    // The coarse mesh mask lives in the undistorted MVS working camera.
    // Reproject the ideal pinhole ray instead of scaling pixels directly so
    // source-resolution and progressive-resolution GGGS targets use exactly
    // the same foreground envelope.
    const float xn =
        (static_cast<float>(x) - output_camera.cx) / output_camera.fx;
    const float yn =
        (static_cast<float>(y) - output_camera.cy) / output_camera.fy;
    const float u = view.fx * xn + view.cx;
    const float v = view.fy * yn + view.cy;
    const int x0 = static_cast<int>(std::floor(u));
    const int y0 = static_cast<int>(std::floor(v));
    const float tx = u - static_cast<float>(x0);
    const float ty = v - static_cast<float>(y0);
    const auto sample = [&](const int px, const int py) {
        if (px < 0 || py < 0 ||
            px >= static_cast<int>(view.width) ||
            py >= static_cast<int>(view.height))
            return 0.F;
        return static_cast<float>(
                   view.foreground_mask[
                       static_cast<std::size_t>(py) * view.width +
                       static_cast<std::size_t>(px)]) /
            mask_denominator;
    };
    return std::clamp(
        (sample(x0, y0) * (1.F - tx) + sample(x0 + 1, y0) * tx) *
                (1.F - ty) +
            (sample(x0, y0 + 1) * (1.F - tx) +
             sample(x0 + 1, y0 + 1) * tx) *
                ty,
        0.F, 1.F);
}

struct HostTrainingView {
    Camera camera;
    // One little-endian RGBA8 word per pixel. Alpha stores the optional
    // binary training mask; opaque 255 is used when no mask is present.
    std::vector<int> rgba;
    std::vector<float> depth;
    std::vector<float> normal;
    bool has_mask{false};
    bool mask_is_validity{false};

    [[nodiscard]] std::size_t bytes() const noexcept {
        return sizeof(*this) + sizeof(int) * rgba.capacity() +
            sizeof(float) * (depth.capacity() + normal.capacity());
    }
};

// Result of a background host load. The duration is carried alongside the view
// so the training thread can fold it into the prefetch-lookahead estimate
// without any cross-thread synchronisation.
struct HostTrainingLoad {
    std::shared_ptr<HostTrainingView> view;
    double host_load_ms{0.0};
};

class PinnedStagingBuffer {
public:
    PinnedStagingBuffer() = default;
    ~PinnedStagingBuffer() {
        if (data_) cudaFreeHost(data_);
    }

    PinnedStagingBuffer(const PinnedStagingBuffer&) = delete;
    PinnedStagingBuffer& operator=(const PinnedStagingBuffer&) = delete;

    void ensure(const std::size_t requested_bytes) {
        if (requested_bytes <= capacity_bytes_) return;
        if (data_) {
            const cudaError_t free_error = cudaFreeHost(data_);
            if (free_error != cudaSuccess)
                throw std::runtime_error(
                    std::string("Failed to free pinned staging memory: ") +
                    cudaGetErrorString(free_error));
            data_ = nullptr;
            capacity_bytes_ = 0;
        }
        if (requested_bytes == 0) return;

        void* pointer = nullptr;
        const cudaError_t error = cudaHostAlloc(
            &pointer, requested_bytes, cudaHostAllocDefault);
        if (error != cudaSuccess)
            throw std::runtime_error(
                std::string("Failed to allocate pinned staging memory: ") +
                cudaGetErrorString(error));
        data_ = static_cast<std::uint8_t*>(pointer);
        capacity_bytes_ = requested_bytes;
    }

    [[nodiscard]] std::uint8_t* data() noexcept { return data_; }
    [[nodiscard]] std::size_t capacity() const noexcept {
        return capacity_bytes_;
    }

private:
    std::uint8_t* data_{};
    std::size_t capacity_bytes_{};
};

// Training-thread-only pool. Buffers are returned only after the CUDA event for
// their transfer has completed, so a subsequent upload can never overwrite
// bytes still being consumed by DMA.
class PinnedStagingPool {
public:
    [[nodiscard]] std::unique_ptr<PinnedStagingBuffer> acquire(
        const std::size_t bytes) {
        std::unique_ptr<PinnedStagingBuffer> buffer;
        if (free_.empty()) {
            buffer = std::make_unique<PinnedStagingBuffer>();
        } else {
            buffer = std::move(free_.back());
            free_.pop_back();
        }
        buffer->ensure(bytes);
        return buffer;
    }

    void release(std::unique_ptr<PinnedStagingBuffer> buffer) {
        if (buffer) free_.push_back(std::move(buffer));
    }

    void clear() { free_.clear(); }

private:
    std::vector<std::unique_ptr<PinnedStagingBuffer>> free_;
};

// Loader-owned device staging ring for the upload pipeline. The copy engine must
// never write into pool memory: the pool hands blocks between streams and
// threads without any ordering, and testing (see CacheStats::async_upload_*) had
// a packed view overwritten behind an in-flight transfer. These slots are plain
// cudaMalloc buffers owned by the loader and reused only after every reader
// finished, so the DMA destination is never something the pool can recycle.
class DeviceUploadRing {
public:
    static constexpr std::size_t k_slot_count = 2;

    struct Slot {
        std::uint8_t* data{};
        std::size_t capacity{};
        cudaEvent_t ready{};      // packed-view H2D on the copy stream completed
        cudaEvent_t reusable{};   // device-to-device hop on the compute stream done
        bool busy{};              // owned by a pending upload
    };

    DeviceUploadRing() {
        for (Slot& slot : slots_) {
            if (cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming) !=
                cudaSuccess)
                slot.ready = nullptr;
            if (cudaEventCreateWithFlags(&slot.reusable, cudaEventDisableTiming) !=
                cudaSuccess)
                slot.reusable = nullptr;
            // An event that was never recorded queries as complete, so a fresh
            // ring starts out fully reusable.
        }
    }

    ~DeviceUploadRing() { clear(); }

    DeviceUploadRing(const DeviceUploadRing&) = delete;
    DeviceUploadRing& operator=(const DeviceUploadRing&) = delete;

    // Hand out a slot whose readers finished. Never blocks: returning nullptr
    // simply keeps the caller on the synchronous upload path.
    [[nodiscard]] Slot* try_acquire(const std::size_t bytes) {
        for (Slot& slot : slots_) {
            if (slot.ready == nullptr || slot.reusable == nullptr) continue;
            if (slot.busy) continue;
            // `reusable` is only recorded once a hop read this slot; an event
            // that was never recorded queries as complete.
            if (cudaEventQuery(slot.reusable) != cudaSuccess) continue;
            if (bytes > slot.capacity && !grow(slot, bytes)) continue;
            slot.busy = true;
            return &slot;
        }
        return nullptr;
    }

    static void release(Slot& slot) { slot.busy = false; }

    static void mark_ready(Slot& slot, const cudaStream_t stream) {
        if (slot.ready != nullptr) cudaEventRecord(slot.ready, stream);
    }

    static void mark_reusable(Slot& slot, const cudaStream_t stream) {
        if (slot.reusable != nullptr) cudaEventRecord(slot.reusable, stream);
    }

    // Block until no slot is being read or written, then release the buffers.
    void clear() noexcept {
        for (Slot& slot : slots_) {
            drain(slot);
            if (slot.data != nullptr) {
                cudaFree(slot.data);
                slot.data = nullptr;
                slot.capacity = 0;
            }
            if (slot.ready != nullptr) {
                cudaEventDestroy(slot.ready);
                slot.ready = nullptr;
            }
            if (slot.reusable != nullptr) {
                cudaEventDestroy(slot.reusable);
                slot.reusable = nullptr;
            }
        }
    }

    // Wait for the slot's readers without giving the buffers back.
    void drain() noexcept {
        for (Slot& slot : slots_) drain(slot);
    }

private:
    static void drain(Slot& slot) noexcept {
        if (slot.ready != nullptr) cudaEventSynchronize(slot.ready);
        if (slot.reusable != nullptr) cudaEventSynchronize(slot.reusable);
    }

    static bool grow(Slot& slot, const std::size_t bytes) {
        void* pointer = nullptr;
        if (cudaMalloc(&pointer, bytes) != cudaSuccess) return false;
        if (slot.data != nullptr) cudaFree(slot.data);
        slot.data = static_cast<std::uint8_t*>(pointer);
        slot.capacity = bytes;
        return true;
    }

    std::array<Slot, k_slot_count> slots_{};
};

std::size_t packed_training_view_bytes(
    const mvs::MvsView& view, const TrainingOptions& options,
    const float resolution_scale) {
    const Camera camera = training_data::training_camera(
        view, options, resolution_scale);
    const std::size_t pixels = saturate_multiply(
        static_cast<std::size_t>(camera.width),
        static_cast<std::size_t>(camera.height));
    std::size_t bytes = saturate_multiply(pixels, sizeof(int));
    const bool full_working_resolution =
        camera.width == view.width && camera.height == view.height;
    const bool can_use_mvs_maps =
        full_working_resolution &&
        !uses_native_splat_projection(camera.model);
    if (can_use_mvs_maps && options.use_mvs_depth &&
        view.depth_map.depth.size() ==
            saturate_multiply(
                static_cast<std::size_t>(view.width),
                static_cast<std::size_t>(view.height))) {
        bytes = saturate_add(bytes, saturate_multiply(pixels, sizeof(float)));
    }
    if (can_use_mvs_maps && options.use_mvs_normals &&
        view.depth_map.normal.size() ==
            saturate_multiply(
                static_cast<std::size_t>(view.width),
                static_cast<std::size_t>(view.height))) {
        bytes = saturate_add(
            bytes, saturate_multiply(pixels, 3 * sizeof(float)));
    }
    return bytes;
}

std::uint8_t quantize_channel(const float value) {
    return static_cast<std::uint8_t>(std::lround(
        std::clamp(value, 0.F, 1.F) * 255.F));
}

int pack_rgba(
    const std::uint8_t red, const std::uint8_t green,
    const std::uint8_t blue, const std::uint8_t alpha) {
    const std::uint32_t packed =
        static_cast<std::uint32_t>(red) |
        (static_cast<std::uint32_t>(green) << 8U) |
        (static_cast<std::uint32_t>(blue) << 16U) |
        (static_cast<std::uint32_t>(alpha) << 24U);
    static_assert(sizeof(int) == sizeof(packed));
    return std::bit_cast<int>(packed);
}

HostTrainingView load_host_training_view(
    const mvs::MvsView& view, const TrainingOptions& options,
    const float resolution_scale = 1.F) {
    tinytensor::TraceScope load_scope("data.host_load_pack");
    if (view.width == 0 || view.height == 0)
        throw std::invalid_argument(
            "Cannot build a GGGS training view with empty dimensions");
    Camera camera =
        training_data::training_camera(view, options, resolution_scale);
    const io::RgbImage source = [&] {
        tinytensor::TraceScope scope("data.read_decode");
        return io::load_rgb_with_minimum_size(view.path, camera.width, camera.height);
    }();
    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    io::GrayImage source_mask;
    bool has_source_mask = false;
    if (options.use_mask) {
        const auto mask_path = resolve_mask_path(view, options);
        if (!mask_path.empty()) {
            source_mask = io::load_gray(mask_path);
            has_source_mask = true;
        } else {
            source_mask = io::load_alpha(view.path);
            has_source_mask = !source_mask.pixels.empty();
        }
    }
    const bool has_projected_mask =
        view.foreground_mask.size() ==
        static_cast<std::size_t>(view.width) * view.height;
    // Older/internal MVS paths used binary 0/1 masks while projected bounds
    // masks use 8-bit 0..255 coverage. Accept both representations so a
    // binary mask cannot silently become 255 times too transparent.
    const float projected_mask_denominator =
        has_projected_mask &&
            *std::max_element(
                view.foreground_mask.begin(), view.foreground_mask.end()) <= 1
        ? 1.F
        : 255.F;
    std::vector<int> rgba(pixels);
    const bool validity_only = options.ignore_undistortion_border &&
        !has_source_mask && !has_projected_mask &&
        !uses_native_splat_projection(camera.model) &&
        (view.source_model == CameraModel::opencv_fisheye ||
         view.k1 != 0.F || view.k2 != 0.F || view.p1 != 0.F || view.p2 != 0.F);
    const bool direct_source =
        camera.width == source.width && camera.height == source.height &&
        (uses_native_splat_projection(camera.model) ||
         (view.k1 == 0.F && view.k2 == 0.F &&
          view.p1 == 0.F && view.p2 == 0.F));
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (pixels >= 4096)
#endif
    for (std::int64_t linear = 0;
         linear < static_cast<std::int64_t>(pixels); ++linear) {
        const auto pixel = static_cast<std::size_t>(linear);
        const auto x = static_cast<std::uint32_t>(
            pixel % camera.width);
        const auto y = static_cast<std::uint32_t>(
            pixel / camera.width);
        std::uint8_t red{};
        std::uint8_t green{};
        std::uint8_t blue{};
        std::uint8_t alpha{255};
        if (direct_source) {
            red = source.pixels[3 * pixel];
            green = source.pixels[3 * pixel + 1];
            blue = source.pixels[3 * pixel + 2];
            if (has_source_mask) {
                const float coverage =
                    source_mask.width == source.width &&
                            source_mask.height == source.height
                        ? static_cast<float>(
                              source_mask.pixels[pixel]) /
                              255.F
                        : sample_mask_coverage(
                              source_mask, source,
                              static_cast<float>(x),
                              static_cast<float>(y));
                alpha = quantize_channel(coverage);
            }
            if (has_projected_mask)
                alpha = std::min(
                    alpha, quantize_channel(
                        sample_projected_foreground_coverage(
                            view, camera, x, y,
                            projected_mask_denominator)));
            rgba[pixel] = pack_rgba(red, green, blue, alpha);
            continue;
        }
        const auto [sx, sy] =
            source_coordinate(view, x, y, camera, source);
        if (validity_only && (sx < 0.F || sy < 0.F ||
                sx > static_cast<float>(source.width - 1) ||
                sy > static_cast<float>(source.height - 1)))
            alpha = 0;
        sample_rgb(source, sx, sy, red, green, blue);
        if (has_source_mask)
            alpha = quantize_channel(
                sample_mask_coverage(source_mask, source, sx, sy));
        if (has_projected_mask)
            alpha = std::min(
                alpha, quantize_channel(
                    sample_projected_foreground_coverage(
                        view, camera, x, y,
                        projected_mask_denominator)));
        rgba[pixel] = pack_rgba(red, green, blue, alpha);
    }

    std::vector<float> depth;
    std::vector<float> normals;
    if (!uses_native_splat_projection(camera.model) && options.use_mvs_depth &&
        camera.width == view.width && camera.height == view.height &&
        view.depth_map.depth.size() == pixels)
        depth = view.depth_map.depth;
    if (!uses_native_splat_projection(camera.model) && options.use_mvs_normals &&
        camera.width == view.width && camera.height == view.height &&
        view.depth_map.normal.size() == pixels) {
        normals.resize(3 * pixels);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            for (int axis = 0; axis < 3; ++axis)
                normals[static_cast<std::size_t>(axis) * pixels + pixel] =
                    view.depth_map.normal[pixel](axis);
        }
    }
    const bool has_mask = has_source_mask || has_projected_mask || validity_only;

    return {
        camera, std::move(rgba), std::move(depth), std::move(normals),
        has_mask, validity_only};
}

TrainingView upload_training_view(
    const HostTrainingView& host, const TrainingOptions& options) {
    TrainingView result;
    result.camera = host.camera;
    const bool decode_gray = options.multi_view_ncc_weight > 0.F;
    if (options.backend == TrainingBackend::vulkan) {
        const std::size_t pixels = host.rgba.size();
        std::vector<float> rgb(3 * pixels);
        std::vector<float> gray(decode_gray ? pixels : 0);
        std::vector<float> mask(host.has_mask ? pixels : 0);
        constexpr float inverse_255 = 1.F / 255.F;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::uint32_t value = std::bit_cast<std::uint32_t>(
                host.rgba[pixel]);
            const float red = float(value & 0xffU) * inverse_255;
            const float green = float((value >> 8U) & 0xffU) * inverse_255;
            const float blue = float((value >> 16U) & 0xffU) * inverse_255;
            rgb[pixel] = red;
            rgb[pixels + pixel] = green;
            rgb[2 * pixels + pixel] = blue;
            if (decode_gray)
                gray[pixel] = 0.299F * red + 0.587F * green + 0.114F * blue;
            if (host.has_mask)
                mask[pixel] = float((value >> 24U) & 0xffU) * inverse_255;
        }
        result.rgb = tinytensor::Tensor::from_vector(
            rgb, {std::size_t{3}, host.camera.height, host.camera.width},
            tinytensor::Device::Vulkan);
        result.gray = decode_gray
            ? tinytensor::Tensor::from_vector(
                  gray, {host.camera.height, host.camera.width},
                  tinytensor::Device::Vulkan)
            : tinytensor::Tensor::zeros(
                  {std::size_t{1}}, tinytensor::Device::Vulkan);
        result.mask = host.has_mask
            ? tinytensor::Tensor::from_vector(
                  mask, {host.camera.height, host.camera.width},
                  tinytensor::Device::Vulkan)
            : tinytensor::Tensor::zeros(
                  {std::size_t{1}}, tinytensor::Device::Vulkan);
        result.depth = host.depth.empty()
            ? tinytensor::Tensor::zeros({1}, tinytensor::Device::Vulkan)
            : tinytensor::Tensor::from_vector(
                  host.depth, {host.camera.height, host.camera.width},
                  tinytensor::Device::Vulkan);
        result.normal = host.normal.empty()
            ? tinytensor::Tensor::zeros({1}, tinytensor::Device::Vulkan)
            : tinytensor::Tensor::from_vector(
                  host.normal, {3, host.camera.height, host.camera.width},
                  tinytensor::Device::Vulkan);
        result.has_mask = host.has_mask;
        result.mask_is_validity = host.mask_is_validity;
        return result;
    }
    auto decoded = detail::upload_packed_training_pixels(
        host.rgba, host.camera.width, host.camera.height, host.has_mask,
        decode_gray);
    result.rgb = std::move(decoded.rgb);
    result.gray = std::move(decoded.gray);
    result.depth = host.depth.empty()
        ? tinytensor::Tensor::zeros({1}, tinytensor::Device::CUDA)
        : tinytensor::Tensor::from_vector(
              host.depth, {host.camera.height, host.camera.width},
              tinytensor::Device::CUDA);
    result.normal = host.normal.empty()
        ? tinytensor::Tensor::zeros({1}, tinytensor::Device::CUDA)
        : tinytensor::Tensor::from_vector(
              host.normal, {3, host.camera.height, host.camera.width},
              tinytensor::Device::CUDA);
    result.mask = std::move(decoded.mask);
    result.has_mask = host.has_mask;
    result.mask_is_validity = host.mask_is_validity;
    return result;
}

}  // namespace

namespace training_data {

struct TrainingDataLoader::Impl {
    Impl(
        const std::vector<mvs::MvsView>& source,
        const TrainingOptions& options, const float resolution_scale = 1.F)
        : source_(source), options_(options),
          capacity_bytes_(
              options.adaptive_training_cache
                  ? std::size_t{0}
                  : options.training_view_cache_bytes),
          resolution_scale_(resolution_scale) {
        if (options_.backend == TrainingBackend::cuda &&
            options_.training_async_upload &&
            options_.training_device_cache_bytes != 0) {
            const cudaError_t error = cudaStreamCreateWithFlags(
                &copy_stream_, cudaStreamNonBlocking);
            if (error != cudaSuccess)
                throw std::runtime_error(
                    std::string("Failed to create splat upload stream: ") +
                    cudaGetErrorString(error));
        }
        try {
            update_cache_budgets();
        } catch (...) {
            if (copy_stream_ != nullptr) cudaStreamDestroy(copy_stream_);
            copy_stream_ = nullptr;
            throw;
        }
        core::Logger::instance().info(
            "splat_data_cache host_budget_bytes=", capacity_bytes_,
            " device_budget_bytes=", device_capacity_bytes_,
            " dataset_packed_bytes=", dataset_packed_bytes_,
            " adaptive=", options_.adaptive_training_cache ? 1 : 0);
    }

    ~Impl() {
        prefetches_.clear();
        clear_pending_device_uploads();
        staging_pool_.clear();
        device_ring_.clear();
        if (copy_stream_ != nullptr) {
            const cudaError_t error = cudaStreamDestroy(copy_stream_);
            if (error != cudaSuccess)
                core::Logger::instance().warning(
                    "Failed to destroy splat upload stream: ",
                    cudaGetErrorString(error));
            copy_stream_ = nullptr;
        }
        // The plan order belongs to the caller; stop referring to it before the
        // caller's storage goes away.
        plan_order_ = nullptr;
    }

    TrainingView get(const std::size_t index) {
        tinytensor::TraceScope get_scope("data.get_wall");
        struct Timer {
            double& total;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~Timer() { total += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count(); }
        } timer{get_wall_ms_};
        tinytensor::VramScope scope("data.decode");
        ++requests_;
        note_request_cadence();
        collect_ready_host_prefetches();
        if (options_.backend == TrainingBackend::vulkan) {
            const auto cached = vulkan_lookup_.find(index);
            if (cached != vulkan_lookup_.end()) {
                ++device_hits_;
                device_hit_rate_ = static_cast<double>(device_hits_) /
                    static_cast<double>(requests_);
                vulkan_entries_.splice(
                    vulkan_entries_.begin(), vulkan_entries_, cached->second);
                return vulkan_entries_.front().view;
            }
            const bool host_resident = lookup_.contains(index);
            const HostTrainingView& host = host_view(index);
            if (host_resident) ++host_hits_;
            TrainingView view = upload_training_view(host, options_);
            uploaded_bytes_ += packed_host_bytes(host);
            device_hit_rate_ = static_cast<double>(device_hits_) /
                static_cast<double>(requests_);
            insert_vulkan_entry(index, view);
            return view;
        }
        const auto found = device_lookup_.find(index);
        if (found != device_lookup_.end()) {
            ++device_hits_;
            device_entries_.splice(
                device_entries_.begin(), device_entries_, found->second);
            // Budget feedback runs only after the entry sits at the front of the
            // list: evaluating may evict entries, and `found` must not be used
            // after that.
            note_device_cache_access(true);
            return decode_device_view(device_entries_.front());
        }
        const auto pending = device_uploads_.find(index);
        if (pending != device_uploads_.end()) {
            // Count it as a cache miss for budget feedback: the transfer was
            // still necessary even when it completed before get(). Budget
            // evaluation is deferred while any transfer owns reserved bytes.
            note_device_cache_access(false);
            DeviceEntry staged_entry;
            if (consume_device_upload(index, staged_entry)) {
                ++async_upload_hits_;
                if (insert_device_entry(std::move(staged_entry)))
                    return decode_device_view(device_entries_.front());
                // No room: the hop already landed in this entry, so serve the
                // view from it instead of uploading again.
                return decode_device_view(staged_entry);
            }
        } else {
            note_device_cache_access(false);
        }
        const bool host_resident = lookup_.contains(index);
        const HostTrainingView& host = host_view(index);
        if (host_resident) ++host_hits_;
        const std::size_t bytes = sizeof(int) * host.rgba.size() +
            sizeof(float) * (host.depth.size() + host.normal.size());
        uploaded_bytes_ += bytes;
        if (device_capacity_bytes_ == 0 || bytes > device_capacity_bytes_)
            return upload_training_view(host, options_);
        if (!make_room_for_device_bytes(bytes))
            return upload_training_view(host, options_);
        DeviceEntry entry = allocate_device_entry(
            index, host.camera, host.has_mask, host.mask_is_validity,
            !host.depth.empty(), !host.normal.empty());
        copy_device_entry_with_pinned_staging(host, entry);
        insert_device_entry(std::move(entry));
        return decode_device_view(device_entries_.front());
    }

    void prefetch(const std::size_t index) {
        if (index >= source_.size())
            throw std::out_of_range(
                "GGGS training prefetch index is out of range");
        if (options_.training_prefetch_views == 0)
            return;
        collect_ready_host_prefetches();
        // Host decoding is the only background-thread work. CUDA allocation,
        // staging and copy enqueue all remain on this training thread.
        if (device_lookup_.contains(index) || device_uploads_.contains(index))
            return;
        const auto cached = lookup_.find(index);
        if (cached != lookup_.end()) {
            if (copy_stream_ != nullptr &&
                device_uploads_.size() < k_max_pending_device_uploads)
                schedule_device_upload(index, *cached->second->view);
            return;
        }
        // Already decoded: decoding it again would only occupy a lookahead slot
        // that a genuinely cold view needs (the earlier implementation re-read
        // every already-cached view on every iteration).
        if (prefetches_.contains(index) ||
            prefetches_.size() >= prefetch_limit()) return;
        const float scale = resolution_scale_;
        prefetches_.emplace(
            index,
            std::async(
                std::launch::async,
                [this, index, scale] {
                    HostTrainingLoad load;
                    const auto start = std::chrono::steady_clock::now();
                    load.view = std::make_shared<HostTrainingView>(
                        load_host_training_view(source_[index], options_, scale));
                    load.host_load_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
                    return load;
                }));
        ++prefetch_issued_;
    }

    bool has_mask(const std::size_t index) {
        if (index >= source_.size())
            throw std::out_of_range(
                "GGGS training view index is out of range");
        const auto found = device_lookup_.find(index);
        if (found != device_lookup_.end())
            return found->second->has_mask && !found->second->mask_is_validity;
        if (const auto cached = cached_host_view(index))
            return cached->has_mask && !cached->mask_is_validity;

        const mvs::MvsView& view = source_[index];
        if (view.foreground_mask.size() ==
            static_cast<std::size_t>(view.width) * view.height)
            return true;
        if (!options_.use_mask) return false;
        if (!resolve_mask_path(view, options_).empty()) return true;
        return io::image_has_alpha(view.path);
    }

    void set_resolution_scale(const float scale) {
        const float clamped = std::clamp(scale, 1e-3F, 1.F);
        if (std::abs(clamped - resolution_scale_) < 1e-6F) return;
        // std::future from std::launch::async joins on destruction. Clear all
        // old-scale work before publishing the new scale and dropping buffers.
        clear_pending_device_uploads();
        entries_.clear();
        lookup_.clear();
        cached_bytes_ = 0;
        device_lookup_.clear();
        device_entries_.clear();
        device_cached_bytes_ = 0;
        vulkan_lookup_.clear();
        vulkan_entries_.clear();
        // Budget feedback starts over at the new scale.
        device_budget_window_requests_ = 0;
        device_budget_window_hits_ = 0;
        device_budget_window_step_ms_ = 0.0;
        device_budget_saturated_ = false;
        device_budget_previous_capacity_ = 0;
        device_budget_baseline_step_ms_ = 0.0;
        prefetches_.clear();
        resolution_scale_ = clamped;
        update_cache_budgets();
        staging_.ensure(0);
        staging_pool_.clear();
        device_ring_.drain();
    }

    void set_epoch_plan(
        const std::vector<std::size_t>* order, const std::size_t cursor) {
        if (order == nullptr || order->empty()) {
            plan_order_ = nullptr;
            plan_position_.clear();
            plan_cursor_ = 0;
            return;
        }
        // The trainer reshuffles in place and resets its cursor to zero, so a
        // cursor that moves backwards marks a fresh epoch that needs a new
        // position map.
        if (order != plan_order_ || order->size() != plan_size_ ||
            cursor < plan_cursor_ || plan_position_.size() != source_.size()) {
            plan_position_.assign(source_.size(), k_absent_plan_position);
            for (std::size_t position = 0; position < order->size(); ++position) {
                const std::size_t index = (*order)[position];
                if (index < plan_position_.size())
                    plan_position_[index] =
                        static_cast<std::uint32_t>(position);
            }
            plan_order_ = order;
            plan_size_ = order->size();
        }
        plan_cursor_ = cursor;
    }

    void ensure_device_headroom(const std::size_t bytes) {
        if (options_.backend == TrainingBackend::vulkan) return;
        if (bytes == 0) return;
        std::size_t free_bytes{}, total_bytes{};
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
            free_bytes >= bytes) {
            return;
        }

        const std::size_t free_before = free_bytes;
        const std::size_t resident =
            device_cached_bytes_ + device_pending_bytes_;
        clear_pending_device_uploads();
        device_lookup_.clear();
        device_entries_.clear();
        device_cached_bytes_ = 0;
        device_capacity_bytes_ = 0;
        tinytensor::Tensor::trim_memory_pool();
        cudaMemGetInfo(&free_bytes, &total_bytes);
        core::Logger::instance().warning(
            "splat_data_cache low_vram_headroom_bytes=", bytes,
            " free_before_bytes=", free_before,
            " released_packed_bytes=", resident,
            " action=disable_device_cache");
    }

    CacheStats stats() const {
        CacheStats result;
        result.requests = requests_;
        result.device_hits = device_hits_;
        result.async_upload_hits = async_upload_hits_;
        result.async_upload_issued = async_upload_issued_;
        result.async_upload_waits = async_upload_waits_;
        result.async_upload_failures = async_upload_failures_;
        result.async_upload_pending = device_uploads_.size();
        result.async_upload_pending_bytes = device_pending_bytes_;
        result.uploaded_bytes = uploaded_bytes_;
        result.device_resident_bytes = device_cached_bytes_;
        result.device_budget_bytes = device_capacity_bytes_;
        result.device_hit_rate = device_hit_rate_;
        result.device_budget_ceiling_bytes = device_ceiling_bytes_;
        result.device_budget_growths = device_budget_growths_;
        result.device_budget_rollbacks = device_budget_rollbacks_;
        result.dataset_packed_bytes = dataset_packed_bytes_;
        result.host_budget_bytes = capacity_bytes_;
        result.get_wall_ms = get_wall_ms_;
        result.host_hits = host_hits_;
        result.prefetch_issued = prefetch_issued_;
        result.host_load_mean_ms = host_load_mean_ms_;
        result.step_mean_ms = step_mean_ms_;
        result.prefetch_depth = prefetch_limit();
        return result;
    }

    [[nodiscard]] std::size_t prefetch_depth() const {
        return prefetch_limit();
    }

private:
    struct DeviceEntry {
        std::size_t index{};
        std::size_t bytes{};
        Camera camera;
        tinytensor::Tensor rgba, depth, normal;
        bool has_mask{};
        bool mask_is_validity{};
    };
    using DeviceEntries = std::list<DeviceEntry>;

    struct VulkanDeviceEntry {
        std::size_t index{};
        std::size_t bytes{};
        TrainingView view;
    };
    using VulkanDeviceEntries = std::list<VulkanDeviceEntry>;

    struct PendingDeviceUpload {
        std::size_t bytes{};
        Camera camera;
        bool has_mask{};
        bool mask_is_validity{};
        bool has_depth{};
        bool has_normal{};
        // Loader-owned device staging the copy engine is allowed to write.
        DeviceUploadRing::Slot* slot{};
        std::unique_ptr<PinnedStagingBuffer> staging;
        // Diagnostic (SPLAT_VERIFY_UPLOAD=1): the packed bytes the transfer is
        // supposed to deliver, so the consumed tensor can be compared against
        // them before use.
        std::vector<int> expected_rgba;
    };

    [[nodiscard]] static std::size_t packed_host_bytes(
        const HostTrainingView& host) {
        return sizeof(int) * host.rgba.size() +
            sizeof(float) * (host.depth.size() + host.normal.size());
    }

    [[nodiscard]] static std::size_t vulkan_view_bytes(
        const TrainingView& view) {
        return saturate_add(
            view.rgb.bytes(),
            saturate_add(
                view.gray.bytes(),
                saturate_add(
                    view.mask.bytes(),
                    saturate_add(view.depth.bytes(), view.normal.bytes()))));
    }

    bool make_room_for_vulkan_bytes(const std::size_t bytes) {
        if (bytes > device_capacity_bytes_) return false;
        while (!vulkan_entries_.empty() &&
               device_cached_bytes_ + bytes > device_capacity_bytes_) {
            const auto evicted = select_eviction_victim(vulkan_entries_);
            device_cached_bytes_ -= evicted->bytes;
            vulkan_lookup_.erase(evicted->index);
            vulkan_entries_.erase(evicted);
        }
        return device_cached_bytes_ + bytes <= device_capacity_bytes_;
    }

    void insert_vulkan_entry(
        const std::size_t index, const TrainingView& view) {
        VulkanDeviceEntry entry;
        entry.index = index;
        entry.bytes = vulkan_view_bytes(view);
        entry.view = view;
        if (!make_room_for_vulkan_bytes(entry.bytes)) return;
        vulkan_entries_.push_front(std::move(entry));
        vulkan_lookup_[index] = vulkan_entries_.begin();
        device_cached_bytes_ += vulkan_entries_.front().bytes;
    }

    // Diagnostic switch (SPLAT_VERIFY_UPLOAD=1): compare the packed bytes a
    // transfer delivered against the host view they were copied from. A
    // mismatch means the copy stream and the pool/allocator disagreed about who
    // owns the memory a transfer touched.
    [[nodiscard]] static bool verify_uploads() {
        static const bool enabled = [] {
            const char* value = std::getenv("SPLAT_VERIFY_UPLOAD");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }

    [[nodiscard]] DeviceEntry allocate_device_entry(
        const std::size_t index, const Camera& camera, const bool has_mask,
        const bool mask_is_validity, const bool has_depth,
        const bool has_normal) const {
        DeviceEntry entry;
        entry.index = index;
        entry.bytes = sizeof(int) * camera.width * camera.height;
        if (has_depth)
            entry.bytes += sizeof(float) * camera.width * camera.height;
        if (has_normal)
            entry.bytes += sizeof(float) * 3 * camera.width * camera.height;
        entry.camera = camera;
        entry.has_mask = has_mask;
        entry.mask_is_validity = mask_is_validity;
        entry.rgba = tinytensor::Tensor::empty(
            {camera.height, camera.width},
            tinytensor::Device::CUDA, tinytensor::DataType::Int32);
        if (has_depth)
            entry.depth = tinytensor::Tensor::empty(
                {camera.height, camera.width},
                tinytensor::Device::CUDA);
        if (has_normal)
            entry.normal = tinytensor::Tensor::empty(
                {std::size_t{3}, camera.height, camera.width},
                tinytensor::Device::CUDA);
        return entry;
    }

    static void pack_device_staging(
        const HostTrainingView& host, PinnedStagingBuffer& staging) {
        const std::size_t rgba_bytes = sizeof(int) * host.rgba.size();
        const std::size_t depth_bytes = sizeof(float) * host.depth.size();
        const std::size_t normal_bytes = sizeof(float) * host.normal.size();
        staging.ensure(rgba_bytes + depth_bytes + normal_bytes);
        std::uint8_t* rgba_destination = staging.data();
        std::uint8_t* depth_destination = rgba_destination + rgba_bytes;
        std::uint8_t* normal_destination = depth_destination + depth_bytes;
        std::memcpy(rgba_destination, host.rgba.data(), rgba_bytes);
        if (depth_bytes != 0)
            std::memcpy(depth_destination, host.depth.data(), depth_bytes);
        if (normal_bytes != 0)
            std::memcpy(normal_destination, host.normal.data(), normal_bytes);
    }

    void copy_device_entry_with_pinned_staging(
        const HostTrainingView& host, DeviceEntry& entry) {
        const std::size_t rgba_bytes =
            sizeof(int) * host.rgba.size();
        const std::size_t depth_bytes =
            sizeof(float) * host.depth.size();
        const std::size_t normal_bytes =
            sizeof(float) * host.normal.size();
        const std::size_t total_bytes =
            rgba_bytes + depth_bytes + normal_bytes;
        staging_.ensure(total_bytes);
        std::uint8_t* destination = staging_.data();
        std::uint8_t* rgba_source = destination;
        std::uint8_t* depth_source = rgba_source + rgba_bytes;
        std::uint8_t* normal_source = depth_source + depth_bytes;
        {
            tinytensor::TraceScope scope("data.pinned_memcpy");
            std::memcpy(rgba_source, host.rgba.data(), rgba_bytes);
            if (depth_bytes != 0) std::memcpy(depth_source, host.depth.data(), depth_bytes);
            if (normal_bytes != 0) std::memcpy(normal_source, host.normal.data(), normal_bytes);
        }
        tinytensor::TraceScope copy_scope("data.h2d_copy");

        const auto check_copy = [](const cudaError_t error) {
            if (error != cudaSuccess)
                throw std::runtime_error(
                    std::string("Failed to copy packed view to CUDA: ") +
                    cudaGetErrorString(error));
        };
        // Synchronous on the calling thread: the data loader never issues CUDA
        // work from a background thread, so there is no cross-stream lifetime
        // to reason about and the pinned buffer is free to reuse on return.
        check_copy(cudaMemcpy(
            entry.rgba.data_ptr(), rgba_source, rgba_bytes,
            cudaMemcpyHostToDevice));
        if (depth_bytes != 0)
            check_copy(cudaMemcpy(
                entry.depth.data_ptr(), depth_source, depth_bytes,
                cudaMemcpyHostToDevice));
        if (normal_bytes != 0)
            check_copy(cudaMemcpy(
                entry.normal.data_ptr(), normal_source, normal_bytes,
                cudaMemcpyHostToDevice));
    }

    bool schedule_device_upload(
        const std::size_t index,
        const HostTrainingView& host) {
        if (copy_stream_ == nullptr ||
            device_uploads_.contains(index) ||
            device_lookup_.contains(index) ||
            device_uploads_.size() >= k_max_pending_device_uploads)
            return false;
        const std::size_t bytes = packed_host_bytes(host);
        if (device_capacity_bytes_ == 0 || bytes > device_capacity_bytes_ ||
            !make_room_for_device_bytes(bytes))
            return false;

        const auto [pending, inserted] = device_uploads_.try_emplace(index);
        if (!inserted) return false;
        PendingDeviceUpload& upload = pending->second;
        DeviceUploadRing::Slot* slot = device_ring_.try_acquire(bytes);
        if (slot == nullptr) {
            device_uploads_.erase(pending);
            return false;
        }
        upload.slot = slot;
        upload.bytes = bytes;
        upload.camera = host.camera;
        upload.has_mask = host.has_mask;
        upload.mask_is_validity = host.mask_is_validity;
        upload.has_depth = !host.depth.empty();
        upload.has_normal = !host.normal.empty();
        const auto check = [](const cudaError_t error, const char* action) {
            if (error != cudaSuccess)
                throw std::runtime_error(
                    std::string(action) + ": " + cudaGetErrorString(error));
        };
        try {
            // Host packing and every CUDA call stay on the training thread; only
            // the DMA itself executes asynchronously, and it writes the
            // loader-owned slot rather than pool memory.
            upload.staging = staging_pool_.acquire(bytes);
            {
                tinytensor::TraceScope scope("data.async_pinned_memcpy");
                pack_device_staging(host, *upload.staging);
            }
            if (verify_uploads()) upload.expected_rgba = host.rgba;
            // The packed layout is contiguous in both the pinned buffer and the
            // slot, so one transfer covers RGBA, depth and normals.
            check(cudaMemcpyAsync(
                      slot->data, upload.staging->data(), bytes,
                      cudaMemcpyHostToDevice, copy_stream_),
                  "Failed to enqueue packed view upload");
            DeviceUploadRing::mark_ready(*slot, copy_stream_);
            device_pending_bytes_ += bytes;
            ++async_upload_issued_;
            return true;
        } catch (const std::exception& error) {
            // A partially enqueued transfer still owns the slot: drain it before
            // the slot can be handed out again.
            cudaStreamSynchronize(copy_stream_);
            staging_pool_.release(std::move(upload.staging));
            if (slot != nullptr) DeviceUploadRing::release(*slot);
            device_uploads_.erase(pending);
            ++async_upload_failures_;
            core::Logger::instance().warning(
                "splat_data_cache async_upload_schedule_failed index=", index,
                " error=", error.what());
            return false;
        }
    }

    static void check_cuda(const cudaError_t error, const char* action) {
        if (error != cudaSuccess)
            throw std::runtime_error(
                std::string(action) + ": " + cudaGetErrorString(error));
    }

    // Adopt a staged packed view. The transfer landed in loader-owned memory, so
    // the packed bytes are hopped into the entry on the compute stream before the
    // slot is recycled: the pool never sees a copy-engine write, and the hop is
    // ordered ahead of the expansion kernels that read it.
    [[nodiscard]] bool consume_device_upload(
        const std::size_t index, DeviceEntry& entry) {
        const auto found = device_uploads_.find(index);
        if (found == device_uploads_.end()) return false;
        PendingDeviceUpload upload = std::move(found->second);
        device_uploads_.erase(found);
        device_pending_bytes_ -= upload.bytes;

        if (upload.slot == nullptr) {
            ++async_upload_failures_;
            return false;
        }
        const cudaError_t query = cudaEventQuery(upload.slot->ready);
        if (query == cudaErrorNotReady) ++async_upload_waits_;
        const cudaError_t sync = query == cudaSuccess
            ? cudaSuccess
            : cudaEventSynchronize(upload.slot->ready);
        if (verify_uploads() && !upload.expected_rgba.empty()) {
            std::vector<int> observed(upload.expected_rgba.size());
            const cudaError_t read = cudaMemcpy(
                observed.data(), upload.slot->data,
                observed.size() * sizeof(int), cudaMemcpyDeviceToHost);
            std::size_t first_mismatch = observed.size();
            if (read == cudaSuccess)
                for (std::size_t i = 0; i < observed.size(); ++i)
                    if (observed[i] != upload.expected_rgba[i]) {
                        first_mismatch = i;
                        break;
                    }
            if (read != cudaSuccess || first_mismatch != observed.size())
                core::Logger::instance().warning(
                    "splat_data_cache async_upload_data_mismatch index=", index,
                    " first_mismatch=", first_mismatch,
                    " words=", observed.size(),
                    " read_error=", cudaGetErrorString(read));
        }
        staging_pool_.release(std::move(upload.staging));
        if (sync != cudaSuccess) {
            ++async_upload_failures_;
            core::Logger::instance().warning(
                "splat_data_cache async_upload_consume_failed index=", index,
                " sync_error=", cudaGetErrorString(sync));
            return false;
        }
        const cudaStream_t compute_stream = tinytensor::getCurrentCUDAStream();
        entry = allocate_device_entry(
            index, upload.camera, upload.has_mask, upload.mask_is_validity,
            upload.has_depth, upload.has_normal);
        const std::size_t rgba_bytes = sizeof(int) * entry.camera.width * entry.camera.height;
        const std::size_t depth_bytes = upload.has_depth
            ? sizeof(float) * entry.camera.width * entry.camera.height
            : 0;
        const std::size_t normal_bytes = upload.has_normal
            ? sizeof(float) * 3 * entry.camera.width * entry.camera.height
            : 0;
        const auto hop = [&](void* target, const std::size_t count,
                             const std::size_t offset) {
            if (count == 0) return;
            check_cuda(cudaMemcpyAsync(
                target, upload.slot->data + offset, count,
                cudaMemcpyDeviceToDevice, compute_stream),
                "Failed to hop packed view into the cache: ");
        };
        hop(entry.rgba.data_ptr(), rgba_bytes, 0);
        hop(entry.depth.data_ptr(), depth_bytes, rgba_bytes);
        hop(entry.normal.data_ptr(), normal_bytes, rgba_bytes + depth_bytes);
        DeviceUploadRing::mark_reusable(*upload.slot, compute_stream);
        DeviceUploadRing::release(*upload.slot);
        uploaded_bytes_ += entry.bytes;
        return true;
    }

    void clear_pending_device_uploads() noexcept {
        for (auto& [index, upload] : device_uploads_) {
            // The transfer owns loader-owned staging: wait for it before the slot
            // can be handed to another upload.
            const cudaError_t sync =
                upload.slot == nullptr || upload.slot->ready == nullptr
                ? cudaSuccess
                : cudaEventSynchronize(upload.slot->ready);
            if (sync != cudaSuccess)
                core::Logger::instance().warning(
                    "splat_data_cache async_upload_cleanup_failed index=", index,
                    " sync_error=", cudaGetErrorString(sync));
            if (upload.slot != nullptr) DeviceUploadRing::release(*upload.slot);
            staging_pool_.release(std::move(upload.staging));
        }
        device_uploads_.clear();
        device_pending_bytes_ = 0;
    }

    [[nodiscard]] bool make_room_for_device_bytes(const std::size_t bytes) {
        while (!device_entries_.empty() &&
               device_cached_bytes_ + device_pending_bytes_ + bytes >
                   device_capacity_bytes_) {
            const auto evicted = select_eviction_victim(device_entries_);
            device_cached_bytes_ -= evicted->bytes;
            device_lookup_.erase(evicted->index);
            device_entries_.erase(evicted);
        }
        return device_cached_bytes_ + device_pending_bytes_ + bytes <=
            device_capacity_bytes_;
    }

    bool insert_device_entry(DeviceEntry&& entry) {
        if (!make_room_for_device_bytes(entry.bytes)) return false;
        const std::size_t index = entry.index;
        device_entries_.push_front(std::move(entry));
        device_lookup_[index] = device_entries_.begin();
        device_cached_bytes_ += device_entries_.front().bytes;
        return true;
    }

    TrainingView decode_device_view(const DeviceEntry& entry) const {
        TrainingView result;
        result.camera = entry.camera;
        result.has_mask = entry.has_mask;
        result.mask_is_validity = entry.mask_is_validity;
        auto decoded = detail::decode_packed_training_pixels(
            entry.rgba, entry.camera.width, entry.camera.height,
            entry.has_mask, options_.multi_view_ncc_weight > 0.F);
        result.rgb = std::move(decoded.rgb);
        result.gray = std::move(decoded.gray);
        result.mask = std::move(decoded.mask);
        // Own the returned supervision tensors independently of the LRU:
        // callers may retain a reference view while loading a neighbour.
        result.depth = entry.depth.is_valid() ? entry.depth
            : tinytensor::Tensor::zeros({1}, tinytensor::Device::CUDA);
        result.normal = entry.normal.is_valid() ? entry.normal
            : tinytensor::Tensor::zeros({1}, tinytensor::Device::CUDA);
        return result;
    }

    struct Entry {
        std::size_t index{};
        std::size_t bytes{};
        std::shared_ptr<HostTrainingView> view;
    };
    using Entries = std::list<Entry>;

    const HostTrainingView& host_view(const std::size_t index) {
        tinytensor::TraceScope scope("data.host_get_wait");
        if (index >= source_.size())
            throw std::out_of_range(
                "GGGS training view index is out of range");
        if (const auto cached = cached_host_view(index))
            return *cached;

        std::shared_ptr<HostTrainingView> loaded;
        const auto prefetched = prefetches_.find(index);
        if (prefetched != prefetches_.end()) {
            HostTrainingLoad result = prefetched->second.get();
            prefetches_.erase(prefetched);
            note_host_load(result.host_load_ms);
            loaded = std::move(result.view);
        } else {
            const auto start = std::chrono::steady_clock::now();
            loaded = std::make_shared<HostTrainingView>(
                load_host_training_view(
                    source_[index], options_, resolution_scale_));
            note_host_load(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        }
        store_host_view(index, std::move(loaded));
        return *entries_.front().view;
    }

    void store_host_view(
        const std::size_t index,
        std::shared_ptr<HostTrainingView> loaded) {
        if (cached_host_view(index)) return;
        const std::size_t loaded_bytes = loaded->bytes();
        while (!entries_.empty() &&
               (capacity_bytes_ == 0 ||
                cached_bytes_ + loaded_bytes > capacity_bytes_)) {
            const auto evicted = select_eviction_victim(entries_);
            cached_bytes_ -= evicted->bytes;
            lookup_.erase(evicted->index);
            entries_.erase(evicted);
        }
        // Keep one decoded view even when caching is disabled or a single
        // image exceeds the budget. It remains valid until the next miss.
        entries_.push_front({index, loaded_bytes, std::move(loaded)});
        lookup_[index] = entries_.begin();
        cached_bytes_ += loaded_bytes;
    }

    void collect_ready_host_prefetches() {
        for (auto it = prefetches_.begin(); it != prefetches_.end();) {
            if (it->second.wait_for(std::chrono::seconds{0}) !=
                std::future_status::ready) {
                ++it;
                continue;
            }
            HostTrainingLoad result = it->second.get();
            const std::size_t index = it->first;
            it = prefetches_.erase(it);
            note_host_load(result.host_load_ms);
            store_host_view(index, std::move(result.view));
        }
    }

    std::shared_ptr<HostTrainingView> cached_host_view(
        const std::size_t index) {
        const auto found = lookup_.find(index);
        if (found == lookup_.end()) return {};
        entries_.splice(entries_.begin(), entries_, found->second);
        return entries_.front().view;
    }

    const std::vector<mvs::MvsView>& source_;
    const TrainingOptions& options_;
    cudaStream_t copy_stream_{};
    // Pinned staging for the packed-view H2D copy, reused by the training
    // thread's synchronous fallback.
    PinnedStagingBuffer staging_;
    PinnedStagingPool staging_pool_;
    // Loader-owned device staging for the copy-stream lookahead. Never pool
    // memory: see DeviceUploadRing.
    DeviceUploadRing device_ring_;
    std::size_t capacity_bytes_{};
    std::size_t cached_bytes_{};
    std::size_t device_capacity_bytes_{};
    std::size_t device_cached_bytes_{};
    std::size_t device_pending_bytes_{};
    std::size_t device_ceiling_bytes_{};
    std::size_t device_floor_bytes_{};
    std::size_t training_reserve_bytes_{};
    std::size_t device_budget_window_requests_{};
    std::size_t device_budget_window_hits_{};
    std::size_t device_budget_growths_{};
    std::size_t device_budget_rollbacks_{};
    std::size_t device_budget_previous_capacity_{};
    double device_budget_window_step_ms_{};
    double device_budget_baseline_step_ms_{};
    bool device_budget_saturated_{false};
    double device_hit_rate_{};
    std::size_t dataset_packed_bytes_{};
    std::size_t requests_{}, device_hits_{}, uploaded_bytes_{};
    std::size_t async_upload_hits_{};
    std::size_t async_upload_issued_{};
    std::size_t async_upload_waits_{};
    std::size_t async_upload_failures_{};
    std::size_t host_hits_{};
    std::size_t prefetch_issued_{};
    double get_wall_ms_{};
    // Prefetch lookahead sizing: how long a host load takes versus how long one
    // training iteration takes. Both are exponentially smoothed on the training
    // thread only.
    double host_load_mean_ms_{};
    double step_mean_ms_{};
    std::chrono::steady_clock::time_point last_request_time_{};
    bool has_last_request_time_{false};
    DeviceEntries device_entries_;
    std::unordered_map<std::size_t, DeviceEntries::iterator> device_lookup_;
    VulkanDeviceEntries vulkan_entries_;
    std::unordered_map<std::size_t, VulkanDeviceEntries::iterator>
        vulkan_lookup_;
    std::unordered_map<std::size_t, PendingDeviceUpload> device_uploads_;
    float resolution_scale_{1.F};
    Entries entries_;
    std::unordered_map<std::size_t, Entries::iterator> lookup_;
    std::unordered_map<std::size_t, std::future<HostTrainingLoad>> prefetches_;
    // Epoch access plan. Positions index the trainer's shuffled order; entries
    // outside the plan (for example held-out views) are evicted first.
    static constexpr std::uint32_t k_absent_plan_position =
        std::numeric_limits<std::uint32_t>::max();
    static constexpr double k_timing_ema = 0.1;
    const std::vector<std::size_t>* plan_order_{};
    std::vector<std::uint32_t> plan_position_;
    std::size_t plan_cursor_{};
    std::size_t plan_size_{};

    void note_request_cadence() {
        const auto now = std::chrono::steady_clock::now();
        if (has_last_request_time_) {
            const double ms = std::chrono::duration<double, std::milli>(
                now - last_request_time_).count();
            step_mean_ms_ = step_mean_ms_ <= 0.0
                ? ms
                : (1.0 - k_timing_ema) * step_mean_ms_ + k_timing_ema * ms;
            device_budget_window_step_ms_ += ms;
        }
        last_request_time_ = now;
        has_last_request_time_ = true;
    }

    void note_host_load(const double host_load_ms) {
        if (!std::isfinite(host_load_ms) || host_load_ms <= 0.0) return;
        host_load_mean_ms_ = host_load_mean_ms_ <= 0.0
            ? host_load_ms
            : (1.0 - k_timing_ema) * host_load_mean_ms_ +
                  k_timing_ema * host_load_ms;
    }

    // A host load must be started far enough ahead of its use to cover one
    // decode plus one iteration of slack. Measured decode and iteration times
    // replace the previous fixed view count, bounded by a fixed share of host
    // memory for the in-flight packed views.
    [[nodiscard]] std::size_t prefetch_limit() const {
        const std::size_t configured = options_.training_prefetch_views;
        if (configured == 0) return 0;
        if (!options_.training_prefetch_adaptive ||
            host_load_mean_ms_ <= 0.0 || step_mean_ms_ <= 0.0)
            return configured;
        std::size_t ceiling =
            std::max<std::size_t>(saturate_multiply(configured, 4), 8);
        ceiling = std::min<std::size_t>(ceiling, 32);
        const std::size_t mean_view_bytes =
            source_.empty() ? 0 : dataset_packed_bytes_ / source_.size();
        if (mean_view_bytes != 0) {
            const std::size_t in_flight_budget = std::size_t{512} * k_mib;
            ceiling = std::min<std::size_t>(
                ceiling,
                std::max<std::size_t>(in_flight_budget / mean_view_bytes, 8));
        }
        if (ceiling <= configured) return configured;
        const double lookahead =
            host_load_mean_ms_ / std::max(step_mean_ms_, 1.0) + 1.0;
        return std::clamp<std::size_t>(
            static_cast<std::size_t>(std::ceil(lookahead)), configured,
            ceiling);
    }

    // Distance, in requests, until the view is needed again. Views outside the
    // current epoch are evicted before any scheduled view.
    [[nodiscard]] std::uint64_t next_use_distance(
        const std::size_t index) const {
        constexpr std::uint64_t absent =
            std::numeric_limits<std::uint64_t>::max();
        if (plan_order_ == nullptr || plan_position_.size() != source_.size())
            return absent;
        if (index >= plan_position_.size() ||
            plan_position_[index] == k_absent_plan_position)
            return absent;
        const std::size_t size = plan_order_->size();
        if (size == 0) return absent;
        const std::size_t position = plan_position_[index];
        const std::size_t cursor = plan_cursor_ % size;
        return position >= cursor ? position - cursor
                                  : position + size - cursor;
    }

    // Evict the entry that is needed farthest in the future. Ties keep the
    // least recently used entry, which is the list tail, so a loader without a
    // published epoch plan still behaves exactly like the previous LRU.
    template <typename EntryList>
    typename EntryList::iterator select_eviction_victim(EntryList& entries) {
        auto victim = std::prev(entries.end());
        if (plan_order_ == nullptr || entries.size() < 2) return victim;
        std::uint64_t victim_distance = next_use_distance(victim->index);
        for (auto candidate = std::prev(victim);; --candidate) {
            const std::uint64_t distance = next_use_distance(candidate->index);
            if (distance > victim_distance) {
                victim = candidate;
                victim_distance = distance;
            }
            if (candidate == entries.begin()) break;
        }
        return victim;
    }

    [[nodiscard]] std::size_t projected_gaussian_count() const noexcept {
        if (options_.enable_densification)
            return options_.densification_cap;
        return 0;
    }

    // Model, optimizer state and per-iteration raster scratch. Used as the
    // reserve the image cache must never eat into.
    [[nodiscard]] std::size_t projected_training_state_bytes() const {
        return saturate_add(
            std::size_t{3} * k_gib / 2,
            saturate_multiply(
                projected_gaussian_count(), std::size_t{2} * 1024));
    }

    // Initial growth decisions observe at least one epoch. Once a growth is
    // pending, 512 subsequent iterations are enough to determine whether the
    // extra hits improved end-to-end cadence without waiting for another full
    // large-dataset epoch.
    [[nodiscard]] std::size_t device_budget_window_requests() const {
        if (device_budget_previous_capacity_ != 0)
            return k_device_budget_window;
        return std::max<std::size_t>(k_device_budget_window, source_.size());
    }

    // Device cache budget feedback. Called once per get(); the expensive part
    // (a VRAM query plus a decision) runs once per window.
    void note_device_cache_access(const bool hit) {
        if (!options_.adaptive_training_cache || device_capacity_bytes_ == 0)
            return;
        ++device_budget_window_requests_;
        if (hit) ++device_budget_window_hits_;
        if (device_budget_window_requests_ >= device_budget_window_requests() &&
            device_uploads_.empty()) {
            const std::size_t intervals =
                device_budget_window_requests_ > 1
                ? device_budget_window_requests_ - 1
                : 1;
            evaluate_device_budget(
                device_budget_window_step_ms_ /
                static_cast<double>(intervals));
        }
    }

    void evaluate_device_budget(const double mean_step_ms) {
        const std::size_t requests = device_budget_window_requests_;
        const std::size_t hits = device_budget_window_hits_;
        device_budget_window_requests_ = 0;
        device_budget_window_hits_ = 0;
        device_budget_window_step_ms_ = 0.0;
        if (requests == 0) return;
        const double hit_rate =
            static_cast<double>(hits) / static_cast<double>(requests);
        device_hit_rate_ = hit_rate;
        std::size_t free_bytes{}, total_bytes{};
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
        const std::size_t reserve =
            std::max(training_reserve_bytes_, projected_training_state_bytes());
        const std::size_t capacity = device_capacity_bytes_;

        // The cache is competing with live training state: give memory back
        // rather than wait for a refinement to force it.
        if (free_bytes < reserve) {
            // Release the grown part first. The effective floor is the safe
            // initial budget after dataset, resolution and free-VRAM guards;
            // the configured value may be larger than that safe budget.
            const std::size_t floor_bytes = device_floor_bytes_;
            const std::size_t target =
                std::max<std::size_t>(capacity / 2, floor_bytes);
            if (target < capacity) {
                device_capacity_bytes_ = target;
                (void)make_room_for_device_bytes(0);
                core::Logger::instance().warning(
                    "splat_data_cache device_budget_shrink budget_bytes=",
                    target, " free_bytes=", free_bytes,
                    " reserve_bytes=", reserve, " hit_rate=", hit_rate);
            }
            device_budget_previous_capacity_ = 0;
            device_budget_baseline_step_ms_ = 0.0;
            return;
        }

        // Judge a growth step over the next complete window. Roll it back when
        // the end-to-end iteration cadence did not improve: a higher cache hit
        // rate alone is not useful when uploads are a negligible part of the
        // critical path.
        if (device_budget_previous_capacity_ != 0) {
            const bool improved =
                std::isfinite(mean_step_ms) && mean_step_ms > 0.0 &&
                device_budget_baseline_step_ms_ > 0.0 &&
                mean_step_ms <= device_budget_baseline_step_ms_ *
                    (1.0 - k_device_step_improvement);
            if (!improved) {
                const std::size_t grown_capacity = device_capacity_bytes_;
                device_capacity_bytes_ = device_budget_previous_capacity_;
                (void)make_room_for_device_bytes(0);
                ++device_budget_rollbacks_;
                device_budget_saturated_ = true;
                core::Logger::instance().info(
                    "splat_data_cache device_budget_rollback budget_bytes=",
                    device_capacity_bytes_, " grown_budget_bytes=", grown_capacity,
                    " baseline_step_ms=", device_budget_baseline_step_ms_,
                    " observed_step_ms=", mean_step_ms,
                    " hit_rate=", hit_rate,
                    " rollbacks=", device_budget_rollbacks_);
            } else {
                core::Logger::instance().info(
                    "splat_data_cache device_budget_accept budget_bytes=",
                    device_capacity_bytes_, " previous_bytes=",
                    device_budget_previous_capacity_,
                    " baseline_step_ms=", device_budget_baseline_step_ms_,
                    " observed_step_ms=", mean_step_ms,
                    " hit_rate=", hit_rate);
            }
            device_budget_previous_capacity_ = 0;
            device_budget_baseline_step_ms_ = 0.0;
            // Keep an accepted step for a full window before considering the
            // next growth, avoiding multiple changes from the same sample.
            return;
        }

        if (device_budget_saturated_ || hit_rate >= k_device_hit_rate_target)
            return;
        if (device_ceiling_bytes_ <= capacity) {
            device_budget_saturated_ = true;
            return;
        }
        const std::size_t idle =
            free_bytes > reserve ? free_bytes - reserve : 0;
        // A transient allocation spike must not permanently disable tuning.
        if (idle < k_min_device_growth) return;
        // Grow by half of the current budget (at least one step) and never past
        // what the idle VRAM, the dataset and the device share allow.
        const std::size_t step = std::max<std::size_t>(
            capacity / 2, k_min_device_growth);
        const std::size_t target = std::min(
            device_ceiling_bytes_,
            saturate_add(capacity, std::min(idle, step)));
        if (target <= capacity) {
            device_budget_saturated_ = true;
            return;
        }
        device_budget_previous_capacity_ = capacity;
        device_budget_baseline_step_ms_ = mean_step_ms;
        device_capacity_bytes_ = target;
        ++device_budget_growths_;
        core::Logger::instance().info(
            "splat_data_cache device_budget_grow budget_bytes=", target,
            " previous_bytes=", capacity, " free_bytes=", free_bytes,
            " reserve_bytes=", reserve, " hit_rate=", hit_rate,
            " step_ms=", mean_step_ms, " growths=", device_budget_growths_);
    }

    void update_cache_budgets() {
        dataset_packed_bytes_ = 0;
        for (const mvs::MvsView& view : source_) {
            dataset_packed_bytes_ = saturate_add(
                dataset_packed_bytes_,
                packed_training_view_bytes(
                    view, options_, resolution_scale_));
        }

        // Vulkan views are materialized as ordinary TinyTensor tensors, then
        // retained in a simple LRU. The CUDA packed-image layout and async
        // copy stream remain separate because their representation is not
        // shared by the Vulkan tensor runtime.
        if (options_.backend == TrainingBackend::vulkan) {
            capacity_bytes_ = options_.training_view_cache_bytes;
            device_capacity_bytes_ = options_.training_device_cache_bytes;
            device_ceiling_bytes_ = 0;
            return;
        }

        if (!options_.adaptive_training_cache) {
            capacity_bytes_ = options_.training_view_cache_bytes;
            device_capacity_bytes_ = 0;
            if (options_.training_device_cache_bytes != 0) {
                std::size_t free_bytes{}, total_bytes{};
                const auto error = cudaMemGetInfo(&free_bytes, &total_bytes);
                if (error != cudaSuccess)
                    throw std::runtime_error(
                        std::string("Training cache VRAM query failed: ") +
                        cudaGetErrorString(error));
                device_capacity_bytes_ = std::min(
                    options_.training_device_cache_bytes, free_bytes / 8);
            }
            return;
        }

        const std::size_t available_ram = available_system_memory_bytes();
        capacity_bytes_ = options_.training_view_cache_bytes;
        if (capacity_bytes_ != 0) {
            const std::size_t desired = std::max(
                capacity_bytes_, dataset_packed_bytes_);
            const std::size_t hard_limit = std::size_t{16} * k_gib;
            const std::size_t ram_limit = available_ram == 0
                ? hard_limit
                : std::min(available_ram - available_ram / 4, hard_limit);
            capacity_bytes_ = std::min(desired, ram_limit);
        }

        device_capacity_bytes_ = 0;
        if (options_.training_device_cache_bytes == 0) return;
        std::size_t free_bytes{}, total_bytes{};
        const auto error = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (error != cudaSuccess)
            throw std::runtime_error(
                std::string("Training cache VRAM query failed: ") +
                cudaGetErrorString(error));

        // At least 75% of VRAM remains for Gaussian/optimizer state and
        // raster scratch. A large projected model shrinks the image-cache
        // share before the cache can starve topology updates.
        const std::size_t projected_training_bytes =
            projected_training_state_bytes();
        training_reserve_bytes_ = projected_training_bytes;
        const double reserve_fraction = total_bytes == 0
            ? 1.0
            : static_cast<double>(projected_training_bytes) /
                  static_cast<double>(total_bytes);
        double cache_fraction = 0.25 - std::max(0.0, reserve_fraction - 0.50);
        if (cache_fraction <= 0.0) {
            // A very large configured Gaussian cap may itself be close to the
            // GPU limit. Keep only the small fixed share; the trainer will
            // drop even that if live free memory becomes low.
            cache_fraction = 0.0;
        }
        std::size_t budget = cache_fraction == 0.0
            ? std::min<std::size_t>(total_bytes / 8, 2 * k_gib)
            : static_cast<std::size_t>(
                  static_cast<double>(total_bytes) *
                  std::min(cache_fraction, 0.25));
        // The configured value is the floor in adaptive mode; the tuner may grow
        // the budget later once the hit rate and the free VRAM say it helps.
        budget = std::min({budget, dataset_packed_bytes_,
                           options_.training_device_cache_bytes});
        // High-resolution views also need large raster/atomic scratch buffers.
        // In that case cap the packed-image share at 1/8 of VRAM once the
        // dataset exceeds the same size; small datasets can still be fully
        // resident.
        if (dataset_packed_bytes_ > total_bytes / 8 &&
            !source_.empty() &&
            dataset_packed_bytes_ / source_.size() > 4 * k_mib) {
            budget = std::min<std::size_t>(budget, total_bytes / 8);
        }
        const std::size_t free_guard =
            free_bytes > k_gib ? free_bytes - k_gib : 0;
        budget = std::min(budget, free_guard);
        device_capacity_bytes_ = budget;
        device_floor_bytes_ = budget;

        // Ceiling for the adaptive growth: idle VRAM beyond the projected
        // training state, capped by the dataset and by a fixed share of the
        // device so a second process still finds room.
        const std::size_t idle = free_bytes > projected_training_bytes
            ? free_bytes - projected_training_bytes
            : 0;
        // Take half of the idle VRAM at most: the pool, the raster scratch and
        // a densification burst all grow into the same space. Without a
        // configured ceiling the budget stays where the caller put it.
        const std::size_t idle_share = static_cast<std::size_t>(
            static_cast<double>(idle) * 0.5);
        device_ceiling_bytes_ = std::min(
            dataset_packed_bytes_,
            std::min(
                options_.training_device_cache_max_bytes,
                std::min(
                    saturate_add(
                        options_.training_device_cache_bytes, idle_share),
                    static_cast<std::size_t>(
                        static_cast<double>(total_bytes) *
                        k_device_cache_max_share))));
        device_ceiling_bytes_ = std::max(device_ceiling_bytes_, budget);
    }
};



TrainingDataLoader::TrainingDataLoader(
    const std::vector<mvs::MvsView>& source,
    const TrainingOptions& options, const float resolution_scale)
    : impl_(std::make_unique<Impl>(source, options, resolution_scale)) {}

TrainingDataLoader::~TrainingDataLoader() = default;
TrainingDataLoader::TrainingDataLoader(TrainingDataLoader&&) noexcept = default;
TrainingDataLoader& TrainingDataLoader::operator=(
    TrainingDataLoader&&) noexcept = default;

TrainingView TrainingDataLoader::get(const std::size_t index) {
    return impl_->get(index);
}

bool TrainingDataLoader::has_mask(const std::size_t index) {
    return impl_->has_mask(index);
}

void TrainingDataLoader::prefetch(const std::size_t index) {
    impl_->prefetch(index);
}

std::size_t TrainingDataLoader::prefetch_depth() const {
    return impl_->prefetch_depth();
}

void TrainingDataLoader::set_resolution_scale(const float scale) {
    impl_->set_resolution_scale(scale);
}

void TrainingDataLoader::set_epoch_plan(
    const std::vector<std::size_t>* order, const std::size_t cursor) {
    impl_->set_epoch_plan(order, cursor);
}

void TrainingDataLoader::ensure_device_headroom(const std::size_t bytes) {
    impl_->ensure_device_headroom(bytes);
}

CacheStats TrainingDataLoader::stats() const {
    return impl_->stats();
}

}  // namespace training_data

Camera camera_from_mvs_view(const mvs::MvsView& view) {
    return make_camera_impl(
        view, uses_native_splat_projection(view.source_model));
}

TrainingView make_training_view(
    const mvs::MvsView& view, const TrainingOptions& options) {
    return upload_training_view(
        load_host_training_view(view, options, 1.F), options);
}

}  // namespace photara::splat
