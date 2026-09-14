#include "training_data_loader.hpp"

#include "core/camera_projection.hpp"
#include "io/image.hpp"
#include "cuda_ops.hpp"
#include "core/logging.hpp"
#include "splat/trainer.hpp"

#include <cuda_runtime_api.h>
#include "internal/cuda_stream_context.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
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

namespace aetherscan::splat {
namespace {

constexpr std::size_t k_mib = std::size_t{1024} * 1024;
constexpr std::size_t k_gib = k_mib * 1024;

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

class PinnedStagingPool {
public:
    std::unique_ptr<PinnedStagingBuffer> acquire(const std::size_t bytes) {
        std::lock_guard lock(mutex_);
        if (free_.empty()) {
            auto buffer = std::make_unique<PinnedStagingBuffer>();
            buffer->ensure(bytes);
            return buffer;
        }
        auto buffer = std::move(free_.back());
        free_.pop_back();
        buffer->ensure(bytes);
        return buffer;
    }

    void release(std::unique_ptr<PinnedStagingBuffer> buffer) {
        if (!buffer) return;
        std::lock_guard lock(mutex_);
        free_.push_back(std::move(buffer));
    }

    void clear() {
        std::lock_guard lock(mutex_);
        free_.clear();
    }

private:
    std::mutex mutex_;
    std::vector<std::unique_ptr<PinnedStagingBuffer>> free_;
};

struct PinnedStagingLease {
    std::unique_ptr<PinnedStagingBuffer> buffer;
    PinnedStagingPool* pool{};

    ~PinnedStagingLease() {
        if (pool) pool->release(std::move(buffer));
    }
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
    if (view.width == 0 || view.height == 0)
        throw std::invalid_argument(
            "Cannot build a GGGS training view with empty dimensions");
    Camera camera =
        training_data::training_camera(view, options, resolution_scale);
    const io::RgbImage source = io::load_rgb_with_minimum_size(
        view.path, camera.width, camera.height);
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
#if defined(AETHERSCAN_HAS_OPENMP)
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
    const HostTrainingView& host, const bool decode_gray) {
    TrainingView result;
    result.camera = host.camera;
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
        const cudaError_t stream_error = cudaStreamCreateWithFlags(
            &copy_stream_, cudaStreamNonBlocking);
        if (stream_error != cudaSuccess)
            throw std::runtime_error(
                std::string("Failed to create splat prefetch CUDA stream: ") +
                cudaGetErrorString(stream_error));
        try {
            update_cache_budgets();
        } catch (...) {
            cudaStreamDestroy(copy_stream_);
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
        device_prefetches_.clear();
        prefetches_.clear();
        if (copy_stream_ != nullptr) {
            const cudaError_t error = cudaStreamDestroy(copy_stream_);
            if (error != cudaSuccess)
                core::Logger::instance().warning(
                    "Failed to destroy splat prefetch CUDA stream: ",
                    cudaGetErrorString(error));
        }
    }

    TrainingView get(const std::size_t index) {
        ++requests_;
        collect_ready_host_prefetches();
        const auto found = device_lookup_.find(index);
        if (found != device_lookup_.end()) {
            ++device_hits_;
            device_entries_.splice(
                device_entries_.begin(), device_entries_, found->second);
            return decode_device_view(device_entries_.front());
        }
        if (consume_device_prefetch(index))
            return decode_device_view(device_entries_.front());
        const HostTrainingView& host = host_view(index);
        const std::size_t bytes = sizeof(int) * host.rgba.size() +
            sizeof(float) * (host.depth.size() + host.normal.size());
        uploaded_bytes_ += bytes;
        if (device_capacity_bytes_ == 0 || bytes > device_capacity_bytes_)
            return upload_training_view(
                host, options_.multi_view_ncc_weight > 0.F);
        if (!make_room_for_device_bytes(bytes))
            return upload_training_view(
                host, options_.multi_view_ncc_weight > 0.F);
        DeviceEntry entry = allocate_device_entry(index, host);
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
        if (device_lookup_.contains(index) ||
            device_prefetches_.contains(index)) {
            return;
        }

        if (const auto host = cached_host_view(index)) {
            const std::size_t bytes = packed_host_bytes(*host);
            if (device_capacity_bytes_ == 0 ||
                bytes > device_capacity_bytes_ ||
                !make_room_for_device_bytes(bytes)) {
                return;
            }

            DeviceEntry entry;
            {
                try {
                    const tinytensor::CUDAStreamGuard stream_guard(
                        copy_stream_);
                    entry = allocate_device_entry(index, *host);
                } catch (const std::exception& error) {
                    core::Logger::instance().warning(
                        "splat_data_cache device_prefetch_allocation_failed ",
                        "index=", index, " error=", error.what());
                    return;
                }
            }
            device_prefetch_bytes_ += bytes;
            try {
                device_prefetches_.emplace(
                    index,
                    std::async(
                        std::launch::async,
                        [this, host = std::move(host),
                         entry = std::move(entry)]() mutable {
                            DevicePrefetchResult result;
                            result.entry = std::move(entry);
                            result.success = true;
                            try {
                                copy_device_entry_with_pinned_staging(
                                    *host, result.entry);
                            } catch (const std::exception& error) {
                                cudaStreamSynchronize(copy_stream_);
                                result.success = false;
                                result.error = error.what();
                            }
                            return result;
                        }));
            } catch (const std::exception& error) {
                device_prefetch_bytes_ -= bytes;
                core::Logger::instance().warning(
                    "splat_data_cache device_prefetch_launch_failed ",
                    "index=", index, " error=", error.what());
            }
            return;
        }

        if (prefetches_.contains(index)) return;
        const float scale = resolution_scale_;
        prefetches_.emplace(
            index,
            std::async(
                std::launch::async,
                [this, index, scale] {
                    return load_host_training_view(
                        source_[index], options_, scale);
                }));
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
        device_prefetches_.clear();
        device_prefetch_bytes_ = 0;
        entries_.clear();
        lookup_.clear();
        cached_bytes_ = 0;
        device_lookup_.clear();
        device_entries_.clear();
        device_cached_bytes_ = 0;
        prefetches_.clear();
        resolution_scale_ = clamped;
        update_cache_budgets();
        pinned_pool_.clear();
    }

    void ensure_device_headroom(const std::size_t bytes) {
        if (bytes == 0) return;
        std::size_t free_bytes{}, total_bytes{};
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
            free_bytes >= bytes) {
            return;
        }

        const std::size_t free_before = free_bytes;
        const std::size_t resident =
            device_cached_bytes_ + device_prefetch_bytes_;
        device_prefetches_.clear();
        device_prefetch_bytes_ = 0;
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
        return {requests_, device_hits_, device_prefetch_hits_,
                uploaded_bytes_, device_cached_bytes_,
                device_capacity_bytes_, device_prefetches_.size(),
                device_prefetch_bytes_, dataset_packed_bytes_,
                capacity_bytes_};
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

    struct DevicePrefetchResult {
        DeviceEntry entry;
        bool success{};
        std::string error;
    };

    [[nodiscard]] static std::size_t packed_host_bytes(
        const HostTrainingView& host) {
        return sizeof(int) * host.rgba.size() +
            sizeof(float) * (host.depth.size() + host.normal.size());
    }

    [[nodiscard]] DeviceEntry allocate_device_entry(
        const std::size_t index, const HostTrainingView& host) const {
        DeviceEntry entry;
        entry.index = index;
        entry.bytes = packed_host_bytes(host);
        entry.camera = host.camera;
        entry.has_mask = host.has_mask;
        entry.mask_is_validity = host.mask_is_validity;
        entry.rgba = tinytensor::Tensor::empty(
            {host.camera.height, host.camera.width},
            tinytensor::Device::CUDA, tinytensor::DataType::Int32);
        if (!host.depth.empty())
            entry.depth = tinytensor::Tensor::empty(
                {host.camera.height, host.camera.width},
                tinytensor::Device::CUDA);
        if (!host.normal.empty())
            entry.normal = tinytensor::Tensor::empty(
                {std::size_t{3}, host.camera.height, host.camera.width},
                tinytensor::Device::CUDA);
        return entry;
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
        PinnedStagingLease lease{pinned_pool_.acquire(total_bytes),
                                 &pinned_pool_};
        std::uint8_t* destination = lease.buffer->data();
        std::uint8_t* rgba_source = destination;
        std::memcpy(rgba_source, host.rgba.data(), rgba_bytes);
        std::uint8_t* depth_source = rgba_source + rgba_bytes;
        if (depth_bytes != 0)
            std::memcpy(depth_source, host.depth.data(), depth_bytes);
        std::uint8_t* normal_source = depth_source + depth_bytes;
        if (normal_bytes != 0)
            std::memcpy(normal_source, host.normal.data(), normal_bytes);

        const auto check_copy = [](const cudaError_t error) {
            if (error != cudaSuccess)
                throw std::runtime_error(
                    std::string("Failed to enqueue packed-view H2D copy: ") +
                    cudaGetErrorString(error));
        };
        check_copy(cudaMemcpyAsync(
            entry.rgba.data_ptr(), rgba_source, rgba_bytes,
            cudaMemcpyHostToDevice, copy_stream_));
        if (depth_bytes != 0)
            check_copy(cudaMemcpyAsync(
                entry.depth.data_ptr(), depth_source, depth_bytes,
                cudaMemcpyHostToDevice, copy_stream_));
        if (normal_bytes != 0)
            check_copy(cudaMemcpyAsync(
                entry.normal.data_ptr(), normal_source, normal_bytes,
                cudaMemcpyHostToDevice, copy_stream_));

        const cudaError_t sync_error = cudaStreamSynchronize(copy_stream_);
        if (sync_error != cudaSuccess)
            throw std::runtime_error(
                std::string("Failed to synchronize packed-view H2D copy: ") +
                cudaGetErrorString(sync_error));
    }

    [[nodiscard]] bool make_room_for_device_bytes(const std::size_t bytes) {
        while (!device_entries_.empty() &&
               device_cached_bytes_ + device_prefetch_bytes_ + bytes >
                   device_capacity_bytes_) {
            const auto& evicted = device_entries_.back();
            device_cached_bytes_ -= evicted.bytes;
            device_lookup_.erase(evicted.index);
            device_entries_.pop_back();
        }
        return device_cached_bytes_ + device_prefetch_bytes_ + bytes <=
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

    bool consume_device_prefetch(const std::size_t index) {
        const auto pending = device_prefetches_.find(index);
        if (pending == device_prefetches_.end()) return false;

        // The view being loaded is the current training view. Waiting here is
        // intentional: the copy stream can overlap already-submitted CUDA
        // training work while this thread waits for staging to finish.
        if (pending->second.wait_for(std::chrono::seconds{0}) !=
            std::future_status::ready) {
            pending->second.wait();
        }
        DevicePrefetchResult result = pending->second.get();
        device_prefetches_.erase(pending);
        device_prefetch_bytes_ -= result.entry.bytes;

        if (!result.success) {
            core::Logger::instance().warning(
                "splat_data_cache device_prefetch_failed index=", index,
                " error=", result.error);
            return false;
        }

        ++device_prefetch_hits_;
        uploaded_bytes_ += result.entry.bytes;
        return insert_device_entry(std::move(result.entry));
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
        if (index >= source_.size())
            throw std::out_of_range(
                "GGGS training view index is out of range");
        if (const auto cached = cached_host_view(index))
            return *cached;

        auto loaded = std::make_shared<HostTrainingView>();
        const auto prefetched = prefetches_.find(index);
        if (prefetched != prefetches_.end()) {
            *loaded = prefetched->second.get();
            prefetches_.erase(prefetched);
        } else {
            *loaded = load_host_training_view(
                source_[index], options_, resolution_scale_);
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
            const auto& evicted = entries_.back();
            cached_bytes_ -= evicted.bytes;
            lookup_.erase(evicted.index);
            entries_.pop_back();
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
            auto loaded = std::make_shared<HostTrainingView>(
                it->second.get());
            const std::size_t index = it->first;
            it = prefetches_.erase(it);
            store_host_view(index, std::move(loaded));
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
    PinnedStagingPool pinned_pool_;
    std::size_t capacity_bytes_{};
    std::size_t cached_bytes_{};
    std::size_t device_capacity_bytes_{};
    std::size_t device_cached_bytes_{};
    std::size_t dataset_packed_bytes_{};
    std::size_t requests_{}, device_hits_{}, uploaded_bytes_{};
    std::size_t device_prefetch_hits_{};
    std::size_t device_prefetch_bytes_{};
    DeviceEntries device_entries_;
    std::unordered_map<std::size_t, DeviceEntries::iterator> device_lookup_;
    std::unordered_map<std::size_t, std::future<DevicePrefetchResult>>
        device_prefetches_;
    float resolution_scale_{1.F};
    Entries entries_;
    std::unordered_map<std::size_t, Entries::iterator> lookup_;
    std::unordered_map<std::size_t, std::future<HostTrainingView>> prefetches_;

    [[nodiscard]] std::size_t projected_gaussian_count() const noexcept {
        if (options_.enable_densification)
            return options_.densification_cap;
        return 0;
    }

    void update_cache_budgets() {
        dataset_packed_bytes_ = 0;
        for (const mvs::MvsView& view : source_) {
            dataset_packed_bytes_ = saturate_add(
                dataset_packed_bytes_,
                packed_training_view_bytes(
                    view, options_, resolution_scale_));
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
        const std::size_t projected_gaussians = projected_gaussian_count();
        const std::size_t projected_training_bytes = saturate_add(
            std::size_t{3} * k_gib / 2,
            saturate_multiply(projected_gaussians, std::size_t{2} * 1024));
        const double reserve_fraction = total_bytes == 0
            ? 1.0
            : static_cast<double>(projected_training_bytes) /
                  static_cast<double>(total_bytes);
        double cache_fraction = 0.25 - std::max(0.0, reserve_fraction - 0.50);
        if (cache_fraction <= 0.0) {
            // A very large configured Gaussian cap may itself be close to the
            // GPU limit. Keep only the legacy small cache share; the trainer
            // will drop even that if live free memory becomes low.
            cache_fraction = 0.0;
        }
        std::size_t budget = cache_fraction == 0.0
            ? std::min<std::size_t>(total_bytes / 8, 2 * k_gib)
            : static_cast<std::size_t>(
                  static_cast<double>(total_bytes) *
                  std::min(cache_fraction, 0.25));
        budget = std::min(budget, dataset_packed_bytes_);
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

void TrainingDataLoader::set_resolution_scale(const float scale) {
    impl_->set_resolution_scale(scale);
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
        load_host_training_view(view, options, 1.F),
        options.multi_view_ncc_weight > 0.F);
}

}  // namespace aetherscan::splat
