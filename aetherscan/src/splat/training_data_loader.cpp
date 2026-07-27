#include "training_data_loader.hpp"

#include "io/image.hpp"
#include "splat/trainer.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <list>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace aetherscan::splat {
namespace {

Camera make_camera_impl(const mvs::MvsView& view) {
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
    camera.fx = view.fx;
    camera.fy = view.fy;
    camera.cx = view.cx;
    camera.cy = view.cy;
    camera.width = view.width;
    camera.height = view.height;
    return camera;
}

}
 
namespace training_data {

Camera training_camera(
    const mvs::MvsView& view, const TrainingOptions& options,
    const float resolution_scale) {
    Camera camera = make_camera_impl(view);
    if (options.use_source_resolution && view.src_width != 0 &&
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

float sample_rgb(
    const io::RgbImage& image, const float x, const float y, const int channel) {
    if (x < 0.F || y < 0.F || x > static_cast<float>(image.width - 1) ||
        y > static_cast<float>(image.height - 1))
        return 0.F;
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, static_cast<int>(image.width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(image.height) - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const auto at = [&](const int px, const int py) {
        return static_cast<float>(image.pixels[
            (static_cast<std::size_t>(py) * image.width + px) * 3 + channel]);
    };
    return ((at(x0, y0) * (1.F - tx) + at(x1, y0) * tx) * (1.F - ty) +
            (at(x0, y1) * (1.F - tx) + at(x1, y1) * tx) * ty) /
           255.F;
}

std::pair<float, float> source_coordinate(
    const mvs::MvsView& view, const std::uint32_t x, const std::uint32_t y,
    const Camera& output_camera, const io::RgbImage& source) {
    const bool distorted = view.k1 != 0.F || view.k2 != 0.F ||
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
    const double radius2 = xn * xn + yn * yn;
    const double radial = 1.0 + view.k1 * radius2 +
                          view.k2 * radius2 * radius2;
    const double xd = xn * radial + 2.0 * view.p1 * xn * yn +
                      view.p2 * (radius2 + 2.0 * xn * xn);
    const double yd = yn * radial + view.p1 * (radius2 + 2.0 * yn * yn) +
                      2.0 * view.p2 * xn * yn;
    return {static_cast<float>(view.src_fx * xd + view.src_cx),
            static_cast<float>(view.src_fy * yd + view.src_cy)};
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
    const int px = static_cast<int>(std::round(x));
    const int py = static_cast<int>(std::round(y));
    if (px < 0 || py < 0 || px >= static_cast<int>(mask.width) ||
        py >= static_cast<int>(mask.height))
        return 0.F;
    return mask.pixels[
               static_cast<std::size_t>(py) * mask.width +
               static_cast<std::size_t>(px)] > 127
        ? 1.F
        : 0.F;
}

struct HostTrainingView {
    Camera camera;
    std::vector<float> rgb;
    std::vector<float> depth;
    std::vector<float> normal;
    std::vector<float> mask;
    bool has_mask{false};

    [[nodiscard]] std::size_t bytes() const noexcept {
        return sizeof(*this) + sizeof(float) *
            (rgb.capacity() + depth.capacity() + normal.capacity() +
             mask.capacity());
    }
};

HostTrainingView load_host_training_view(
    const mvs::MvsView& view, const TrainingOptions& options,
    const float resolution_scale = 1.F) {
    if (view.width == 0 || view.height == 0)
        throw std::invalid_argument(
            "Cannot build a GGGS training view with empty dimensions");
    const io::RgbImage source = io::load_rgb(view.path);
    Camera camera =
        training_data::training_camera(view, options, resolution_scale);
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
    std::vector<float> rgb(3 * pixels);
    std::vector<float> mask(pixels, 1.F);
    for (std::uint32_t y = 0; y < camera.height; ++y) {
        for (std::uint32_t x = 0; x < camera.width; ++x) {
            const auto [sx, sy] =
                source_coordinate(view, x, y, camera, source);
            const std::size_t pixel =
                static_cast<std::size_t>(y) * camera.width + x;
            for (int channel = 0; channel < 3; ++channel)
                rgb[static_cast<std::size_t>(channel) * pixels + pixel] =
                    sample_rgb(source, sx, sy, channel);
            if (has_source_mask)
                mask[pixel] =
                    sample_mask_coverage(source_mask, source, sx, sy);
        }
    }

    std::vector<float> depth;
    std::vector<float> normals;
    if (options.use_mvs_depth &&
        camera.width == view.width && camera.height == view.height &&
        view.depth_map.depth.size() == pixels)
        depth = view.depth_map.depth;
    if (options.use_mvs_normals &&
        camera.width == view.width && camera.height == view.height &&
        view.depth_map.normal.size() == pixels) {
        normals.resize(3 * pixels);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            for (int axis = 0; axis < 3; ++axis)
                normals[static_cast<std::size_t>(axis) * pixels + pixel] =
                    view.depth_map.normal[pixel](axis);
        }
    }
    bool has_mask = has_source_mask;
    if (options.use_mask && !has_source_mask &&
        camera.width == view.width && camera.height == view.height &&
        view.foreground_mask.size() == pixels) {
        has_mask = true;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel)
            mask[pixel] =
                view.foreground_mask[pixel] != 0 ? 1.F : 0.F;
    }

    return {
        camera, std::move(rgb), std::move(depth), std::move(normals),
        std::move(mask), has_mask};
}

TrainingView upload_training_view(const HostTrainingView& host) {
    TrainingView result;
    result.camera = host.camera;
    result.rgb = tinytensor::Tensor::from_vector(
        host.rgb, {3, host.camera.height, host.camera.width},
        tinytensor::Device::CUDA);
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
    result.mask = tinytensor::Tensor::from_vector(
        host.mask, {host.camera.height, host.camera.width},
        tinytensor::Device::CUDA);
    result.has_mask = host.has_mask;
    return result;
}

}  // namespace

namespace training_data {

struct TrainingDataLoader::Impl {
    Impl(
        const std::vector<mvs::MvsView>& source,
        const TrainingOptions& options, const float resolution_scale = 1.F)
        : source_(source), options_(options),
          capacity_bytes_(options.training_view_cache_bytes),
          resolution_scale_(resolution_scale) {}

    TrainingView get(const std::size_t index) {
        return upload_training_view(host_view(index));
    }

    bool has_mask(const std::size_t index) {
        return host_view(index).has_mask;
    }

    void set_resolution_scale(const float scale) {
        const float clamped = std::clamp(scale, 1e-3F, 1.F);
        if (std::abs(clamped - resolution_scale_) < 1e-6F) return;
        entries_.clear();
        lookup_.clear();
        cached_bytes_ = 0;
        resolution_scale_ = clamped;
    }

private:
    struct Entry {
        std::size_t index{};
        std::size_t bytes{};
        HostTrainingView view;
    };
    using Entries = std::list<Entry>;

    const HostTrainingView& host_view(const std::size_t index) {
        if (index >= source_.size())
            throw std::out_of_range(
                "GGGS training view index is out of range");
        const auto found = lookup_.find(index);
        if (found != lookup_.end()) {
            entries_.splice(entries_.begin(), entries_, found->second);
            return entries_.front().view;
        }

        HostTrainingView loaded = load_host_training_view(
            source_[index], options_, resolution_scale_);
        const std::size_t loaded_bytes = loaded.bytes();
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
        return entries_.front().view;
    }

    const std::vector<mvs::MvsView>& source_;
    const TrainingOptions& options_;
    std::size_t capacity_bytes_{};
    std::size_t cached_bytes_{};
    float resolution_scale_{1.F};
    Entries entries_;
    std::unordered_map<std::size_t, Entries::iterator> lookup_;
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

void TrainingDataLoader::set_resolution_scale(const float scale) {
    impl_->set_resolution_scale(scale);
}

}  // namespace training_data

Camera camera_from_mvs_view(const mvs::MvsView& view) {
    return make_camera_impl(view);
}

TrainingView make_training_view(
    const mvs::MvsView& view, const TrainingOptions& options) {
    return upload_training_view(
        load_host_training_view(view, options, 1.F));
}

}  // namespace aetherscan::splat
