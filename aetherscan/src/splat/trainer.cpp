#include "splat/trainer.hpp"

#include "cuda_ops.hpp"
#include "io/image.hpp"

#include <cuda_runtime.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace aetherscan::splat {
namespace {

constexpr float k_sh0 = 0.28209479177387814F;

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
    const io::RgbImage& source) {
    const bool distorted = view.k1 != 0.F || view.k2 != 0.F ||
                           view.p1 != 0.F || view.p2 != 0.F;
    if (!distorted) {
        const float scale_x = static_cast<float>(view.width) / source.width;
        const float scale_y = static_cast<float>(view.height) / source.height;
        return {(static_cast<float>(x) + 0.5F) / scale_x - 0.5F,
                (static_cast<float>(y) + 0.5F) / scale_y - 0.5F};
    }
    const double xn = (static_cast<double>(x) - view.cx) / view.fx;
    const double yn = (static_cast<double>(y) - view.cy) / view.fy;
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

template <typename T>
std::vector<T> download(const tinytensor::Tensor& tensor) {
    std::vector<T> values(tensor.numel());
    if (!values.empty()) {
        const cudaError_t error = cudaMemcpy(
            values.data(), tensor.data_ptr(), values.size() * sizeof(T),
            cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            throw std::runtime_error(
                std::string("Failed to download tensor: ") +
                cudaGetErrorString(error));
    }
    return values;
}

void write_float(std::ofstream& stream, const float value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
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

float sample_binary_mask(
    const io::GrayImage& mask, const io::RgbImage& source,
    const float source_x, const float source_y) {
    const float x = (source_x + 0.5F) * mask.width / source.width - 0.5F;
    const float y = (source_y + 0.5F) * mask.height / source.height - 0.5F;
    const int ix = static_cast<int>(std::lround(x));
    const int iy = static_cast<int>(std::lround(y));
    if (ix < 0 || iy < 0 || ix >= static_cast<int>(mask.width) ||
        iy >= static_cast<int>(mask.height))
        return 0.F;
    return mask.pixels[static_cast<std::size_t>(iy) * mask.width + ix] > 127
        ? 1.F
        : 0.F;
}

}  // namespace

Camera camera_from_mvs_view(const mvs::MvsView& view) {
    return make_camera_impl(view);
}

GaussianModel initialize_from_dense_cloud(
    const mvs::MvsScene& scene, const TrainingOptions& options) {
    if (scene.dense_cloud.points.empty())
        throw std::invalid_argument("GGGS initialization requires a non-empty dense cloud");
    if (options.sh_degree > 3)
        throw std::invalid_argument("The current GGGS CUDA backend supports SH degree <= 3");
    const std::size_t source_count = scene.dense_cloud.points.size();
    const std::size_t count = options.max_gaussians == 0
        ? source_count
        : std::min(source_count, options.max_gaussians);
    const std::size_t bases = static_cast<std::size_t>(options.sh_degree + 1U) *
                              (options.sh_degree + 1U);
    std::vector<float> means(count * 3);
    std::vector<float> scales(count * 3);
    std::vector<float> quaternions(count * 4);
    std::vector<float> opacities(count);
    std::vector<float> sh(count * bases * 3, 0.F);

    const auto source_index = [source_count, count](const std::size_t index) {
        return count == source_count
            ? index
            : std::min(source_count - 1, index * source_count / count);
    };
    mvs::Vec3f minimum =
        scene.dense_cloud.points[source_index(0)].position;
    mvs::Vec3f maximum = minimum;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& point = scene.dense_cloud.points[source_index(index)];
        minimum = minimum.cwiseMin(point.position);
        maximum = maximum.cwiseMax(point.position);
    }
    const float fallback_scale = std::max(
        (maximum - minimum).norm() /
            std::sqrt(static_cast<float>(std::max<std::size_t>(count, 1))),
        1e-6F);
    const float opacity = std::clamp(options.initial_opacity, 1e-6F, 1.F - 1e-6F);
    const float opacity_logit = std::log(opacity / (1.F - opacity));

    for (std::size_t index = 0; index < count; ++index) {
        const auto& point = scene.dense_cloud.points[source_index(index)];
        for (int axis = 0; axis < 3; ++axis)
            means[3 * index + axis] = point.position(axis);
        float footprint = std::numeric_limits<float>::infinity();
        for (const auto view_id : point.views) {
            if (view_id >= scene.views.size()) continue;
            const auto& view = scene.views[view_id];
            const auto camera_point = view.pose.transform_world_to_camera(
                point.position.cast<double>());
            if (camera_point.z() > 0.0)
                footprint = std::min(
                    footprint,
                    static_cast<float>(camera_point.z()) /
                        std::max(view.fx, view.fy));
        }
        const float scale = std::max(
            std::isfinite(footprint) ? footprint * options.initial_scale
                                     : fallback_scale * options.initial_scale,
            1e-7F);
        for (int axis = 0; axis < 3; ++axis)
            scales[3 * index + axis] = std::log(scale);

        mvs::Vec3f normal = point.normal;
        if (!normal.allFinite() || normal.squaredNorm() < 1e-12F)
            normal = mvs::Vec3f::UnitZ();
        else
            normal.normalize();
        Eigen::Quaternionf rotation = Eigen::Quaternionf::FromTwoVectors(
            mvs::Vec3f::UnitZ(), normal).normalized();
        quaternions[4 * index + 0] = rotation.w();
        quaternions[4 * index + 1] = rotation.x();
        quaternions[4 * index + 2] = rotation.y();
        quaternions[4 * index + 3] = rotation.z();
        opacities[index] = opacity_logit;
        for (int channel = 0; channel < 3; ++channel)
            sh[(index * bases) * 3 + channel] =
                (std::clamp(point.color(channel), 0.F, 1.F) - 0.5F) / k_sh0;
    }

    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        means, {count, 3}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        scales, {count, 3}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        quaternions, {count, 4}, tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        opacities, {count, 1}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(
        sh, {count, bases, 3}, tinytensor::Device::CUDA);
    model.sh_degree = options.sh_degree;
    return model;
}

TrainingView make_training_view(
    const mvs::MvsView& view, const TrainingOptions& options) {
    if (view.width == 0 || view.height == 0)
        throw std::invalid_argument("Cannot build a GGGS training view with empty dimensions");
    const io::RgbImage source = io::load_rgb(view.path);
    const std::size_t pixels = static_cast<std::size_t>(view.width) * view.height;
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
    for (std::uint32_t y = 0; y < view.height; ++y) {
        for (std::uint32_t x = 0; x < view.width; ++x) {
            const auto [sx, sy] = source_coordinate(view, x, y, source);
            const std::size_t pixel = static_cast<std::size_t>(y) * view.width + x;
            for (int channel = 0; channel < 3; ++channel)
                rgb[static_cast<std::size_t>(channel) * pixels + pixel] =
                    sample_rgb(source, sx, sy, channel);
            if (has_source_mask)
                mask[pixel] = sample_binary_mask(source_mask, source, sx, sy);
        }
    }

    std::vector<float> depth(pixels, 0.F);
    std::vector<float> normals(3 * pixels, 0.F);
    if (view.depth_map.depth.size() == pixels) depth = view.depth_map.depth;
    if (view.depth_map.normal.size() == pixels) {
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            for (int axis = 0; axis < 3; ++axis)
                normals[static_cast<std::size_t>(axis) * pixels + pixel] =
                    view.depth_map.normal[pixel](axis);
        }
    }
    bool has_mask = has_source_mask;
    if (options.use_mask && !has_source_mask &&
        view.foreground_mask.size() == pixels) {
        has_mask = true;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel)
            mask[pixel] = view.foreground_mask[pixel] != 0 ? 1.F : 0.F;
    }

    TrainingView result;
    result.camera = camera_from_mvs_view(view);
    result.rgb = tinytensor::Tensor::from_vector(
        rgb, {3, view.height, view.width}, tinytensor::Device::CUDA);
    result.depth = tinytensor::Tensor::from_vector(
        depth, {view.height, view.width}, tinytensor::Device::CUDA);
    result.normal = tinytensor::Tensor::from_vector(
        normals, {3, view.height, view.width}, tinytensor::Device::CUDA);
    result.mask = tinytensor::Tensor::from_vector(
        mask, {view.height, view.width}, tinytensor::Device::CUDA);
    result.has_mask = has_mask;
    return result;
}

Trainer::Trainer(TrainingOptions options) : options_(std::move(options)) {}

GaussianModel Trainer::train(
    const mvs::MvsScene& scene, ProgressCallback progress) const {
    if (scene.views.empty())
        throw std::invalid_argument("GGGS training requires at least one MVS view");
    GaussianModel model = initialize_from_dense_cloud(scene, options_);
    std::vector<TrainingView> views;
    views.reserve(scene.views.size());
    for (const auto& view : scene.views)
        views.push_back(make_training_view(view, options_));
    if (options_.use_mask &&
        std::none_of(views.begin(), views.end(),
                     [](const TrainingView& view) { return view.has_mask; }))
        throw std::invalid_argument(
            "GGGS mask training was requested, but no matching mask files or "
            "source alpha channels were found");

    detail::AdamState means_state = detail::make_adam_state(model.means);
    detail::AdamState scales_state = detail::make_adam_state(model.log_scales);
    detail::AdamState rotations_state = detail::make_adam_state(model.quaternions);
    detail::AdamState opacity_state = detail::make_adam_state(model.opacity_logits);
    detail::AdamState sh_state = detail::make_adam_state(model.sh);
    Rasterizer rasterizer;
    std::mt19937 random(options_.seed);
    std::uniform_int_distribution<std::size_t> choose_view(0, views.size() - 1);
    mvs::Vec3f scene_minimum = scene.dense_cloud.points.front().position;
    mvs::Vec3f scene_maximum = scene_minimum;
    for (const auto& point : scene.dense_cloud.points) {
        scene_minimum = scene_minimum.cwiseMin(point.position);
        scene_maximum = scene_maximum.cwiseMax(point.position);
    }
    const float scene_extent = std::max(
        (scene_maximum - scene_minimum).norm(), 1e-6F);
    const float minimum_log_scale = std::log(
        scene_extent * std::max(options_.minimum_scale_fraction, 1e-8F));
    const float maximum_log_scale = std::log(
        scene_extent * std::max(
            options_.maximum_scale_fraction,
            options_.minimum_scale_fraction));

    for (unsigned iteration = 1; iteration <= options_.iterations; ++iteration) {
        const auto started = std::chrono::steady_clock::now();
        const bool report_progress = progress &&
            (iteration == 1 || iteration == options_.iterations ||
             (options_.log_interval != 0 &&
              iteration % options_.log_interval == 0));
        const std::size_t view_index = choose_view(random);
        const auto& target = views[view_index];
        RasterizeOptions raster_options;
        raster_options.active_sh_degree = std::min(
            options_.sh_degree,
            options_.sh_degree_interval == 0
                ? options_.sh_degree
                : (iteration - 1) / options_.sh_degree_interval);
        raster_options.kernel_size = options_.kernel_size;
        raster_options.scale_modifier = options_.scale_modifier;
        raster_options.require_depth = options_.use_mvs_depth ||
                                       options_.use_mvs_normals;
        RenderResult rendered = rasterizer.forward(model, target.camera, raster_options);
        detail::LossGradients loss = detail::compute_training_loss(
            rendered, target, options_, report_progress);
        ModelGradients gradients = rasterizer.backward(
            model, rendered, loss.color, loss.alpha, loss.depth, loss.normal);

        const float progress_fraction = static_cast<float>(iteration - 1) /
                                        std::max(1U, options_.iterations);
        const float means_lr = options_.means_lr * scene_extent *
                               std::pow(0.01F, progress_fraction);
        detail::adam_step(
            model.means, gradients.means, means_state, means_lr, iteration, options_);
        detail::adam_step(
            model.log_scales, gradients.log_scales, scales_state,
            options_.scales_lr, iteration, options_, 0, 0.F,
            minimum_log_scale, maximum_log_scale);
        detail::constrain_scale_ratio(
            model.log_scales, options_.max_scale_ratio);
        detail::adam_step(
            model.quaternions, gradients.quaternions, rotations_state,
            options_.quaternions_lr, iteration, options_);
        detail::adam_step(
            model.opacity_logits, gradients.opacity_logits, opacity_state,
            options_.opacities_lr, iteration, options_, 0, 0.F, -12.F, 12.F);
        detail::adam_step(
            model.sh, gradients.sh, sh_state, options_.sh0_lr, iteration,
            options_, model.sh.shape()[1] * 3, options_.sh_rest_lr);

        if (report_progress) {
            const cudaError_t report_error = cudaDeviceSynchronize();
            if (report_error != cudaSuccess)
                throw std::runtime_error(
                    std::string("GGGS training step failed: ") +
                    cudaGetErrorString(report_error));
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            if (!progress({iteration, options_.iterations, model.size(), view_index,
                           loss.total, loss.rgb, loss.alpha_value,
                           loss.depth_value, loss.normal_value, milliseconds}))
                break;
        }
    }
    const cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string("GGGS training synchronization failed: ") +
            cudaGetErrorString(error));
    return model;
}

RenderMetrics render_evaluation_png(
    const GaussianModel& model, const mvs::MvsView& view,
    const std::filesystem::path& path, const TrainingOptions& training_options) {
    const TrainingView target = make_training_view(view, training_options);
    RasterizeOptions options;
    options.active_sh_degree = model.sh_degree;
    options.require_depth = false;
    const RenderResult rendered = Rasterizer().forward(
        model, target.camera, options);
    const std::vector<float> color = download<float>(rendered.color);
    const std::vector<float> alpha = download<float>(rendered.alpha);
    const std::vector<float> target_rgb = download<float>(target.rgb);
    const std::vector<float> mask = download<float>(target.mask);
    const std::size_t pixels =
        static_cast<std::size_t>(view.width) * view.height;

    io::RgbImage image;
    image.width = view.width;
    image.height = view.height;
    image.pixels.resize(3 * pixels);
    double absolute_error = 0.0;
    double squared_error = 0.0;
    std::size_t samples = 0;
    std::size_t covered = 0;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        if (alpha[pixel] > 0.01F) ++covered;
        for (int channel = 0; channel < 3; ++channel) {
            const std::size_t planar =
                static_cast<std::size_t>(channel) * pixels + pixel;
            const float prediction = std::clamp(color[planar], 0.F, 1.F);
            image.pixels[3 * pixel + channel] = static_cast<std::uint8_t>(
                std::lround(prediction * 255.F));
            if (mask[pixel] > 0.F) {
                const double difference =
                    static_cast<double>(prediction - target_rgb[planar]);
                absolute_error += std::abs(difference);
                squared_error += difference * difference;
                ++samples;
            }
        }
    }
    io::save_rgb_png(image, path);
    const double inverse_samples = 1.0 / std::max<std::size_t>(samples, 1);
    const double mse = squared_error * inverse_samples;
    RenderMetrics metrics;
    metrics.mae = static_cast<float>(absolute_error * inverse_samples);
    metrics.psnr = mse > 0.0
        ? static_cast<float>(-10.0 * std::log10(mse))
        : std::numeric_limits<float>::infinity();
    metrics.alpha_coverage = pixels != 0
        ? static_cast<float>(covered) / static_cast<float>(pixels)
        : 0.F;
    return metrics;
}

void save_gaussians_ply(
    const GaussianModel& model, const std::filesystem::path& path) {
    const std::size_t count = model.size();
    const std::size_t bases = model.sh.shape()[1];
    const auto means = download<float>(model.means);
    const auto log_scales = download<float>(model.log_scales);
    const auto rotations = download<float>(model.quaternions);
    const auto opacities = download<float>(model.opacity_logits);
    const auto sh = download<float>(model.sh);
    const auto require_finite = [](const std::vector<float>& values,
                                   const char* name) {
        const auto invalid = std::find_if(
            values.begin(), values.end(),
            [](const float value) { return !std::isfinite(value); });
        if (invalid != values.end()) {
            const auto index = static_cast<std::size_t>(
                std::distance(values.begin(), invalid));
            throw std::runtime_error(
                std::string("Refusing to write non-finite GGGS parameter ") +
                name + " at scalar index " + std::to_string(index));
        }
    };
    require_finite(means, "means");
    require_finite(log_scales, "log_scales");
    require_finite(rotations, "quaternions");
    require_finite(opacities, "opacity_logits");
    require_finite(sh, "SH");
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Failed to create Gaussian PLY: " + path.string());
    output << "ply\nformat binary_little_endian 1.0\n"
           << "comment AetherScan GGGS (3DGS-compatible SH layout)\n"
           << "element vertex " << count << '\n'
           << "property float x\nproperty float y\nproperty float z\n"
           << "property float nx\nproperty float ny\nproperty float nz\n"
           << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    for (std::size_t i = 0; i < (bases - 1) * 3; ++i)
        output << "property float f_rest_" << i << '\n';
    output << "property float opacity\n"
           << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
           << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
           << "end_header\n";
    for (std::size_t gaussian = 0; gaussian < count; ++gaussian) {
        for (int axis = 0; axis < 3; ++axis) write_float(output, means[3 * gaussian + axis]);
        write_float(output, 0.F); write_float(output, 0.F); write_float(output, 0.F);
        for (int channel = 0; channel < 3; ++channel)
            write_float(output, sh[(gaussian * bases) * 3 + channel]);
        // Standard 3DGS PLY stores all remaining bases for R, then G, then B.
        for (int channel = 0; channel < 3; ++channel)
            for (std::size_t basis = 1; basis < bases; ++basis)
                write_float(output, sh[(gaussian * bases + basis) * 3 + channel]);
        write_float(output, opacities[gaussian]);
        for (int axis = 0; axis < 3; ++axis)
            write_float(output, log_scales[3 * gaussian + axis]);
        for (int component = 0; component < 4; ++component)
            write_float(output, rotations[4 * gaussian + component]);
    }
    if (!output) throw std::runtime_error("Failed while writing Gaussian PLY: " + path.string());
}

}  // namespace aetherscan::splat
