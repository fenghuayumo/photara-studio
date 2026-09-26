#include "io/image.hpp"
#include "mvs/densify.hpp"
#include "splat/dataset.hpp"
#include "splat/formats.hpp"
#include "splat/rasterizer.hpp"
#include "splat/trainer.hpp"
#include "../splat/optimizer.hpp"
#include "../splat/photometric_loss.hpp"
#include "../splat/training_data_loader.hpp"
#include "../splat/densification.hpp"
#include "sfm/asfm.hpp"

#include "splat_drender/vulkan_api.h"
#include "vulkan/backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Manual verification tool (not a ctest entry): renders a trained model with
// the CUDA forward rasterizer used by training (splat_drender through
// photara::splat::Rasterizer) and with the Vulkan forward rasterizer the editor
// previews through (splat_drender::vulkan::SplatRasterizer), then reports every
// forward channel and writes the images side by side.
//
// Usage:
//   photara_splat_vulkan_parity MODEL IMAGES OUTPUT_DIR [STRIDE] [SFM]
//
// MODEL is a trained splat PLY. IMAGES is the image directory the cameras
// index. SFM selects the camera source: a working SfM cache (*.bin, what the
// trainer reads) or a COLMAP/OpenMVS dataset directory; when it is omitted the
// COLMAP dataset next to IMAGES is used.

namespace {

void require(bool condition, const std::string& message);

struct Image {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixels;  // RGB8
};

template <typename T>
double relative_l2(const std::vector<T>& got, const std::vector<T>& reference) {
    if (got.size() != reference.size()) return std::numeric_limits<double>::infinity();
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double delta = static_cast<double>(got[i]) - static_cast<double>(reference[i]);
        numerator += delta * delta;
        denominator += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
    }
    if (!(denominator > 1e-30)) return std::sqrt(numerator);
    return std::sqrt(numerator / denominator);
}

double relative_l2_slice(
    const std::vector<float>& packed, const std::size_t offset,
    const std::vector<float>& reference) {
    if (offset + reference.size() > packed.size())
        return std::numeric_limits<double>::infinity();
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double delta = static_cast<double>(packed[offset + i]) - reference[i];
        numerator += delta * delta;
        denominator += static_cast<double>(reference[i]) * reference[i];
    }
    return denominator > 1e-30 ? std::sqrt(numerator / denominator)
                               : std::sqrt(numerator);
}

Image to_image(const std::vector<float>& planar, const std::uint32_t width, const std::uint32_t height) {
    Image image;
    image.width = width;
    image.height = height;
    const auto area = static_cast<std::size_t>(width) * height;
    image.pixels.resize(area * 3U);
    for (std::size_t pixel = 0; pixel < area; ++pixel) {
        for (std::size_t channel = 0; channel < 3U; ++channel) {
            const float value = std::clamp(planar[channel * area + pixel], 0.0F, 1.0F);
            image.pixels[pixel * 3U + channel] = static_cast<std::uint8_t>(std::lround(255.0F * value));
        }
    }
    return image;
}

Image difference_image(const Image& left, const Image& right, const int gain) {
    Image image;
    image.width = left.width;
    image.height = left.height;
    image.pixels.resize(left.pixels.size());
    for (std::size_t i = 0; i < left.pixels.size(); ++i) {
        const int delta = std::abs(int(left.pixels[i]) - int(right.pixels[i])) * gain;
        image.pixels[i] = static_cast<std::uint8_t>(std::min(delta, 255));
    }
    return image;
}

double psnr_8bit(const Image& left, const Image& right) {
    double squared = 0.0;
    for (std::size_t i = 0; i < left.pixels.size(); ++i) {
        const double delta = double(left.pixels[i]) - double(right.pixels[i]);
        squared += delta * delta;
    }
    const double count = static_cast<double>(left.pixels.size());
    const double mse = squared / count;
    if (mse <= 0.0) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

double mean_absolute_8bit(const Image& left, const Image& right) {
    double total = 0.0;
    for (std::size_t i = 0; i < left.pixels.size(); ++i)
        total += std::abs(int(left.pixels[i]) - int(right.pixels[i]));
    return total / static_cast<double>(left.pixels.size());
}

template <typename T>
double max_absolute_difference(const std::vector<T>& got, const std::vector<T>& reference, std::size_t* above = nullptr,
                               const double threshold = 0.0) {
    double worst = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < got.size() && i < reference.size(); ++i) {
        const double delta = std::abs(static_cast<double>(got[i]) - static_cast<double>(reference[i]));
        worst = std::max(worst, delta);
        if (above != nullptr && delta > threshold) ++count;
    }
    if (above != nullptr) *above = count;
    return worst;
}

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// The flat model dump the trainer writes for the editor (`--splat-packed-model`
// / the GUI handoff): header, then means, log scales, quaternions, opacity
// logits, SH, optional normals and 3D filter. Reading it here exercises the
// exact bytes the editor maps after training.
photara::splat::GaussianModel load_packed_model(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open packed model: " + path.string());
    input.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(input.tellg());
    input.seekg(0);
    std::vector<std::byte> data(bytes);
    input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));

    constexpr std::uint32_t k_magic = 0x4D475350U;  // "PSGM"
    constexpr std::size_t k_header = 88;
    if (bytes < k_header) throw std::runtime_error("packed model is truncated");
    std::uint32_t magic = 0, version = 0, sh_bases = 1, sh_degree = 0, flags = 0;
    std::uint64_t count = 0;
    std::array<std::uint64_t, 7> offsets{};
    std::memcpy(&magic, data.data() + 0, 4);
    std::memcpy(&version, data.data() + 4, 4);
    std::memcpy(&count, data.data() + 8, 8);
    std::memcpy(&sh_bases, data.data() + 16, 4);
    std::memcpy(&sh_degree, data.data() + 20, 4);
    std::memcpy(&flags, data.data() + 24, 4);
    std::memcpy(offsets.data(), data.data() + 32, 7 * 8);
    require(magic == k_magic && version == 1 && count != 0,
            "packed model has an unexpected header");
    require(offsets[0] == k_header, "packed model does not start with the attribute block");

    const auto read = [&](const std::size_t index, const std::size_t elements) {
        require(offsets[index] + elements * sizeof(float) <= bytes, "packed model array is out of range");
        std::vector<float> values(elements);
        std::memcpy(values.data(), data.data() + offsets[index], elements * sizeof(float));
        return values;
    };
    const std::size_t bases = std::max(1U, sh_bases);
    const auto means = read(0, count * 3U);
    const auto log_scales = read(1, count * 3U);
    const auto quaternions = read(2, count * 4U);
    const auto opacity = read(3, count);
    const auto sh = read(4, count * bases * 3U);

    photara::splat::GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(means, {count, 3U}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(log_scales, {count, 3U}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(quaternions, {count, 4U}, tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(opacity, {count, 1U}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(sh, {count, bases, 3U}, tinytensor::Device::CUDA);
    model.sh_degree = sh_degree;
    std::cout << "packed model: gaussians=" << count << " sh_bases=" << bases << " sh_degree=" << sh_degree
              << " flags=" << flags << '\n';
    return model;
}

// Working SfM cache (*.bin, what the trainer reads) or an external dataset
// directory. Cameras must come from the reconstruction the model was trained
// against, otherwise the view is meaningless.
std::vector<photara::mvs::MvsView> load_views(
    const std::filesystem::path& images, const std::filesystem::path& sfm) {
    std::vector<photara::mvs::MvsView> views;
    if (!sfm.empty() && sfm.extension() == ".bin") {
        photara::sfm::AsfmOptions asfm;
        asfm.path_base = images;
        const auto sfm_scene = photara::sfm::load_asfm(sfm, asfm);
        photara::mvs::DensifyOptions densify;
        views = photara::mvs::build_mvs_scene(sfm_scene, densify).views;
    } else {
        photara::splat::DatasetLoadRequest request;
        request.source = sfm.empty() ? images.parent_path() : sfm;
        request.image_directory = images;
        views = photara::splat::load_splat_dataset(request).scene.views;
    }
    return views;
}

double median_ms(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    return samples.empty() ? 0.0 : samples[samples.size() / 2];
}

// Raw channel dumps for offline analysis, enabled with SPLAT_PARITY_DUMP=<dir>.
// Little-endian, channel-major (matching SplatForwardOutput): one flat array
// per channel and backend, so a Python reader only needs shape and dtype.
void dump_channel(const std::filesystem::path& directory, const std::string& name,
                  const void* data, const std::size_t elements, const std::size_t element_bytes) {
    std::ofstream out(directory / (name + ".bin"), std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + (directory / (name + ".bin")).string());
    out.write(static_cast<const char*>(data),
              static_cast<std::streamsize>(elements * element_bytes));
}

template <typename T>
void dump_channel(const std::filesystem::path& directory, const std::string& name,
                  const std::vector<T>& values) {
    dump_channel(directory, name, values.data(), values.size(), sizeof(T));
}

// Runs `body` `warmup` times untimed and `iterations` times timed, and returns
// the median wall time in milliseconds.
template <typename Body>
double time_ms(Body&& body, const int warmup, const int iterations) {
    for (int i = 0; i < warmup; ++i) body();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        body();
        const auto finish = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
    }
    return median_ms(samples);
}

splat_drender::vulkan::SplatCamera to_vulkan_camera(const photara::splat::Camera& camera) {
    splat_drender::vulkan::SplatCamera vulkan_camera;
    vulkan_camera.width = camera.width;
    vulkan_camera.height = camera.height;
    vulkan_camera.fx = camera.fx;
    vulkan_camera.fy = camera.fy;
    vulkan_camera.cx = camera.cx;
    vulkan_camera.cy = camera.cy;
    vulkan_camera.k1 = camera.k1;
    vulkan_camera.k2 = camera.k2;
    vulkan_camera.k3 = camera.k3;
    vulkan_camera.k4 = camera.k4;
    vulkan_camera.mode = static_cast<std::uint32_t>(camera.model);
    vulkan_camera.world_to_camera =
        std::span<const float>(camera.world_to_camera.data(), camera.world_to_camera.size());
    vulkan_camera.center = std::span<const float>(camera.position.data(), camera.position.size());
    return vulkan_camera;
}

// Editor-style orbit camera (same math as the editor's make_preview_camera and
// orbit_intrinsics), aimed at the model's subject.
photara::splat::Camera make_orbit_camera(
    const std::vector<float>& means, const float distance, const float yaw_degrees,
    const float pitch_degrees, const float fov_degrees, const std::uint32_t width,
    const std::uint32_t height) {
    constexpr float k_pi = 3.14159265358979323846F;
    const std::size_t count = means.size() / 3U;
    std::vector<float> xs(count), ys(count), zs(count);
    for (std::size_t i = 0; i < count; ++i) {
        xs[i] = means[i * 3U];
        ys[i] = means[i * 3U + 1U];
        zs[i] = means[i * 3U + 2U];
    }
    const auto median = [](std::vector<float>& values) {
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        return values[values.size() / 2];
    };
    const float cx_world = median(xs), cy_world = median(ys), cz_world = median(zs);
    std::vector<float> radii(count);
    for (std::size_t i = 0; i < count; ++i)
        radii[i] = std::sqrt((xs[i] - cx_world) * (xs[i] - cx_world) + (ys[i] - cy_world) * (ys[i] - cy_world) +
                             (zs[i] - cz_world) * (zs[i] - cz_world));
    const float target[3] = {cx_world, cy_world, cz_world};

    const float yaw = yaw_degrees * k_pi / 180.0F;
    const float pitch = std::clamp(pitch_degrees * k_pi / 180.0F, -1.53F, 1.53F);
    const float offset[3] = {
        std::cos(pitch) * std::sin(yaw), -std::sin(pitch), std::cos(pitch) * std::cos(yaw)};
    const float eye[3] = {
        target[0] + offset[0] * distance, target[1] + offset[1] * distance,
        target[2] + offset[2] * distance};
    float forward[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]};
    const auto normalize = [](float* v) {
        const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (length > 0.0F) { v[0] /= length; v[1] /= length; v[2] /= length; }
    };
    normalize(forward);
    float right[3] = {
        forward[1] * 0.0F - forward[2] * 1.0F, forward[2] * 0.0F - forward[0] * 0.0F,
        forward[0] * 1.0F - forward[1] * 0.0F};
    normalize(right);
    const float up[3] = {
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0]};
    // OpenCV basis: X right, Y down, Z forward.
    const float x[3] = {right[0], right[1], right[2]};
    const float y[3] = {-up[0], -up[1], -up[2]};
    const float z[3] = {forward[0], forward[1], forward[2]};
    const float tx = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
    const float ty = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
    const float tz = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);

    photara::splat::Camera camera;
    camera.world_to_camera = {x[0], y[0], z[0], 0.0F, x[1], y[1], z[1], 0.0F,
                              x[2], y[2], z[2], 0.0F, tx,   ty,   tz,   1.0F};
    camera.position = {eye[0], eye[1], eye[2]};
    camera.width = width;
    camera.height = height;
    const float half = std::clamp(fov_degrees, 10.0F, 170.0F) * 0.5F * k_pi / 180.0F;
    camera.fx = camera.fy = static_cast<float>(height) * 0.5F / std::max(1e-4F, std::tan(half));
    camera.cx = 0.5F * static_cast<float>(width);
    camera.cy = 0.5F * static_cast<float>(height);
    camera.model = photara::CameraModel::pinhole;
    return camera;
}

// Times the forward pass of both rasterizers on one trained model. The CUDA
// model stays resident on the device, exactly like training; the Vulkan model
// is uploaded once, exactly like the editor.
int run_benchmark(int argc, char** argv) {
    require(argc >= 6,
            "Usage: photara_splat_vulkan_parity --bench MODEL IMAGES OUTPUT SFM [WIDTH] [HEIGHT] [ITERS] [GAUSSIANS]");
    const std::filesystem::path model_path(argv[2]);
    const std::filesystem::path images(argv[3]);
    const std::filesystem::path output(argv[4]);
    const std::filesystem::path sfm(argv[5]);
    const std::uint32_t width = argc >= 7 ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 1000U;
    const std::uint32_t height = argc >= 8 ? static_cast<std::uint32_t>(std::stoul(argv[7])) : 1000U;
    const int iterations = argc >= 9 ? std::stoi(argv[8]) : 20;
    const std::size_t limit = argc >= 10 ? std::stoull(argv[9]) : 0;
    std::filesystem::create_directories(output);

    const auto views = load_views(images, sfm);
    require(!views.empty(), "no registered views were found");
    auto model = photara::splat::load_gaussians(model_path);

    photara::splat::TrainingOptions options;
    options.use_mask = false;
    options.use_source_resolution = true;
    options.ignore_undistortion_border = true;
    options.progressive_resolution = false;
    const auto view = photara::splat::make_training_view(views.front(), options);
    photara::splat::Camera camera = view.camera;
    camera.width = width;
    camera.height = height;
    const float scale_x = static_cast<float>(width) / static_cast<float>(view.camera.width);
    const float scale_y = static_cast<float>(height) / static_cast<float>(view.camera.height);
    camera.fx = view.camera.fx * scale_x;
    camera.fy = view.camera.fy * scale_y;
    camera.cx = view.camera.cx * scale_x;
    camera.cy = view.camera.cy * scale_y;

    // Activated attributes for the Vulkan rasterizer (see the parity path).
    std::size_t count = model.size();
    std::vector<float> means = model.means.to_vector();
    std::vector<float> sh = model.sh.to_vector();
    std::vector<float> log_scales = model.log_scales.to_vector();
    std::vector<float> raw_quaternions = model.quaternions.to_vector();
    std::vector<float> opacity_logits = model.opacity_logits.to_vector();
    std::vector<float> filter_3d =
        model.filter_3d.is_valid() ? model.filter_3d.to_vector() : std::vector<float>{};
    if (limit != 0 && limit < count) {
        // Truncating the model keeps both rasterizers on identical input and
        // shows how the frame time scales with the Gaussian count.
        const std::size_t bases = sh.size() / (count * 3U);
        count = limit;
        means.resize(count * 3U);
        sh.resize(count * bases * 3U);
        log_scales.resize(count * 3U);
        raw_quaternions.resize(count * 4U);
        opacity_logits.resize(count);
        if (!filter_3d.empty()) filter_3d.resize(count);
    }
    // Both sides run off the same host data, so the comparison cannot drift on
    // activation or layout.
    const std::size_t bases = sh.size() / (count * 3U);
    model.means = tinytensor::Tensor::from_vector(means, {count, 3U}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(log_scales, {count, 3U}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(raw_quaternions, {count, 4U}, tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(opacity_logits, {count, 1U}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(sh, {count, bases, 3U}, tinytensor::Device::CUDA);
    model.filter_3d = filter_3d.empty()
        ? tinytensor::Tensor{}
        : tinytensor::Tensor::from_vector(filter_3d, {count, 1U}, tinytensor::Device::CUDA);
    model.normal_features = {};
    std::vector<float> scales(count * 3U);
    std::vector<float> quaternions(count * 4U);
    std::vector<float> opacities(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float filter_squared = filter_3d.empty() ? 0.0F : filter_3d[i] * filter_3d[i];
        float determinant_ratio = 1.0F;
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            const float raw = std::exp(log_scales[i * 3U + axis]);
            const float filtered = std::sqrt(raw * raw + filter_squared);
            scales[i * 3U + axis] = filtered;
            determinant_ratio *= raw / filtered;
        }
        const float w = raw_quaternions[i * 4U + 0U];
        const float x = raw_quaternions[i * 4U + 1U];
        const float y = raw_quaternions[i * 4U + 2U];
        const float z = raw_quaternions[i * 4U + 3U];
        const float inverse_norm = 1.0F / std::sqrt(std::max(w * w + x * x + y * y + z * z, 1e-20F));
        quaternions[i * 4U + 0U] = w * inverse_norm;
        quaternions[i * 4U + 1U] = x * inverse_norm;
        quaternions[i * 4U + 2U] = y * inverse_norm;
        quaternions[i * 4U + 3U] = z * inverse_norm;
        opacities[i] = (1.0F / (1.0F + std::exp(-opacity_logits[i]))) * determinant_ratio;
    }

    // Adopt TinyTensor's device so its tensors can be bound directly by the
    // differentiable rasterizer without copies or external-memory export.
    const auto tensor_device = tinytensor::vulkan::device_handles();
    splat_drender::vulkan::ContextOptions context_options;
    context_options.external_device.instance = tensor_device.instance;
    context_options.external_device.physical_device = tensor_device.physical_device;
    context_options.external_device.device = tensor_device.device;
    context_options.external_device.queue = tensor_device.queue;
    context_options.external_device.queue_family = tensor_device.queue_family;
    context_options.external_device.push_descriptors =
        tinytensor::vulkan::device_info().push_descriptors;
    context_options.external_device.buffer_float32_atomic_add =
        tinytensor::vulkan::device_info().buffer_atomic_f32;
    splat_drender::vulkan::Context context(context_options);
    splat_drender::vulkan::SplatRasterizer vulkan(context);
    splat_drender::vulkan::SplatGaussians vulkan_gaussians;
    vulkan_gaussians.means = means;
    vulkan_gaussians.sh = sh;
    vulkan_gaussians.opacities = opacities;
    vulkan_gaussians.scales = scales;
    vulkan_gaussians.rotations = quaternions;
    vulkan_gaussians.log_scales = log_scales;
    vulkan_gaussians.raw_rotations = raw_quaternions;
    vulkan_gaussians.opacity_logits = opacity_logits;
    vulkan_gaussians.filter_3d = filter_3d;
    vulkan_gaussians.sh_degree = model.sh_degree;
    vulkan_gaussians.sh_bases = static_cast<std::uint32_t>(bases);
    const auto upload_start = std::chrono::steady_clock::now();
    vulkan.upload_model(vulkan_gaussians);
    const auto upload_finish = std::chrono::steady_clock::now();
    const auto vulkan_camera = to_vulkan_camera(camera);

    photara::splat::RasterizeOptions color_only;
    color_only.active_sh_degree = model.sh_degree;
    color_only.require_depth = false;
    photara::splat::RasterizeOptions with_geometry = color_only;
    with_geometry.require_depth = true;
    splat_drender::vulkan::SplatSettings vulkan_color;
    vulkan_color.need_depth = false;
    splat_drender::vulkan::SplatSettings vulkan_training = vulkan_color;
    vulkan_training.pixel_snapshots = true;
    splat_drender::vulkan::SplatSettings vulkan_geometry;
    vulkan_geometry.need_depth = true;

    std::cout << "model=" << model_path.string() << " gaussians=" << count
              << " view=" << views.front().path.filename().string() << " size=" << width << 'x' << height
              << " device=" << context.device_info().name << " iterations=" << iterations << '\n';
    std::cout << "vulkan upload_model (one-off) = " << std::chrono::duration<double, std::milli>(
                     upload_finish - upload_start).count() << " ms\n\n";
    std::cout << "phase                                  vulkan_ms  cuda_ms  ratio\n";

    volatile double sink = 0.0;
    const auto row = [&](const char* name, const double vulkan_ms, const double cuda_ms) {
        char line[256];
        char ratio[16] = "   n/a";
        if (cuda_ms > 0.0) std::snprintf(ratio, sizeof(ratio), "%5.2fx", vulkan_ms / cuda_ms);
        std::snprintf(line, sizeof(line), "%-38.38s %9.3f  %7.3f  %s\n", name, vulkan_ms, cuda_ms, ratio);
        std::cout << line;
    };

    {
        const auto vulkan_info = vulkan.render(vulkan_camera, vulkan_color);
        const auto cuda_info = photara::splat::Rasterizer().forward(model, camera, color_only);
        cudaDeviceSynchronize();
        std::cout << "instances@" << width << 'x' << height << " vulkan=" << vulkan_info.instance_count
                  << " visible=" << vulkan_info.visible_count
                  << " | cuda instances=" << cuda_info.rendered_instances << "\n\n";
    }

    // Color only, the shading the editor actually previews.
    const double vk_color = time_ms(
        [&] {
            const auto result = vulkan.render(vulkan_camera, vulkan_color);
            sink += result.color[0];
        },
        3, iterations);
    const double cuda_color = time_ms(
        [&] {
            const auto result = photara::splat::Rasterizer().forward(model, camera, color_only);
            cudaDeviceSynchronize();
            sink += result.rendered_instances;
        },
        3, iterations);
    // Same, but the caller copies the image back to the host like the editor does.
    const double cuda_color_readback = time_ms(
        [&] {
            const auto result = photara::splat::Rasterizer().forward(model, camera, color_only);
            cudaDeviceSynchronize();
            const auto color = result.color.to_vector();
            const auto alpha = result.alpha.to_vector();
            sink += color[0] + alpha[0];
        },
        3, iterations);
    row("vulkan render (color, need_depth=0)", vk_color, cuda_color);
    row("vulkan render (training snapshots)", time_ms(
            [&] {
                const auto result = vulkan.render(vulkan_camera, vulkan_training);
                sink += result.color[0];
            },
            3, iterations), cuda_color);
    // The host row includes upload/readback. The device rows bind TinyTensor
    // buffers directly and measure the training path without PCIe staging.
    const std::size_t benchmark_pixels = static_cast<std::size_t>(width) * height;
    std::vector<float> backward_color(benchmark_pixels * 3U, 1.0F / 3.0F);
    std::vector<float> backward_alpha(benchmark_pixels, 0.125F);
    const auto backward_forward = vulkan.render(vulkan_camera, vulkan_training);
    sink += backward_forward.instance_count;
    row("vulkan blend backward (host I/O)", time_ms(
            [&] {
                const auto gradients = vulkan.backward_blend(backward_color, backward_alpha);
                sink += gradients.colors.empty() ? 0.0 : gradients.colors[0];
            },
            3, iterations), 0.0);
    auto device_backward_color = tinytensor::Tensor::from_vector(
        backward_color, {backward_color.size()}, tinytensor::Device::Vulkan);
    auto device_backward_alpha = tinytensor::Tensor::from_vector(
        backward_alpha, {backward_alpha.size()}, tinytensor::Device::Vulkan);
    auto device_backward_gradient = tinytensor::Tensor::zeros(
        {static_cast<std::size_t>(vulkan.blend_gradient_float_count())},
        tinytensor::Device::Vulkan, tinytensor::DataType::Float32);
    tinytensor::vulkan::synchronize();
    const auto device_color_view = tinytensor::vulkan::buffer_view(device_backward_color);
    const auto device_alpha_view = tinytensor::vulkan::buffer_view(device_backward_alpha);
    const auto device_gradient_view = tinytensor::vulkan::buffer_view(device_backward_gradient);
    const splat_drender::vulkan::SplatBufferView device_color_loss{
        device_color_view.buffer, device_color_view.offset, device_color_view.bytes};
    const splat_drender::vulkan::SplatBufferView device_alpha_loss{
        device_alpha_view.buffer, device_alpha_view.offset, device_alpha_view.bytes};
    const splat_drender::vulkan::SplatBufferView device_packed_gradient{
        device_gradient_view.buffer, device_gradient_view.offset, device_gradient_view.bytes};
    const double vulkan_blend_device = time_ms(
        [&] {
            vulkan.backward_blend_device(
                device_color_loss, device_alpha_loss, device_packed_gradient);
        },
        3, iterations);
    auto device_model_gradient = tinytensor::Tensor::zeros(
        {static_cast<std::size_t>(vulkan.model_gradient_float_count())},
        tinytensor::Device::Vulkan, tinytensor::DataType::Float32);
    tinytensor::vulkan::synchronize();
    const auto device_model_gradient_view = tinytensor::vulkan::buffer_view(device_model_gradient);
    const splat_drender::vulkan::SplatBufferView device_packed_model_gradient{
        device_model_gradient_view.buffer, device_model_gradient_view.offset,
        device_model_gradient_view.bytes};
    const double vulkan_full_device = time_ms(
        [&] {
            vulkan.backward_device(
                device_color_loss, device_alpha_loss, device_packed_model_gradient);
        },
        3, iterations);
    auto cuda_backward_color = tinytensor::Tensor::from_vector(
        backward_color, {3U, height, width}, tinytensor::Device::CUDA);
    auto cuda_backward_alpha = tinytensor::Tensor::from_vector(
        backward_alpha, {height, width}, tinytensor::Device::CUDA);
    photara::splat::Rasterizer cuda_backward_rasterizer;
    const auto cuda_backward_frame =
        cuda_backward_rasterizer.forward(model, camera, color_only);
    const double cuda_full_backward = time_ms(
        [&] {
            const auto gradients = cuda_backward_rasterizer.backward(
                model, cuda_backward_frame, cuda_backward_color,
                cuda_backward_alpha, {}, {});
            cudaDeviceSynchronize();
            sink += gradients.means.numel();
        },
        3, iterations);
    row("vulkan blend backward (device)", vulkan_blend_device, 0.0);
    row("full backward (device)", vulkan_full_device, cuda_full_backward);
    const auto cuda_reference_gradients = cuda_backward_rasterizer.backward(
        model, cuda_backward_frame, cuda_backward_color,
        cuda_backward_alpha, {}, {});
    cudaDeviceSynchronize();
    const auto packed_model_gradients = device_model_gradient.to_vector();
    const std::size_t mean_offset = 0;
    const std::size_t feature_offset = count * 3U;
    const std::size_t feature_count = count * bases * 3U;
    const std::size_t opacity_offset = feature_offset + feature_count;
    const std::size_t scale_offset = opacity_offset + count;
    const std::size_t rotation_offset = scale_offset + count * 3U;
    const std::size_t covariance_offset = rotation_offset + count * 4U;
    const std::size_t log_scale_offset = covariance_offset + count * 6U;
    const std::size_t raw_rotation_offset = log_scale_offset + count * 3U;
    const std::size_t logit_offset = raw_rotation_offset + count * 4U;
    std::cout << "gradient rel_l2: mean="
              << relative_l2_slice(
                     packed_model_gradients, mean_offset,
                     cuda_reference_gradients.means.to_vector())
              << " sh="
              << relative_l2_slice(
                     packed_model_gradients, feature_offset,
                     cuda_reference_gradients.sh.to_vector())
              << " log_scale="
              << relative_l2_slice(
                     packed_model_gradients, log_scale_offset,
                     cuda_reference_gradients.log_scales.to_vector())
              << " quaternion="
              << relative_l2_slice(
                     packed_model_gradients, raw_rotation_offset,
                     cuda_reference_gradients.quaternions.to_vector())
              << " opacity_logit="
              << relative_l2_slice(
                     packed_model_gradients, logit_offset,
                     cuda_reference_gradients.opacity_logits.to_vector())
              << '\n';
    row("vulkan render_rgba (editor path)", time_ms(
            [&] {
                std::vector<std::uint8_t> rgba;
                vulkan.render_rgba(vulkan_camera, vulkan_color, rgba);
                sink += rgba[0];
            },
            3, iterations), cuda_color_readback);
    row("vulkan render_rgba_device (no copy)", time_ms(
            [&] {
                const auto frame = vulkan.render_rgba_device(vulkan_camera, vulkan_color);
                sink += static_cast<double>(frame.bytes);
            },
            3, iterations), 0.0);
    row("vulkan render (geometry, need_depth=1)", time_ms(
            [&] {
                const auto result = vulkan.render(vulkan_camera, vulkan_geometry);
                sink += result.median_depth[0];
            },
            3, iterations), time_ms(
            [&] {
                const auto result = photara::splat::Rasterizer().forward(model, camera, with_geometry);
                cudaDeviceSynchronize();
                sink += result.median_depth.to_vector()[0];
            },
            3, iterations));

    // Resolution probe: the same scene at 64x64 exposes the cost that does not
    // scale with the pixel count (preprocessing and the instance sort).
    photara::splat::Camera small = camera;
    small.width = 64;
    small.height = 64;
    small.fx = 64.0F * camera.fx / static_cast<float>(width);
    small.fy = 64.0F * camera.fy / static_cast<float>(height);
    small.cx = 64.0F * camera.cx / static_cast<float>(width);
    small.cy = 64.0F * camera.cy / static_cast<float>(height);
    const auto small_vulkan_camera = to_vulkan_camera(small);
    row("vulkan render at 64x64 (color)", time_ms(
            [&] {
                const auto result = vulkan.render(small_vulkan_camera, vulkan_color);
                sink += result.color[0];
            },
            3, iterations), time_ms(
            [&] {
                const auto result = photara::splat::Rasterizer().forward(model, small, color_only);
                cudaDeviceSynchronize();
                sink += result.rendered_instances;
            },
            3, iterations));
    std::cout << "\n(sink=" << sink << ")\n";
    (void)output;
    return 0;
}

// A deliberately small optimizer smoke test over one real training image.
// It keeps the full Gaussian model and CUDA-compatible Vulkan backward, but
// updates only activated opacity. Backtracking makes this a deterministic test
// of gradient direction rather than a learning-rate tuning benchmark.
int run_train_smoke(int argc, char** argv) {
    require(argc >= 6,
            "Usage: photara_splat_vulkan_parity --train-smoke MODEL IMAGES OUTPUT SFM [ITERS]");
    const std::filesystem::path model_path(argv[2]);
    const std::filesystem::path images(argv[3]);
    const std::filesystem::path output(argv[4]);
    const std::filesystem::path sfm(argv[5]);
    const int iterations = argc >= 7 ? std::max(1, std::stoi(argv[6])) : 5;
    std::filesystem::create_directories(output);

    const auto views = load_views(images, sfm);
    require(!views.empty(), "no registered views were found");
    auto model = photara::splat::load_gaussians(model_path);
    photara::splat::TrainingOptions options;
    options.use_mask = false;
    options.use_source_resolution = true;
    options.ignore_undistortion_border = true;
    options.progressive_resolution = false;
    const auto view = photara::splat::make_training_view(views.front(), options);
    const photara::splat::Camera camera = view.camera;
    require(camera.model == photara::CameraModel::pinhole,
            "Vulkan train smoke follows the pinhole-only training path");
    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    const std::vector<float> target = view.rgb.to_vector();
    require(target.size() == 3 * pixels, "training target has the wrong shape");

    const std::size_t count = model.size();
    std::vector<float> means = model.means.to_vector();
    std::vector<float> sh = model.sh.to_vector();
    std::vector<float> log_scales = model.log_scales.to_vector();
    std::vector<float> raw_quaternions = model.quaternions.to_vector();
    std::vector<float> opacity_logits = model.opacity_logits.to_vector();
    std::vector<float> filter_3d = model.filter_3d.is_valid()
        ? model.filter_3d.to_vector() : std::vector<float>{};
    const std::size_t bases = sh.size() / (count * 3U);
    std::vector<float> scales(count * 3U);
    std::vector<float> quaternions(count * 4U);
    std::vector<float> opacities(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float filter_squared = filter_3d.empty()
            ? 0.0F : filter_3d[i] * filter_3d[i];
        float determinant_ratio = 1.0F;
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            const float raw = std::exp(log_scales[3U * i + axis]);
            const float filtered = std::sqrt(raw * raw + filter_squared);
            scales[3U * i + axis] = filtered;
            determinant_ratio *= raw / filtered;
        }
        const float w = raw_quaternions[4U * i];
        const float x = raw_quaternions[4U * i + 1U];
        const float y = raw_quaternions[4U * i + 2U];
        const float z = raw_quaternions[4U * i + 3U];
        const float inverse_norm = 1.0F / std::sqrt(
            std::max(w * w + x * x + y * y + z * z, 1e-20F));
        quaternions[4U * i] = w * inverse_norm;
        quaternions[4U * i + 1U] = x * inverse_norm;
        quaternions[4U * i + 2U] = y * inverse_norm;
        quaternions[4U * i + 3U] = z * inverse_norm;
        opacities[i] =
            (1.0F / (1.0F + std::exp(-opacity_logits[i]))) *
            determinant_ratio;
    }

    require(tinytensor::vulkan::available(),
            "TinyTensor Vulkan backend is unavailable");
    const auto handles = tinytensor::vulkan::device_handles();
    splat_drender::vulkan::ContextOptions context_options;
    context_options.external_device.instance = handles.instance;
    context_options.external_device.physical_device = handles.physical_device;
    context_options.external_device.device = handles.device;
    context_options.external_device.queue = handles.queue;
    context_options.external_device.queue_family = handles.queue_family;
    splat_drender::vulkan::Context context(context_options);
    splat_drender::vulkan::SplatRasterizer rasterizer(context);
    splat_drender::vulkan::SplatGaussians gaussians;
    gaussians.means = means;
    gaussians.sh = sh;
    gaussians.opacities = opacities;
    gaussians.scales = scales;
    gaussians.rotations = quaternions;
    gaussians.sh_degree = model.sh_degree;
    gaussians.sh_bases = static_cast<std::uint32_t>(bases);
    rasterizer.upload_model(gaussians);
    const auto vk_camera = to_vulkan_camera(camera);
    splat_drender::vulkan::SplatSettings training_settings;
    training_settings.need_depth = false;
    training_settings.pixel_snapshots = true;
    splat_drender::vulkan::SplatSettings evaluation_settings = training_settings;
    evaluation_settings.pixel_snapshots = false;
    const std::vector<float> zero_alpha(pixels, 0.0F);

    const auto objective = [&](const std::vector<float>& color,
                               std::vector<float>* gradient) {
        double sum = 0.0;
        if (gradient != nullptr) gradient->resize(3 * pixels);
        const float inverse = 1.0F / static_cast<float>(3 * pixels);
        for (std::size_t i = 0; i < 3 * pixels; ++i) {
            const float difference = color[i] - target[i];
            sum += static_cast<double>(difference) * difference;
            if (gradient != nullptr) (*gradient)[i] = 2.0F * difference * inverse;
        }
        return static_cast<float>(sum * inverse);
    };

    float first_loss = -1.0F;
    float current_loss = std::numeric_limits<float>::infinity();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto frame = rasterizer.render(vk_camera, training_settings);
        std::vector<float> color_gradient;
        current_loss = objective(frame.color, &color_gradient);
        if (iteration == 0) first_loss = current_loss;
        const auto gradients = rasterizer.backward(color_gradient, zero_alpha);
        require(gradients.opacities.size() == count,
                "Vulkan train smoke returned the wrong opacity gradient shape");
        float maximum_gradient = 0.0F;
        for (float value : gradients.opacities) {
            require(std::isfinite(value),
                    "Vulkan train smoke produced a non-finite gradient");
            maximum_gradient = std::max(maximum_gradient, std::abs(value));
        }
        require(maximum_gradient > 0.0F,
                "Vulkan train smoke produced a zero gradient");
        const std::vector<float> base = opacities;
        float step = 0.02F / maximum_gradient;
        bool accepted = false;
        float accepted_loss = current_loss;
        for (int trial = 0; trial < 12; ++trial) {
            for (std::size_t i = 0; i < count; ++i)
                opacities[i] = std::clamp(
                    base[i] - step * gradients.opacities[i], 1.0e-5F, 0.999F);
            rasterizer.update_means_and_opacities(means, opacities);
            const auto candidate = rasterizer.render(vk_camera, evaluation_settings);
            accepted_loss = objective(candidate.color, nullptr);
            if (std::isfinite(accepted_loss) && accepted_loss < current_loss) {
                accepted = true;
                break;
            }
            step *= 0.5F;
        }
        require(accepted, "Vulkan train smoke could not find a descending step");
        current_loss = accepted_loss;
        std::cout << "vulkan train iteration=" << iteration + 1 << '/' << iterations
                  << " mse=" << current_loss << " step=" << step
                  << " max_grad=" << maximum_gradient << '\n';
    }
    require(current_loss < first_loss,
            "Vulkan train smoke did not reduce the real-image objective");
    std::ofstream report(output / "vulkan_train_smoke.txt");
    report << "model=" << model_path.string() << '\n'
           << "view=" << views.front().path.string() << '\n'
           << "gaussians=" << count << '\n'
           << "size=" << camera.width << 'x' << camera.height << '\n'
           << "iterations=" << iterations << '\n'
           << "initial_mse=" << first_loss << '\n'
           << "final_mse=" << current_loss << '\n';
    std::cout << "Vulkan real-data train smoke: " << first_loss << " -> "
              << current_loss << " report="
              << (output / "vulkan_train_smoke.txt").string() << '\n';
    return 0;
}


}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--bench") return run_benchmark(argc, argv);
        if (argc > 1 && std::string(argv[1]) == "--train-smoke")
            return run_train_smoke(argc, argv);
        if (argc < 4 || argc > 12)
            throw std::runtime_error(
                "Usage: photara_splat_vulkan_parity MODEL IMAGES OUTPUT_DIR [STRIDE] [SFM]"
                " [ORBIT_DISTANCE ORBIT_YAW ORBIT_PITCH ORBIT_FOV [WIDTH HEIGHT]]");
        const unsigned stride = argc >= 5 ? std::stoul(argv[4]) : 8U;
        if (stride == 0) throw std::runtime_error("STRIDE must be >= 1");
        const std::filesystem::path images(argv[2]);
        const std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);

        const std::vector<photara::mvs::MvsView> views = load_views(
            images, argc == 6 ? std::filesystem::path(argv[5]) : std::filesystem::path{});
        require(!views.empty(), "no registered views were found");
        const std::filesystem::path model_path(argv[1]);
        const bool packed_model = model_path.extension() == ".bin";
        const auto model = packed_model ? load_packed_model(model_path)
                                        : photara::splat::load_gaussians(model_path);
        require(model.size() > 0, "the model has no Gaussians");
        std::cout << "model=" << argv[1] << " gaussians=" << model.size()
                  << " sh_degree=" << model.sh_degree << " views=" << views.size() << '\n';

        // Same view preparation as the quality evaluation tool, so the rendered
        // camera is the training camera at source resolution.
        photara::splat::TrainingOptions options;
        options.use_mask = false;
        options.use_source_resolution = true;
        options.ignore_undistortion_border = true;
        options.progressive_resolution = false;

        // Activated attributes for the Vulkan rasterizer. photara::splat's
        // activate_kernel applies exactly this: exp(log scale) with the Mip 3D
        // filter, a normalised quaternion and sigmoid(logit) scaled by the
        // footprint determinant ratio.
        const std::size_t count = model.size();
        const std::vector<float> means = model.means.to_vector();
        const std::vector<float> log_scales = model.log_scales.to_vector();
        const std::vector<float> raw_quaternions = model.quaternions.to_vector();
        const std::vector<float> opacity_logits = model.opacity_logits.to_vector();
        const std::vector<float> sh = model.sh.to_vector();
        const std::vector<float> filter_3d =
            model.filter_3d.is_valid() ? model.filter_3d.to_vector() : std::vector<float>{};
        std::vector<float> scales(count * 3U);
        std::vector<float> quaternions(count * 4U);
        std::vector<float> opacities(count);
        for (std::size_t i = 0; i < count; ++i) {
            const float filter_squared =
                filter_3d.empty() ? 0.0F : filter_3d[i] * filter_3d[i];
            float determinant_ratio = 1.0F;
            for (std::size_t axis = 0; axis < 3U; ++axis) {
                const float raw = std::exp(log_scales[i * 3U + axis]);
                const float filtered = std::sqrt(raw * raw + filter_squared);
                scales[i * 3U + axis] = filtered;
                determinant_ratio *= raw / filtered;
            }
            const float w = raw_quaternions[i * 4U + 0U];
            const float x = raw_quaternions[i * 4U + 1U];
            const float y = raw_quaternions[i * 4U + 2U];
            const float z = raw_quaternions[i * 4U + 3U];
            const float inverse_norm =
                1.0F / std::sqrt(std::max(w * w + x * x + y * y + z * z, 1e-20F));
            quaternions[i * 4U + 0U] = w * inverse_norm;
            quaternions[i * 4U + 1U] = x * inverse_norm;
            quaternions[i * 4U + 2U] = y * inverse_norm;
            quaternions[i * 4U + 3U] = z * inverse_norm;
            opacities[i] = (1.0F / (1.0F + std::exp(-opacity_logits[i]))) * determinant_ratio;
        }

        splat_drender::vulkan::Context context;
        splat_drender::vulkan::SplatRasterizer vulkan(context);
        splat_drender::vulkan::SplatGaussians vulkan_gaussians;
        vulkan_gaussians.means = means;
        vulkan_gaussians.sh = sh;
        vulkan_gaussians.opacities = opacities;
        vulkan_gaussians.scales = scales;
        vulkan_gaussians.rotations = quaternions;
        vulkan_gaussians.sh_degree = model.sh_degree;
        vulkan_gaussians.sh_bases = static_cast<std::uint32_t>(sh.size() / (count * 3U));
        vulkan.upload_model(vulkan_gaussians);

        std::cout << "device=" << context.device_info().name << '\n';
        std::cout << "\nview                     size        color      alpha      depth     normal"
                     "   inst(c/v)   vis(c/v)  radius(max/num)  flag_mm  psnr8(pair/c/v)\n";

        double worst_color = 0.0;
        double worst_alpha = 0.0;
        double worst_depth = 0.0;
        double worst_normal = 0.0;
        double total_psnr_cuda = 0.0;
        double total_psnr_vulkan = 0.0;
        double total_psnr_pair = 0.0;
        std::size_t compared = 0;
        // Optional editor-style orbit camera instead of the training views:
        // DISTANCE YAW PITCH FOV [WIDTH HEIGHT]. The editor lets the user orbit
        // freely, so this is where "close to the model" behaviour is checked.
        const bool orbit_override = argc >= 7;
        const float orbit_distance = orbit_override ? std::stof(argv[6]) : 0.0F;
        const float orbit_yaw = argc >= 8 ? std::stof(argv[7]) : 45.0F;
        const float orbit_pitch = argc >= 9 ? std::stof(argv[8]) : 25.0F;
        const float orbit_fov = argc >= 10 ? std::stof(argv[9]) : 50.0F;
        const auto orbit_width = argc >= 11 ? static_cast<std::uint32_t>(std::stoul(argv[10])) : 872U;
        const auto orbit_height = argc >= 12 ? static_cast<std::uint32_t>(std::stoul(argv[11])) : 558U;
        // Narrowing the field of view of a training view is the same close-up
        // the editor reaches by moving the eye in, without inventing a basis.
        const float zoom_fov = std::getenv("SPLAT_PROBE_FOV") != nullptr
            ? static_cast<float>(std::atof(std::getenv("SPLAT_PROBE_FOV")))
            : 0.0F;
        if (orbit_override) {
            std::cout << "orbit camera: distance=" << orbit_distance << " yaw=" << orbit_yaw
                      << " pitch=" << orbit_pitch << " fov=" << orbit_fov << " raster=" << orbit_width << 'x'
                      << orbit_height << '\n';
        }
        for (std::size_t index = 0; index < views.size(); index += stride) {
            const auto target = photara::splat::make_training_view(views[index], options);
            photara::splat::Camera orbit_camera;
            std::vector<float> orbit_target;
            if (orbit_override) {
                if (index != 0) break;
                orbit_camera = make_orbit_camera(
                    means, orbit_distance, orbit_yaw, orbit_pitch, orbit_fov, orbit_width, orbit_height);
                orbit_target.assign(
                    static_cast<std::size_t>(orbit_width) * orbit_height * 3U, 0.0F);
            }
            photara::splat::Camera camera = orbit_override ? orbit_camera : target.camera;
            if (zoom_fov > 1.0F) {
                const auto zoom_width = orbit_width;
                const auto zoom_height = orbit_height;
                const float half = zoom_fov * 0.5F * 3.14159265358979323846F / 180.0F;
                camera.width = zoom_width;
                camera.height = zoom_height;
                camera.fx = camera.fy = static_cast<float>(zoom_height) * 0.5F / std::tan(half);
                camera.cx = 0.5F * static_cast<float>(zoom_width);
                camera.cy = 0.5F * static_cast<float>(zoom_height);
            }
            require(camera.width > 0 && camera.height > 0, "the view has no resolution");

            photara::splat::RasterizeOptions raster;
            raster.active_sh_degree = model.sh_degree;
            raster.require_depth = true;
            const auto cuda = photara::splat::Rasterizer().forward(model, camera, raster);

            splat_drender::vulkan::SplatCamera vulkan_camera;
            vulkan_camera = to_vulkan_camera(camera);
            splat_drender::vulkan::SplatSettings settings;
            settings.need_depth = true;
            const auto vulkan_output = vulkan.render(vulkan_camera, settings);

            const std::size_t area = static_cast<std::size_t>(camera.width) * camera.height;
            const std::vector<float> cuda_color = cuda.color.to_vector();
            const std::vector<float> cuda_alpha = cuda.alpha.to_vector();
            const std::vector<float> cuda_depth = cuda.median_depth.to_vector();
            const std::vector<float> cuda_normal = cuda.normal.to_vector();
            const std::vector<float> cuda_visibility = cuda.visibility.to_vector();
            const std::vector<int> cuda_radii = cuda.radii.to_vector_int();

            require(vulkan_output.color.size() == area * 3U, "Vulkan color has the wrong size");
            const double color_error = relative_l2(vulkan_output.color, cuda_color);
            const double alpha_error = relative_l2(vulkan_output.alpha, cuda_alpha);
            const double depth_error = relative_l2(vulkan_output.median_depth, cuda_depth);
            const double normal_error = relative_l2(vulkan_output.normal, cuda_normal);
            worst_color = std::max(worst_color, color_error);
            worst_alpha = std::max(worst_alpha, alpha_error);
            worst_depth = std::max(worst_depth, depth_error);
            worst_normal = std::max(worst_normal, normal_error);

            int radius_delta = 0;
            int radius_mismatches = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const int delta = std::abs(vulkan_output.radii[i] - cuda_radii[i]);
                radius_delta = std::max(radius_delta, delta);
                if (delta != 0) ++radius_mismatches;
            }
            std::size_t visibility_mismatches = 0;
            const std::size_t cuda_flag_count =
                static_cast<std::size_t>(std::count_if(cuda_visibility.begin(), cuda_visibility.end(),
                                                       [](float value) { return value > 0.0F; }));
            std::size_t vulkan_flag_count = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const bool left = vulkan_output.visibility[i] > 0.0F;
                const bool right = cuda_visibility[i] > 0.0F;
                if (left) ++vulkan_flag_count;
                if (left != right) ++visibility_mismatches;
            }

            std::size_t depth_pixels_over = 0;
            const double depth_max = max_absolute_difference(vulkan_output.median_depth, cuda_depth, &depth_pixels_over, 1e-3);
            const double color_max = max_absolute_difference(vulkan_output.color, cuda_color);
            const double normal_max = max_absolute_difference(vulkan_output.normal, cuda_normal);
            std::size_t depth_zero_mismatches = 0;
            for (std::size_t i = 0; i < cuda_depth.size(); ++i) {
                const bool left = vulkan_output.median_depth[i] == 0.0F;
                const bool right = cuda_depth[i] == 0.0F;
                if (left != right) ++depth_zero_mismatches;
            }

            const std::string name = orbit_override ? std::string("orbit-camera") : views[index].path.filename().string();

            // SPLAT_PARITY_DUMP=<dir> writes every forward channel of both
            // backends to disk, which is how median depth and normals are
            // inspected pixel by pixel instead of through one relative-L2
            // number. Channel-major float32, C order, plus a small manifest.
            const char* dump_directory = std::getenv("SPLAT_PARITY_DUMP");
            if (dump_directory != nullptr && *dump_directory != '\0') {
                const std::filesystem::path dump_root(dump_directory);
                const std::string stem = std::to_string(index);
                std::filesystem::create_directories(dump_root);
                dump_channel(dump_root, "cuda_color_" + stem, cuda_color);
                dump_channel(dump_root, "vulkan_color_" + stem, vulkan_output.color);
                dump_channel(dump_root, "cuda_alpha_" + stem, cuda_alpha);
                dump_channel(dump_root, "vulkan_alpha_" + stem, vulkan_output.alpha);
                dump_channel(dump_root, "cuda_depth_" + stem, cuda_depth);
                dump_channel(dump_root, "vulkan_depth_" + stem, vulkan_output.median_depth);
                dump_channel(dump_root, "cuda_normal_" + stem, cuda_normal);
                dump_channel(dump_root, "vulkan_normal_" + stem, vulkan_output.normal);
                dump_channel(dump_root, "cuda_visibility_" + stem, cuda_visibility);
                dump_channel(dump_root, "vulkan_visibility_" + stem, vulkan_output.visibility);
                dump_channel(dump_root, "cuda_radii_" + stem, cuda_radii);
                dump_channel(dump_root, "vulkan_radii_" + stem, vulkan_output.radii);
                std::ofstream meta(dump_root / ("meta_" + stem + ".txt"));
                meta << "view " << name << '\n'
                     << "width " << camera.width << '\n'
                     << "height " << camera.height << '\n'
                     << "fx " << camera.fx << '\n'
                     << "fy " << camera.fy << '\n'
                     << "cx " << camera.cx << '\n'
                     << "cy " << camera.cy << '\n'
                     << "gaussians " << count << '\n'
                     << "cuda_instances " << cuda.rendered_instances << '\n'
                     << "vulkan_instances " << vulkan_output.instance_count << '\n'
                     << "cuda_visible " << cuda_flag_count << '\n'
                     << "vulkan_visible " << vulkan_flag_count << '\n'
                     << "visibility_mismatches " << visibility_mismatches << '\n'
                     << "radius_max_delta " << radius_delta << '\n'
                     << "radius_mismatches " << radius_mismatches << '\n'
                     << "depth_max_abs " << depth_max << '\n'
                     << "depth_pixels_over_1e-3 " << depth_pixels_over << '\n'
                     << "depth_zero_mismatches " << depth_zero_mismatches << '\n'
                     << "normal_max_abs " << normal_max << '\n'
                     << "color_max_abs " << color_max << '\n';
            }

            const Image cuda_image = to_image(cuda_color, camera.width, camera.height);
            const Image vulkan_image = to_image(vulkan_output.color, camera.width, camera.height);
            const Image target_image = to_image(
                orbit_override ? orbit_target : target.rgb.to_vector(), camera.width, camera.height);
            const double pair_psnr = psnr_8bit(cuda_image, vulkan_image);
            const double cuda_psnr = psnr_8bit(cuda_image, target_image);
            const double vulkan_psnr = psnr_8bit(vulkan_image, target_image);
            total_psnr_pair += pair_psnr;
            total_psnr_cuda += cuda_psnr;
            total_psnr_vulkan += vulkan_psnr;
            ++compared;

            char line[512];
            std::snprintf(
                line, sizeof(line),
                "%-22.22s %4ux%-5u %9.2e %9.2e %9.2e %9.2e %7d/%-7d %7zu/%-7zu %4d/%-9d %9zu  %5.2f/%.2f/%.2f\n",
                name.c_str(), camera.width, camera.height, color_error, alpha_error, depth_error,
                normal_error, cuda.rendered_instances, vulkan_output.instance_count, cuda_flag_count,
                vulkan_flag_count, radius_delta, radius_mismatches, visibility_mismatches,
                pair_psnr, cuda_psnr, vulkan_psnr);
            std::cout << line;

            const std::string stem = std::to_string(index);
            photara::io::save_rgb_png(
                photara::io::RgbImage{camera.width, camera.height, cuda_image.pixels}, output / ("cuda_" + stem + ".png"));
            photara::io::save_rgb_png(
                photara::io::RgbImage{camera.width, camera.height, vulkan_image.pixels}, output / ("vulkan_" + stem + ".png"));
            photara::io::save_rgb_png(
                photara::io::RgbImage{camera.width, camera.height, target_image.pixels}, output / ("target_" + stem + ".png"));
            const Image delta = difference_image(cuda_image, vulkan_image, 8);
            photara::io::save_rgb_png(
                photara::io::RgbImage{camera.width, camera.height, delta.pixels}, output / ("delta_" + stem + ".png"));
            std::cout << "    mean|delta|=" << mean_absolute_8bit(cuda_image, vulkan_image)
                      << "/255 max|color|=" << color_max
                      << " depth: max|d|=" << depth_max << " >1e-3 px=" << depth_pixels_over
                      << " zero_mm=" << depth_zero_mismatches
                      << " normal: max|d|=" << normal_max
                      << " files=cuda_/vulkan_/target_/delta_" << stem << ".png\n";
        }
        require(compared > 0, "no views were rendered");
        std::cout << "\nworst relative L2 over " << compared << " views: color=" << worst_color
                  << " alpha=" << worst_alpha << " depth=" << worst_depth << " normal=" << worst_normal << '\n';
        std::cout << "mean 8-bit PSNR: cuda-vs-vulkan=" << total_psnr_pair / compared
                  << " cuda-vs-photo=" << total_psnr_cuda / compared
                  << " vulkan-vs-photo=" << total_psnr_vulkan / compared << '\n';
        std::cout << "images in " << output.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
