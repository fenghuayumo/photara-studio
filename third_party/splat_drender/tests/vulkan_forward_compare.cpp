#include <cuda_runtime.h>

#include <splat_drender/vulkan_api.h>
#include <splat_drender/api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

struct Rng {
    std::uint32_t state;
    float next() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) * (1.0F / 16777216.0F);
    }
    float range(float lo, float hi) { return lo + (hi - lo) * next(); }
};

struct CudaBuffer {
    void* ptr = nullptr;
    std::size_t bytes = 0;
    CudaBuffer() = default;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    ~CudaBuffer() { cudaFree(ptr); }

    void* ensure(std::size_t size) {
        if (size > bytes) {
            check_cuda(cudaFree(ptr), "cudaFree");
            ptr = nullptr;
            check_cuda(cudaMalloc(&ptr, size), "cudaMalloc");
            bytes = size;
        }
        return ptr;
    }

    template <typename T>
    T* upload(const std::vector<T>& values) {
        auto* device = static_cast<T*>(ensure(std::max(values.size() * sizeof(T), std::size_t{4})));
        if (!values.empty()) {
            check_cuda(cudaMemcpy(device, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
        }
        return device;
    }

    template <typename T>
    T* zeros(std::size_t count) {
        auto* device = static_cast<T*>(ensure(std::max(count * sizeof(T), std::size_t{4})));
        check_cuda(cudaMemset(device, 0, count * sizeof(T)), "memset");
        return device;
    }

    template <typename T>
    std::vector<T> download(std::size_t count) const {
        std::vector<T> values(count);
        if (count != 0) {
            check_cuda(cudaMemcpy(values.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
        }
        return values;
    }
};

struct Scene {
    std::string name;
    std::uint32_t width = 64;
    std::uint32_t height = 48;
    std::uint32_t mode = 0;
    float fx = 48;
    float fy = 48;
    float cx = 32;
    float cy = 24;
    float k1 = 0, k2 = 0, k3 = 0, k4 = 0;
    float kernel_size = 0;
    float scale_modifier = 1;
    bool need_depth = true;
    bool use_sh = false;
    bool use_covariance = false;
    bool expect_double_sort = false;
    std::uint32_t sh_degree = 0;
    std::vector<float> means;
    std::vector<float> colors;
    std::vector<float> sh;
    std::vector<float> opacities;
    std::vector<float> scales;
    std::vector<float> rotations;
    std::vector<float> covariances;
};

void add_gaussian(Scene& scene, Rng& rng, float x, float y, float z, float scale, bool color) {
    scene.means.insert(scene.means.end(), {x, y, z});
    scene.opacities.push_back(rng.range(0.15F, 0.95F));
    float qx = rng.range(-1, 1), qy = rng.range(-1, 1), qz = rng.range(-1, 1), qw = rng.range(-1, 1);
    const float qn = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    scene.rotations.insert(scene.rotations.end(), {qw / qn, qx / qn, qy / qn, qz / qn});
    const float sx = scale * rng.range(0.6F, 1.4F);
    const float sy = scale * rng.range(0.6F, 1.4F);
    const float sz = scale * rng.range(0.6F, 1.4F);
    scene.scales.insert(scene.scales.end(), {sx, sy, sz});
    if (color) scene.colors.insert(scene.colors.end(), {rng.range(0.05F, 0.95F), rng.range(0.05F, 0.95F), rng.range(0.05F, 0.95F)});
}

void fill_sh(Scene& scene, Rng& rng) {
    const std::uint32_t bases = (scene.sh_degree + 1) * (scene.sh_degree + 1);
    const auto count = static_cast<std::uint32_t>(scene.means.size() / 3);
    scene.sh.assign(static_cast<std::size_t>(count) * bases * 3, 0.0F);
    for (std::uint32_t i = 0; i < count; ++i) {
        scene.sh[(i * bases) * 3 + 0] = rng.range(-0.4F, 0.4F);
        scene.sh[(i * bases) * 3 + 1] = rng.range(-0.4F, 0.4F);
        scene.sh[(i * bases) * 3 + 2] = rng.range(-0.4F, 0.4F);
        for (std::uint32_t b = 1; b < bases; ++b) {
            for (int channel = 0; channel < 3; ++channel) {
                scene.sh[(static_cast<std::size_t>(i) * bases + b) * 3 + channel] = rng.range(-0.15F, 0.15F);
            }
        }
    }
}

void fill_covariance(Scene& scene) {
    const auto count = scene.means.size() / 3;
    scene.covariances.resize(count * 6);
    for (std::size_t i = 0; i < count; ++i) {
        const float sx = scene.scales[i * 3];
        const float sy = scene.scales[i * 3 + 1];
        const float sz = scene.scales[i * 3 + 2];
        const float w = scene.rotations[i * 4];
        const float x = scene.rotations[i * 4 + 1];
        const float y = scene.rotations[i * 4 + 2];
        const float z = scene.rotations[i * 4 + 3];
        const float r00 = 1 - 2 * (y * y + z * z);
        const float r01 = 2 * (x * y - w * z);
        const float r02 = 2 * (x * z + w * y);
        const float r10 = 2 * (x * y + w * z);
        const float r11 = 1 - 2 * (x * x + z * z);
        const float r12 = 2 * (y * z - w * x);
        const float r20 = 2 * (x * z - w * y);
        const float r21 = 2 * (y * z + w * x);
        const float r22 = 1 - 2 * (x * x + y * y);
        const float m00 = sx * r00, m01 = sx * r01, m02 = sx * r02;
        const float m10 = sy * r10, m11 = sy * r11, m12 = sy * r12;
        const float m20 = sz * r20, m21 = sz * r21, m22 = sz * r22;
        // Sigma = R^T S^2 R, stored as the upper triangle of (S R)^T (S R) = R^T S^2 R.
        // Rows of M = S * R are scale * row of R. Vrk = M^T M.
        auto dot = [](float a, float b, float c, float d, float e, float f) { return a * d + b * e + c * f; };
        scene.covariances[i * 6 + 0] = dot(m00, m10, m20, m00, m10, m20);
        scene.covariances[i * 6 + 1] = dot(m00, m10, m20, m01, m11, m21);
        scene.covariances[i * 6 + 2] = dot(m00, m10, m20, m02, m12, m22);
        scene.covariances[i * 6 + 3] = dot(m01, m11, m21, m01, m11, m21);
        scene.covariances[i * 6 + 4] = dot(m01, m11, m21, m02, m12, m22);
        scene.covariances[i * 6 + 5] = dot(m02, m12, m22, m02, m12, m22);
    }
}

double rel_l2(const std::vector<float>& got, const std::vector<float>& reference) {
    double numerator = 0;
    double denominator = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double delta = static_cast<double>(got[i]) - reference[i];
        numerator += delta * delta;
        denominator += static_cast<double>(reference[i]) * reference[i];
    }
    if (denominator < 1e-30) return std::sqrt(numerator);
    return std::sqrt(numerator / denominator);
}

void compare_case(const Scene& scene) {
    const std::vector<float> view{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const std::vector<float> center{0, 0, 0};
    const float background[3] = {0.1F, 0.2F, 0.3F};
    const auto count = static_cast<int>(scene.means.size() / 3);
    const auto pixels = static_cast<std::size_t>(scene.width) * scene.height;

    splat_drender::vulkan::SplatCamera camera;
    camera.width = scene.width;
    camera.height = scene.height;
    camera.fx = scene.fx;
    camera.fy = scene.fy;
    camera.cx = scene.cx;
    camera.cy = scene.cy;
    camera.mode = scene.mode;
    camera.k1 = scene.k1;
    camera.k2 = scene.k2;
    camera.k3 = scene.k3;
    camera.k4 = scene.k4;
    camera.world_to_camera = view;
    camera.center = center;
    splat_drender::vulkan::SplatGaussians gaussians;
    gaussians.means = scene.means;
    gaussians.opacities = scene.opacities;
    if (scene.use_sh) {
        gaussians.sh = scene.sh;
        gaussians.sh_degree = scene.sh_degree;
        gaussians.sh_bases = (scene.sh_degree + 1) * (scene.sh_degree + 1);
    } else {
        gaussians.colors = scene.colors;
    }
    if (scene.use_covariance) gaussians.covariances = scene.covariances;
    else {
        gaussians.scales = scene.scales;
        gaussians.rotations = scene.rotations;
    }
    splat_drender::vulkan::SplatSettings settings;
    settings.background[0] = background[0];
    settings.background[1] = background[1];
    settings.background[2] = background[2];
    settings.kernel_size = scene.kernel_size;
    settings.scale_modifier = scene.scale_modifier;
    settings.need_depth = scene.need_depth;

    splat_drender::vulkan::Context context;
    splat_drender::vulkan::SplatRasterizer rasterizer(context);
    const auto vulkan = rasterizer.forward(gaussians, camera, settings);

    CudaBuffer means, color, opacity, scales, rotations, covariance, view_dev, center_dev;
    CudaBuffer out_color, out_alpha, out_depth, out_normal, out_visibility, out_radii;
    CudaBuffer pool_g, pool_grad, pool_i, pool_p, pool_t, pool_point;
    const float* sh_ptr = scene.use_sh ? color.upload(scene.sh) : nullptr;
    const float* colors_ptr = scene.use_sh ? nullptr : color.upload(scene.colors);
    splat_drender::Gaussians cuda_g;
    cuda_g.count = count;
    cuda_g.means = means.upload(scene.means);
    cuda_g.sh = sh_ptr;
    cuda_g.colors = colors_ptr;
    cuda_g.opacities = opacity.upload(scene.opacities);
    cuda_g.scales = scene.use_covariance ? nullptr : scales.upload(scene.scales);
    cuda_g.rotations = scene.use_covariance ? nullptr : rotations.upload(scene.rotations);
    cuda_g.covariances = scene.use_covariance ? covariance.upload(scene.covariances) : nullptr;
    cuda_g.sh_degree = scene.use_sh ? static_cast<int>(scene.sh_degree) : 0;
    cuda_g.sh_bases = scene.use_sh ? static_cast<int>((scene.sh_degree + 1) * (scene.sh_degree + 1)) : 0;

    splat_drender::CameraView cuda_camera;
    cuda_camera.width = static_cast<int>(scene.width);
    cuda_camera.height = static_cast<int>(scene.height);
    cuda_camera.fx = scene.fx;
    cuda_camera.fy = scene.fy;
    cuda_camera.cx = scene.cx;
    cuda_camera.cy = scene.cy;
    cuda_camera.mode = static_cast<splat_drender::CameraMode>(scene.mode);
    cuda_camera.k1 = scene.k1;
    cuda_camera.k2 = scene.k2;
    cuda_camera.k3 = scene.k3;
    cuda_camera.k4 = scene.k4;
    cuda_camera.world_to_camera = view_dev.upload(view);
    cuda_camera.center = center_dev.upload(center);

    splat_drender::RenderSettings cuda_settings;
    cuda_settings.background[0] = background[0];
    cuda_settings.background[1] = background[1];
    cuda_settings.background[2] = background[2];
    cuda_settings.scale_modifier = scene.scale_modifier;
    cuda_settings.kernel_size = scene.kernel_size;
    cuda_settings.need_depth = scene.need_depth;

    splat_drender::RenderOutputs cuda_out;
    cuda_out.color = out_color.zeros<float>(pixels * 3);
    cuda_out.alpha = out_alpha.zeros<float>(pixels);
    cuda_out.median_depth = scene.need_depth ? out_depth.zeros<float>(pixels) : nullptr;
    cuda_out.normal = scene.need_depth ? out_normal.zeros<float>(pixels * 3) : nullptr;
    cuda_out.visibility = out_visibility.zeros<float>(static_cast<std::size_t>(count));
    cuda_out.radii = out_radii.zeros<int>(static_cast<std::size_t>(count));

    splat_drender::WorkspacePools pools;
    pools.gaussian = [&](std::size_t bytes) { return static_cast<char*>(pool_g.ensure(bytes)); };
    pools.grad = [&](std::size_t bytes) { return static_cast<char*>(pool_grad.ensure(bytes)); };
    pools.instance = [&](std::size_t bytes) { return static_cast<char*>(pool_i.ensure(bytes)); };
    pools.pixel = [&](std::size_t bytes) { return static_cast<char*>(pool_p.ensure(bytes)); };
    pools.tile = [&](std::size_t bytes) { return static_cast<char*>(pool_t.ensure(bytes)); };
    pools.point = [&](std::size_t bytes) { return static_cast<char*>(pool_point.ensure(bytes)); };

    const auto cuda_result = splat_drender::Rasterizer::forward(pools, cuda_g, cuda_camera, cuda_settings, cuda_out);
    check_cuda(cudaGetLastError(), "cuda forward");
    check_cuda(cudaDeviceSynchronize(), "cuda synchronize");

    const auto cuda_color = out_color.download<float>(pixels * 3);
    const auto cuda_alpha = out_alpha.download<float>(pixels);
    const auto cuda_visibility = out_visibility.download<float>(static_cast<std::size_t>(count));
    const auto cuda_radii = out_radii.download<int>(static_cast<std::size_t>(count));
    const double color_error = rel_l2(vulkan.color, cuda_color);
    const double alpha_error = rel_l2(vulkan.alpha, cuda_alpha);
    std::cout << scene.name << " instances cuda=" << cuda_result.instance_count
              << " vulkan=" << vulkan.instance_count
              << " visible cuda=" << cuda_result.visible_count
              << " vulkan=" << vulkan.visible_count
              << " color=" << color_error << " alpha=" << alpha_error;
    double depth_error = 0;
    double normal_error = 0;
    if (scene.need_depth) {
        depth_error = rel_l2(vulkan.median_depth, out_depth.download<float>(pixels));
        normal_error = rel_l2(vulkan.normal, out_normal.download<float>(pixels * 3));
        std::cout << " depth=" << depth_error << " normal=" << normal_error;
    }
    std::cout << '\n';

    require(vulkan.instance_count == cuda_result.instance_count && vulkan.visible_count == cuda_result.visible_count,
            scene.name + " instance or visible count differs");
    if (scene.expect_double_sort) {
        require(vulkan.instance_count > 131072, scene.name + " did not reach the double-sort path");
    }
    int radius_delta = 0;
    int radius_mismatches = 0;
    for (int i = 0; i < count; ++i) {
        const int delta = std::abs(vulkan.radii[static_cast<std::size_t>(i)] - cuda_radii[static_cast<std::size_t>(i)]);
        radius_delta = std::max(radius_delta, delta);
        if (delta != 0) ++radius_mismatches;
    }
    std::cout << "  radius_max_abs=" << radius_delta << " mismatches=" << radius_mismatches << '\n';
    // CUDA builds the 3-sigma radius with fast sqrt, so ceil can move by one
    // without changing the opacity-bounded tile list or the blended image.
    require(radius_delta <= 1, scene.name + " radii differ by more than one pixel");
    for (int i = 0; i < count; ++i) {
        require(vulkan.visibility[static_cast<std::size_t>(i)] == cuda_visibility[static_cast<std::size_t>(i)],
                scene.name + " visibility differs");
    }
    require(color_error < 1e-4 && alpha_error < 1e-4, scene.name + " color or alpha exceeds 1e-4 relative L2");
    if (scene.need_depth) {
        require(depth_error < 1e-4 && normal_error < 1e-4, scene.name + " depth or normal exceeds 1e-4 relative L2");
    }
}

Scene pinhole_scene(const char* name, int count, float scale, bool depth) {
    Scene scene;
    scene.name = name;
    scene.need_depth = depth;
    Rng rng{19};
    for (int i = 0; i < count; ++i) {
        add_gaussian(scene, rng, rng.range(-0.6F, 0.6F), rng.range(-0.45F, 0.45F), rng.range(1.2F, 3.4F), scale, true);
    }
    return scene;
}

// One centered Gaussian in front of the camera, held together with the spans
// that point at its own storage.
struct SingleGaussianScene {
    const std::vector<float> identity{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
    const std::vector<float> origin{0, 0, 0};
    const std::vector<float> means{0, 0, 2};
    const std::vector<float> colors{1, 0, 0};
    const std::vector<float> opacity{0.8F};
    const std::vector<float> scales{0.12F, 0.12F, 0.12F};
    const std::vector<float> rotation{1, 0, 0, 0};
    splat_drender::vulkan::SplatCamera camera;
    splat_drender::vulkan::SplatGaussians gaussians;
    splat_drender::vulkan::SplatSettings settings;

    SingleGaussianScene() {
        camera.width = 32;
        camera.height = 32;
        camera.fx = 40;
        camera.fy = 40;
        camera.cx = 16;
        camera.cy = 16;
        camera.world_to_camera = identity;
        camera.center = origin;
        gaussians.means = means;
        gaussians.colors = colors;
        gaussians.opacities = opacity;
        gaussians.scales = scales;
        gaussians.rotations = rotation;
        settings.background[0] = 0.05F;
        settings.background[1] = 0.10F;
        settings.background[2] = 0.20F;
        settings.need_depth = true;
    }
};

// Single centered Gaussian: covers the Vulkan backend's own contract (finite
// channels, alpha over the background, coverage that stops at the 1/255 edge)
// without comparing against CUDA.
void smoke_case() {
    SingleGaussianScene scene;
    splat_drender::vulkan::Context context;
    splat_drender::vulkan::SplatRasterizer rasterizer(context);
    const auto output = rasterizer.forward(scene.gaussians, scene.camera, scene.settings);
    require(output.color.size() == 3u * 32u * 32u, "Splat forward returned the wrong color size");
    require(output.instance_count > 0 && output.radii[0] > 0, "Centered Gaussian was culled");
    require(std::ranges::all_of(output.color, [](float value) { return std::isfinite(value); }),
            "Splat color contains non-finite values");
    require(std::ranges::all_of(output.median_depth, [](float value) { return std::isfinite(value); }),
            "Splat depth contains non-finite values");
    const auto center = 16u * 32u + 16u;
    require(output.alpha[center] > 0.2F, "Centered Gaussian did not cover the image center");
    require(std::abs(output.color[center] - (output.alpha[center] + 0.05F * (1.0F - output.alpha[center]))) < 0.05F,
            "Centered Gaussian did not contribute red");
    require(output.alpha[0] < 0.05F, "Splat coverage reached the far corner");
    std::cout << "single-gaussian smoke: instances=" << output.instance_count
              << " center_alpha=" << output.alpha[center] << '\n';
}

// A Context may adopt a device the caller already owns. It then creates no
// instance and no device, keeps only its own pools, and has to produce exactly
// the same image as the Context that owns the device.
void adopted_device_case() {
    SingleGaussianScene scene;
    splat_drender::vulkan::Context owner;
    splat_drender::vulkan::SplatRasterizer owner_rasterizer(owner);
    const auto reference = owner_rasterizer.forward(scene.gaussians, scene.camera, scene.settings);

    splat_drender::vulkan::ContextOptions options;
    options.external_device.instance = owner.instance();
    options.external_device.physical_device = owner.physical_device();
    options.external_device.device = owner.device();
    options.external_device.queue = owner.queue();
    options.external_device.queue_family = owner.queue_family();
    splat_drender::vulkan::Context adopted(options);
    require(adopted.device() == owner.device(), "The adopted Context created a device of its own");
    require(adopted.instance() == owner.instance(), "The adopted Context lost the caller instance");
    splat_drender::vulkan::SplatRasterizer adopted_rasterizer(adopted);
    const auto output = adopted_rasterizer.forward(scene.gaussians, scene.camera, scene.settings);
    require(output.color == reference.color, "Adopting a device changed the color");
    require(output.alpha == reference.alpha, "Adopting a device changed the alpha");
    require(output.median_depth == reference.median_depth, "Adopting a device changed the depth");
    require(output.instance_count == reference.instance_count, "Adopting a device changed the instance count");

    // The queue family is required: vkGetDeviceQueue cannot recover the index of
    // a family the device was not created with.
    splat_drender::vulkan::ContextOptions missing_family;
    missing_family.external_device.physical_device = owner.physical_device();
    missing_family.external_device.device = owner.device();
    missing_family.external_device.queue = owner.queue();
    bool rejected = false;
    try {
        splat_drender::vulkan::Context invalid(missing_family);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Adopting a device without a compute queue family must fail");
    std::cout << "adopted-device smoke: instances=" << output.instance_count
              << " image matches owner\n";
}

} // namespace

int main() {
    try {
        check_cuda(cudaSetDevice(0), "cudaSetDevice");
        smoke_case();
        adopted_device_case();
        compare_case(pinhole_scene("pinhole-color", 24, 0.08F, true));

        auto sh = pinhole_scene("pinhole-sh", 12, 0.1F, true);
        sh.use_sh = true;
        sh.sh_degree = 3;
        sh.kernel_size = 0.3F;
        sh.colors.clear();
        Rng rng{7};
        fill_sh(sh, rng);
        compare_case(sh);

        auto covariance = pinhole_scene("pinhole-covariance", 10, 0.09F, true);
        covariance.use_covariance = true;
        covariance.scale_modifier = 1.25F;
        fill_covariance(covariance);
        compare_case(covariance);

        auto tied = pinhole_scene("equal-depth", 1, 0.1F, false);
        tied.means.insert(tied.means.end(), {tied.means[0], tied.means[1], tied.means[2]});
        tied.opacities.push_back(0.6F);
        tied.scales.insert(tied.scales.end(), {tied.scales[0], tied.scales[1], tied.scales[2]});
        tied.rotations.insert(tied.rotations.end(), {tied.rotations[0], tied.rotations[1], tied.rotations[2], tied.rotations[3]});
        tied.colors.insert(tied.colors.end(), {0.1F, 0.8F, 0.2F});
        compare_case(tied);

        auto fisheye = pinhole_scene("fisheye", 16, 0.07F, true);
        fisheye.mode = 1;
        fisheye.k1 = -0.08F;
        fisheye.k2 = 0.02F;
        fisheye.fx = 30;
        fisheye.fy = 30;
        compare_case(fisheye);

        Scene equirect;
        equirect.name = "equirect";
        equirect.width = 64;
        equirect.height = 32;
        equirect.mode = 3;
        equirect.fx = 64.0F / (2.0F * 3.14159265F);
        equirect.fy = equirect.fx;
        equirect.cx = 32;
        equirect.cy = 16;
        Rng equirect_rng{11};
        for (int i = 0; i < 18; ++i) {
            const float azimuth = equirect_rng.range(-3.05F, 3.05F);
            const float elevation = equirect_rng.range(-1.0F, 1.0F);
            const float radius = equirect_rng.range(1.0F, 2.5F);
            const float ce = std::cos(elevation);
            add_gaussian(equirect, equirect_rng, radius * ce * std::sin(azimuth), radius * std::sin(elevation),
                         radius * ce * std::cos(azimuth), 0.12F, true);
        }
        add_gaussian(equirect, equirect_rng, 0.05F, 0.0F, -1.4F, 0.45F, true);
        add_gaussian(equirect, equirect_rng, -0.08F, 0.1F, -1.3F, 0.45F, true);
        compare_case(equirect);

        Scene large = pinhole_scene("large-sort", 8000, 2.5F, false);
        large.expect_double_sort = true;
        large.width = 96;
        large.height = 64;
        large.fx = 70;
        large.fy = 70;
        large.cx = 48;
        large.cy = 32;
        compare_case(large);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
