#include <cuda_runtime.h>

#include <splat_drender/vulkan_api.h>
#include <splat_drender/api.h>
#include "../../../photara/src/splat/cuda_ops.hpp"
#include "../../../photara/src/splat/optimizer.hpp"
#include "../../../photara/src/splat/photometric_loss.hpp"
#include "internal/tensor_impl.hpp"
#include "vulkan/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
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
    bool check_backward = false;
    std::uint32_t sh_degree = 0;
    std::vector<float> means;
    std::vector<float> colors;
    std::vector<float> sh;
    std::vector<float> opacities;
    std::vector<float> scales;
    std::vector<float> rotations;
    std::vector<float> covariances;
    std::vector<float> log_scales;
    std::vector<float> raw_rotations;
    std::vector<float> opacity_logits;
    std::vector<float> filter_3d;
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

void enable_activation_chain(Scene& scene) {
    const std::size_t count = scene.means.size() / 3;
    scene.log_scales.resize(count * 3);
    scene.raw_rotations.resize(count * 4);
    scene.opacity_logits.resize(count);
    scene.filter_3d.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float filter = 0.012F + 0.003F * static_cast<float>(i % 4);
        scene.filter_3d[i] = filter;
        float determinant_ratio = 1.0F;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            const std::size_t offset = i * 3 + axis;
            const float raw_scale = scene.scales[offset];
            scene.log_scales[offset] = std::log(raw_scale);
            const float filtered = std::sqrt(raw_scale * raw_scale + filter * filter);
            scene.scales[offset] = filtered;
            determinant_ratio *= raw_scale / filtered;
        }
        const float base_opacity = scene.opacities[i];
        scene.opacity_logits[i] = std::log(base_opacity / (1.0F - base_opacity));
        scene.opacities[i] = base_opacity * determinant_ratio;
        const float quaternion_scale = 1.25F + 0.1F * static_cast<float>(i % 3);
        for (std::size_t component = 0; component < 4; ++component) {
            const std::size_t offset = i * 4 + component;
            scene.raw_rotations[offset] = scene.rotations[offset] * quaternion_scale;
        }
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
    gaussians.log_scales = scene.log_scales;
    gaussians.raw_rotations = scene.raw_rotations;
    gaussians.opacity_logits = scene.opacity_logits;
    gaussians.filter_3d = scene.filter_3d;
    splat_drender::vulkan::SplatSettings settings;
    settings.background[0] = background[0];
    settings.background[1] = background[1];
    settings.background[2] = background[2];
    settings.kernel_size = scene.kernel_size;
    settings.scale_modifier = scene.scale_modifier;
    settings.need_depth = scene.need_depth;
    settings.pixel_snapshots = scene.check_backward;

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

    if (scene.check_backward) {
        std::vector<float> loss_color(pixels * 3);
        std::vector<float> loss_alpha(pixels);
        std::vector<float> loss_depth;
        std::vector<float> loss_normal;
        for (std::size_t i = 0; i < loss_color.size(); ++i) {
            loss_color[i] = 0.3F * std::sin(static_cast<float>(i) * 0.013F) + 0.05F;
        }
        for (std::size_t i = 0; i < loss_alpha.size(); ++i) {
            loss_alpha[i] = 0.2F * std::cos(static_cast<float>(i) * 0.017F) - 0.03F;
        }
        if (scene.need_depth) {
            loss_depth.resize(pixels);
            loss_normal.resize(pixels * 3);
            // Photara enables depth/MV training for pinhole cameras only.
            // Equirect still exercises its normal and RGB geometry paths.
            for (std::size_t i = 0; i < pixels; ++i)
                loss_depth[i] = scene.mode == 3 ? 0.0F :
                    0.04F * std::sin(static_cast<float>(i) * 0.019F);
            for (std::size_t i = 0; i < loss_normal.size(); ++i)
                loss_normal[i] = 0.03F * std::cos(static_cast<float>(i) * 0.023F);
        }
        const auto vulkan_grad = rasterizer.backward(
            loss_color, loss_alpha, loss_depth, loss_normal);

        CudaBuffer loss_color_dev, loss_alpha_dev, loss_depth_dev, loss_normal_dev;
        CudaBuffer grad_means, grad_features, grad_opacity, grad_scales, grad_rotations, grad_covariance;
        splat_drender::ForwardOutputsView cuda_fwd_view;
        cuda_fwd_view.alpha = static_cast<const float*>(out_alpha.ptr);
        cuda_fwd_view.radii = static_cast<const int*>(out_radii.ptr);
        if (scene.need_depth) {
            cuda_fwd_view.median_depth = static_cast<const float*>(out_depth.ptr);
            cuda_fwd_view.normal = static_cast<const float*>(out_normal.ptr);
        }
        splat_drender::LossGradients cuda_loss;
        cuda_loss.color = loss_color_dev.upload(loss_color);
        cuda_loss.alpha = loss_alpha_dev.upload(loss_alpha);
        if (scene.need_depth) {
            cuda_loss.median_depth = loss_depth_dev.upload(loss_depth);
            cuda_loss.normal = loss_normal_dev.upload(loss_normal);
        }
        splat_drender::ModelGradients cuda_grad;
        cuda_grad.means = grad_means.zeros<float>(static_cast<std::size_t>(count) * 3);
        if (scene.use_sh) {
            cuda_grad.sh = grad_features.zeros<float>(scene.sh.size());
        } else {
            cuda_grad.colors = grad_features.zeros<float>(static_cast<std::size_t>(count) * 3);
        }
        cuda_grad.opacities = grad_opacity.zeros<float>(static_cast<std::size_t>(count));
        if (scene.use_covariance) {
            cuda_grad.covariances = grad_covariance.zeros<float>(static_cast<std::size_t>(count) * 6);
        } else {
            cuda_grad.scales = grad_scales.zeros<float>(static_cast<std::size_t>(count) * 3);
            cuda_grad.rotations = grad_rotations.zeros<float>(static_cast<std::size_t>(count) * 4);
        }
        splat_drender::Rasterizer::backward(
            pools, cuda_g, cuda_camera, cuda_settings, cuda_result,
            cuda_fwd_view, cuda_loss, cuda_grad);
        check_cuda(cudaGetLastError(), "cuda backward");
        check_cuda(cudaDeviceSynchronize(), "cuda backward synchronize");
        const auto cuda_mean_grad = grad_means.download<float>(static_cast<std::size_t>(count) * 3);
        const auto cuda_feature_grad = grad_features.download<float>(
            scene.use_sh ? scene.sh.size() : static_cast<std::size_t>(count) * 3);
        const auto cuda_opacity_grad = grad_opacity.download<float>(static_cast<std::size_t>(count));
        const auto& vulkan_feature_grad = scene.use_sh ? vulkan_grad.sh : vulkan_grad.colors;
        const double mean_grad_error = rel_l2(vulkan_grad.means, cuda_mean_grad);
        const double feature_grad_error = rel_l2(vulkan_feature_grad, cuda_feature_grad);
        const double opacity_grad_error = rel_l2(vulkan_grad.opacities, cuda_opacity_grad);
        std::cout << "  backward mean=" << mean_grad_error
                  << " feature=" << feature_grad_error
                  << " opacity=" << opacity_grad_error;
        double geometry_grad_error = 0.0;
        if (scene.use_covariance) {
            geometry_grad_error = rel_l2(
                vulkan_grad.covariances,
                grad_covariance.download<float>(static_cast<std::size_t>(count) * 6));
            std::cout << " covariance=" << geometry_grad_error;
        } else {
            const double scale_error = rel_l2(
                vulkan_grad.scales,
                grad_scales.download<float>(static_cast<std::size_t>(count) * 3));
            const double rotation_error = rel_l2(
                vulkan_grad.rotations,
                grad_rotations.download<float>(static_cast<std::size_t>(count) * 4));
            geometry_grad_error = std::max(scale_error, rotation_error);
            std::cout << " scale=" << scale_error << " rotation=" << rotation_error;
        }
        double activation_grad_error = 0.0;
        if (!scene.log_scales.empty()) {
            const auto cuda_scale_grad = grad_scales.download<float>(static_cast<std::size_t>(count) * 3);
            const auto cuda_rotation_grad = grad_rotations.download<float>(static_cast<std::size_t>(count) * 4);
            std::vector<float> expected_log_scale(static_cast<std::size_t>(count) * 3);
            std::vector<float> expected_raw_rotation(static_cast<std::size_t>(count) * 4);
            std::vector<float> expected_logit(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i) {
                const float filter2 = scene.filter_3d[static_cast<std::size_t>(i)] *
                                      scene.filter_3d[static_cast<std::size_t>(i)];
                for (int axis = 0; axis < 3; ++axis) {
                    const std::size_t offset = static_cast<std::size_t>(i) * 3 + axis;
                    const float raw_scale = std::exp(scene.log_scales[offset]);
                    const float filtered_scale = scene.scales[offset];
                    expected_log_scale[offset] =
                        cuda_scale_grad[offset] * raw_scale * raw_scale / filtered_scale +
                        cuda_opacity_grad[static_cast<std::size_t>(i)] *
                            scene.opacities[static_cast<std::size_t>(i)] * filter2 /
                            (filtered_scale * filtered_scale);
                }
                const std::size_t qbase = static_cast<std::size_t>(i) * 4;
                float norm2 = 0.0F;
                for (int q = 0; q < 4; ++q) {
                    const float raw = scene.raw_rotations[qbase + q];
                    norm2 += raw * raw;
                }
                const float inv_norm = 1.0F / std::sqrt(std::max(norm2, 1e-20F));
                float product = 0.0F;
                for (int q = 0; q < 4; ++q) {
                    product += scene.raw_rotations[qbase + q] * inv_norm *
                               cuda_rotation_grad[qbase + q];
                }
                for (int q = 0; q < 4; ++q) {
                    const float normalized = scene.raw_rotations[qbase + q] * inv_norm;
                    expected_raw_rotation[qbase + q] =
                        inv_norm * (cuda_rotation_grad[qbase + q] - normalized * product);
                }
                const float logit = scene.opacity_logits[static_cast<std::size_t>(i)];
                const float opacity = 1.0F / (1.0F + std::exp(-logit));
                expected_logit[static_cast<std::size_t>(i)] =
                    cuda_opacity_grad[static_cast<std::size_t>(i)] *
                    scene.opacities[static_cast<std::size_t>(i)] * (1.0F - opacity);
            }
            const double log_scale_error = rel_l2(vulkan_grad.log_scales, expected_log_scale);
            const double raw_rotation_error = rel_l2(vulkan_grad.raw_rotations, expected_raw_rotation);
            const double logit_error = rel_l2(vulkan_grad.opacity_logits, expected_logit);
            activation_grad_error = std::max({log_scale_error, raw_rotation_error, logit_error});
            std::cout << " log_scale=" << log_scale_error
                      << " raw_rotation=" << raw_rotation_error
                      << " logit=" << logit_error;
        }
        std::cout << '\n';
        require(mean_grad_error < 5e-4 && feature_grad_error < 5e-4 &&
                    opacity_grad_error < 5e-4 && geometry_grad_error < 5e-4 &&
                    activation_grad_error < 5e-4,
                scene.name + " full backward exceeds 5e-4 relative L2");
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

    // Snapshot generation is the forward half of the Vulkan backward pass.
    // It must preserve the normal render while building its bucket
    // prefix and per-pixel transmittance state entirely on the GPU.
    scene.settings.pixel_snapshots = true;
    const auto snapshot_output = rasterizer.forward(scene.gaussians, scene.camera, scene.settings);
    require(snapshot_output.color == output.color, "Enabling snapshots changed the forward color");
    require(snapshot_output.alpha == output.alpha, "Enabling snapshots changed the forward alpha");
    require(snapshot_output.median_depth == output.median_depth, "Enabling snapshots changed the forward depth");
    require(snapshot_output.instance_count == output.instance_count, "Enabling snapshots changed the instance count");
    std::cout << "single-gaussian smoke: instances=" << output.instance_count
              << " center_alpha=" << output.alpha[center] << " snapshots=ok\n";
}

void sample_depth_case() {
    SingleGaussianScene scene;
    scene.settings.point_depth_bracket = 0.8F;
    scene.settings.point_depth_tolerance = 1.0e-5F;
    const std::vector<float> points{0.0F, 0.0F, 2.0F, 0.015F, -0.01F, 2.0F};
    splat_drender::vulkan::Context context;
    splat_drender::vulkan::SplatRasterizer vulkan(context);
    vulkan.upload_model(scene.gaussians);
    const auto got = vulkan.sample_depth(points, scene.camera, scene.settings);

    CudaBuffer means, colors, opacity, scales, rotations, view, center, point_dev;
    CudaBuffer out_points, out_depth, out_contrib, out_inside;
    CudaBuffer pool_g, pool_grad, pool_i, pool_p, pool_t, pool_point;
    splat_drender::Gaussians g;
    g.count = 1;
    g.means = means.upload(scene.means); g.colors = colors.upload(scene.colors);
    g.opacities = opacity.upload(scene.opacity); g.scales = scales.upload(scene.scales);
    g.rotations = rotations.upload(scene.rotation);
    splat_drender::CameraView camera;
    camera.width = static_cast<int>(scene.camera.width); camera.height = static_cast<int>(scene.camera.height);
    camera.fx = scene.camera.fx; camera.fy = scene.camera.fy; camera.cx = scene.camera.cx; camera.cy = scene.camera.cy;
    camera.world_to_camera = view.upload(scene.identity); camera.center = center.upload(scene.origin);
    splat_drender::RenderSettings settings;
    settings.need_depth = true; settings.point_depth_bracket = 0.8F;
    settings.point_depth_tolerance = 1.0e-5F;
    splat_drender::WorkspacePools pools;
    pools.gaussian=[&](std::size_t n){return static_cast<char*>(pool_g.ensure(n));};
    pools.grad=[&](std::size_t n){return static_cast<char*>(pool_grad.ensure(n));};
    pools.instance=[&](std::size_t n){return static_cast<char*>(pool_i.ensure(n));};
    pools.pixel=[&](std::size_t n){return static_cast<char*>(pool_p.ensure(n));};
    pools.tile=[&](std::size_t n){return static_cast<char*>(pool_t.ensure(n));};
    pools.point=[&](std::size_t n){return static_cast<char*>(pool_point.ensure(n));};
    splat_drender::SampleOutputs out;
    out.ray_points = out_points.zeros<float3>(2); out.median_depth = out_depth.zeros<float>(2);
    out.n_contrib = out_contrib.zeros<unsigned>(2); out.inside = out_inside.zeros<bool>(2);
    const auto counts = splat_drender::Rasterizer::sample_depth(
        pools, g, camera, settings, point_dev.upload(points), 2, out);
    check_cuda(cudaDeviceSynchronize(), "sample depth synchronize");
    const auto expected_points = out_points.download<float>(6);
    const auto expected_depth = out_depth.download<float>(2);
    require(rel_l2(got.camera_points, expected_points) < 2e-4,
            "Vulkan sample_depth camera points differ from CUDA");
    require(rel_l2(got.median_depth, expected_depth) < 2e-4,
            "Vulkan sample_depth median differs from CUDA");

    const std::vector<float> upstream{0.2F,-0.1F,0.3F,-0.15F,0.25F,0.1F};
    const auto got_grad = vulkan.sample_depth_backward(upstream);
    CudaBuffer upstream_dev, grad_points, grad_means, grad_colors, grad_opacity, grad_scales, grad_rotations;
    splat_drender::SampleOutputsView fwd;
    fwd.median_depth = static_cast<const float*>(out_depth.ptr);
    fwd.n_contrib = static_cast<const unsigned*>(out_contrib.ptr);
    fwd.inside = static_cast<const bool*>(out_inside.ptr);
    splat_drender::SampleGradients point_grads;
    point_grads.points = grad_points.zeros<float3>(2);
    splat_drender::ModelGradients model_grads;
    model_grads.means=grad_means.zeros<float>(3); model_grads.colors=grad_colors.zeros<float>(3);
    model_grads.opacities=grad_opacity.zeros<float>(1); model_grads.scales=grad_scales.zeros<float>(3);
    model_grads.rotations=grad_rotations.zeros<float>(4);
    splat_drender::Rasterizer::sample_depth_backward(
        pools,g,camera,settings,static_cast<const float*>(point_dev.ptr),2,counts,fwd,
        reinterpret_cast<const float3*>(upstream_dev.upload(upstream)),point_grads,model_grads);
    check_cuda(cudaDeviceSynchronize(), "sample depth backward synchronize");
    require(rel_l2(got_grad.points, grad_points.download<float>(6)) < 2e-4,
            "Vulkan sample_depth point gradients differ from CUDA");
    require(rel_l2(got_grad.model.means, grad_means.download<float>(3)) < 2e-3,
            "Vulkan sample_depth mean gradients differ from CUDA");
    require(rel_l2(got_grad.model.opacities, grad_opacity.download<float>(1)) < 2e-3,
            "Vulkan sample_depth opacity gradients differ from CUDA");
    require(rel_l2(got_grad.model.scales, grad_scales.download<float>(3)) < 2e-3,
            "Vulkan sample_depth scale gradients differ from CUDA");
    require(rel_l2(got_grad.model.rotations, grad_rotations.download<float>(4)) < 2e-3,
            "Vulkan sample_depth rotation gradients differ from CUDA");
    std::cout << "sample-depth forward/backward: depth=" << got.median_depth[0]
              << " mean_grad_rel_l2=" << rel_l2(got_grad.model.means, grad_means.download<float>(3)) << '\n';
}

void multi_view_loss_case() {
    using photara::splat::Camera;
    using photara::splat::RenderResult;
    using photara::splat::TrainingOptions;
    using photara::splat::TrainingView;
    namespace detail = photara::splat::detail;
    constexpr std::uint32_t width = 32;
    constexpr std::uint32_t height = 32;
    constexpr std::size_t pixels = width * height;
    const auto make_camera = [](float center_x) {
        Camera camera;
        camera.world_to_camera[0] = 1.0F;
        camera.world_to_camera[5] = 1.0F;
        camera.world_to_camera[10] = 1.0F;
        camera.world_to_camera[15] = 1.0F;
        camera.world_to_camera[12] = -center_x;
        camera.position[0] = center_x;
        camera.fx = camera.fy = 40.0F;
        camera.cx = camera.cy = 15.5F;
        camera.width = width;
        camera.height = height;
        return camera;
    };
    TrainingView reference;
    TrainingView neighbour;
    reference.camera = make_camera(0.0F);
    neighbour.camera = make_camera(0.1F);
    std::vector<float> reference_gray(pixels);
    std::vector<float> neighbour_gray(pixels);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            reference_gray[pixel] = 0.5F + 0.22F * std::sin(0.37F * x) +
                                    0.18F * std::cos(0.29F * y);
            neighbour_gray[pixel] = 0.5F +
                0.22F * std::sin(0.37F * (static_cast<float>(x) + 2.0F)) +
                0.18F * std::cos(0.29F * y);
        }
    }
    // Keep the sampled neighbour surface at z=2 but perturb the reference
    // depth so the NCC derivative is well-conditioned rather than ~0 at the
    // exact optimum.
    const std::vector<float> depths(pixels, 2.08F);
    std::vector<float> normals(3 * pixels, 0.0F);
    std::fill(normals.begin() + 2 * pixels, normals.end(), 1.0F);
    std::vector<float> sampled_points(3 * pixels);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t pixel = static_cast<std::size_t>(y) * width + x;
            sampled_points[3 * pixel] =
                (static_cast<float>(x) - 15.5F) / 40.0F * 2.0F - 0.1F;
            sampled_points[3 * pixel + 1] =
                (static_cast<float>(y) - 15.5F) / 40.0F * 2.0F;
            sampled_points[3 * pixel + 2] = 2.0F;
        }
    }

    reference.gray = tinytensor::Tensor::from_vector(
        reference_gray, {height, width}, tinytensor::Device::CUDA);
    neighbour.gray = tinytensor::Tensor::from_vector(
        neighbour_gray, {height, width}, tinytensor::Device::CUDA);
    RenderResult reference_render;
    reference_render.median_depth = tinytensor::Tensor::from_vector(
        depths, {height, width}, tinytensor::Device::CUDA);
    reference_render.normal = tinytensor::Tensor::from_vector(
        normals, {3, height, width}, tinytensor::Device::CUDA);
    const auto cuda_sampled = tinytensor::Tensor::from_vector(
        sampled_points, {pixels, std::size_t{3}}, tinytensor::Device::CUDA);
    const auto cuda_inside = tinytensor::Tensor::ones_bool(
        {pixels}, tinytensor::Device::CUDA);
    detail::LossGradients cuda_gradients;
    cuda_gradients.depth = tinytensor::Tensor::zeros(
        {height, width}, tinytensor::Device::CUDA);
    cuda_gradients.normal = tinytensor::Tensor::zeros(
        {3, height, width}, tinytensor::Device::CUDA);
    TrainingOptions options;
    options.multi_view_geo_weight = 0.02F;
    options.multi_view_ncc_weight = 0.6F;
    tinytensor::Tensor cuda_sampled_gradient;
    const auto cuda_loss = detail::add_multi_view_loss(
        cuda_sampled, cuda_inside, reference_render, reference, neighbour,
        options, cuda_gradients, cuda_sampled_gradient, true);

    const auto make_vk_camera = [](const Camera& camera) {
        splat_drender::vulkan::SplatCamera result;
        result.width = camera.width;
        result.height = camera.height;
        result.fx = camera.fx; result.fy = camera.fy;
        result.cx = camera.cx; result.cy = camera.cy;
        result.world_to_camera = camera.world_to_camera;
        result.center = camera.position;
        return result;
    };
    std::vector<std::uint32_t> inside(pixels, 1u);
    splat_drender::vulkan::SplatMultiViewInput input;
    input.reference_camera = make_vk_camera(reference.camera);
    input.neighbour_camera = make_vk_camera(neighbour.camera);
    input.reference_depth = depths;
    input.reference_normal = normals;
    input.reference_gray = reference_gray;
    input.neighbour_gray = neighbour_gray;
    input.sampled_neighbour_points = sampled_points;
    input.sampled_inside = inside;
    input.geometry_weight = options.multi_view_geo_weight;
    input.ncc_weight = options.multi_view_ncc_weight;
    input.pixel_noise_threshold = options.multi_view_pixel_noise_threshold;
    input.robust_ncc = options.multi_view_robust_ncc;
    input.ncc_lambda_reference = options.multi_view_ncc_lambda_reference;
    input.ncc_sharpness = options.multi_view_ncc_sharpness;
    input.ncc_min_weight = options.multi_view_ncc_min_weight;
    splat_drender::vulkan::Context context;
    splat_drender::vulkan::SplatRasterizer vulkan(context);
    const auto got = vulkan.multi_view_loss(input);

    require(got.geometry_pixels == cuda_loss.geometry_pixels &&
                got.geometry_candidates == cuda_loss.geometry_candidates &&
                got.ncc_pixels == cuda_loss.ncc_pixels,
            "Vulkan multi-view accepted-pixel counts differ from CUDA");
    require(std::abs(got.geometry - cuda_loss.geometry) < 2.0e-5F,
            "Vulkan multi-view geometry loss differs from CUDA");
    require(std::abs(got.ncc - cuda_loss.ncc) < 2.0e-4F,
            "Vulkan multi-view NCC loss differs from CUDA");
    const auto cuda_depth_gradient = cuda_gradients.depth.to_vector();
    const auto cuda_normal_gradient = cuda_gradients.normal.to_vector();
    const auto cuda_point_gradient = cuda_sampled_gradient.to_vector();
    const double depth_gradient_error =
        rel_l2(got.reference_depth_gradient, cuda_depth_gradient);
    const double normal_gradient_error =
        rel_l2(got.reference_normal_gradient, cuda_normal_gradient);
    const double point_gradient_error =
        rel_l2(got.sampled_point_gradient, cuda_point_gradient);
    const auto norm = [](const std::vector<float>& values) {
        double sum = 0.0;
        for (float value : values) sum += static_cast<double>(value) * value;
        return std::sqrt(sum);
    };
    std::cout << "multi-view raw parity: counts=" << got.geometry_pixels
              << '/' << got.geometry_candidates << '/' << got.ncc_pixels
              << " cuda=" << cuda_loss.geometry_pixels << '/'
              << cuda_loss.geometry_candidates << '/' << cuda_loss.ncc_pixels
              << " grad_rel_l2=" << depth_gradient_error << '/'
              << normal_gradient_error << '/' << point_gradient_error
              << " norms=" << norm(got.reference_depth_gradient) << '/'
              << norm(cuda_depth_gradient) << ','
              << norm(got.reference_normal_gradient) << '/'
              << norm(cuda_normal_gradient) << '\n';
    require(depth_gradient_error < 2.0e-3,
            "Vulkan multi-view depth gradient differs from CUDA");
    require(normal_gradient_error < 3.0e-3,
            "Vulkan multi-view normal gradient differs from CUDA");
    require(point_gradient_error < 2.0e-3,
            "Vulkan multi-view point gradient differs from CUDA");
    std::cout << "multi-view geometry/NCC: geo=" << got.geometry
              << " ncc=" << got.ncc
              << " depth_grad_rel_l2="
              << depth_gradient_error
              << '\n';
    const std::vector<float> empty_reference_mask(pixels, 0.0F);
    const std::vector<float> full_neighbour_mask(pixels, 1.0F);
    input.reference_mask = empty_reference_mask;
    input.neighbour_mask = full_neighbour_mask;
    const auto masked = vulkan.multi_view_loss(input);
    require(masked.geometry_pixels == 0 && masked.geometry_candidates == 0 &&
                masked.ncc_pixels == 0,
            "Vulkan multi-view loss ignored the foreground masks");
}

void ssim_loss_case() {
    constexpr std::uint32_t width = 37;
    constexpr std::uint32_t height = 29;
    constexpr std::size_t pixels = width * height;
    Rng random{0x5a17c9e3u};
    std::vector<float> prediction(3 * pixels);
    std::vector<float> target(3 * pixels);
    std::vector<float> mask(pixels, 1.0F);
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        prediction[i] = random.next();
        target[i] = random.next();
    }
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
            if ((x + 2 * y) % 11 == 0) mask[static_cast<std::size_t>(y) * width + x] = 0.0F;
    const auto cuda_prediction = tinytensor::Tensor::from_vector(
        prediction, {3U, height, width}, tinytensor::Device::CUDA);
    const auto cuda_target = tinytensor::Tensor::from_vector(
        target, {3U, height, width}, tinytensor::Device::CUDA);
    const auto cuda_mask = tinytensor::Tensor::from_vector(
        mask, {height, width}, tinytensor::Device::CUDA);
    for (int masked = 0; masked < 2; ++masked) {
        auto cuda_gradient = tinytensor::Tensor::zeros(
            {3U, height, width}, tinytensor::Device::CUDA);
        photara::splat::detail::photometric_loss(
            cuda_prediction, cuda_target, cuda_mask, masked != 0,
            cuda_gradient, nullptr, width, height);
        splat_drender::vulkan::Context context;
        splat_drender::vulkan::SplatRasterizer vulkan(context);
        const auto got = vulkan.photometric_loss(
            prediction, target, width, height, 0.2F, 1.0F,
            masked != 0 ? std::span<const float>(mask) : std::span<const float>{});
        const double error = rel_l2(got.gradient, cuda_gradient.to_vector());
        std::cout << "SSIM gradient parity mask=" << masked
                  << " rel_l2=" << error << " loss=" << got.loss << '\n';
        require(error < 3.0e-4,
                "Vulkan fused L1+SSIM gradient differs from CUDA");
        require(std::isfinite(got.loss) && got.loss >= 0.0F,
                "Vulkan fused L1+SSIM returned an invalid loss");
    }
}

void geometry_gradient_descent_case() {
    SingleGaussianScene scene;
    scene.settings.need_depth = true;
    scene.settings.pixel_snapshots = true;
    splat_drender::vulkan::Context context;
    splat_drender::vulkan::SplatRasterizer rasterizer(context);
    const auto target = rasterizer.forward(
        scene.gaussians, scene.camera, scene.settings);
    std::vector<float> means{0.0F, 0.0F, 2.35F};
    splat_drender::vulkan::SplatGaussians trainable = scene.gaussians;
    trainable.means = means;
    const std::size_t pixels =
        static_cast<std::size_t>(scene.camera.width) * scene.camera.height;
    const std::vector<float> zero_color(3 * pixels, 0.0F);
    const std::vector<float> zero_alpha(pixels, 0.0F);
    const std::vector<float> zero_normal(3 * pixels, 0.0F);
    float first_loss = -1.0F;
    float previous_loss = std::numeric_limits<float>::infinity();
    float final_loss = previous_loss;
    for (int iteration = 0; iteration < 8; ++iteration) {
        const auto rendered = rasterizer.forward(
            trainable, scene.camera, scene.settings);
        std::vector<float> depth_gradient(pixels, 0.0F);
        float loss = 0.0F;
        std::size_t valid = 0;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            if (!(target.median_depth[pixel] > 0.0F) ||
                !(rendered.median_depth[pixel] > 0.0F))
                continue;
            const float difference =
                rendered.median_depth[pixel] - target.median_depth[pixel];
            loss += 0.5F * difference * difference;
            depth_gradient[pixel] = difference;
            ++valid;
        }
        require(valid != 0, "Vulkan geometry descent found no valid depth pixels");
        loss /= static_cast<float>(valid);
        for (float& gradient : depth_gradient)
            gradient /= static_cast<float>(valid);
        if (iteration == 0) first_loss = loss;
        require(loss <= previous_loss * 1.001F,
                "Vulkan geometry gradient descent increased the depth loss");
        previous_loss = loss;
        final_loss = loss;
        const auto gradients = rasterizer.backward(
            zero_color, zero_alpha, depth_gradient, zero_normal);
        require(gradients.means.size() == 3 &&
                    std::isfinite(gradients.means[2]),
                "Vulkan geometry descent produced a non-finite mean gradient");
        means[2] -= 0.5F * gradients.means[2];
        trainable.means = means;
    }
    require(final_loss < first_loss * 0.02F,
            "Vulkan geometry gradients did not minimize the depth objective");
    require(std::abs(means[2] - 2.0F) < 0.06F,
            "Vulkan geometry descent did not recover the target surface");
    std::cout << "geometry gradient descent: loss=" << first_loss << " -> "
              << final_loss << " z=" << means[2] << '\n';
}

// Adam is the last stage of the training chain.  A forward/backward match is
// insufficient if bias correction, grouped SH learning rates, inactive SH
// bands, or parameter clamps drift after repeated updates.  Exercise the same
// public optimizer dispatch on CUDA and Vulkan and compare the complete
// parameter/moment trajectory after several non-identical gradients.
void optimizer_parity_case() {
    namespace detail = photara::splat::detail;
    using tinytensor::Device;
    using tinytensor::Tensor;

    photara::splat::TrainingOptions options;
    options.beta1 = 0.9F;
    options.beta2 = 0.999F;
    options.adam_epsilon = 1.0e-15F;

    constexpr std::size_t rows = 5;
    constexpr std::size_t stride = 12;
    std::vector<float> initial(rows * stride);
    std::vector<float> gradient(rows * stride);
    for (std::size_t i = 0; i < initial.size(); ++i) {
        initial[i] = -0.35F + 0.013F * static_cast<float>(i);
        gradient[i] = 0.025F * std::sin(0.37F * static_cast<float>(i + 1));
    }
    auto cuda_parameter = Tensor::from_vector(
        initial, {rows, stride}, Device::CUDA);
    auto vulkan_parameter = Tensor::from_vector(
        initial, {rows, stride}, Device::Vulkan);
    auto cuda_state = detail::make_adam_state(cuda_parameter);
    auto vulkan_state = detail::make_adam_state(vulkan_parameter);

    for (unsigned step = 1; step <= 9; ++step) {
        std::vector<float> step_gradient = gradient;
        for (std::size_t i = 0; i < step_gradient.size(); ++i)
            step_gradient[i] *= 0.7F + 0.03F * static_cast<float>(step) +
                0.01F * static_cast<float>(i % 5);
        const auto cuda_gradient = Tensor::from_vector(
            step_gradient, {rows, stride}, Device::CUDA);
        const auto vulkan_gradient = Tensor::from_vector(
            step_gradient, {rows, stride}, Device::Vulkan);
        // Three RGB DC values use the primary rate; the rest of each SH row
        // uses the secondary rate.  Keep the last basis inactive.
        detail::adam_step_active_prefix(
            cuda_parameter, cuda_gradient, cuda_state, 2.5e-3F, step,
            options, stride, 9, 1.25e-4F);
        detail::adam_step_active_prefix(
            vulkan_parameter, vulkan_gradient, vulkan_state, 2.5e-3F, step,
            options, stride, 9, 1.25e-4F);
    }

    const double parameter_error = rel_l2(
        vulkan_parameter.to_vector(), cuda_parameter.to_vector());
    const double first_error = rel_l2(
        vulkan_state.first.to_vector(), cuda_state.first.to_vector());
    const double second_error = rel_l2(
        vulkan_state.second.to_vector(), cuda_state.second.to_vector());
    std::cout << "optimizer grouped trajectory: parameter=" << parameter_error
              << " moments=" << first_error << '/' << second_error << '\n';
    // The second moment is around 1e-6 here, so a few last-bit differences in
    // fused multiply/add order produce a larger relative error than they do in
    // either the parameter or first moment.
    require(parameter_error < 3.0e-6 && first_error < 3.0e-6 &&
                second_error < 2.0e-5,
            "Vulkan grouped/active-prefix Adam trajectory differs from CUDA");

    // Structure parameters cover independent learning rates and the scale and
    // opacity clamps used by real training.
    const std::vector<float> means{
        -0.2F, 0.1F, 1.8F, 0.3F, -0.15F, 2.2F,
        0.05F, 0.25F, 2.7F, -0.4F, -0.2F, 3.1F,
        0.45F, 0.05F, 1.5F};
    const std::vector<float> scales{
        -2.0F, -2.1F, -2.2F, -1.8F, -1.9F, -2.0F,
        -2.4F, -2.3F, -2.2F, -1.6F, -1.7F, -1.8F,
        -2.05F, -2.0F, -1.95F};
    const std::vector<float> rotations{
        1.F, 0.1F, -0.05F, 0.02F, 0.9F, -0.15F, 0.08F, 0.03F,
        1.1F, 0.04F, 0.02F, -0.09F, 0.95F, 0.12F, -0.07F, 0.05F,
        1.05F, -0.03F, 0.06F, 0.11F};
    const std::vector<float> opacity{-1.8F, -2.1F, -1.5F, -2.4F, -1.9F};
    const auto make_model = [&](Device device) {
        photara::splat::GaussianModel model;
        model.means = Tensor::from_vector(means, {rows, 3}, device);
        model.log_scales = Tensor::from_vector(scales, {rows, 3}, device);
        model.quaternions = Tensor::from_vector(rotations, {rows, 4}, device);
        model.opacity_logits = Tensor::from_vector(opacity, {rows, 1}, device);
        return model;
    };
    auto cuda_model = make_model(Device::CUDA);
    auto vulkan_model = make_model(Device::Vulkan);
    auto cuda_means_state = detail::make_adam_state(cuda_model.means);
    auto cuda_scales_state = detail::make_adam_state(cuda_model.log_scales);
    auto cuda_rotations_state = detail::make_adam_state(cuda_model.quaternions);
    auto cuda_opacity_state = detail::make_adam_state(cuda_model.opacity_logits);
    auto vk_means_state = detail::make_adam_state(vulkan_model.means);
    auto vk_scales_state = detail::make_adam_state(vulkan_model.log_scales);
    auto vk_rotations_state = detail::make_adam_state(vulkan_model.quaternions);
    auto vk_opacity_state = detail::make_adam_state(vulkan_model.opacity_logits);
    for (unsigned step = 1; step <= 7; ++step) {
        const auto make_gradient = [&](Device device) {
            photara::splat::ModelGradients result;
            const float factor = 0.4F + 0.09F * static_cast<float>(step);
            std::vector<float> gm(means.size()), gs(scales.size());
            std::vector<float> gr(rotations.size()), go(opacity.size());
            for (std::size_t i = 0; i < gm.size(); ++i) {
                gm[i] = factor * 0.03F * std::cos(0.23F * static_cast<float>(i + step));
                gs[i] = factor * 0.04F * std::sin(0.19F * static_cast<float>(i + 2 * step));
            }
            for (std::size_t i = 0; i < gr.size(); ++i)
                gr[i] = factor * 0.02F * std::cos(0.17F * static_cast<float>(i + step));
            for (std::size_t i = 0; i < go.size(); ++i)
                go[i] = factor * 0.05F * std::sin(0.31F * static_cast<float>(i + step));
            result.means = Tensor::from_vector(gm, {rows, 3}, device);
            result.log_scales = Tensor::from_vector(gs, {rows, 3}, device);
            result.quaternions = Tensor::from_vector(gr, {rows, 4}, device);
            result.opacity_logits = Tensor::from_vector(go, {rows, 1}, device);
            return result;
        };
        const auto cuda_gradient = make_gradient(Device::CUDA);
        const auto vulkan_gradient = make_gradient(Device::Vulkan);
        detail::adam_step_structure(
            cuda_model, cuda_gradient, cuda_means_state, cuda_scales_state,
            cuda_rotations_state, cuda_opacity_state, 1.6e-4F, step,
            options, -2.35F, -1.65F);
        detail::adam_step_structure(
            vulkan_model, vulkan_gradient, vk_means_state, vk_scales_state,
            vk_rotations_state, vk_opacity_state, 1.6e-4F, step,
            options, -2.35F, -1.65F);
    }
    const double means_error = rel_l2(
        vulkan_model.means.to_vector(), cuda_model.means.to_vector());
    const double scales_error = rel_l2(
        vulkan_model.log_scales.to_vector(), cuda_model.log_scales.to_vector());
    const double rotations_error = rel_l2(
        vulkan_model.quaternions.to_vector(), cuda_model.quaternions.to_vector());
    const double opacity_error = rel_l2(
        vulkan_model.opacity_logits.to_vector(),
        cuda_model.opacity_logits.to_vector());
    require(means_error < 3.0e-6 && scales_error < 3.0e-6 &&
                rotations_error < 3.0e-6 && opacity_error < 3.0e-6,
            "Vulkan structure Adam trajectory differs from CUDA");
    std::cout << "optimizer trajectory parity: grouped=" << parameter_error
              << " moments=" << first_error << '/' << second_error
              << " structure=" << means_error << '/' << scales_error << '/'
              << rotations_error << '/' << opacity_error << '\n';
}

// TinyTensor owns the device and all loss/gradient storage; splat_drender
// adopts that device and binds the tensor buffers directly. This is the path
// training will use, and catches accidental host staging or cross-device use.
void tinytensor_interop_case() {
    require(tinytensor::vulkan::available(), "TinyTensor Vulkan backend is unavailable");
    SingleGaussianScene scene;
    scene.settings.need_depth = false;
    scene.settings.pixel_snapshots = true;
    const auto handles = tinytensor::vulkan::device_handles();
    splat_drender::vulkan::ContextOptions options;
    options.external_device.instance = handles.instance;
    options.external_device.physical_device = handles.physical_device;
    options.external_device.device = handles.device;
    options.external_device.queue = handles.queue;
    options.external_device.queue_family = handles.queue_family;
    splat_drender::vulkan::Context context(options);
    splat_drender::vulkan::SplatRasterizer rasterizer(context);
    const auto forward = rasterizer.forward(scene.gaussians, scene.camera, scene.settings);
    const std::size_t pixels = static_cast<std::size_t>(scene.camera.width) * scene.camera.height;
    std::vector<float> color_loss(pixels * 3, 0.0F);
    std::vector<float> alpha_loss(pixels, 0.0F);
    for (std::size_t i = 0; i < pixels; ++i) {
        color_loss[i] = 0.1F;
        color_loss[pixels + i] = -0.2F;
        color_loss[2 * pixels + i] = 0.05F;
        alpha_loss[i] = 0.03F;
    }
    const auto expected = rasterizer.backward_blend(color_loss, alpha_loss);
    const auto expected_model = rasterizer.backward(color_loss, alpha_loss);
    auto device_color = tinytensor::Tensor::from_vector(
        color_loss, {color_loss.size()}, tinytensor::Device::Vulkan);
    auto device_alpha = tinytensor::Tensor::from_vector(
        alpha_loss, {alpha_loss.size()}, tinytensor::Device::Vulkan);
    auto device_gradient = tinytensor::Tensor::zeros(
        {static_cast<std::size_t>(rasterizer.blend_gradient_float_count())},
        tinytensor::Device::Vulkan, tinytensor::DataType::Float32);
    tinytensor::vulkan::synchronize();
    const auto color_view = tinytensor::vulkan::buffer_view(device_color);
    const auto alpha_view = tinytensor::vulkan::buffer_view(device_alpha);
    const auto gradient_view = tinytensor::vulkan::buffer_view(device_gradient);
    rasterizer.backward_blend_device(
        {color_view.buffer, color_view.offset, color_view.bytes},
        {alpha_view.buffer, alpha_view.offset, alpha_view.bytes},
        {gradient_view.buffer, gradient_view.offset, gradient_view.bytes});
    const auto got = device_gradient.to_vector();
    std::vector<float> reference;
    reference.reserve(expected.mean2d.size() + expected.conic_opacity.size() + expected.colors.size());
    reference.insert(reference.end(), expected.mean2d.begin(), expected.mean2d.end());
    reference.insert(reference.end(), expected.conic_opacity.begin(), expected.conic_opacity.end());
    reference.insert(reference.end(), expected.colors.begin(), expected.colors.end());
    reference.insert(reference.end(), expected.ray_plane.begin(), expected.ray_plane.end());
    reference.insert(reference.end(), expected.normal.begin(), expected.normal.end());
    reference.insert(reference.end(), expected_model.refine_weight.begin(),
                     expected_model.refine_weight.end());
    require(rel_l2(got, reference) < 1e-6, "TinyTensor zero-copy blend gradients differ");
    auto device_model_gradient = tinytensor::Tensor::zeros(
        {static_cast<std::size_t>(rasterizer.model_gradient_float_count())},
        tinytensor::Device::Vulkan, tinytensor::DataType::Float32);
    // Flush TinyTensor's initialization before another owner records writes
    // against the shared queue and buffer.
    tinytensor::vulkan::synchronize();
    const auto model_gradient_view = tinytensor::vulkan::buffer_view(device_model_gradient);
    rasterizer.backward_device(
        {color_view.buffer, color_view.offset, color_view.bytes},
        {alpha_view.buffer, alpha_view.offset, alpha_view.bytes},
        {model_gradient_view.buffer, model_gradient_view.offset, model_gradient_view.bytes});
    const auto got_model = device_model_gradient.to_vector();
    std::vector<float> model_reference;
    const auto append = [&](const std::vector<float>& values) {
        model_reference.insert(model_reference.end(), values.begin(), values.end());
    };
    append(expected_model.means);
    append(expected_model.colors);
    append(expected_model.opacities);
    append(expected_model.scales);
    append(expected_model.rotations);
    model_reference.insert(model_reference.end(), 6, 0.0F);
    model_reference.insert(model_reference.end(), 8, 0.0F);
    append(expected_model.refine_weight);
    require(got_model.size() == model_reference.size(), "TinyTensor model gradient layout differs");
    require(rel_l2(got_model, model_reference) < 1e-6,
            "TinyTensor zero-copy model gradients differ");

    // Full training chain: render color -> fused L1+SSIM -> raster backward.
    // Only the final packed model gradient and one logging scalar cross to the
    // host; the image-sized photometric gradient is never downloaded.
    std::vector<float> photo_target = forward.color;
    for (std::size_t i = 0; i < photo_target.size(); ++i)
        photo_target[i] = std::clamp(
            photo_target[i] + 0.03F * std::sin(static_cast<float>(i) * 0.17F),
            0.0F, 1.0F);
    const auto expected_photo = rasterizer.photometric_loss(
        forward.color, photo_target, scene.camera.width, scene.camera.height);
    const std::vector<float> zero_alpha(pixels, 0.0F);
    const auto expected_photo_model = rasterizer.backward(
        expected_photo.gradient, zero_alpha);

    const auto device_frame = rasterizer.render_device(scene.camera, scene.settings);
    auto device_target = tinytensor::Tensor::from_vector(
        photo_target, {photo_target.size()}, tinytensor::Device::Vulkan);
    auto device_zero_alpha = tinytensor::Tensor::zeros(
        {pixels}, tinytensor::Device::Vulkan, tinytensor::DataType::Float32);
    auto device_photo_model_gradient = tinytensor::Tensor::zeros(
        {static_cast<std::size_t>(rasterizer.model_gradient_float_count())},
        tinytensor::Device::Vulkan, tinytensor::DataType::Float32);
    tinytensor::vulkan::synchronize();
    const auto target_view = tinytensor::vulkan::buffer_view(device_target);
    const auto zero_alpha_view = tinytensor::vulkan::buffer_view(device_zero_alpha);
    const auto photo_model_view =
        tinytensor::vulkan::buffer_view(device_photo_model_gradient);
    const auto device_photo = photara::splat::detail::photometric_loss(
        rasterizer, device_frame,
        {target_view.buffer, target_view.offset, target_view.bytes});
    const float device_photo_scalar =
        photara::splat::detail::photometric_loss_scalar(rasterizer, device_photo);
    rasterizer.backward_device(
        device_photo.gradient,
        {zero_alpha_view.buffer, zero_alpha_view.offset, zero_alpha_view.bytes},
        {photo_model_view.buffer, photo_model_view.offset, photo_model_view.bytes});
    const auto got_photo_model = device_photo_model_gradient.to_vector();
    std::vector<float> photo_model_reference;
    const auto append_photo = [&](const std::vector<float>& values) {
        photo_model_reference.insert(
            photo_model_reference.end(), values.begin(), values.end());
    };
    append_photo(expected_photo_model.means);
    append_photo(expected_photo_model.colors);
    append_photo(expected_photo_model.opacities);
    append_photo(expected_photo_model.scales);
    append_photo(expected_photo_model.rotations);
    photo_model_reference.insert(photo_model_reference.end(), 6, 0.0F);
    photo_model_reference.insert(photo_model_reference.end(), 8, 0.0F);
    append_photo(expected_photo_model.refine_weight);
    require(rel_l2(got_photo_model, photo_model_reference) < 3.0e-4,
            "device-resident photometric/backward chain differs");
    require(std::abs(device_photo_scalar - expected_photo.loss) < 2.0e-5F,
            "device photometric scalar reduction differs");

    // Bind the trainable raw tensors themselves. This is the optimizer-facing
    // path: preprocessing activates exp(scale), normalized quaternion and
    // sigmoid(opacity) directly from the same storage Adam updates in place.
    constexpr float sh_c0 = 0.28209479177387814F;
    auto raw_means = tinytensor::Tensor::from_vector(
        scene.means, {1, 3}, tinytensor::Device::Vulkan);
    auto raw_sh = tinytensor::Tensor::from_vector(
        std::vector<float>{
            (scene.colors[0] - 0.5F) / sh_c0,
            (scene.colors[1] - 0.5F) / sh_c0,
            (scene.colors[2] - 0.5F) / sh_c0},
        {1, 1, 3}, tinytensor::Device::Vulkan);
    auto raw_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{
            std::log(scene.scales[0]), std::log(scene.scales[1]),
            std::log(scene.scales[2])},
        {1, 3}, tinytensor::Device::Vulkan);
    auto raw_rotations = tinytensor::Tensor::from_vector(
        scene.rotation, {1, 4}, tinytensor::Device::Vulkan);
    auto raw_opacity = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(scene.opacity[0] / (1.0F - scene.opacity[0]))},
        {1}, tinytensor::Device::Vulkan);
    auto raw_filter = tinytensor::Tensor::zeros(
        {1}, tinytensor::Device::Vulkan, tinytensor::DataType::Float32);
    tinytensor::vulkan::synchronize();
    const auto splat_view = [](const tinytensor::Tensor& tensor) {
        const auto view = tinytensor::vulkan::buffer_view(tensor);
        return splat_drender::vulkan::SplatBufferView{
            view.buffer, view.offset, view.bytes};
    };
    splat_drender::vulkan::SplatDeviceGaussians raw_model;
    raw_model.means = splat_view(raw_means);
    raw_model.sh = splat_view(raw_sh);
    raw_model.log_scales = splat_view(raw_scales);
    raw_model.raw_rotations = splat_view(raw_rotations);
    raw_model.opacity_logits = splat_view(raw_opacity);
    raw_model.filter_3d = splat_view(raw_filter);
    raw_model.count = 1;
    raw_model.sh_degree = 0;
    raw_model.sh_bases = 1;
    rasterizer.bind_model_device(raw_model);
    const auto raw_forward = rasterizer.render(scene.camera, scene.settings);
    require(raw_forward.instance_count == forward.instance_count,
            "raw device model changed the projected instance count");
    require(rel_l2(raw_forward.color, forward.color) < 2.0e-6 &&
                rel_l2(raw_forward.alpha, forward.alpha) < 2.0e-6 &&
                rel_l2(raw_forward.median_depth, forward.median_depth) < 2.0e-6,
            "raw device model activation differs from the uploaded model");
    std::cout << "tinytensor zero-copy backward: device=" << context.device_info().name
              << " gradients=" << got.size() << " rel_l2=" << rel_l2(got, reference)
              << " model_rel_l2=" << rel_l2(got_model, model_reference)
              << " photo_model_rel_l2="
              << rel_l2(got_photo_model, photo_model_reference)
              << " raw_model_rel_l2=" << rel_l2(raw_forward.color, forward.color)
              << " instances=" << forward.instance_count << '\n';
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
        sample_depth_case();
        multi_view_loss_case();
        ssim_loss_case();
        geometry_gradient_descent_case();
        optimizer_parity_case();
        tinytensor_interop_case();
        adopted_device_case();
        auto color_backward = pinhole_scene("pinhole-color", 24, 0.08F, false);
        color_backward.check_backward = true;
        color_backward.scale_modifier = 1.3F;
        enable_activation_chain(color_backward);
        compare_case(color_backward);

        auto sh = pinhole_scene("pinhole-sh", 12, 0.1F, true);
        sh.use_sh = true;
        sh.sh_degree = 3;
        sh.kernel_size = 0.3F;
        sh.colors.clear();
        sh.check_backward = true;
        Rng rng{7};
        fill_sh(sh, rng);
        compare_case(sh);

        auto covariance = pinhole_scene("pinhole-covariance", 10, 0.09F, true);
        covariance.use_covariance = true;
        covariance.scale_modifier = 1.25F;
        covariance.check_backward = true;
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
        fisheye.check_backward = true;
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
        equirect.check_backward = true;
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
