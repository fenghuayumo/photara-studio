#include "splat/visualize.hpp"

#include "core/camera_projection.hpp"
#include "cuda_ops.hpp"
#include "splat/rasterizer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace photara::splat {
namespace {

constexpr int k_threads = 256;
constexpr float k_near = 1e-4F;
__device__ constexpr float k_sh0 = 0.28209479177387814F;
__device__ constexpr float k_sh1 = 0.4886025119029199F;
__device__ constexpr float k_sh2[] = {
    1.0925484305920792F, -1.0925484305920792F, 0.31539156525252005F,
    -1.0925484305920792F, 0.5462742152960396F};
__device__ constexpr float k_sh3[] = {
    -0.5900435899266435F, 2.890611442640554F, -0.4570457994644658F,
    0.3731763325901154F, -0.4570457994644658F, 1.445305721320277F,
    -0.5900435899266435F};
constexpr int k_ring_segments = 32;

void check_cuda(const cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    throw std::runtime_error(
        std::string("Splat visualize ") + operation +
        " failed: " + cudaGetErrorString(error));
}

__device__ void camera_point(
    const float* w2c, const float X, const float Y, const float Z, float& x,
    float& y, float& z) {
    x = w2c[0] * X + w2c[4] * Y + w2c[8] * Z + w2c[12];
    y = w2c[1] * X + w2c[5] * Y + w2c[9] * Z + w2c[13];
    z = w2c[2] * X + w2c[6] * Y + w2c[10] * Z + w2c[14];
}

__device__ void camera_vector(
    const float* w2c, const float X, const float Y, const float Z, float& x,
    float& y, float& z) {
    x = w2c[0] * X + w2c[4] * Y + w2c[8] * Z;
    y = w2c[1] * X + w2c[5] * Y + w2c[9] * Z;
    z = w2c[2] * X + w2c[6] * Y + w2c[10] * Z;
}

__device__ void eval_sh_color(
    const float* sh, const int index, const int bases, const int degree,
    const float mx, const float my, const float mz, const float camx,
    const float camy, const float camz, float& r, float& g, float& b) {
    const float* coeff = sh + static_cast<std::size_t>(index) * bases * 3;
    r = k_sh0 * coeff[0];
    g = k_sh0 * coeff[1];
    b = k_sh0 * coeff[2];
    if (degree > 0 && bases >= 4) {
        float dx = mx - camx;
        float dy = my - camy;
        float dz = mz - camz;
        const float inv = rsqrtf(fmaxf(dx * dx + dy * dy + dz * dz, 1e-12F));
        dx *= inv;
        dy *= inv;
        dz *= inv;
        r += -k_sh1 * dy * coeff[3] + k_sh1 * dz * coeff[6] - k_sh1 * dx * coeff[9];
        g += -k_sh1 * dy * coeff[4] + k_sh1 * dz * coeff[7] - k_sh1 * dx * coeff[10];
        b += -k_sh1 * dy * coeff[5] + k_sh1 * dz * coeff[8] - k_sh1 * dx * coeff[11];
        if (degree > 1 && bases >= 9) {
            const float xx = dx * dx, yy = dy * dy, zz = dz * dz;
            const float xy = dx * dy, yz = dy * dz, xz = dx * dz;
            const float b4 = k_sh2[0] * xy;
            const float b5 = k_sh2[1] * yz;
            const float b6 = k_sh2[2] * (2.F * zz - xx - yy);
            const float b7 = k_sh2[3] * xz;
            const float b8 = k_sh2[4] * (xx - yy);
            r += b4 * coeff[12] + b5 * coeff[15] + b6 * coeff[18] +
                 b7 * coeff[21] + b8 * coeff[24];
            g += b4 * coeff[13] + b5 * coeff[16] + b6 * coeff[19] +
                 b7 * coeff[22] + b8 * coeff[25];
            b += b4 * coeff[14] + b5 * coeff[17] + b6 * coeff[20] +
                 b7 * coeff[23] + b8 * coeff[26];
            if (degree > 2 && bases >= 16) {
                const float c9 = k_sh3[0] * dy * (3.F * xx - yy);
                const float c10 = k_sh3[1] * xy * dz;
                const float c11 = k_sh3[2] * dy * (4.F * zz - xx - yy);
                const float c12 = k_sh3[3] * dz * (2.F * zz - 3.F * xx - 3.F * yy);
                const float c13 = k_sh3[4] * dx * (4.F * zz - xx - yy);
                const float c14 = k_sh3[5] * dz * (xx - yy);
                const float c15 = k_sh3[6] * dx * (xx - 3.F * yy);
                r += c9 * coeff[27] + c10 * coeff[30] + c11 * coeff[33] +
                     c12 * coeff[36] + c13 * coeff[39] + c14 * coeff[42] +
                     c15 * coeff[45];
                g += c9 * coeff[28] + c10 * coeff[31] + c11 * coeff[34] +
                     c12 * coeff[37] + c13 * coeff[40] + c14 * coeff[43] +
                     c15 * coeff[46];
                b += c9 * coeff[29] + c10 * coeff[32] + c11 * coeff[35] +
                     c12 * coeff[38] + c13 * coeff[41] + c14 * coeff[44] +
                     c15 * coeff[47];
            }
        }
    }
    r = fmaxf(r + 0.5F, 0.F);
    g = fmaxf(g + 0.5F, 0.F);
    b = fmaxf(b + 0.5F, 0.F);
}

__device__ bool project_overlay_pixel(
    const float x, const float y, const float z, const int width,
    const int height, const float fx, const float fy, const float cx,
    const float cy, const int model, const float k1, const float k2,
    const float k3, const float k4, float& u, float& v) {
    if (model == static_cast<int>(photara::CameraModel::equirectangular)) {
        const auto pixel = project_equirectangular_camera(
            x, y, z, width, height);
        if (!pixel.valid) return false;
        u = static_cast<float>(pixel.u);
        v = static_cast<float>(pixel.v);
        return true;
    }
    if (model == static_cast<int>(photara::CameraModel::opencv_fisheye)) {
        const auto pixel = project_fisheye_camera(
            x, y, z, fx, fy, cx, cy, k1, k2, k3, k4);
        if (!pixel.valid) return false;
        u = static_cast<float>(pixel.u);
        v = static_cast<float>(pixel.v);
        return true;
    }
    if (z <= k_near) return false;
    u = fx * x / z + cx;
    v = fy * y / z + cy;
    return true;
}

__device__ void blend_pixel(
    float* color, float* alpha, const int pixel, const int pixels,
    const float r, const float g, const float b, const float a) {
    if (pixel < 0 || pixel >= pixels || a <= 0.F) return;
    atomicAdd(color + pixel, r * a);
    atomicAdd(color + pixels + pixel, g * a);
    atomicAdd(color + 2 * pixels + pixel, b * a);
    atomicAdd(alpha + pixel, a);
}

__device__ void quat_scaled_axes(
    const float* quaternion, const float* scale, const float modifier,
    float ax[3], float ay[3], float az[3]) {
    const float qw = quaternion[0];
    const float qx = quaternion[1];
    const float qy = quaternion[2];
    const float qz = quaternion[3];
    const float xx = qx * qx;
    const float yy = qy * qy;
    const float zz = qz * qz;
    const float xy = qx * qy;
    const float xz = qx * qz;
    const float yz = qy * qz;
    const float wx = qw * qx;
    const float wy = qw * qy;
    const float wz = qw * qz;
    const float sx = scale[0] * modifier;
    const float sy = scale[1] * modifier;
    const float sz = scale[2] * modifier;
    ax[0] = (1.F - 2.F * (yy + zz)) * sx;
    ax[1] = (2.F * (xy + wz)) * sx;
    ax[2] = (2.F * (xz - wy)) * sx;
    ay[0] = (2.F * (xy - wz)) * sy;
    ay[1] = (1.F - 2.F * (xx + zz)) * sy;
    ay[2] = (2.F * (yz + wx)) * sy;
    az[0] = (2.F * (xz + wy)) * sz;
    az[1] = (2.F * (yz - wx)) * sz;
    az[2] = (1.F - 2.F * (xx + yy)) * sz;
}

__global__ void visualize_points_kernel(
    const float* means, const float* sh, const float* w2c, float* color,
    float* alpha, const int count, const int width, const int height,
    const int bases, const int sh_degree, const float fx, const float fy,
    const float cx, const float cy, const int model, const float k1,
    const float k2, const float k3, const float k4, const float camx,
    const float camy, const float camz, const float point_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float mx = means[3 * index];
    const float my = means[3 * index + 1];
    const float mz = means[3 * index + 2];
    float x, y, z;
    camera_point(w2c, mx, my, mz, x, y, z);
    float u = 0.F, v = 0.F;
    if (!project_overlay_pixel(
            x, y, z, width, height, fx, fy, cx, cy, model, k1, k2, k3, k4, u,
            v))
        return;
    const int radius = max(1, static_cast<int>(ceilf(point_size)));
    const int pixels = width * height;
    float r, g, b;
    eval_sh_color(
        sh, index, bases, sh_degree, mx, my, mz, camx, camy, camz, r, g, b);
    const float radius_sq = point_size * point_size;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (static_cast<float>(dx * dx + dy * dy) > radius_sq) continue;
            const int px = static_cast<int>(floorf(u)) + dx;
            const int py = static_cast<int>(floorf(v)) + dy;
            if (px < 0 || py < 0 || px >= width || py >= height) continue;
            blend_pixel(color, alpha, py * width + px, pixels, r, g, b, 1.F);
        }
    }
}

__global__ void visualize_rings_kernel(
    const float* means, const float* scales, const float* quaternions,
    const float* sh, const float* w2c, float* color, float* alpha,
    const int count, const int width, const int height, const int bases,
    const int sh_degree, const float fx, const float fy, const float cx,
    const float cy, const int model, const float k1, const float k2,
    const float k3, const float k4, const float camx, const float camy,
    const float camz, const float scale_modifier, const float ring_scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float mx = means[3 * index];
    const float my = means[3 * index + 1];
    const float mz = means[3 * index + 2];
    float x, y, z;
    camera_point(w2c, mx, my, mz, x, y, z);
    float u = 0.F, v = 0.F;
    if (!project_overlay_pixel(
            x, y, z, width, height, fx, fy, cx, cy, model, k1, k2, k3, k4, u,
            v))
        return;
    if (z <= k_near) z = k_near;

    float ax[3], ay[3], az[3];
    quat_scaled_axes(
        quaternions + 4 * index, scales + 3 * index, scale_modifier, ax, ay,
        az);
    float cax[3], cay[3], caz[3];
    camera_vector(w2c, ax[0], ax[1], ax[2], cax[0], cax[1], cax[2]);
    camera_vector(w2c, ay[0], ay[1], ay[2], cay[0], cay[1], cay[2]);
    camera_vector(w2c, az[0], az[1], az[2], caz[0], caz[1], caz[2]);

    const float inv_z = 1.F / z;
    const float inv_z2 = inv_z * inv_z;
    const float jx_x = fx * inv_z;
    const float jx_z = -fx * x * inv_z2;
    const float jy_y = fy * inv_z;
    const float jy_z = -fy * y * inv_z2;
    const float p0x = jx_x * cax[0] + jx_z * cax[2];
    const float p0y = jy_y * cax[1] + jy_z * cax[2];
    const float p1x = jx_x * cay[0] + jx_z * cay[2];
    const float p1y = jy_y * cay[1] + jy_z * cay[2];
    const float p2x = jx_x * caz[0] + jx_z * caz[2];
    const float p2y = jy_y * caz[1] + jy_z * caz[2];
    const float a = p0x * p0x + p1x * p1x + p2x * p2x;
    const float b = p0x * p0y + p1x * p1y + p2x * p2y;
    const float c = p0y * p0y + p1y * p1y + p2y * p2y;
    const float mid = 0.5F * (a + c);
    const float ext =
        0.5F * sqrtf(fmaxf(0.F, (a - c) * (a - c) + 4.F * b * b));
    const float rx = fminf(80.F, ring_scale * sqrtf(fmaxf(0.F, mid + ext)));
    const float ry = fminf(80.F, ring_scale * sqrtf(fmaxf(0.F, mid - ext)));
    if (rx < 0.75F && ry < 0.75F) return;
    const float rotation = 0.5F * atan2f(2.F * b, a - c);
    const float cos_r = cosf(rotation);
    const float sin_r = sinf(rotation);
    const int pixels = width * height;
    float red, green, blue;
    eval_sh_color(
        sh, index, bases, sh_degree, mx, my, mz, camx, camy, camz, red,
        green, blue);

    float prev_x = 0.F;
    float prev_y = 0.F;
    for (int segment = 0; segment <= k_ring_segments; ++segment) {
        const float theta =
            6.28318530718F * static_cast<float>(segment) /
            static_cast<float>(k_ring_segments);
        const float local_x = rx * cosf(theta);
        const float local_y = ry * sinf(theta);
        const float px = u + local_x * cos_r - local_y * sin_r;
        const float py = v + local_x * sin_r + local_y * cos_r;
        if (segment > 0) {
            const float dx = px - prev_x;
            const float dy = py - prev_y;
            const int steps = max(
                1, static_cast<int>(ceilf(sqrtf(dx * dx + dy * dy))));
            for (int step = 0; step <= steps; ++step) {
                const float t = static_cast<float>(step) /
                                static_cast<float>(steps);
                const int ix = static_cast<int>(
                    floorf(prev_x + dx * t));
                const int iy = static_cast<int>(
                    floorf(prev_y + dy * t));
                if (ix < 0 || iy < 0 || ix >= width || iy >= height) continue;
                blend_pixel(
                    color, alpha, iy * width + ix, pixels, red, green, blue,
                    1.F);
            }
        }
        prev_x = px;
        prev_y = py;
    }
}

__global__ void composite_background_kernel(
    float* color, const float* alpha, const float br, const float bg,
    const float bb, const int pixels) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    const float weight = fmaxf(alpha[pixel], 0.F);
    const float coverage = fminf(weight, 1.F);
    const float inv = weight > 1e-6F ? coverage / weight : 0.F;
    color[pixel] = fminf(color[pixel] * inv + br * (1.F - coverage), 1.F);
    color[pixels + pixel] =
        fminf(color[pixels + pixel] * inv + bg * (1.F - coverage), 1.F);
    color[2 * pixels + pixel] =
        fminf(color[2 * pixels + pixel] * inv + bb * (1.F - coverage), 1.F);
}

tinytensor::Tensor render_debug_overlay(
    const GaussianModel& model, const Camera& camera,
    const VisualizeOptions& options) {
    const int width = static_cast<int>(camera.width);
    const int height = static_cast<int>(camera.height);
    const int count = static_cast<int>(model.size());
    const int pixels = width * height;
    auto color = tinytensor::Tensor::zeros(
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    auto alpha = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    if (count == 0 || pixels == 0) return color;

    const auto activated = detail::activate_parameters(model);
    const auto view = tinytensor::Tensor::from_vector(
        std::vector<float>(
            camera.world_to_camera.begin(), camera.world_to_camera.end()),
        {4, 4}, tinytensor::Device::CUDA);
    const int bases = static_cast<int>(model.sh.shape()[1]);
    const int sh_degree = static_cast<int>(std::min(
        options.active_sh_degree, model.sh_degree));
    const int blocks = (count + k_threads - 1) / k_threads;
    if (options.mode == VisualizationMode::points) {
        visualize_points_kernel<<<blocks, k_threads>>>(
            model.means.ptr<float>(), model.sh.ptr<float>(),
            view.ptr<float>(), color.ptr<float>(), alpha.ptr<float>(), count,
            width, height, bases, sh_degree, camera.fx, camera.fy, camera.cx,
            camera.cy, static_cast<int>(camera.model), camera.k1, camera.k2,
            camera.k3, camera.k4, camera.position[0], camera.position[1],
            camera.position[2], options.point_size_px);
    } else {
        visualize_rings_kernel<<<blocks, k_threads>>>(
            model.means.ptr<float>(), activated.scales.ptr<float>(),
            activated.quaternions.ptr<float>(), model.sh.ptr<float>(),
            view.ptr<float>(), color.ptr<float>(), alpha.ptr<float>(), count,
            width, height, bases, sh_degree, camera.fx, camera.fy, camera.cx,
            camera.cy, static_cast<int>(camera.model), camera.k1, camera.k2,
            camera.k3, camera.k4, camera.position[0], camera.position[1],
            camera.position[2], options.scale_modifier, options.ring_scale);
    }
    check_cuda(cudaGetLastError(), "launch overlay kernel");
    const int composite_blocks = (pixels + k_threads - 1) / k_threads;
    composite_background_kernel<<<composite_blocks, k_threads>>>(
        color.ptr<float>(), alpha.ptr<float>(), options.background[0],
        options.background[1], options.background[2], pixels);
    check_cuda(cudaGetLastError(), "launch composite kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize overlay");
    return color;
}

}  // namespace

tinytensor::Tensor visualize(
    const GaussianModel& model, const Camera& camera,
    const VisualizeOptions& options) {
    if (options.mode == VisualizationMode::splat) {
        Rasterizer rasterizer;
        RasterizeOptions raster;
        raster.active_sh_degree = options.active_sh_degree;
        raster.background = options.background;
        raster.kernel_size = options.kernel_size;
        raster.scale_modifier = options.scale_modifier;
        raster.require_depth = false;
        return rasterizer.forward(model, camera, raster).color;
    }
    return render_debug_overlay(model, camera, options);
}

}  // namespace photara::splat
