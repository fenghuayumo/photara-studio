#include "focus_mask.hpp"

#include "core/camera_projection.hpp"
#include "internal/cuda_stream_context.hpp"

#include <stdexcept>

#include <vector>

namespace photara::splat::detail {
namespace {

bool focus_ray_axis(
    float& t_near,
    float& t_far,
    const float origin,
    const float direction,
    const float half_extent) {
    if (fabsf(direction) < 1e-9F)
        return fabsf(origin) <= half_extent;
    float enter = (-half_extent - origin) / direction;
    float exit = (half_extent - origin) / direction;
    if (enter > exit) {
        const float swap = enter;
        enter = exit;
        exit = swap;
    }
    t_near = fmaxf(t_near, enter);
    t_far = fminf(t_far, exit);
    return t_far >= t_near;
}

bool focus_ray_hits(
    const float origin_x,
    const float origin_y,
    const float origin_z,
    const float direction_x,
    const float direction_y,
    const float direction_z,
    const float half_x,
    const float half_y,
    const float half_z) {
    float t_near = 0.F;
    float t_far = 3.402823466e+38F;
    return focus_ray_axis(t_near, t_far, origin_x, direction_x, half_x) &&
           focus_ray_axis(t_near, t_far, origin_y, direction_y, half_y) &&
           focus_ray_axis(t_near, t_far, origin_z, direction_z, half_z) && t_far >= 0.F;
}

float focus_mask_value(
    const FocusMaskFrame& frame,
    const unsigned x,
    const unsigned y) {
    CameraRay ray{};
    if (frame.model == static_cast<unsigned>(CameraModel::opencv_fisheye))
        ray = unproject_fisheye_camera(
            x, y, frame.fx, frame.fy, frame.cx, frame.cy, frame.k1, frame.k2, frame.k3, frame.k4);
    else if (frame.model == static_cast<unsigned>(CameraModel::equirectangular))
        ray = unproject_equirectangular_camera(
            x, y, static_cast<int>(frame.width), static_cast<int>(frame.height));
    else
        ray = CameraRay{
            (static_cast<double>(x) - frame.cx) / frame.fx,
            (static_cast<double>(y) - frame.cy) / frame.fy, 1.0, true};
    if (!ray.valid)
        return 0.F;
    const float ray_x = static_cast<float>(ray.x);
    const float ray_y = static_cast<float>(ray.y);
    const float ray_z = static_cast<float>(ray.z);
    const float direction_x = frame.camera_to_box[0] * ray_x + frame.camera_to_box[1] * ray_y +
                              frame.camera_to_box[2] * ray_z;
    const float direction_y = frame.camera_to_box[3] * ray_x + frame.camera_to_box[4] * ray_y +
                              frame.camera_to_box[5] * ray_z;
    const float direction_z = frame.camera_to_box[6] * ray_x + frame.camera_to_box[7] * ray_y +
                              frame.camera_to_box[8] * ray_z;
    return focus_ray_hits(
               frame.origin_x, frame.origin_y, frame.origin_z, direction_x, direction_y,
               direction_z, frame.half_x, frame.half_y, frame.half_z)
               ? 1.F
               : 0.F;
}

} // namespace

tinytensor::Tensor focus_view_mask_cpu(
    const FocusMaskFrame& frame,
    const tinytensor::Device device) {
    const std::size_t pixels = static_cast<std::size_t>(frame.width) * frame.height;
    std::vector<float> mask(pixels, 0.F);
    for (unsigned y = 0; y < frame.height; ++y)
        for (unsigned x = 0; x < frame.width; ++x)
            mask[static_cast<std::size_t>(y) * frame.width + x] = focus_mask_value(frame, x, y);
    return tinytensor::Tensor::from_vector(mask, {frame.height, frame.width}, device);
}

tinytensor::Tensor focus_view_mask_cuda(const FocusMaskFrame&) {
    throw std::runtime_error("CUDA focus mask backend is disabled");
}
} // namespace photara::splat::detail
