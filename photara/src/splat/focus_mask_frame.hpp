#pragma once

#include "focus_mask.hpp"
#include "mvs/types.hpp"
#include "splat/types.hpp"

namespace photara::splat::detail {

inline FocusMaskFrame make_focus_mask_frame(
    const Camera& camera, const mvs::OrientedBoundingBox& bounds) {
    FocusMaskFrame frame;
    frame.width = camera.width;
    frame.height = camera.height;
    frame.model = static_cast<unsigned>(camera.model);
    frame.fx = camera.fx;
    frame.fy = camera.fy;
    frame.cx = camera.cx;
    frame.cy = camera.cy;
    frame.k1 = camera.k1;
    frame.k2 = camera.k2;
    frame.k3 = camera.k3;
    frame.k4 = camera.k4;
    const mvs::Vec3f origin = bounds.local(mvs::Vec3f(
        camera.position[0], camera.position[1], camera.position[2]));
    frame.origin_x = origin.x();
    frame.origin_y = origin.y();
    frame.origin_z = origin.z();
    mvs::Mat3f camera_to_world;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            camera_to_world(row, column) = camera.world_to_camera[
                static_cast<std::size_t>(row) * 4 + column];
    const mvs::Mat3f camera_to_box = bounds.axes.transpose() * camera_to_world;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            frame.camera_to_box[row * 3 + column] =
                camera_to_box(row, column);
    frame.half_x = bounds.half_extent.x();
    frame.half_y = bounds.half_extent.y();
    frame.half_z = bounds.half_extent.z();
    return frame;
}

}  // namespace photara::splat::detail
