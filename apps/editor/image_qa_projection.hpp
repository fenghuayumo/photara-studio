#pragma once

#include "sparse_view.hpp"
#include "splat/training_image.hpp"

#include <bit>
#include <sstream>

namespace editor {

inline constexpr std::uint32_t k_image_qa_long_edge = 1600;

inline photara::mvs::MvsView image_qa_training_view(
    const ViewPose& pose, const std::filesystem::path& path) {
    photara::mvs::MvsView view;
    view.path = path;
    view.width = pose.width;
    view.height = pose.height;
    view.fx = pose.fx;
    view.fy = pose.fy;
    view.cx = pose.cx >= 0.F ? pose.cx : 0.5F * pose.width;
    view.cy = pose.cy >= 0.F ? pose.cy : 0.5F * pose.height;
    view.source_model = pose.model;
    const bool source = pose.src_width != 0 && pose.src_height != 0;
    view.src_width = source ? pose.src_width : view.width;
    view.src_height = source ? pose.src_height : view.height;
    view.src_fx = source ? pose.src_fx : view.fx;
    view.src_fy = source ? pose.src_fy : view.fy;
    view.src_cx = source ? pose.src_cx : view.cx;
    view.src_cy = source ? pose.src_cy : view.cy;
    view.k1 = pose.k1;
    view.k2 = pose.k2;
    view.p1 = pose.k3;
    view.p2 = pose.k4;
    for (int row = 0; row < 3; ++row) {
        view.pose.C[row] = row == 0 ? pose.centre.x : row == 1 ? pose.centre.y : pose.centre.z;
        for (int col = 0; col < 3; ++col)
            view.pose.R(row, col) = pose.rotation[row * 3 + col];
    }
    return view;
}

inline photara::splat::TrainingOptions image_qa_training_options(
    const bool use_mask, const std::filesystem::path& mask_dir,
    const bool undistort_to_pinhole) {
    photara::splat::TrainingOptions options;
    options.use_source_resolution = true;
    options.max_image_dimension = k_image_qa_long_edge;
    options.progressive_resolution = false;
    options.ignore_undistortion_border = true;
    options.use_mask = use_mask;
    options.mask_dir = mask_dir;
    options.undistort_to_pinhole = undistort_to_pinhole;
    return options;
}

// Exact float bits distinguish small calibration updates, even on the same
// file and camera index. Includes source and output projections separately.
inline std::string image_qa_camera_key(const photara::splat::Camera& camera) {
    std::ostringstream key;
    key << camera.width << ',' << camera.height << ',' << static_cast<unsigned>(camera.model);
    const auto append = [&](float value) { key << ',' << std::bit_cast<std::uint32_t>(value); };
    for (float value : {camera.fx, camera.fy, camera.cx, camera.cy,
                        camera.k1, camera.k2, camera.k3, camera.k4}) append(value);
    for (float value : camera.world_to_camera) append(value);
    for (float value : camera.position) append(value);
    return key.str();
}

inline std::string image_qa_projection_key(
    const photara::mvs::MvsView& view, const photara::splat::TrainingOptions& options) {
    photara::splat::Camera source;
    source.model = view.source_model;
    source.fx = view.src_fx; source.fy = view.src_fy;
    source.cx = view.src_cx; source.cy = view.src_cy;
    source.width = view.src_width; source.height = view.src_height;
    source.k1 = view.k1; source.k2 = view.k2; source.k3 = view.p1; source.k4 = view.p2;
    return image_qa_camera_key(source) + '|' + image_qa_camera_key(
        photara::splat::training_data::training_camera(view, options));
}

}  // namespace editor
