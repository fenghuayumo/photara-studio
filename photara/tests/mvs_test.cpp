#include "mvs/densify.hpp"
#include "mvs/export.hpp"
#include "mvs/internal.hpp"
#include "mvs/maxflow.hpp"
#include "io/image.hpp"
#include "io/mesh.hpp"
#include "sfm/scene.hpp"
#include "texture/projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace photara::mvs;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

MvsView make_plane_view(const Index id, const std::uint32_t size) {
    MvsView view;
    view.id = id;
    view.sfm_image_id = id;
    view.path = "this-file-does-not-exist.png";
    view.pose = photara::sfm::Pose3D::identity();
    view.width = size;
    view.height = size;
    view.fx = 100.F;
    view.fy = 100.F;
    view.cx = static_cast<float>(size - 1) * 0.5F;
    view.cy = static_cast<float>(size - 1) * 0.5F;
    view.src_fx = view.fx;
    view.src_fy = view.fy;
    view.src_cx = view.cx;
    view.src_cy = view.cy;
    view.src_width = size;
    view.src_height = size;
    view.depth_map.view_id = id;
    view.depth_map.resize(size, size);
    view.depth_map.depth_min = 1.F;
    view.depth_map.depth_max = 3.F;
    std::fill(view.depth_map.depth.begin(), view.depth_map.depth.end(), 2.F);
    std::fill(
        view.depth_map.normal.begin(), view.depth_map.normal.end(),
        Vec3f{0.F, 0.F, -1.F});
    std::fill(
        view.depth_map.confidence.begin(), view.depth_map.confidence.end(), 0.1F);
    return view;
}

MvsScene make_plane_scene() {
    constexpr std::uint32_t size = 16;
    MvsScene scene;
    scene.thread_count = 4;
    scene.views.push_back(make_plane_view(0, size));
    scene.views.push_back(make_plane_view(1, size));
    scene.views[0].neighbors.push_back(NeighborScore{1, 1.F, 10.F, 1.F, 10});
    scene.views[1].neighbors.push_back(NeighborScore{0, 1.F, 10.F, 1.F, 10});
    SparsePoint sparse;
    sparse.position = Vec3f{0.F, 0.F, 2.F};
    sparse.view_ids = {0, 1};
    scene.sparse_points.push_back(std::move(sparse));
    return scene;
}

void test_build_mvs_scene_samples_sparse_colors() {
    using photara::io::RgbImage;
    using photara::io::save_rgb_png;
    using photara::sfm::Image;
    using photara::sfm::PinholeCamera;
    using photara::sfm::Scene;
    using photara::sfm::Track;

    const auto root = std::filesystem::temp_directory_path() /
                      "photara_mvs_sparse_color_test";
    std::filesystem::create_directories(root);
    const auto red_path = root / "red.png";
    const auto green_path = root / "green.png";
    RgbImage red{2, 2, std::vector<std::uint8_t>(2 * 2 * 3, 0)};
    for (std::size_t i = 0; i < red.pixels.size(); i += 3) red.pixels[i] = 255;
    RgbImage green{2, 2, std::vector<std::uint8_t>(2 * 2 * 3, 0)};
    for (std::size_t i = 1; i < green.pixels.size(); i += 3)
        green.pixels[i] = 255;
    save_rgb_png(red, red_path);
    save_rgb_png(green, green_path);

    Scene source;
    PinholeCamera camera;
    camera.id = 0;
    camera.width = 2;
    camera.height = 2;
    camera.fx = 1;
    camera.fy = 1;
    camera.cx = 0.5;
    camera.cy = 0.5;
    source.cameras.push_back(camera);
    for (std::uint32_t index = 0; index < 2; ++index) {
        Image image;
        image.id = index;
        image.camera_id = 0;
        image.path = index == 0 ? red_path : green_path;
        image.registered = true;
        image.pose.C = Eigen::Vector3d(index, 0, 0);
        image.features.image_width = 2;
        image.features.image_height = 2;
        image.features.keypoints.push_back({0.F, 0.F});
        source.images.push_back(std::move(image));
    }
    Track track;
    track.position = Eigen::Vector3d(0.5, 0, 1);
    track.observations = {{0, 0}, {1, 0}};
    track.num_inliers = 2;
    source.tracks.push_back(std::move(track));

    DensifyOptions options;
    const MvsScene scene = build_mvs_scene(source, options);
    require(
        scene.sparse_points.size() == 1,
        "SfM track was not converted into a sparse MVS point");
    const Vec3f color = scene.sparse_points[0].color;
    require(
        std::abs(color.x() - 0.5F) < 1e-3F &&
            std::abs(color.y() - 0.5F) < 1e-3F &&
            std::abs(color.z()) < 1e-3F,
        "SfM sparse point did not average photo colours");
    source.cameras[0].model = photara::CameraModel::opencv_fisheye;
    const auto fisheye_scene = build_mvs_scene(source, options);
    require(fisheye_scene.views[0].source_model == photara::CameraModel::opencv_fisheye,
            "MVS must preserve the source fisheye projection for rectification");
    std::filesystem::remove_all(root);
}

void test_parallel_fusion() {
    MvsScene scene = make_plane_scene();
    DensifyOptions options;
    options.speckle_size = 1;
    options.min_views_fuse = 2;
    options.thread_count = 4;
    fuse_depth_maps(scene, options);
    require(!scene.dense_cloud.points.empty(), "plane fusion produced no points");
    require(
        scene.dense_cloud.points.size() >= 220 &&
            scene.dense_cloud.points.size() <= 280,
        "plane fusion did not weld duplicate views at pixel scale");
    for (const DensePoint& point : scene.dense_cloud.points) {
        require(point.position.allFinite(), "fused position is non-finite");
        require(point.normal.allFinite(), "fused normal is non-finite");
        require(point.views.size() >= 2, "fused point lacks multi-view support");
        require(
            point.view_weights.size() == point.views.size(),
            "fused per-view weights are incomplete");
        require(
            std::all_of(
                point.view_weights.begin(), point.view_weights.end(),
                [](const float weight) { return weight > 0.F; }),
            "fused per-view weight is not positive");
    }
}

void test_mask_constrained_fusion() {
    MvsScene scene = make_plane_scene();
    for (auto& view : scene.views) {
        view.foreground_mask.assign(view.depth_map.size(), 0);
        for (std::uint32_t y = 0; y < view.height; ++y)
            for (std::uint32_t x = 0; x < view.width / 2; ++x)
                view.foreground_mask[view.depth_map.index(x, y)] = 1;
    }
    DensifyOptions options;
    options.speckle_size = 1;
    options.min_views_fuse = 2;
    fuse_depth_maps(scene, options);
    require(!scene.dense_cloud.points.empty(), "masked fusion produced no points");
    require(scene.dense_cloud.points.size() < 150, "fusion ignored foreground mask");
}

#if 0  // Removed legacy MVS ROI/mask-bootstrap tests.
void test_projected_mesh_mask() {
    MvsScene scene;
    scene.views.push_back(make_plane_view(0, 32));
    scene.mesh.vertices = {
        Vec3f{-0.12F, -0.12F, 2.F}, Vec3f{0.12F, -0.12F, 2.F},
        Vec3f{0.12F, 0.12F, 2.F}, Vec3f{-0.12F, 0.12F, 2.F}};
    scene.mesh.faces = {Eigen::Vector3i{0,1,2}, Eigen::Vector3i{0,2,3}};
    DensifyOptions options;
    options.auto_roi_mask_dilate_px = 0;
    detail::build_projected_foreground_masks(scene, options);
    const auto& mask = scene.views[0].foreground_mask;
    require(mask.size() == 32U * 32U, "projected mask has wrong size");
    require(mask[16U * 32U + 16U] != 0, "projected mesh missed image center");
    require(mask[0] == 0, "projected mesh mask filled background corner");
    require(
        std::any_of(
            mask.begin(), mask.end(),
            [](const std::uint8_t value) {
                return value > 0 && value < 255;
            }),
        "projected mesh mask has no soft silhouette coverage");
}

void test_projected_mesh_mask_fills_enclosed_holes() {
    MvsScene scene;
    scene.views.push_back(make_plane_view(0, 32));
    scene.mesh.vertices = {
        Vec3f{-0.20F, -0.20F, 2.F}, Vec3f{0.20F, -0.20F, 2.F},
        Vec3f{0.20F, 0.20F, 2.F}, Vec3f{-0.20F, 0.20F, 2.F},
        Vec3f{-0.06F, -0.06F, 2.F}, Vec3f{0.06F, -0.06F, 2.F},
        Vec3f{0.06F, 0.06F, 2.F}, Vec3f{-0.06F, 0.06F, 2.F}};
    // Four strips form a projected frame with a deliberate central hole.
    scene.mesh.faces = {
        Eigen::Vector3i{0,1,5}, Eigen::Vector3i{0,5,4},
        Eigen::Vector3i{1,2,6}, Eigen::Vector3i{1,6,5},
        Eigen::Vector3i{3,7,6}, Eigen::Vector3i{3,6,2},
        Eigen::Vector3i{0,4,7}, Eigen::Vector3i{0,7,3}};
    DensifyOptions options;
    options.auto_roi_mask_dilate_px = 0;
    detail::build_projected_foreground_masks(scene, options);
    const auto& mask = scene.views[0].foreground_mask;
    require(
        mask[16U * 32U + 16U] != 0,
        "projected coarse-mesh mask retained an enclosed hole");
    require(
        mask[0] == 0,
        "coarse-mesh hole filling leaked into exterior background");
}

void test_projected_mesh_mask_preserves_large_enclosed_holes() {
    MvsScene scene;
    scene.views.push_back(make_plane_view(0, 64));
    scene.mesh.vertices = {
        Vec3f{-0.40F, -0.40F, 2.F}, Vec3f{0.40F, -0.40F, 2.F},
        Vec3f{0.40F, 0.40F, 2.F}, Vec3f{-0.40F, 0.40F, 2.F},
        Vec3f{-0.18F, -0.18F, 2.F}, Vec3f{0.18F, -0.18F, 2.F},
        Vec3f{0.18F, 0.18F, 2.F}, Vec3f{-0.18F, 0.18F, 2.F}};
    scene.mesh.faces = {
        Eigen::Vector3i{0,1,5}, Eigen::Vector3i{0,5,4},
        Eigen::Vector3i{1,2,6}, Eigen::Vector3i{1,6,5},
        Eigen::Vector3i{3,7,6}, Eigen::Vector3i{3,6,2},
        Eigen::Vector3i{0,4,7}, Eigen::Vector3i{0,7,3}};
    DensifyOptions options;
    options.auto_roi_mask_dilate_px = 0;
    options.auto_roi_mask_close_px = 1;
    options.auto_roi_mask_feather_px = 0;
    detail::build_projected_foreground_masks(scene, options);
    require(
        scene.views[0].foreground_mask[32U * 64U + 32U] == 0,
        "projected coarse-mesh mask filled a real subject opening");
    require(
        scene.views[0].foreground_mask[0] == 0,
        "large-hole preservation leaked into exterior background");
}

void test_projected_mesh_mask_closes_open_notches() {
    MvsScene scene;
    scene.views.push_back(make_plane_view(0, 32));
    scene.mesh.vertices = {
        Vec3f{-0.20F, -0.20F, 2.F}, Vec3f{0.20F, -0.20F, 2.F},
        Vec3f{0.20F, 0.20F, 2.F}, Vec3f{-0.20F, 0.20F, 2.F},
        Vec3f{-0.06F, -0.06F, 2.F}, Vec3f{0.06F, -0.06F, 2.F},
        Vec3f{0.06F, 0.06F, 2.F}, Vec3f{-0.06F, 0.06F, 2.F}};
    // Deliberately omit the top strip, connecting the central void to the
    // exterior. Hole filling alone cannot repair this boundary notch.
    scene.mesh.faces = {
        Eigen::Vector3i{1,2,6}, Eigen::Vector3i{1,6,5},
        Eigen::Vector3i{3,7,6}, Eigen::Vector3i{3,6,2},
        Eigen::Vector3i{0,4,7}, Eigen::Vector3i{0,7,3}};
    DensifyOptions options;
    options.auto_roi_mask_dilate_px = 0;
    options.auto_roi_mask_close_px = 4;
    detail::build_projected_foreground_masks(scene, options);
    require(
        scene.views[0].foreground_mask[16U * 32U + 16U] != 0,
        "coarse-mesh mask retained a boundary-connected notch");
    require(
        scene.views[0].foreground_mask[0] == 0,
        "coarse-mesh closing leaked into distant background");
}

void test_depth_roi_mask_without_mesh() {
    MvsScene scene;
    scene.views.push_back(make_plane_view(0, 32));
    scene.roi.valid = true;
    scene.roi.center = Vec3f{0.F, 0.F, 2.F};
    scene.roi.half_extent = Vec3f{0.10F, 0.10F, 0.10F};
    DensifyOptions options;
    options.auto_roi_mask_close_px = 1;
    options.auto_roi_mask_dilate_px = 0;
    options.auto_roi_mask_feather_px = 0;
    detail::build_depth_roi_foreground_masks(scene, options);
    const auto& mask = scene.views[0].foreground_mask;
    require(mask.size() == 32U * 32U, "depth ROI mask has wrong size");
    require(mask[16U * 32U + 16U] == 255,
            "depth ROI mask missed the subject center");
    require(mask[0] == 0, "depth ROI mask retained distant background");
    require(scene.mesh.faces.empty(),
            "depth ROI mask unexpectedly required a mesh");

    std::fill(
        scene.views[0].depth_map.depth.begin(),
        scene.views[0].depth_map.depth.end(), 3.F);
    detail::build_depth_roi_foreground_masks(scene, options);
    require(scene.views[0].foreground_mask[16U * 32U + 16U] == 0,
            "depth ROI mask retained geometry outside the 3D OBB");
}

#endif

void test_subject_bounds_from_sparse_points() {
    std::vector<SparsePoint> sparse_points;
    for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                SparsePoint point;
                point.position = Vec3f{
                    0.1F * static_cast<float>(x),
                    0.1F * static_cast<float>(y),
                    0.1F * static_cast<float>(z)};
                point.view_ids = {0, 1, 2};
                sparse_points.push_back(std::move(point));
            }
        }
    }
    SparsePoint supported_extension;
    supported_extension.position = Vec3f{0.6F, 0.15F, 0.15F};
    supported_extension.view_ids = {0, 1, 2, 3};
    for (int i = 0; i < 12; ++i) {
        SparsePoint point = supported_extension;
        point.position.x() += 0.002F * static_cast<float>(i);
        sparse_points.push_back(std::move(point));
    }
    SparsePoint isolated;
    isolated.position = Vec3f{100.F, 100.F, 100.F};
    isolated.view_ids = {0, 1};
    sparse_points.push_back(std::move(isolated));

    OrientedBoundingBox bounds;
    require(
        detail::estimate_subject_bounds(sparse_points, bounds, 2, 1.15F),
        "SubjectBounds estimation failed");
    require(bounds.valid, "SubjectBounds were not marked valid");
    require(
        bounds.contains(Vec3f{0.61F, 0.15F, 0.15F}),
        "SubjectBounds clipped a supported thin extension");
    require(
        !bounds.contains(Vec3f{100.F, 100.F, 100.F}),
        "SubjectBounds retained an isolated SfM outlier");
}

#if 0  // Removed legacy MVS ROI/mask-bootstrap tests.
void test_input_mask_is_not_cut_by_coarse_mesh_holes() {
    MvsScene scene;
    scene.views.push_back(make_plane_view(0, 32));
    scene.views[0].foreground_mask.assign(32U * 32U, 255);
    // A deliberately incomplete coarse mesh that does not cover the center.
    scene.mesh.vertices = {
        Vec3f{-0.20F, -0.20F, 2.F}, Vec3f{-0.08F, -0.20F, 2.F},
        Vec3f{-0.20F, -0.08F, 2.F}};
    scene.mesh.faces = {Eigen::Vector3i{0,1,2}};
    scene.roi.valid = true;
    scene.roi.center = Vec3f{0.F, 0.F, 2.F};
    scene.roi.half_extent = Vec3f{0.25F, 0.25F, 0.25F};
    DensifyOptions options;
    options.auto_roi_mask_dilate_px = 0;
    detail::build_projected_foreground_masks(scene, options);
    require(
        scene.views[0].foreground_mask[16U * 32U + 16U] != 0,
        "coarse mesh hole was baked into the final input mask");
    require(
        scene.views[0].foreground_mask[0] == 0,
        "projected OBB envelope failed to reject distant background");
}

void test_manual_obb_file() {
    const auto path = std::filesystem::temp_directory_path() /
                      "photara-mvs-test-roi.txt";
    {
        std::ofstream output(path);
        output << "1 2 3  1 0.01 0  0 1 0  0 0 1  4 5 6\n";
    }
    OrientedBoundingBox roi;
    const bool loaded = detail::load_manual_roi(path, roi);
    std::filesystem::remove(path);
    require(loaded && roi.valid, "manual OBB file did not load");
    require((roi.axes.transpose() * roi.axes - Mat3f::Identity()).norm() < 1e-4F,
            "manual OBB axes were not orthonormalized");
    require(roi.contains(Vec3f{1.F,2.F,3.F}), "manual OBB missed its center");
    require(!roi.contains(Vec3f{20.F,2.F,3.F}), "manual OBB accepted far point");
    save_roi(roi, path);
    OrientedBoundingBox roundtrip;
    require(detail::load_manual_roi(path, roundtrip), "saved OBB did not reload");
    std::filesystem::remove(path);
    require((roundtrip.center - roi.center).norm() < 1e-5F,
            "OBB save/load changed its center");
}

#endif

void test_aether_texture_camera_projection() {
    MvsView view;
    view.width = 641;
    view.height = 479;
    view.fx = 713.25F;
    view.fy = 698.75F;
    view.cx = 301.2F;
    view.cy = 245.8F;
    view.pose.R = (
        Eigen::AngleAxisd(
            0.41, Eigen::Vector3d{0.2, 0.9, -0.3}.normalized()) *
        Eigen::AngleAxisd(-0.17, Eigen::Vector3d::UnitX()))
                      .toRotationMatrix();
    view.pose.C = Eigen::Vector3d{-0.4, 0.25, 1.1};
    constexpr float near_z = 0.2F;
    constexpr float far_z = 8.F;
    const auto matrix = photara::texture::world_to_clip_row_major(
        view, near_z, far_z);

    const Vec3f camera_point{0.13F, -0.09F, 2.4F};
    const Vec3f world =
        view.pose
            .transform_camera_to_world(camera_point.cast<double>())
            .cast<float>();
    std::array<float, 4> clip{};
    for (int row = 0; row < 4; ++row)
        clip[static_cast<std::size_t>(row)] =
            matrix[static_cast<std::size_t>(4 * row)] * world.x() +
            matrix[static_cast<std::size_t>(4 * row + 1)] * world.y() +
            matrix[static_cast<std::size_t>(4 * row + 2)] * world.z() +
            matrix[static_cast<std::size_t>(4 * row + 3)];
    require(clip[3] > 0.F, "aether texture camera reversed positive depth");
    const float pixel_corner_x =
        (clip[0] / clip[3] * 0.5F + 0.5F) *
        static_cast<float>(view.width);
    const float pixel_corner_y =
        (clip[1] / clip[3] * 0.5F + 0.5F) *
        static_cast<float>(view.height);
    const float expected_x =
        view.fx * camera_point.x() / camera_point.z() + view.cx;
    const float expected_y =
        view.fy * camera_point.y() / camera_point.z() + view.cy;
    // aether_drender samples photo texels at pixel_corner - 0.5.
    require(
        std::abs((pixel_corner_x - 0.5F) - expected_x) < 1e-4F &&
            std::abs((pixel_corner_y - 0.5F) - expected_y) < 1e-4F,
        "Photara/aether_drender texture projection has a half-pixel offset");
    const float ndc_depth = clip[2] / clip[3];
    require(
        ndc_depth > -1.F && ndc_depth < 1.F,
        "aether_drender texture projection produced invalid clip depth");

    view.foreground_mask.assign(
        static_cast<std::size_t>(view.width) * view.height, 0);
    constexpr std::size_t mask_pixel = 1234;
    view.foreground_mask[mask_pixel] = 128;
    require(
        photara::texture::has_effective_foreground_mask(view) &&
            std::abs(
                photara::texture::effective_foreground_coverage(
                    view, mask_pixel) -
                128.F / 255.F) < 1e-6F,
        "texture baking did not preserve the in-memory soft foreground mask");
}

#if 0  // Removed dense-cloud automatic ROI test.
void test_automatic_ground_and_subject_roi() {
    MvsScene scene;
    const Vec3f target{0.F, 0.5F, 0.F};
    const std::array<Vec3f, 4> cameras{
        Vec3f{2.F, 1.5F, 0.F}, Vec3f{-2.F, 1.5F, 0.F},
        Vec3f{0.F, 1.5F, 2.F}, Vec3f{0.F, 1.5F, -2.F}};
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        MvsView view;
        view.id = static_cast<Index>(i);
        view.pose.C = cameras[i].cast<double>();
        const Vec3f forward = (target - cameras[i]).normalized();
        Vec3f right = Vec3f::UnitY().cross(forward).normalized();
        const Vec3f down = forward.cross(right).normalized();
        view.pose.R.row(0) = right.cast<double>();
        view.pose.R.row(1) = down.cast<double>();
        view.pose.R.row(2) = forward.cast<double>();
        scene.views.push_back(view);
    }
    for (int z = -10; z <= 10; ++z)
        for (int x = -10; x <= 10; ++x) {
            DensePoint p;
            p.position = Vec3f{0.1F * x, 0.F, 0.1F * z};
            p.views = {0,1};
            scene.dense_cloud.points.push_back(p);
        }
    for (int z = -4; z <= 4; ++z)
        for (int y = 2; y <= 8; ++y)
            for (int x = -4; x <= 4; ++x) {
                DensePoint p;
                p.position = Vec3f{0.05F*x, 0.1F*y, 0.05F*z};
                p.views = {0,1,2};
                scene.dense_cloud.points.push_back(p);
            }
    MvsScene reversed = scene;
    std::reverse(
        reversed.dense_cloud.points.begin(),
        reversed.dense_cloud.points.end());
    DensifyOptions options;
    options.auto_roi_component_voxel_fraction = 0.04F;
    options.roi_margin_fraction = 0.05F;
    require(detail::estimate_automatic_roi(scene, options), "automatic ROI failed");
    require(
        detail::estimate_automatic_roi(reversed, options),
        "reversed automatic ROI failed");
    require(
        (scene.roi.center - reversed.roi.center).norm() < 1e-4F &&
            (scene.roi.half_extent - reversed.roi.half_extent).norm() < 1e-4F,
        "automatic ROI depends on parallel fusion point ordering");
    require(scene.has_ground_plane, "automatic ROI missed dominant ground plane");
    require(scene.roi.contains(target), "automatic ROI missed subject target");
    require(!scene.roi.contains(Vec3f{0.9F, 0.F, 0.9F}),
            "automatic ROI retained distant ground");
    for (const auto& point : scene.dense_cloud.points)
        require(point.position.y() > 0.05F, "subject component retained ground");
}

#endif

void test_quality_presets() {
    DensifyOptions options;
    apply_quality_preset(options, DensifyQuality::high);
    require(options.resolution_level == 0, "high preset is not full resolution");
    require(
        options.mesh_method == MeshMethod::delaunay_cut,
        "high preset does not select global meshing");
    require(options.min_patch_views == 3, "high preset patch support is weak");
    require(options.min_views_filter == 2, "high preset filter support is weak");
    require(options.min_views_fuse == 3, "high preset fusion support is weak");
    require(options.mask_border_px == 1, "high preset silhouette guard mismatch");
    require(
        options.grazing_weight_floor > 0.F &&
            options.grazing_weight_floor < 0.2F,
        "high preset grazing samples are not softly weighted");
    require(
        std::abs(options.mesh_k_behind - 1.F) < 1e-6F,
        "global mesh surface thickness is not one sigma");
    require(
        std::abs(options.mesh_dist_insert_px - 0.75F) < 1e-6F,
        "high preset global mesh spacing is too coarse");

    apply_quality_preset(options, DensifyQuality::default_quality);
    require(
        std::abs(options.mesh_dist_insert_px - 0.75F) < 1e-6F,
        "default preset global mesh spacing mismatch");

    apply_quality_preset(options, DensifyQuality::preview);
    require(options.resolution_level == 2, "preview preset resolution mismatch");
    require(
        options.mesh_method == MeshMethod::delaunay_cut,
        "preview preset does not select CGAL global meshing");
    require(
        std::abs(options.depth_diff_threshold - 0.01F) < 1e-6F,
        "preset application leaked high-quality thresholds");
}

#if !defined(PHOTARA_HAS_CGAL)
void test_missing_cgal_fails_before_densify() {
    MvsScene scene;
    DensifyOptions options;
    options.mesh_method = MeshMethod::delaunay_cut;
    bool rejected = false;
    try {
        densify(scene, options);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "missing CGAL did not reject global meshing early");
}
#endif

void test_scalable_maxflow_cut() {
    maxflow::Graph graph(2);
    graph.add_tweights(0, 5.F, 0.F);
    graph.add_tweights(1, 0.F, 5.F);
    graph.add_edge(0, 1, 2.F, 2.F);
    require(std::abs(graph.maxflow() - 2.F) < 1e-5F, "maxflow value mismatch");
    require(graph.is_source_side(0), "source-constrained node crossed cut");
    require(!graph.is_source_side(1), "sink-constrained node crossed cut");
    maxflow::Graph unary_overlap(1);
    unary_overlap.add_tweights(0, 7.F, 3.F);
    require(
        std::abs(unary_overlap.maxflow() - 3.F) < 1e-5F,
        "maxflow omitted the constant unary offset");

    struct Arc {
        int a{};
        int b{};
        float forward{};
        float reverse{};
    };
    std::mt19937 random(41);
    std::uniform_int_distribution<int> capacity(0, 5);
    for (int trial = 0; trial < 40; ++trial) {
        constexpr int nodes = 7;
        std::array<float, nodes> source{};
        std::array<float, nodes> sink{};
        std::vector<Arc> arcs;
        maxflow::Graph candidate(nodes);
        for (int i = 0; i < nodes; ++i) {
            source[static_cast<std::size_t>(i)] =
                static_cast<float>(capacity(random));
            sink[static_cast<std::size_t>(i)] =
                static_cast<float>(capacity(random));
            candidate.add_tweights(
                static_cast<std::size_t>(i),
                source[static_cast<std::size_t>(i)],
                sink[static_cast<std::size_t>(i)]);
        }
        for (int i = 0; i < nodes; ++i) {
            for (int j = i + 1; j < nodes; ++j) {
                if ((capacity(random) & 1) == 0) continue;
                const Arc arc{
                    i, j, static_cast<float>(capacity(random)),
                    static_cast<float>(capacity(random))};
                arcs.push_back(arc);
                candidate.add_edge(i, j, arc.forward, arc.reverse);
            }
        }
        const auto cut_energy = [&](const unsigned mask) {
            float energy = 0.F;
            for (int i = 0; i < nodes; ++i) {
                const bool source_side = (mask & (1U << i)) != 0;
                energy += source_side
                    ? sink[static_cast<std::size_t>(i)]
                    : source[static_cast<std::size_t>(i)];
            }
            for (const Arc& arc : arcs) {
                const bool a_source = (mask & (1U << arc.a)) != 0;
                const bool b_source = (mask & (1U << arc.b)) != 0;
                if (a_source && !b_source) energy += arc.forward;
                if (b_source && !a_source) energy += arc.reverse;
            }
            return energy;
        };
        float optimum = std::numeric_limits<float>::infinity();
        for (unsigned mask = 0; mask < (1U << nodes); ++mask)
            optimum = std::min(optimum, cut_energy(mask));
        candidate.maxflow();
        unsigned result_mask = 0;
        for (int i = 0; i < nodes; ++i)
            if (candidate.is_source_side(static_cast<std::size_t>(i)))
                result_mask |= 1U << i;
        require(
            std::abs(cut_energy(result_mask) - optimum) < 1e-4F,
            "push-relabel cut differs from brute-force optimum");
    }
}

void test_openmvs_energy_conventions() {
    const std::array<std::array<int, 3>, 4> expected{
        std::array<int, 3>{2, 1, 3}, std::array<int, 3>{2, 3, 0},
        std::array<int, 3>{0, 3, 1}, std::array<int, 3>{0, 1, 2}};
    for (int opposite = 0; opposite < 4; ++opposite) {
        require(
            detail::oriented_tetrahedron_facet_vertices(opposite) ==
                expected[static_cast<std::size_t>(opposite)],
            "Delaunay facet orientation differs from CGAL/OpenMVS");
    }

    detail::DepthEvidence evidence;
    evidence.positive_confidence = 1.F;
    evidence.negative_confidence = 1.1F;
    evidence.supporting_views = 2;
    evidence.conflicting_views = 1;
    require(
        !detail::accepts_depth_evidence(evidence, 1),
        "free-space conflict did not outweigh positive depth evidence");
    evidence.positive_confidence = 1.2F;
    require(
        detail::accepts_depth_evidence(evidence, 2),
        "consistent depth evidence was rejected");
    require(
        !detail::accepts_depth_evidence(evidence, 3),
        "depth evidence ignored the minimum support count");

    DensifyOptions options;
    require(
        std::abs(detail::weak_surface_sink_multiplier(1600.F, 100.F, options) -
                 1500.F) < 1e-5F,
        "weak-surface beta/gamma evidence was not reinforced");
    require(
        detail::weak_surface_sink_multiplier(900.F, 10.F, options) == 0.F,
        "weak-surface absolute threshold was ignored");
    require(
        detail::weak_surface_sink_multiplier(1600.F, 300.F, options) == 0.F,
        "weak-surface relative threshold was ignored");
    require(
        detail::weak_surface_sink_multiplier(6000.F, 500.F, options) == 0.F,
        "weak-surface outlier threshold was ignored");
    const float support_scale =
        detail::weak_surface_support_scale(1'500'000.F, options);
    require(
        std::abs(support_scale - 1500.F) < 1e-5F,
        "weak-surface support calibration did not preserve OpenMVS scale");
    require(
        std::abs(detail::weak_surface_sink_multiplier(
                     2'400'000.F / support_scale,
                     150'000.F / support_scale, options) -
                 1500.F) < 1e-5F,
        "calibrated weak-surface evidence was not reinforced");
}

void test_mesh_clean() {
    Mesh mesh;
    mesh.vertices = {
        Vec3f{0.F, 0.F, 0.F}, Vec3f{1.F, 0.F, 0.F},
        Vec3f{1.F, 1.F, 0.F}, Vec3f{0.F, 1.F, 0.F},
        Vec3f{5.F, 5.F, 5.F}};
    mesh.faces = {
        Eigen::Vector3i{0, 1, 2}, Eigen::Vector3i{0, 3, 2},
        Eigen::Vector3i{2, 1, 0}, Eigen::Vector3i{0, 0, 1}};
    DensifyOptions options;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 0;
    options.mesh_spurious_factor = 0.F;
    options.mesh_remove_spikes = false;
    detail::clean_mesh(mesh, options);
    require(mesh.faces.size() == 2, "mesh clean kept invalid faces");
    require(mesh.vertices.size() == 4, "mesh clean did not compact vertices");
    require(mesh.normals.size() == 4, "mesh clean did not rebuild normals");

    int shared_forward = 0;
    int shared_reverse = 0;
    for (const Eigen::Vector3i& face : mesh.faces) {
        for (int i = 0; i < 3; ++i) {
            const int a = face[i];
            const int b = face[(i + 1) % 3];
            if (a == 0 && b == 2) ++shared_forward;
            if (a == 2 && b == 0) ++shared_reverse;
        }
    }
    require(
        shared_forward == 1 && shared_reverse == 1,
        "mesh clean did not orient the shared edge consistently");
}

void test_subject_bounds_aware_mesh_clean() {
    Mesh mesh;
    mesh.vertices = {
        Vec3f{-0.2F, -0.2F, 0.F}, Vec3f{0.2F, -0.2F, 0.F},
        Vec3f{0.F, 0.2F, 0.F}, Vec3f{2.F, 0.F, 0.F},
        Vec3f{-0.2F, -0.2F, -0.2F}, Vec3f{0.2F, -0.2F, -0.2F},
        Vec3f{0.F, 0.2F, -0.2F}, Vec3f{0.F, 0.F, 0.2F}};
    mesh.faces = {
        Eigen::Vector3i{0,1,2}, Eigen::Vector3i{1,3,2},
        // Tetrahedron with the (4, 6, 5) face missing: this is an internal
        // reconstruction hole and must still close under SubjectBounds.
        Eigen::Vector3i{4,5,7}, Eigen::Vector3i{5,6,7},
        Eigen::Vector3i{6,4,7}};
    OrientedBoundingBox subject_bounds;
    subject_bounds.valid = true;
    subject_bounds.half_extent = Vec3f{0.5F,0.5F,0.5F};
    DensifyOptions options;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 8;
    options.mesh_spurious_factor = 0.F;
    options.mesh_remove_spikes = false;
    detail::clean_mesh(mesh, options, &subject_bounds);
    require(
        mesh.faces.size() == 7,
        "bounds-aware Clean confused a bounds cut with an internal hole");
    for (const Vec3f& vertex : mesh.vertices)
        require(
            subject_bounds.contains(vertex),
            "bounds-aware Clean kept outside vertex");
}

void test_bow_tie_holes_are_split_and_closed() {
    Mesh mesh;
    mesh.vertices = {
        Vec3f{0.F, 0.F, 0.F}, Vec3f{1.F, 0.F, 0.F},
        Vec3f{0.F, 1.F, 0.F}, Vec3f{0.F, 0.F, 1.F},
        Vec3f{-1.F, 0.F, 0.F}, Vec3f{0.F, -1.F, 0.F},
        Vec3f{0.F, 0.F, -1.F}};
    mesh.faces = {
        Eigen::Vector3i{0,3,1}, Eigen::Vector3i{1,3,2},
        Eigen::Vector3i{2,3,0}, Eigen::Vector3i{0,6,4},
        Eigen::Vector3i{4,6,5}, Eigen::Vector3i{5,6,0}};
    DensifyOptions options;
    options.mesh_method = MeshMethod::delaunay_cut;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 8;
    options.mesh_spurious_factor = 0.F;
    options.mesh_remove_spikes = false;
    detail::clean_mesh(mesh, options);
    require(
        mesh.faces.size() == 12,
        "bow-tie boundary fans were not split and closed independently");
    require(
        mesh.vertices.size() == 10,
        "bow-tie cleanup did not duplicate the shared vertex and add caps");
}

void test_sparse_tsdf_mesh() {
    MvsScene scene;
    MvsView view;
    view.id = 0;
    view.pose = photara::sfm::Pose3D::identity();
    view.width = view.height = 32;
    view.fx = view.fy = 32.F;
    view.cx = view.cy = 15.5F;
    view.depth_map.view_id = 0;
    view.depth_map.resize(view.width, view.height);
    std::fill(
        view.depth_map.depth.begin(), view.depth_map.depth.end(), 1.F);
    std::fill(
        view.depth_map.normal.begin(), view.depth_map.normal.end(),
        Vec3f{0.F, 0.F, -1.F});
    std::fill(
        view.depth_map.confidence.begin(), view.depth_map.confidence.end(),
        0.F);
    scene.views.push_back(std::move(view));

    DensifyOptions options;
    options.mesh_method = MeshMethod::tsdf;
    options.mesh_tsdf_voxel_size = 0.025F;
    options.mesh_tsdf_truncation_voxels = 3.F;
    options.mesh_tsdf_pixel_step = 1;
    options.mesh_tsdf_min_weight = 0.01F;
    options.mesh_tsdf_min_component_fraction = 0.F;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 0;
    MvsScene parallel_scene = scene;
    options.thread_count = 1;
    require(
        detail::reconstruct_mesh_tsdf(scene, options),
        "sparse TSDF rejected a valid depth plane");
    options.thread_count = 4;
    require(
        detail::reconstruct_mesh_tsdf(parallel_scene, options),
        "parallel sparse TSDF rejected a valid depth plane");
    require(
        scene.mesh.vertices.size() == parallel_scene.mesh.vertices.size() &&
            scene.mesh.faces.size() == parallel_scene.mesh.faces.size(),
        "parallel Marching Cubes changed mesh sizes");
    for (std::size_t index = 0; index < scene.mesh.vertices.size(); ++index) {
        const Vec3f& single = scene.mesh.vertices[index];
        const Vec3f& parallel = parallel_scene.mesh.vertices[index];
        require(
            single.x() == parallel.x() &&
                single.y() == parallel.y() &&
                single.z() == parallel.z(),
            "parallel Marching Cubes changed vertex order or position");
    }
    for (std::size_t index = 0; index < scene.mesh.faces.size(); ++index) {
        const Eigen::Vector3i& single = scene.mesh.faces[index];
        const Eigen::Vector3i& parallel = parallel_scene.mesh.faces[index];
        require(
            single.x() == parallel.x() &&
                single.y() == parallel.y() &&
                single.z() == parallel.z(),
            "parallel Marching Cubes changed face order or indices");
    }
    detail::clean_mesh(scene.mesh, options);
    require(!scene.mesh.faces.empty(), "sparse TSDF plane mesh is empty");
    const auto [minimum, maximum] = std::minmax_element(
        scene.mesh.vertices.begin(), scene.mesh.vertices.end(),
        [](const Vec3f& a, const Vec3f& b) { return a.z() < b.z(); });
    require(
        minimum != scene.mesh.vertices.end() &&
            std::abs(minimum->z() - 1.F) < 0.03F &&
            std::abs(maximum->z() - 1.F) < 0.03F,
        "sparse TSDF zero crossing does not match the input depth");
}

std::size_t boundary_edge_count(const Mesh& mesh) {
    std::map<std::pair<int, int>, unsigned> edge_counts;
    for (const Eigen::Vector3i& face : mesh.faces) {
        for (int edge = 0; edge < 3; ++edge) {
            int a = face[edge];
            int b = face[(edge + 1) % 3];
            if (a > b) std::swap(a, b);
            ++edge_counts[{a, b}];
        }
    }
    return static_cast<std::size_t>(std::count_if(
        edge_counts.begin(), edge_counts.end(),
        [](const auto& item) { return item.second == 1; }));
}

void test_tsdf_support_closing_repairs_internal_one_voxel_gap() {
    const auto make_scene = [](const bool internal_gap) {
        MvsScene scene;
        MvsView view;
        view.id = 0;
        view.pose = photara::sfm::Pose3D::identity();
        view.width = view.height = 64;
        view.fx = view.fy = 64.F;
        view.cx = view.cy = 31.5F;
        view.depth_map.view_id = 0;
        view.depth_map.resize(view.width, view.height);
        std::fill(
            view.depth_map.depth.begin(), view.depth_map.depth.end(), 1.F);
        std::fill(
            view.depth_map.normal.begin(), view.depth_map.normal.end(),
            Vec3f{0.F, 0.F, -1.F});
        std::fill(
            view.depth_map.confidence.begin(),
            view.depth_map.confidence.end(), 0.F);
        if (internal_gap) {
            // A single missing depth ray creates an internal zero-weight
            // support column while surrounding surface samples remain valid.
            view.depth_map.depth[
                view.depth_map.index(32, 32)] = 0.F;
        }
        scene.views.push_back(std::move(view));
        return scene;
    };

    DensifyOptions options;
    options.mesh_method = MeshMethod::tsdf;
    options.mesh_tsdf_voxel_size = 0.0125F;
    options.mesh_tsdf_truncation_voxels = 3.F;
    options.mesh_tsdf_min_weight = 0.01F;
    options.mesh_tsdf_min_component_fraction = 0.F;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 0;
    options.mesh_tsdf_support_closing_axes = 0;

    MvsScene open_scene = make_scene(true);
    require(
        detail::reconstruct_mesh_tsdf(open_scene, options),
        "TSDF without support closing rejected the test plane");
    const std::size_t open_boundary_edges =
        boundary_edge_count(open_scene.mesh);

    options.mesh_tsdf_support_closing_axes = 2;
    MvsScene closed_scene = make_scene(true);
    require(
        detail::reconstruct_mesh_tsdf(closed_scene, options),
        "TSDF support closing rejected the test plane");
    const std::size_t closed_boundary_edges =
        boundary_edge_count(closed_scene.mesh);
    require(
        closed_scene.mesh.faces.size() > open_scene.mesh.faces.size(),
        "TSDF support closing did not restore missing surface faces");
    require(
        closed_boundary_edges < open_boundary_edges,
        "TSDF support closing did not reduce the internal gap boundary");

    options.mesh_tsdf_support_closing_axes = 0;
    MvsScene open_silhouette = make_scene(false);
    require(
        detail::reconstruct_mesh_tsdf(open_silhouette, options),
        "TSDF without support closing rejected the intact plane");
    options.mesh_tsdf_support_closing_axes = 2;
    MvsScene closed_silhouette = make_scene(false);
    require(
        detail::reconstruct_mesh_tsdf(closed_silhouette, options),
        "TSDF support closing rejected the intact plane");
    require(
        closed_silhouette.mesh.faces.size() ==
            open_silhouette.mesh.faces.size() &&
        closed_silhouette.mesh.vertices.size() ==
            open_silhouette.mesh.vertices.size(),
        "TSDF support closing grew an open silhouette");
}

void test_imported_scene_resolution() {
    MvsScene scene;
    MvsView view;
    view.width = view.src_width = 1000;
    view.height = view.src_height = 800;
    view.fx = view.src_fx = 900.F;
    view.fy = view.src_fy = 880.F;
    view.cx = view.src_cx = 500.F;
    view.cy = view.src_cy = 400.F;
    view.pose.C = photara::sfm::Vec3{1., 2., 3.};
    scene.views.push_back(view);
    DensifyOptions options;
    prepare_imported_scene(scene, options);
    const auto& resized = scene.views.front();
    require(resized.width == 640 && resized.height == 512,
        "imported cameras ignored working-resolution preset");
    require(std::abs(resized.fx - 576.F) < 1e-4F &&
            std::abs(resized.cx - 320.F) < 1e-4F &&
            (resized.pose.C - view.pose.C).norm() == 0. && resized.src_fx == view.src_fx,
        "imported camera resize changed pose/source calibration");
    prepare_imported_scene(scene, options);
    require(scene.views.front().width == 640 && std::abs(scene.views.front().fx - 576.F) < 1e-4F,
        "imported camera preparation is not idempotent");
}

void test_dense_ply_round_trip() {
    DenseCloud source;
    DensePoint point;
    point.position = Vec3f{1.25F, -2.5F, 3.75F};
    point.normal = Vec3f{0.F, 1.F, 0.F};
    point.color = Vec3f{0.2F, 0.4F, 0.8F};
    point.weight = 3.5F;
    point.views = {0, 7, 300};
    point.view_weights = {0.5F, 1.F, 2.F};
    source.points.push_back(point);
    const auto path = std::filesystem::temp_directory_path() /
                      "photara_dense_ply_round_trip.ply";
    save_dense_ply(source, path);
    const DenseCloud loaded = load_dense_ply(path);
    std::filesystem::remove(path);
    require(loaded.points.size() == 1, "dense PLY loader lost a point");
    require(loaded.points[0].views == point.views &&
            loaded.points[0].view_weights == point.view_weights &&
            loaded.points[0].weight == point.weight,
            "dense PLY round trip lost graph-cut visibility weights");
    require(
        (loaded.points[0].position - point.position).norm() < 1e-6F,
        "dense PLY loader changed point position");
    require(
        (loaded.points[0].normal - point.normal).norm() < 1e-6F,
        "dense PLY loader changed point normal");
    require(
        (loaded.points[0].color - point.color).cwiseAbs().maxCoeff() < 0.005F,
        "dense PLY loader changed point color");
}

void test_tsdf_holes_use_boundary_triangulation() {
    Mesh mesh;
    mesh.vertices = {
        Vec3f{0.F, 0.F, 0.F}, Vec3f{1.F, 0.F, 0.F},
        Vec3f{0.F, 1.F, 0.F}, Vec3f{0.F, 0.F, 1.F}};
    // Tetrahedron with one triangular face missing.
    mesh.faces = {
        Eigen::Vector3i{0, 1, 3}, Eigen::Vector3i{1, 2, 3},
        Eigen::Vector3i{2, 0, 3}};
    DensifyOptions options;
    options.mesh_method = MeshMethod::tsdf;
    options.mesh_min_component_faces = 1;
    options.mesh_tsdf_min_component_fraction = 0.F;
    options.mesh_close_hole_edges = 8;
    options.mesh_tsdf_smooth_iters = 0;
    detail::clean_mesh(mesh, options);
    require(
        mesh.faces.size() == 4,
        "TSDF boundary triangulation did not close a triangular hole");
    require(
        mesh.vertices.size() == 4,
        "TSDF boundary triangulation introduced a center-fan vertex");
}

void test_mesh_ply_round_trip() {
    Mesh source;
    source.vertices = {
        Vec3f{0.F, 0.F, 0.F},
        Vec3f{1.F, 0.F, 0.F},
        Vec3f{0.F, 1.F, 0.F}};
    source.normals.assign(3, Vec3f{0.F, 0.F, 1.F});
    source.colors = {
        Vec3f{1.F, 0.F, 0.F},
        Vec3f{0.F, 1.F, 0.F},
        Vec3f{0.F, 0.F, 1.F}};
    source.faces.emplace_back(0, 1, 2);
    const auto path = std::filesystem::temp_directory_path() /
                      "photara_mesh_ply_round_trip.ply";
    save_mesh_ply(source, path);
    const Mesh loaded = load_mesh_ply(path);
    std::filesystem::remove(path);
    require(loaded.vertices.size() == 3, "mesh PLY loader lost vertices");
    require(loaded.faces.size() == 1, "mesh PLY loader lost a face");
    require(
        loaded.faces.front() == Eigen::Vector3i(0, 1, 2),
        "mesh PLY loader changed face indices");
    require(
        loaded.normals.size() == source.normals.size(),
        "mesh PLY loader lost normals");
    require(
        loaded.colors.size() == source.colors.size(),
        "mesh PLY loader lost colors");
}

std::string glb_json_chunk(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(static_cast<bool>(in), "failed to open exported GLB");
    std::uint32_t header[5]{};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    require(static_cast<bool>(in), "truncated GLB header");
    require(header[0] == 0x46546c67U, "GLB magic is wrong");
    require(header[1] == 2U, "GLB version is not 2.0");
    require(header[4] == 0x4e4f534aU, "GLB JSON chunk type is wrong");
    std::string json(header[3], '\0');
    in.read(json.data(), static_cast<std::streamsize>(header[3]));
    require(static_cast<bool>(in), "truncated GLB JSON chunk");
    return json;
}

void test_mesh_glb_export() {
    Mesh source;
    source.vertices = {
        Vec3f{0.F, 0.F, 0.F},
        Vec3f{1.F, 0.F, 0.F},
        Vec3f{0.F, 1.F, 0.F}};
    source.normals.assign(3, Vec3f{0.F, 0.F, 1.F});
    source.colors = {
        Vec3f{1.F, 0.F, 0.F},
        Vec3f{0.F, 1.F, 0.F},
        Vec3f{0.F, 0.F, 1.F}};
    source.faces.emplace_back(0, 1, 2);
    const auto geometry = std::filesystem::temp_directory_path() /
                          "photara_mesh_glb_geometry.glb";
    photara::io::save_mesh_glb(source, geometry);
    const std::string geometry_json = glb_json_chunk(geometry);
    std::filesystem::remove(geometry);
    require(
        geometry_json.find("\"POSITION\"") != std::string::npos,
        "geometry GLB is missing POSITION");
    require(
        geometry_json.find("\"NORMAL\"") != std::string::npos,
        "geometry GLB is missing NORMAL");
    require(
        geometry_json.find("\"COLOR_0\"") != std::string::npos,
        "geometry GLB is missing COLOR_0");
    require(
        geometry_json.find("image/png") == std::string::npos,
        "geometry GLB should not embed an image");

    photara::io::RgbImage atlas;
    atlas.width = 1;
    atlas.height = 1;
    atlas.pixels = {255, 128, 64};
    const auto png_path = std::filesystem::temp_directory_path() /
                          "photara_mesh_glb_atlas.png";
    photara::io::save_rgb_png(atlas, png_path);
    std::ifstream png_in(png_path, std::ios::binary);
    require(static_cast<bool>(png_in), "failed to read test atlas PNG");
    const std::vector<std::uint8_t> png(
        (std::istreambuf_iterator<char>(png_in)),
        std::istreambuf_iterator<char>());
    png_in.close();
    std::filesystem::remove(png_path);
    const std::array<Vec2f, 3> uvs{
        Vec2f{0.F, 0.F}, Vec2f{1.F, 0.F}, Vec2f{0.F, 1.F}};
    const auto textured = std::filesystem::temp_directory_path() /
                          "photara_mesh_glb_textured.glb";
    photara::io::save_mesh_glb(source, textured, uvs, png);
    const std::string textured_json = glb_json_chunk(textured);
    std::filesystem::remove(textured);
    require(
        textured_json.find("\"TEXCOORD_0\"") != std::string::npos,
        "textured GLB is missing TEXCOORD_0");
    require(
        textured_json.find("image/png") != std::string::npos,
        "textured GLB is missing an embedded PNG");
    require(
        textured_json.find("KHR_materials_unlit") != std::string::npos,
        "textured GLB should use an unlit albedo material");
}

#if defined(PHOTARA_HAS_CGAL)
void test_global_delaunay_mesh() {
    MvsScene scene;
    const std::array<Vec3f, 6> cameras{
        Vec3f{3.F, 0.F, 0.F}, Vec3f{-3.F, 0.F, 0.F},
        Vec3f{0.F, 3.F, 0.F}, Vec3f{0.F, -3.F, 0.F},
        Vec3f{0.F, 0.F, 3.F}, Vec3f{0.F, 0.F, -3.F}};
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        MvsView view;
        view.id = static_cast<Index>(i);
        view.pose = photara::sfm::Pose3D::identity();
        view.pose.C = cameras[i].cast<double>();
        view.fx = view.fy = 500.F;
        scene.views.push_back(std::move(view));
    }
    for (int latitude = 1; latitude < 12; ++latitude) {
        const float phi = std::numbers::pi_v<float> *
                          static_cast<float>(latitude) / 12.F;
        for (int longitude = 0; longitude < 24; ++longitude) {
            const float theta = 2.F * std::numbers::pi_v<float> *
                                static_cast<float>(longitude) / 24.F;
            DensePoint point;
            point.position = Vec3f{
                std::sin(phi) * std::cos(theta),
                std::sin(phi) * std::sin(theta), std::cos(phi)};
            point.normal = point.position;
            point.color = (point.position.array() * 0.5F + 0.5F).matrix();
            point.weight = 1.F;
            for (std::size_t i = 0; i < cameras.size(); ++i)
                if (cameras[i].dot(point.position) > 0.F)
                    point.views.push_back(static_cast<Index>(i));
            scene.dense_cloud.points.push_back(std::move(point));
        }
    }
    DensifyOptions options;
    options.mesh_method = MeshMethod::delaunay_cut;
    options.mesh_max_points = 0;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 0;
    options.mesh_k_inf = 1.0e4F;
    options.mesh_k_qual = 0.05F;
    options.mesh_k_behind = 1.F;
    // Disable the unsupported-webbing gate for the synthetic closed fixture;
    // its behavior is covered by real-scene topology/visual regression.
    options.mesh_max_edge_scale = 0.F;
    require(
        detail::reconstruct_mesh_global_cgal(scene, options),
        "global Delaunay backend rejected the sphere");
    std::map<std::pair<int, int>, unsigned> edge_counts;
    double signed_volume = 0.;
    for (const Eigen::Vector3i& face : scene.mesh.faces) {
        const auto a = scene.mesh.vertices[face[0]].cast<double>().eval();
        const auto b = scene.mesh.vertices[face[1]].cast<double>().eval();
        const auto c = scene.mesh.vertices[face[2]].cast<double>().eval();
        signed_volume += a.dot(b.cross(c)) / 6.;
        for (int edge = 0; edge < 3; ++edge) {
            int a = face[edge];
            int b = face[(edge + 1) % 3];
            if (a > b) std::swap(a, b);
            ++edge_counts[{a, b}];
        }
    }
    require(
        std::none_of(
            edge_counts.begin(), edge_counts.end(),
            [](const auto& item) { return item.second == 1; }),
        "global graph-cut surface was punctured by facet filtering");
    require(signed_volume > 0., "graph-cut convex-hull facets face inward");
    detail::clean_mesh(scene.mesh, options);
    require(!scene.mesh.faces.empty(), "global Delaunay mesh is empty");
    require(
        scene.mesh.normals.size() == scene.mesh.vertices.size(),
        "global Delaunay mesh normals are incomplete");

    // The graph-cut energy must be invariant to the large world-coordinate
    // offsets commonly found in georeferenced projects. This used to feed
    // large absolute floats directly into all distance calculations.
    MvsScene shifted = scene;
    shifted.mesh = {};
    const Vec3f offset{1.0e6F, -2.0e6F, 3.0e6F};
    constexpr float object_scale = 10.F;
    for (std::size_t i = 0; i < shifted.views.size(); ++i)
        shifted.views[i].pose.C =
            (offset + cameras[i] * object_scale).cast<double>();
    for (DensePoint& point : shifted.dense_cloud.points)
        point.position = offset + point.position * object_scale;
    require(
        detail::reconstruct_mesh_global_cgal(shifted, options),
        "canonical-coordinate Delaunay backend rejected a shifted scene");
    require(
        !shifted.mesh.faces.empty(),
        "canonical-coordinate Delaunay backend produced no shifted surface");
    Vec3f shifted_center = Vec3f::Zero();
    for (const Vec3f& vertex : shifted.mesh.vertices)
        shifted_center += vertex;
    shifted_center /= static_cast<float>(shifted.mesh.vertices.size());
    require(
        (shifted_center - offset).norm() < object_scale,
        "canonical-coordinate Delaunay output did not map back to world space");
}
#endif

}  // namespace

int main() {
    try {
        test_parallel_fusion();
        test_build_mvs_scene_samples_sparse_colors();
        test_mask_constrained_fusion();
        test_subject_bounds_from_sparse_points();
        test_aether_texture_camera_projection();
        test_quality_presets();
#if !defined(PHOTARA_HAS_CGAL)
        test_missing_cgal_fails_before_densify();
#endif
        test_scalable_maxflow_cut();
        test_openmvs_energy_conventions();
        test_mesh_clean();
        test_subject_bounds_aware_mesh_clean();
        test_bow_tie_holes_are_split_and_closed();
        test_tsdf_holes_use_boundary_triangulation();
        test_sparse_tsdf_mesh();
        test_tsdf_support_closing_repairs_internal_one_voxel_gap();
        test_dense_ply_round_trip();
        test_imported_scene_resolution();
        test_mesh_ply_round_trip();
        test_mesh_glb_export();
#if defined(PHOTARA_HAS_CGAL)
        test_global_delaunay_mesh();
#endif
        std::cout << "mvs tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mvs test failed: " << error.what() << '\n';
        return 1;
    }
}
