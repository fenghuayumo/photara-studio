#pragma once

#include "image_qa_projection.hpp"
#include "io/image.hpp"
#include "mvs/densify.hpp"
#include "sfm/asfm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

inline void test_studio_image_qa_projection() {
    using namespace photara;
    const auto check = [](const bool value, const char* message) {
        if (!value) throw std::runtime_error(message);
    };
    const auto root = std::filesystem::temp_directory_path() /
        ("photara-qa-projection-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root / "masks");
    struct Cleanup {
        std::filesystem::path root;
        ~Cleanup() { std::error_code ec; std::filesystem::remove_all(root, ec); }
    } cleanup{root};
    const auto path = root / "view.png";
    io::RgbImage source{64, 48, std::vector<std::uint8_t>(64 * 48 * 3)};
    io::RgbImage mask{64, 48, std::vector<std::uint8_t>(64 * 48 * 3)};
    for (unsigned y = 0; y < 48; ++y) for (unsigned x = 0; x < 64; ++x) {
        const auto i = (y * 64 + x) * 3;
        source.pixels[i] = static_cast<std::uint8_t>(3 * x);
        source.pixels[i + 1] = static_cast<std::uint8_t>(4 * y);
        source.pixels[i + 2] = static_cast<std::uint8_t>(x + y);
        for (unsigned c = 0; c < 3; ++c) mask.pixels[i + c] = x >= 32 ? 255 : 0;
    }
    io::save_rgb_png(source, path);
    io::save_rgb_png(mask, root / "masks/view.png");
    editor::ViewPose pose;
    pose.registered = true;
    pose.width = 64; pose.height = 48;
    pose.fx = pose.fy = 40.F; pose.cx = 32.F; pose.cy = 24.F;
    auto options = editor::image_qa_training_options(false, {}, false);
    auto view = editor::image_qa_training_view(pose, path);
    const auto original = splat::training_data::load_training_image(view, options);
    check(original.rgb.pixels == source.pixels && original.mask.pixels.empty(),
        "Zero-distortion QA must preserve the source pixels exactly");

    pose.k1 = .2F; pose.k2 = -.01F; pose.k3 = .003F; pose.k4 = -.002F;
    view = editor::image_qa_training_view(pose, path);
    const auto corrected = splat::training_data::load_training_image(view, options);
    check(corrected.rgb.pixels != source.pixels,
        "Distorted pinhole QA must compare against corrected pixels");
    check(corrected.camera.model == CameraModel::pinhole && corrected.camera.k1 == 0.F,
        "Corrected pixels must use an ideal pinhole render camera");
    check(corrected.mask.pixels.size() == 64 * 48 && corrected.mask.pixels[0] == 0 &&
          corrected.mask.pixels[24 * 64 + 32] == 255,
        "Undistortion outside the capture must be excluded from QA");
    // Independent Brown formula and a linear RGB ramp make the expected
    // resampled pixel known without calling the production projection helper.
    constexpr unsigned x = 49, y = 31;
    const double xn = (x - 32.) / 40., yn = (y - 24.) / 40.;
    const double r2 = xn * xn + yn * yn;
    const double radial = 1 + .2 * r2 - .01 * r2 * r2;
    const double sx = 40 * (xn * radial + 2 * .003 * xn * yn - .002 * (r2 + 2 * xn * xn)) + 32;
    const double sy = 40 * (yn * radial + .003 * (r2 + 2 * yn * yn) - 2 * .002 * xn * yn) + 24;
    check(std::abs(int(corrected.rgb.pixels[(y * 64 + x) * 3]) - int(std::lround(3 * sx))) <= 1 &&
          std::abs(int(corrected.rgb.pixels[(y * 64 + x) * 3 + 1]) - int(std::lround(4 * sy))) <= 1,
        "QA RGB must sample the original photo at the Brown-distorted coordinate");
    auto masked_options = options;
    masked_options.use_mask = true;
    masked_options.mask_dir = root / "masks";
    const auto masked = splat::training_data::load_training_image(view, masked_options);
    check(masked.rgb.pixels == corrected.rgb.pixels && masked.mask.pixels[24 * 64 + 16] == 0 &&
          masked.mask.pixels[24 * 64 + 48] == 255 && masked.mask.pixels[0] == 0,
        "Source masks must be reprojected with RGB, including missing border pixels");

    const auto key = editor::image_qa_projection_key(view, options);
    view.k1 = std::nextafter(view.k1, 1.F);
    check(key != editor::image_qa_projection_key(view, options),
        "Even a small calibration update must invalidate cached QA pixels");
    view.source_model = CameraModel::opencv_fisheye;
    const auto native = splat::training_data::load_training_image(view, options);
    check(native.rgb.pixels == source.pixels && native.camera.model == CameraModel::opencv_fisheye &&
          native.camera.k1 == view.k1,
        "Native fisheye QA must retain source pixels and the distorted render camera");
    const auto native_key = editor::image_qa_projection_key(view, options);
    options.undistort_to_pinhole = true;
    const auto fish_pinhole = splat::training_data::load_training_image(view, options);
    check(fish_pinhole.camera.model == CameraModel::pinhole && fish_pinhole.rgb.pixels != source.pixels &&
          native_key != editor::image_qa_projection_key(view, options),
        "Fisheye training in pinhole mode must change both QA pixels and camera/cache");
    options.undistort_to_pinhole = false;
    view.source_model = CameraModel::equirectangular;
    const auto pano = splat::training_data::load_training_image(view, options);
    check(pano.rgb.pixels == source.pixels && pano.camera.model == CameraModel::equirectangular,
        "Panorama QA must retain the native image projection");

    pose.src_width = 2560; pose.src_height = 1440;
    pose.src_fx = pose.src_fy = 1700.F; pose.src_cx = 1280.F; pose.src_cy = 720.F;
    pose.width = 640; pose.height = 360;
    pose.centre = {1.F, 2.F, 3.F};
    view = editor::image_qa_training_view(pose, path);
    const auto camera = splat::training_data::training_camera(view, options);
    check(camera.width == 1600 && camera.height == 900 && camera.fx == 1062.5F &&
          std::abs(camera.cx - 799.8125F) < 1e-5F && camera.world_to_camera[12] == -1.F &&
          camera.world_to_camera[13] == -2.F && camera.world_to_camera[14] == -3.F,
        "Imported source intrinsics, pixel-center resizing and pose must reach the QA camera");
}

// Optional local-dataset check: native SfM -> Studio view metadata -> QA must
// retain the exact camera/pixels/mask produced by the trainer's MVS input path.
inline void test_studio_image_qa_dataset(
    const std::filesystem::path& sfm, const std::filesystem::path& images) {
    photara::sfm::AsfmOptions load;
    load.path_base = images;
    const auto source = photara::sfm::load_asfm(sfm, load);
    const auto editor_scene = editor::sparse_scene_from_sfm(source, false);
    if (!editor_scene.ok) throw std::runtime_error("Cannot load Studio camera metadata");
    const auto training_scene = photara::mvs::build_mvs_scene(source, {});
    const auto options = editor::image_qa_training_options(false, {}, false);
    std::size_t checked = 0;
    for (const auto& view : training_scene.views) {
        const auto& pose = editor_scene.scene.views.at(view.sfm_image_id);
        const auto reference = photara::splat::training_data::load_training_image(view, options);
        const auto qa = photara::splat::training_data::load_training_image(
            editor::image_qa_training_view(pose, view.path), options);
        const auto& a = reference.camera;
        const auto& b = qa.camera;
        const auto scalar_close = [](float x, float y) {
            return std::abs(x - y) <= 1e-5F * std::max({1.F, std::abs(x), std::abs(y)});
        };
        // Studio stores display poses as floats; allow their rounding while
        // requiring identical projection intrinsics and reference pixels.
        const bool same_camera = a.width == b.width && a.height == b.height && a.model == b.model &&
            a.fx == b.fx && a.fy == b.fy && a.cx == b.cx && a.cy == b.cy &&
            a.k1 == b.k1 && a.k2 == b.k2 && a.k3 == b.k3 && a.k4 == b.k4 &&
            std::equal(a.world_to_camera.begin(), a.world_to_camera.end(), b.world_to_camera.begin(), scalar_close) &&
            std::equal(a.position.begin(), a.position.end(), b.position.begin(), scalar_close);
        if (!same_camera || reference.rgb.pixels != qa.rgb.pixels || reference.mask.pixels != qa.mask.pixels)
            throw std::runtime_error("Studio/training projection mismatch: " + view.path.string());
        ++checked;
    }
    if (checked == 0) throw std::runtime_error("No registered QA dataset views");
    std::cout << "QA dataset views=" << checked << " cameras match; RGB/masks identical to training\n";
}
