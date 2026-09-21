// Manual verification tool for equirectangular (360) alignment.
//
// Runs the full reconstruction on a folder of panoramas and writes the result
// as a COLMAP text model, so the poses can be cross-checked against an
// independent reconstruction of the same capture (see
// scripts/compare_equirect_reference.py).
//
//   photara_equirect_dataset_check <images> <out_colmap_dir>
//       [--stride N] [--limit N] [--mode global|incremental|hierarchical]
//       [--camera-model auto|equirectangular|pinhole|opencv_fisheye]
//       [--max-features N] [--window N]
#include "core/logging.hpp"
#include "sfm/export_colmap.hpp"
#include "sfm/reconstruct.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool is_image(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
           extension == ".tif" || extension == ".tiff" || extension == ".bmp";
}

struct Options {
    std::filesystem::path images;
    std::filesystem::path output;
    std::size_t stride{1};
    std::size_t limit{0};
    std::string mode{"global"};
    std::string camera_model{"equirectangular"};
    unsigned max_features{8000};
    std::size_t window{0};
};

bool parse_options(const int argc, char** argv, Options& options) {
    if (argc < 3) return false;
    options.images = argv[1];
    options.output = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto value = [&](std::string& target) {
            if (i + 1 >= argc) return false;
            target = argv[++i];
            return true;
        };
        if (flag == "--stride") {
            std::string text;
            if (!value(text)) return false;
            options.stride = std::max<std::size_t>(1, std::stoul(text));
        } else if (flag == "--limit") {
            std::string text;
            if (!value(text)) return false;
            options.limit = std::stoul(text);
        } else if (flag == "--mode") {
            if (!value(options.mode)) return false;
        } else if (flag == "--camera-model") {
            if (!value(options.camera_model)) return false;
        } else if (flag == "--max-features") {
            std::string text;
            if (!value(text)) return false;
            options.max_features = std::stoul(text);
        } else if (flag == "--window") {
            std::string text;
            if (!value(text)) return false;
            options.window = std::stoul(text);
        } else {
            std::cerr << "unknown flag: " << flag << '\n';
            return false;
        }
    }
    return true;
}

}  // namespace

int main(const int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        std::cerr << "usage: equirect_dataset_check <images> <out_colmap_dir> "
                     "[--stride N] [--limit N] [--mode M] [--camera-model M] "
                     "[--max-features N] [--window N]\n";
        return 2;
    }

    std::vector<std::filesystem::path> files;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(
             options.images, error)) {
        if (!entry.is_regular_file() || !is_image(entry.path())) continue;
        files.push_back(entry.path());
    }
    if (error) {
        std::cerr << "cannot read " << options.images << ": " << error.message()
                  << '\n';
        return 2;
    }
    std::sort(files.begin(), files.end());
    if (options.stride > 1) {
        std::vector<std::filesystem::path> sampled;
        for (std::size_t i = 0; i < files.size(); i += options.stride)
            sampled.push_back(files[i]);
        files = std::move(sampled);
    }
    if (options.limit > 0 && files.size() > options.limit)
        files.resize(options.limit);
    if (files.size() < 2) {
        std::cerr << "need at least two images, found " << files.size() << '\n';
        return 2;
    }

    photara::sfm::ReconstructionConfig config;
    if (options.mode == "incremental")
        config.mode = photara::sfm::ReconstructionMode::incremental;
    else if (options.mode == "hierarchical")
        config.mode = photara::sfm::ReconstructionMode::hierarchical;
    else
        config.mode = photara::sfm::ReconstructionMode::global;
    config.frontend.camera_model =
        photara::CameraModel::equirectangular;
    if (options.camera_model == "auto" || options.camera_model == "automatic")
        config.frontend.camera_model = photara::CameraModel::automatic;
    else if (options.camera_model == "pinhole")
        config.frontend.camera_model = photara::CameraModel::pinhole;
    else if (options.camera_model == "fisheye" ||
             options.camera_model == "opencv_fisheye")
        config.frontend.camera_model = photara::CameraModel::opencv_fisheye;
    config.frontend.max_features = options.max_features;
    config.frontend.neighbor_window = options.window;

    std::cout << "images=" << files.size() << " mode=" << options.mode
              << " camera_model=" << options.camera_model << '\n';

    // Verification tool: keep the calibration diagnostics visible.
    photara::core::Logger::instance().set_levels(
        photara::core::LogLevel::info, photara::core::LogLevel::off);

    photara::sfm::Scene scene;
    const auto started = std::chrono::steady_clock::now();
    const auto summary =
        photara::sfm::reconstruct(scene, files, config);
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();

    if (!summary.valid) {
        std::cerr << "reconstruction failed after " << seconds << " s\n";
        return 1;
    }

    unsigned equirectangular_cameras = 0;
    for (const auto& camera : scene.cameras)
        if (camera.model == photara::CameraModel::equirectangular)
            ++equirectangular_cameras;

    std::cout << "registered=" << summary.registered_views << "/"
              << files.size() << " landmarks=" << summary.landmarks
              << " mean_reproj_px=" << summary.mean_reprojection_error_pixels
              << " rms_reproj_px=" << summary.rms_reprojection_error_pixels
              << " equirect_cameras=" << equirectangular_cameras << "/"
              << scene.cameras.size() << " seconds=" << seconds << '\n';

    if (!options.output.empty()) {
        photara::sfm::save_colmap_text(
            scene, options.output, options.images, true);
        std::cout << "colmap_model=" << options.output.string() << '\n';
    }
    return 0;
}
