#include "aetherscan/sfm/mapping.hpp"
#include "aetherscan/sfm/frontend.hpp"
#include "aetherscan/sfm/export_mvs.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double number(const char* text) {
    double value = 0;
    const std::string input = text;
    const auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    if (result.ec != std::errc{} || value <= 0)
        throw std::invalid_argument("Invalid positive number: " + input);
    return value;
}

std::string lower_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

void save_ply(const aetherscan::sfm::Scene& scene, const std::filesystem::path& path) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Failed to create PLY: " + path.string());
    output << "ply\nformat ascii 1.0\nelement vertex " << scene.landmarks.size()
           << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto& landmark : scene.landmarks)
        output << landmark.position[0] << ' ' << landmark.position[1] << ' '
               << landmark.position[2] << '\n';
}

void save_scene(
    const aetherscan::sfm::Scene& scene,
    const std::filesystem::path& path,
    const std::filesystem::path& /*image_directory*/) {
    const std::string extension = lower_extension(path);
    if (extension == ".mvs") {
        // Absolute image paths so OpenMVS Viewer resolves them from any cwd.
        aetherscan::sfm::export_openmvs_interface(scene, path);
        return;
    }
    if (extension == ".ply") {
        save_ply(scene, path);
        return;
    }
    throw std::invalid_argument("Output must end with .mvs or .ply");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 6) {
            std::cout << "Usage: aetherscan images_dir focal_pixels "
                         "incremental|global output.(mvs|ply) [neighbor_window]\n"
                         "  .mvs  OpenMVS Interface (open in Viewer)\n"
                         "  .ply  sparse XYZ point cloud\n";
            return argc == 1 ? 0 : 1;
        }
        const std::filesystem::path directory = argv[1];
        const std::filesystem::path output_path = argv[4];
        const double focal = number(argv[2]);
        const std::string mode = argv[3];
        const std::size_t window =
            argc == 6 ? static_cast<std::size_t>(number(argv[5])) : 3;
        if (mode != "incremental" && mode != "global")
            throw std::invalid_argument("Mode must be incremental or global");

        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;
            std::string extension = entry.path().extension().string();
            std::transform(
                extension.begin(), extension.end(), extension.begin(),
                [](const unsigned char value) {
                    return static_cast<char>(std::tolower(value));
                });
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
                extension == ".tif" || extension == ".tiff")
                files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
        if (files.size() < 2) throw std::runtime_error("Need at least two images");

        const auto started = std::chrono::steady_clock::now();
        aetherscan::sfm::FrontEndOptions frontend_options;
        frontend_options.focal_pixels = focal;
        frontend_options.neighbor_window = window;
        auto frontend = aetherscan::sfm::run_frontend(files, frontend_options);
        auto& scene = frontend.scene;
        std::cout << "threads=" << frontend.timing.threads_used
                  << " extract_s=" << frontend.timing.extract_seconds
                  << " match_verify_s=" << frontend.timing.match_verify_seconds
                  << " tracks_s=" << frontend.timing.tracks_seconds
                  << " verified_pairs=" << frontend.edges.size()
                  << " tracks=" << scene.tracks.size() << '\n';

        aetherscan::sfm::MapperSummary summary;
        const auto mapping_started = std::chrono::steady_clock::now();
        if (mode == "incremental") {
            aetherscan::sfm::IncrementalMapperOptions options;
            options.bundle.optimizer.maximum_iterations = 8;
            summary = aetherscan::sfm::run_incremental_mapping(
                scene, frontend.edges[frontend.seed_edge_index], options);
        } else {
            aetherscan::sfm::GlobalMapperOptions options;
            options.bundle.maximum_iterations = 12;
            summary = aetherscan::sfm::run_global_mapping(scene, frontend.edges, options);
        }
        const double mapping_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - mapping_started)
                .count();

        save_scene(scene, output_path, directory);
        // Convenience: also emit companion .mvs next to .ply for OpenMVS Viewer.
        if (lower_extension(output_path) == ".ply") {
            const auto mvs_path = output_path.parent_path() / (output_path.stem().string() + ".mvs");
            aetherscan::sfm::export_openmvs_interface(scene, mvs_path);
            std::cout << "mvs=" << mvs_path << '\n';
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << "mode=" << mode << " valid=" << summary.valid
                  << " registered=" << summary.registered_views << '/' << scene.views.size()
                  << " landmarks=" << summary.landmarks << " failed=" << summary.failed_views
                  << " mapping_s=" << mapping_seconds << " elapsed_s=" << elapsed
                  << " output=" << output_path << '\n';
        return summary.valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
