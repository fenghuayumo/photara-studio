#include "sfm/reconstruct.hpp"
#include "sfm/export_mvs.hpp"

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
    std::size_t count = 0;
    for (const auto& track : scene.tracks) {
        if (track.is_triangulated()) ++count;
    }
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Failed to create PLY: " + path.string());
    output << "ply\nformat ascii 1.0\nelement vertex " << count
           << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        output << track.position.x() << ' ' << track.position.y() << ' '
               << track.position.z() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 6) {
            std::cout << "Usage: aetherscan images_dir focal_pixels incremental|hierarchical|global output.(mvs|ply) "
                         "[neighbor_window]\n"
                         "  incremental: star initialization + PnP resection\n"
                         "  hierarchical: clustered incremental SfM + Sim(3) merge\n"
                         "  global: rotation averaging + global positioning + BA\n"
                         "  .mvs  OpenMVS Interface (open in Viewer)\n"
                         "  .ply  sparse XYZ point cloud\n";
            return argc == 1 ? 0 : 1;
        }
        const std::filesystem::path directory = argv[1];
        const double focal = number(argv[2]);
        const std::string mode = argv[3];
        if (mode != "incremental" && mode != "hierarchical" && mode != "global")
            throw std::invalid_argument(
                "Mode must be incremental, hierarchical, or global");
        const std::filesystem::path output_path = argv[4];
        const std::size_t window =
            argc == 6 ? static_cast<std::size_t>(number(argv[5])) : 3;

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

        aetherscan::sfm::ReconstructionConfig config;
        if (mode == "global")
            config.mode = aetherscan::sfm::ReconstructionMode::global;
        else if (mode == "hierarchical")
            config.mode = aetherscan::sfm::ReconstructionMode::hierarchical;
        else
            config.mode = aetherscan::sfm::ReconstructionMode::incremental;
        config.frontend.focal_pixels = focal;
        config.frontend.neighbor_window = window;

        const auto started = std::chrono::steady_clock::now();
        aetherscan::sfm::Scene scene;
        const auto summary = aetherscan::sfm::reconstruct(scene, files, config);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

        if (!summary.valid) {
            std::cerr << "reconstruction failed: registered="
                      << summary.registered_views << '/' << scene.images.size()
                      << " landmarks=" << summary.landmarks << '\n';
            return 2;
        }

        if (lower_extension(output_path) == ".mvs") {
            aetherscan::sfm::export_openmvs_interface(scene, output_path);
        } else if (lower_extension(output_path) == ".ply") {
            save_ply(scene, output_path);
            const auto mvs_path =
                output_path.parent_path() / (output_path.stem().string() + ".mvs");
            aetherscan::sfm::export_openmvs_interface(scene, mvs_path);
            std::cout << "mvs=" << mvs_path << '\n';
        } else {
            throw std::invalid_argument("Output must end with .mvs or .ply");
        }

        std::cout << "valid=" << summary.valid
                  << " registered=" << summary.registered_views << '/' << scene.images.size()
                  << " landmarks=" << summary.landmarks
                  << " failed=" << summary.failed_views << " elapsed_s=" << elapsed
                  << " output=" << output_path << '\n';
        return summary.valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
