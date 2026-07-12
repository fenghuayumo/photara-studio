#include "sfm/reconstruct.hpp"
#include "sfm/export_mvs.hpp"
#include "core/logging.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

std::uint64_t peak_working_set_bytes() noexcept {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (!K32GetProcessMemoryInfo(
            GetCurrentProcess(), &counters, sizeof(counters)))
        return 0;
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#endif
}

double number(const std::string& input) {
    double value = 0;
    const auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    if (result.ec != std::errc{} || value <= 0)
        throw std::invalid_argument("Invalid positive number: " + input);
    return value;
}

bool boolean_flag(const std::string& input) {
    if (input == "1" || input == "true") return true;
    if (input == "0" || input == "false") return false;
    throw std::invalid_argument("Expected boolean flag 0/1: " + input);
}

#if defined(_WIN32)
std::string argument_text(const wchar_t* text) {
    if (*text == L'\0') return {};
    const int size = WideCharToMultiByte(
        CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) return {};
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
    result.pop_back();
    return result;
}

std::filesystem::path argument_path(const wchar_t* text) {
    return std::filesystem::path(text);
}
#else
std::string argument_text(const char* text) {
    return text;
}

std::filesystem::path argument_path(const char* text) {
    return std::filesystem::path(text);
}
#endif

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

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc < 5 || argc > 13) {
            std::cout << "Usage: aetherscan images_dir focal_pixels incremental|hierarchical|global output.(mvs|ply) "
                         "[neighbor_window] [match_ratio] [mutual_check] [sift_contrast] [cache_dir|-] "
                         "[sift|siftgpu] [mutual_ratio|siftgpu] [max_features]\n"
                         "  incremental: star initialization + PnP resection\n"
                         "  hierarchical: clustered incremental SfM + Sim(3) merge\n"
                         "  global: rotation averaging + global positioning + BA\n"
                         "  .mvs  OpenMVS Interface (open in Viewer)\n"
                         "  .ply  sparse XYZ point cloud\n"
                         "  log level: set AETHERSCAN_LOG_LEVEL=error|warning|info|debug|trace|off\n";
            return argc == 1 ? 0 : 1;
        }
        const std::filesystem::path directory = argument_path(argv[1]);
        const double focal = number(argument_text(argv[2]));
        const std::string mode = argument_text(argv[3]);
        if (mode != "incremental" && mode != "hierarchical" && mode != "global")
            throw std::invalid_argument(
                "Mode must be incremental, hierarchical, or global");
        const std::filesystem::path output_path = argument_path(argv[4]);
        const std::size_t window =
            argc >= 6
                ? static_cast<std::size_t>(
                      number(argument_text(argv[5])))
                : 3;
        const float match_ratio =
            argc >= 7
                ? static_cast<float>(number(argument_text(argv[6])))
                : 0.85F;
        if (match_ratio > 1.F)
            throw std::invalid_argument("match_ratio must be in (0, 1]");
        const bool mutual_check =
            argc >= 8 ? boolean_flag(argument_text(argv[7])) : true;
        const double sift_contrast =
            argc >= 9 ? number(argument_text(argv[8])) : 0.005;
        const std::filesystem::path cache_directory =
            argc >= 10
                ? (argument_text(argv[9]) == "-"
                       ? std::filesystem::path{}
                       : argument_path(argv[9]))
                 : directory / ".aetherscan-cache";
        const std::string extractor =
            argc >= 11 ? argument_text(argv[10]) : "sift";
        const std::string matcher =
            argc >= 12 ? argument_text(argv[11]) : "mutual_ratio";
        const unsigned max_features =
            argc >= 13
                ? static_cast<unsigned>(number(argument_text(argv[12])))
                : 27000U;

        const char* configured_level = std::getenv("AETHERSCAN_LOG_LEVEL");
        const auto console_level = configured_level
            ? aetherscan::core::parse_log_level(
                  configured_level, aetherscan::core::LogLevel::info)
            : aetherscan::core::LogLevel::info;
        std::filesystem::path log_directory = output_path.parent_path();
        if (log_directory.empty()) log_directory = std::filesystem::current_path();
        const std::filesystem::path log_path =
            aetherscan::core::Logger::instance().configure(
                log_directory, "aetherscan", console_level,
                aetherscan::core::LogLevel::trace);
        aetherscan::core::Logger::instance().info(
            "AetherScan started: mode=", mode, " images_dir=", directory,
            " output=", output_path, " log=", log_path);

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
        config.frontend.sift_contrast_threshold = sift_contrast;
        config.frontend.match_ratio = match_ratio;
        config.frontend.mutual_check = mutual_check;
        config.frontend.extractor = extractor;
        config.frontend.matcher = matcher;
        config.frontend.max_features = max_features;
        // Sequential window + BoW retrieval (learned vocabulary).
        config.frontend.augment_sequential_with_retrieval = true;
        config.frontend.checkpoint.directory = cache_directory;

        const auto started = std::chrono::steady_clock::now();
        aetherscan::sfm::Scene scene;
        const auto summary = aetherscan::sfm::reconstruct(scene, files, config);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

        if (!summary.valid) {
            aetherscan::core::Logger::instance().error(
                "reconstruction failed: registered=", summary.registered_views,
                '/', scene.images.size(), " landmarks=", summary.landmarks);
            return 2;
        }

        if (lower_extension(output_path) == ".mvs") {
            aetherscan::sfm::export_openmvs_interface(scene, output_path);
        } else if (lower_extension(output_path) == ".ply") {
            save_ply(scene, output_path);
            auto mvs_path =
                output_path.parent_path() / output_path.stem();
            mvs_path += ".mvs";
            aetherscan::sfm::export_openmvs_interface(scene, mvs_path);
            aetherscan::core::Logger::instance().info("mvs=", mvs_path);
        } else {
            throw std::invalid_argument("Output must end with .mvs or .ply");
        }

        aetherscan::core::Logger::instance().info(
            "valid=", summary.valid, " registered=", summary.registered_views,
            '/', scene.images.size(), " landmarks=", summary.landmarks,
            " reprojection_mean_px=",
            summary.mean_reprojection_error_pixels,
            " reprojection_rms_px=",
            summary.rms_reprojection_error_pixels,
            " reprojection_observations=",
            summary.reprojection_observations,
            " peak_working_set_mb=",
            static_cast<double>(peak_working_set_bytes()) / (1024.0 * 1024.0),
            " failed=", summary.failed_views, " elapsed_s=", elapsed,
            " output=", output_path, " log=", log_path);
        return summary.valid ? 0 : 2;
    } catch (const std::exception& error) {
        aetherscan::core::Logger::instance().error("error: ", error.what());
        return 1;
    }
}
