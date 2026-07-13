#include "sfm/reconstruct.hpp"
#include "sfm/export_mvs.hpp"
#include "core/logging.hpp"

#include <cxxopts.hpp>

#include <algorithm>
#include <cctype>
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

struct ReconstructCli {
    std::filesystem::path images_dir;
    double focal_pixels{};
    std::string mode;
    std::filesystem::path output;
    std::size_t neighbor_window{3};
    float match_ratio{0.85F};
    bool mutual_check{true};
    double sift_contrast{0.005};
    std::filesystem::path cache_dir;
    std::string extractor{"siftgpu"};
    std::string matcher{"gpu_mutual_ratio"};
    std::string pipeline;  // empty / none | lightglue_end2end
    unsigned max_features{27000U};
    std::filesystem::path extractor_model;
    std::uint32_t extractor_width{1024U};
    std::uint32_t extractor_height{1024U};
    bool extractor_cpu{false};
    std::filesystem::path lightglue_model;
    std::string lightglue_extractor{"disk"};
    std::uint32_t lightglue_width{1024U};
    std::uint32_t lightglue_height{1024U};
    float lightglue_min_score{0.0F};
    bool lightglue_cpu{false};
};

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

#if defined(_WIN32)
std::string wide_to_utf8(const wchar_t* text) {
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

std::filesystem::path utf8_to_path(const std::string& utf8) {
    if (utf8.empty()) return {};
    const int wide_size = MultiByteToWideChar(
        CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (wide_size <= 1) return {};
    std::wstring wide(static_cast<std::size_t>(wide_size), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, utf8.c_str(), -1, wide.data(), wide_size);
    wide.pop_back();
    return std::filesystem::path(wide);
}
#else
std::filesystem::path utf8_to_path(const std::string& utf8) {
    return std::filesystem::path(utf8);
}
#endif

struct Utf8Argv {
    std::vector<std::string> storage;
    std::vector<char*> pointers;

#if defined(_WIN32)
    explicit Utf8Argv(int argc, wchar_t** argv) {
        storage.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i)
            storage.push_back(wide_to_utf8(argv[i]));
        pointers.reserve(storage.size());
        for (auto& argument : storage) pointers.push_back(argument.data());
    }
#else
    explicit Utf8Argv(int argc, char** argv) {
        storage.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i) storage.emplace_back(argv[i]);
        pointers.reserve(storage.size());
        for (auto& argument : storage) pointers.push_back(argument.data());
    }
#endif

    int argc() const { return static_cast<int>(pointers.size()); }
    char** argv() { return pointers.data(); }
};

void print_help(const cxxopts::Options& options) {
    std::cout << options.help() << '\n'
              << "Feature backends:\n"
              << "  default   --extractor siftgpu --matcher gpu_mutual_ratio\n"
              << "  compose   any compatible --extractor × --matcher\n"
              << "  fused     --pipeline lightglue_end2end (optional recipe)\n"
              << "Modes:\n"
              << "  incremental  star initialization + PnP resection\n"
              << "  hierarchical clustered incremental SfM + Sim(3) merge\n"
              << "  global       rotation averaging + global positioning + BA\n"
              << "Output formats:\n"
              << "  .mvs  OpenMVS Interface (open in Viewer)\n"
              << "  .ply  sparse XYZ point cloud\n"
              << "Log level: set AETHERSCAN_LOG_LEVEL=error|warning|info|debug|trace|off\n";
}

ReconstructCli parse_cli(int argc, char** argv) {
    cxxopts::Options options(
        "aetherscan", "High-performance Structure from Motion reconstruction");
    options.custom_help("[options]");
    options.add_options()
        ("h,help", "Print usage")
        ("i,images", "Image directory", cxxopts::value<std::string>())
        ("f,focal",
         "Initial focal length in pixels (0 = 1.2 * max(width,height); "
         "refined by view-graph consensus + BA unless trusted)",
         cxxopts::value<double>()->default_value("0"))
        ("m,mode",
         "Reconstruction mode: incremental, hierarchical, or global",
         cxxopts::value<std::string>())
        ("o,output", "Output path (.mvs or .ply)", cxxopts::value<std::string>())
        ("window", "Sequential neighbor window",
         cxxopts::value<std::size_t>()->default_value("3"))
        ("match-ratio", "Lowe ratio test threshold",
         cxxopts::value<float>()->default_value("0.85"))
        ("mutual-check", "Mutual match consistency check",
         cxxopts::value<bool>()->default_value("true"))
        ("sift-contrast", "SIFT contrast threshold",
         cxxopts::value<double>()->default_value("0.005"))
        ("cache-dir", "Feature cache directory (- to disable)",
         cxxopts::value<std::string>()->default_value(""))
        ("extractor",
         "Feature extractor: siftgpu (default), sift, superpoint, disk",
         cxxopts::value<std::string>()->default_value("siftgpu"))
        ("matcher",
         "Feature matcher: gpu_mutual_ratio (default), mutual_ratio, lightglue. "
         "Legacy alias: siftgpu→gpu_mutual_ratio",
         cxxopts::value<std::string>()->default_value("gpu_mutual_ratio"))
        ("pipeline",
         "Optional fused pair recipe: none (default) or lightglue_end2end",
         cxxopts::value<std::string>()->default_value(""))
        ("max-features", "Maximum features per image",
         cxxopts::value<unsigned>()->default_value("27000"))
        ("extractor-model",
         "ONNX weights for --extractor superpoint|disk",
         cxxopts::value<std::string>()->default_value(""))
        ("extractor-width", "Learned extractor network input width",
         cxxopts::value<std::uint32_t>()->default_value("1024"))
        ("extractor-height", "Learned extractor network input height",
         cxxopts::value<std::uint32_t>()->default_value("1024"))
        ("extractor-cpu", "Force learned extractor ONNX on CPU",
         cxxopts::value<bool>()->default_value("false"))
        ("lightglue-model",
         "ONNX for --matcher lightglue (*_lightglue_fused.onnx) or "
         "--pipeline lightglue_end2end",
         cxxopts::value<std::string>()->default_value(""))
        ("lightglue-extractor",
         "End2end pipeline head only: disk or superpoint",
         cxxopts::value<std::string>()->default_value("disk"))
        ("lightglue-width", "End2end pipeline network width",
         cxxopts::value<std::uint32_t>()->default_value("1024"))
        ("lightglue-height", "End2end pipeline network height",
         cxxopts::value<std::uint32_t>()->default_value("1024"))
        ("lightglue-min-score", "Drop LightGlue matches below this score",
         cxxopts::value<float>()->default_value("0"))
        ("lightglue-cpu", "Force LightGlue matcher / end2end ONNX on CPU",
         cxxopts::value<bool>()->default_value("false"));

    const auto result = options.parse(argc, argv);
    if (result.count("help") || argc <= 1) {
        print_help(options);
        std::exit(0);
    }

    if (!result.count("images") || !result.count("mode") ||
        !result.count("output")) {
        throw std::invalid_argument(
            "Missing required options: --images, --mode, --output");
    }

    ReconstructCli cli;
    cli.images_dir = utf8_to_path(result["images"].as<std::string>());
    cli.focal_pixels = result["focal"].as<double>();
    cli.mode = result["mode"].as<std::string>();
    cli.output = utf8_to_path(result["output"].as<std::string>());
    cli.neighbor_window = result["window"].as<std::size_t>();
    cli.match_ratio = result["match-ratio"].as<float>();
    cli.mutual_check = result["mutual-check"].as<bool>();
    cli.sift_contrast = result["sift-contrast"].as<double>();
    cli.extractor = result["extractor"].as<std::string>();
    cli.matcher = result["matcher"].as<std::string>();
    cli.pipeline = result["pipeline"].as<std::string>();
    cli.max_features = result["max-features"].as<unsigned>();
    const auto extractor_model_text =
        result["extractor-model"].as<std::string>();
    if (!extractor_model_text.empty())
        cli.extractor_model = utf8_to_path(extractor_model_text);
    cli.extractor_width = result["extractor-width"].as<std::uint32_t>();
    cli.extractor_height = result["extractor-height"].as<std::uint32_t>();
    cli.extractor_cpu = result["extractor-cpu"].as<bool>();
    const auto lightglue_model_text =
        result["lightglue-model"].as<std::string>();
    if (!lightglue_model_text.empty())
        cli.lightglue_model = utf8_to_path(lightglue_model_text);
    cli.lightglue_extractor = result["lightglue-extractor"].as<std::string>();
    cli.lightglue_width = result["lightglue-width"].as<std::uint32_t>();
    cli.lightglue_height = result["lightglue-height"].as<std::uint32_t>();
    cli.lightglue_min_score = result["lightglue-min-score"].as<float>();
    cli.lightglue_cpu = result["lightglue-cpu"].as<bool>();

    const auto cache_text = result["cache-dir"].as<std::string>();
    if (!cache_text.empty() && cache_text != "-")
        cli.cache_dir = utf8_to_path(cache_text);

    if (cli.focal_pixels < 0.0)
        throw std::invalid_argument("--focal must be >= 0");
    if (cli.mode != "incremental" && cli.mode != "hierarchical" &&
        cli.mode != "global") {
        throw std::invalid_argument(
            "--mode must be incremental, hierarchical, or global");
    }
    if (cli.match_ratio <= 0.F || cli.match_ratio > 1.F)
        throw std::invalid_argument("--match-ratio must be in (0, 1]");
    if (cli.neighbor_window == 0)
        throw std::invalid_argument("--window must be positive");
    if (cli.sift_contrast <= 0.0)
        throw std::invalid_argument("--sift-contrast must be positive");
    if (cli.max_features == 0U)
        throw std::invalid_argument("--max-features must be positive");

    if (cli.pipeline == "none") cli.pipeline.clear();
    if (!cli.pipeline.empty() && cli.pipeline != "lightglue_end2end")
        throw std::invalid_argument(
            "--pipeline must be empty/none or lightglue_end2end");
    if (!cli.pipeline.empty() && cli.matcher == "lightglue")
        throw std::invalid_argument(
            "Use either --matcher lightglue or --pipeline lightglue_end2end");

    const bool need_lg_model =
        cli.pipeline == "lightglue_end2end" || cli.matcher == "lightglue";
    if (need_lg_model) {
        if (cli.lightglue_model.empty())
            throw std::invalid_argument(
                "--lightglue-model is required for matcher lightglue or "
                "pipeline lightglue_end2end");
        if (cli.lightglue_width == 0U || cli.lightglue_height == 0U)
            throw std::invalid_argument(
                "--lightglue-width/height must be positive");
        if (cli.lightglue_min_score < 0.F || cli.lightglue_min_score > 1.F)
            throw std::invalid_argument(
                "--lightglue-min-score must be in [0, 1]");
    }
    if (cli.pipeline == "lightglue_end2end") {
        if (cli.lightglue_extractor != "disk" &&
            cli.lightglue_extractor != "superpoint")
            throw std::invalid_argument(
                "--lightglue-extractor must be disk or superpoint");
    }
    if (cli.matcher == "lightglue") {
        if (cli.extractor != "superpoint" && cli.extractor != "disk")
            throw std::invalid_argument(
                "--matcher lightglue currently accepts descriptors from "
                "--extractor superpoint|disk (see features/compat.hpp)");
        if (cli.extractor_model.empty())
            throw std::invalid_argument(
                "--extractor superpoint|disk requires --extractor-model");
    }
    if ((cli.extractor == "superpoint" || cli.extractor == "disk")) {
        if (cli.extractor_model.empty() && cli.pipeline.empty())
            throw std::invalid_argument(
                "--extractor superpoint|disk requires --extractor-model");
        if (cli.extractor_width == 0U || cli.extractor_height == 0U)
            throw std::invalid_argument(
                "--extractor-width/height must be positive");
    }
    return cli;
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

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        Utf8Argv utf8_argv(argc, argv);
        const ReconstructCli cli = parse_cli(utf8_argv.argc(), utf8_argv.argv());

        const char* configured_level = std::getenv("AETHERSCAN_LOG_LEVEL");
        const auto console_level = configured_level
            ? aetherscan::core::parse_log_level(
                  configured_level, aetherscan::core::LogLevel::info)
            : aetherscan::core::LogLevel::info;
        std::filesystem::path log_directory = cli.output.parent_path();
        if (log_directory.empty()) log_directory = std::filesystem::current_path();
        const std::filesystem::path log_path =
            aetherscan::core::Logger::instance().configure(
                log_directory, "aetherscan", console_level,
                aetherscan::core::LogLevel::trace);
        aetherscan::core::Logger::instance().info(
            "AetherScan started: mode=", cli.mode, " images_dir=", cli.images_dir,
            " output=", cli.output, " log=", log_path);

        std::vector<std::filesystem::path> files;
        for (const auto& entry :
             std::filesystem::directory_iterator(cli.images_dir)) {
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
        if (cli.mode == "global")
            config.mode = aetherscan::sfm::ReconstructionMode::global;
        else if (cli.mode == "hierarchical")
            config.mode = aetherscan::sfm::ReconstructionMode::hierarchical;
        else
            config.mode = aetherscan::sfm::ReconstructionMode::incremental;
        config.frontend.focal_pixels = cli.focal_pixels;
        config.frontend.neighbor_window = cli.neighbor_window;
        config.frontend.sift_contrast_threshold = cli.sift_contrast;
        config.frontend.match_ratio = cli.match_ratio;
        config.frontend.mutual_check = cli.mutual_check;
        config.frontend.extractor = cli.extractor;
        config.frontend.matcher = cli.matcher;
        config.frontend.pipeline = cli.pipeline;
        config.frontend.max_features = cli.max_features;
        config.frontend.extractor_model_path = cli.extractor_model;
        config.frontend.extractor_input_width = cli.extractor_width;
        config.frontend.extractor_input_height = cli.extractor_height;
        config.frontend.extractor_use_cuda = !cli.extractor_cpu;
        config.frontend.lightglue_model_path = cli.lightglue_model;
        config.frontend.lightglue_extractor = cli.lightglue_extractor;
        config.frontend.lightglue_input_width = cli.lightglue_width;
        config.frontend.lightglue_input_height = cli.lightglue_height;
        config.frontend.lightglue_min_score = cli.lightglue_min_score;
        config.frontend.lightglue_use_cuda = !cli.lightglue_cpu;
        // Sequential window + BoW retrieval (learned vocabulary).
        // lightglue_end2end uses a temporary SiftGPU extract for BoW only.
        config.frontend.augment_sequential_with_retrieval = true;
        config.frontend.checkpoint.directory = cli.cache_dir;

        const auto started = std::chrono::steady_clock::now();
        aetherscan::sfm::Scene scene;
        const auto summary = aetherscan::sfm::reconstruct(scene, files, config);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();

        if (!summary.valid) {
            aetherscan::core::Logger::instance().error(
                "reconstruction failed: registered=", summary.registered_views,
                '/', scene.images.size(), " landmarks=", summary.landmarks);
            return 2;
        }

        if (lower_extension(cli.output) == ".mvs") {
            aetherscan::sfm::export_openmvs_interface(scene, cli.output);
        } else if (lower_extension(cli.output) == ".ply") {
            save_ply(scene, cli.output);
            auto mvs_path = cli.output.parent_path() / cli.output.stem();
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
            " output=", cli.output, " log=", log_path);
        return summary.valid ? 0 : 2;
    } catch (const std::exception& error) {
        aetherscan::core::Logger::instance().error("error: ", error.what());
        return 1;
    }
}
