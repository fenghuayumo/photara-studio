#include "core/logging.hpp"
#include "sfm/asfm.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/reconstruct.hpp"
#include <charconv>
#include <chrono>
#include <iostream>

namespace {

unsigned parse_unsigned(const char *text, const char *name) {
    const std::string_view input(text);
    unsigned value = 0;
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size() || value == 0)
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return value;
}

} // namespace

int main(int argc, char **argv) {
    try {
        using namespace photara::sfm;
        photara::core::Logger::instance().configure({});
        if (argc < 4 || argc > 7)
            throw std::invalid_argument("Usage: photara_sfm_positioning_benchmark tracks-HEX.bin "
                                        "cpu|cuda|vulkan NEW_OUTPUT.asfm [tracks_per_image] "
                                        "[warm_start_min_images] [automatic|cpu|cuda|vulkan BA]");
        const std::filesystem::path input(argv[1]), output(argv[3]);
        if (std::filesystem::exists(output))
            throw std::invalid_argument("Output already exists");
        auto stem = input.stem().string();
        auto key = std::stoull(stem.substr(stem.find('-') + 1), nullptr, 16);
        CheckpointOptions options;
        options.directory = input.parent_path();
        options.write = false;
        Scene scene;
        if (!CheckpointStore(options).load_scene(CheckpointStage::tracks, key, scene))
            throw std::runtime_error("Cannot load tracks checkpoint");
        GlobalPositioningOptions positioning;
        const std::string backend_name(argv[2]);
        if (backend_name == "cuda") {
            positioning.prefer_cuda = true;
            positioning.backend = PositioningBackend::cuda;
        } else if (backend_name == "cpu") {
            positioning.prefer_cuda = false;
            positioning.backend = PositioningBackend::cpu;
        } else if (backend_name == "vulkan") {
            positioning.prefer_cuda = false;
            positioning.backend = PositioningBackend::vulkan;
        } else
            throw std::invalid_argument("Unknown backend");
        if (argc >= 5)
            positioning.tracks_per_registered_image = parse_unsigned(argv[4], "tracks_per_image");
        if (argc >= 6)
            positioning.camera_warm_start_min_images =
                parse_unsigned(argv[5], "warm_start_min_images");
        if (argc >= 7) {
            const std::string bundle_backend(argv[6]);
            if (bundle_backend == "automatic")
                set_bundle_backend_preference(BundleBackendPreference::automatic);
            else if (bundle_backend == "cpu")
                set_bundle_backend_preference(BundleBackendPreference::cpu);
            else if (bundle_backend == "cuda")
                set_bundle_backend_preference(BundleBackendPreference::cuda);
            else if (bundle_backend == "vulkan")
                set_bundle_backend_preference(BundleBackendPreference::vulkan);
            else
                throw std::invalid_argument("Unknown bundle backend");
        }
        const auto start = std::chrono::steady_clock::now();
        auto result = run_global_mapping(scene, {}, positioning, {});
        std::cout << "mapping_seconds="
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << " registered=" << scene.registered_count() << " landmarks=" << result.landmarks
                  << " reprojection_mean_px=" << result.mean_reprojection_error_pixels
                  << " reprojection_rms_px=" << result.rms_reprojection_error_pixels
                  << " reprojection_observations=" << result.reprojection_observations << '\n';
        save_asfm(scene, output);
        return scene.registered_count() ? 0 : 2;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
