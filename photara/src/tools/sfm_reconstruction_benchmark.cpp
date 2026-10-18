#include "core/logging.hpp"
#include "sfm/asfm.hpp"
#include "sfm/appearance.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/export_colmap.hpp"
#include "sfm/reconstruct.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

int main(int argc, char** argv) {
    try {
        using namespace photara::sfm;
        if (argc < 4 || argc > 7)
            throw std::invalid_argument("Usage: photara_sfm_reconstruction_benchmark tracks-HEX.bin automatic|cpu|cuda|vulkan NEW_OUTPUT.asfm [automatic|cpu|cuda|vulkan BA [global|hierarchical [CLUSTER_CAPACITY]]]");
        const auto backend = [](const std::string& value) {
            if (value == "automatic") return PositioningBackend::automatic;
            if (value == "cpu") return PositioningBackend::cpu;
            if (value == "cuda") return PositioningBackend::cuda;
            if (value == "vulkan") return PositioningBackend::vulkan;
            throw std::invalid_argument("Unknown positioning backend");
        };
        ReconstructionConfig config;
        // Offline ablation only; never changes the application's defaults.
        if (const auto* camera_only = std::getenv("PHOTARA_BENCHMARK_CAMERA_ONLY_INITIALIZER");
            camera_only && std::string_view(camera_only) == "1")
            config.global_positioning.constraint = GlobalPositioningConstraint::only_cameras;
        if (const auto* enable = std::getenv("PHOTARA_BENCHMARK_ENABLE_ORBIT");
            enable && std::string_view(enable) == "1")
            config.global_positioning.regularize_complete_orbits = true;
        if (const auto* disable = std::getenv("PHOTARA_BENCHMARK_DISABLE_ORBIT");
            disable && std::string_view(disable) == "1")
            config.global_positioning.regularize_complete_orbits = false;
        config.global_positioning.backend = backend(argv[2]);
        const std::string ba = argc >= 5 ? argv[4] : "automatic";
        if (ba == "automatic") set_bundle_backend_preference(BundleBackendPreference::automatic);
        else if (ba == "cpu") set_bundle_backend_preference(BundleBackendPreference::cpu);
        else if (ba == "cuda") set_bundle_backend_preference(BundleBackendPreference::cuda);
        else if (ba == "vulkan") set_bundle_backend_preference(BundleBackendPreference::vulkan);
        else throw std::invalid_argument("Unknown BA backend");
        const std::string mode = argc >= 6 ? argv[5] : "global";
        if (mode == "hierarchical") config.mode = ReconstructionMode::hierarchical;
        else if (mode != "global") throw std::invalid_argument("Unknown reconstruction mode");
        if (argc == 7) {
            const std::string value = argv[6];
            if (mode != "hierarchical" || value.empty() ||
                value.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("Cluster capacity requires hierarchical mode and a positive integer");
            const auto capacity = std::stoull(value);
            if (capacity == 0 || capacity > std::numeric_limits<unsigned>::max())
                throw std::invalid_argument("Cluster capacity out of range");
            config.hierarchical.cluster.max_views_per_cluster = static_cast<unsigned>(capacity);
        }
        const std::filesystem::path input(argv[1]), output(argv[3]);
        if (std::filesystem::exists(output)) throw std::invalid_argument("Output already exists");
        if (const auto* export_path = std::getenv("PHOTARA_BENCHMARK_EXPORT_COLMAP");
            export_path && *export_path && std::filesystem::exists(export_path))
            throw std::invalid_argument("COLMAP export directory already exists");
        const auto stem = input.stem().string();
        if (!stem.starts_with("tracks-")) throw std::invalid_argument("Expected a tracks checkpoint filename");
        std::size_t consumed = 0;
        const auto key = std::stoull(stem.substr(7), &consumed, 16);
        if (consumed != stem.size() - 7) throw std::invalid_argument("Invalid tracks checkpoint key");
        const auto level = std::getenv("PHOTARA_LOG_LEVEL");
        photara::core::Logger::instance().configure({}, "photara",
            level ? photara::core::parse_log_level(level, photara::core::LogLevel::info) : photara::core::LogLevel::info);
        CheckpointOptions checkpoint;
        checkpoint.directory = input.parent_path();
        checkpoint.write = false;
        Scene scene;
        const auto load_start = std::chrono::steady_clock::now();
        if (!CheckpointStore(checkpoint).load_scene(CheckpointStage::tracks, key, scene))
            throw std::runtime_error("Cannot load tracks checkpoint");
        const auto seconds = [](auto start) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        };
        const double load_seconds = seconds(load_start);
        const auto start = std::chrono::steady_clock::now();
        // Includes metadata restoration, mapping/BA, final recovery and position
        // audit. No reconstruction cache or reference model enters this path.
        const auto result = reconstruct_from_frontend(scene, config);
        const double alignment_seconds = seconds(start);
        const auto save_start = std::chrono::steady_clock::now();
        // Exported scenes are also used as 3DGS initializers in downstream
        // comparisons. Match the CLI's colour sampling without timing it as BA.
        if (const auto* export_path = std::getenv("PHOTARA_BENCHMARK_EXPORT_COLMAP");
            export_path && *export_path)
            colour_triangulated_tracks(scene);
        save_asfm(scene, output);
        if (const auto* export_path = std::getenv("PHOTARA_BENCHMARK_EXPORT_COLMAP");
            export_path && *export_path)
            save_colmap_text(scene, export_path);
        // ASFM deliberately omits the pair graph. Retain final active flags and
        // matched features separately so diagnostics do not use pre-solve edges.
        CheckpointOptions graph;
        graph.directory = output.parent_path() / (output.stem().string() + ".graph");
        graph.read = false;
        CheckpointStore(graph).save_scene(CheckpointStage::reconstruction, key, scene);
        std::cout << std::setprecision(12)
                  << "regularize_complete_orbits=" << config.global_positioning.regularize_complete_orbits << '\n'
                  << "reconstruction_mode=" << mode << '\n'
                  << "cluster_capacity=" << config.hierarchical.cluster.max_views_per_cluster << '\n'
                  << "checkpoint_load_seconds=" << load_seconds << '\n'
                  << "alignment_seconds=" << alignment_seconds << '\n'
                  << "serialization_seconds=" << seconds(save_start) << '\n'
                  << "valid=" << result.valid << '\n'
                  << "images=" << scene.images.size() << '\n'
                  << "registered=" << result.registered_views << '\n'
                  << "failed_views=" << result.failed_views << '\n'
                  << "landmarks=" << result.landmarks << '\n'
                  << "position_audited_views=" << result.position_audited_views << '\n'
                  << "position_unverified_views=" << result.position_unverified_views << '\n'
                  << "reprojection_mean_pixels=" << result.mean_reprojection_error_pixels << '\n'
                  << "reprojection_rms_pixels=" << result.rms_reprojection_error_pixels << '\n'
                  << "reprojection_observations=" << result.reprojection_observations << '\n';
        return result.valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
