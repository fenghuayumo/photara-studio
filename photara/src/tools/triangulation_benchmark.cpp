#include "sfm/triangulation.hpp"
#include "sfm/checkpoint.hpp"
#include <bit>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace photara::sfm;

struct Config {
    std::size_t tracks{100'000};
    std::size_t views{6};
    std::size_t iterations{5};
    unsigned threads{0};
    std::filesystem::path checkpoint;
};

std::size_t parse_size(const std::string_view text, const char* option) {
    std::size_t value = 0;
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} ||
        result.ptr != text.data() + text.size() || value == 0) {
        throw std::invalid_argument(std::string("Invalid value for ") + option);
    }
    return value;
}

Config parse_arguments(const int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--help") {
            std::cout
                << "Usage: photara_triangulation_benchmark [options]\n"
                << "  --tracks N      Track count (default 100000)\n"
                << "  --views N       Observations per track (default 6)\n"
                << "  --iterations N  Timed iterations (default 5)\n"
                << "  --threads N     Worker threads, 0=hardware (default 0)\n";
            std::cout << "  --checkpoint PATH  Use a real tracks/reconstruction checkpoint\n";
            std::exit(0);
        }
        if (i + 1 >= argc)
            throw std::invalid_argument(
                "Missing value after " + std::string(argument));
        const std::string_view value = argv[++i];
        if (argument == "--checkpoint")
            config.checkpoint = std::filesystem::path(value);
        else if (argument == "--tracks")
            config.tracks = parse_size(value, "--tracks");
        else if (argument == "--views")
            config.views = parse_size(value, "--views");
        else if (argument == "--iterations")
            config.iterations = parse_size(value, "--iterations");
        else if (argument == "--threads")
            config.threads = value == "0"
                ? 0U
                : static_cast<unsigned>(parse_size(value, "--threads"));
        else
            throw std::invalid_argument(
                "Unknown option: " + std::string(argument));
    }
    if (config.views < 2)
        throw std::invalid_argument("--views must be at least 2");
    if (config.views > std::numeric_limits<std::uint8_t>::max())
        throw std::invalid_argument("--views must not exceed 255");
    if (config.tracks > std::numeric_limits<Index>::max())
        throw std::invalid_argument("--tracks exceeds the index range");
    return config;
}

Scene make_scene(const Config& config) {
    Scene scene;
    scene.thread_count = config.threads;
    PinholeCamera camera;
    camera.id = 0;
    camera.width = 1920;
    camera.height = 1080;
    camera.fx = camera.fy = 1400.0;
    camera.cx = 960.0;
    camera.cy = 540.0;
    scene.cameras.push_back(camera);

    scene.images.resize(config.views);
    for (std::size_t view = 0; view < config.views; ++view) {
        Image& image = scene.images[view];
        image.id = static_cast<Index>(view);
        image.camera_id = 0;
        image.registered = true;
        image.pose.C.x() =
            (static_cast<double>(view) -
             0.5 * static_cast<double>(config.views - 1)) *
            0.15;
        image.features.image_width = camera.width;
        image.features.image_height = camera.height;
        image.features.keypoints.reserve(config.tracks);
    }

    std::mt19937_64 random(0xA37E5CAFull);
    std::uniform_real_distribution<double> x_distribution(-2.0, 2.0);
    std::uniform_real_distribution<double> y_distribution(-1.2, 1.2);
    std::uniform_real_distribution<double> z_distribution(4.0, 12.0);
    std::normal_distribution<double> noise(0.0, 0.25);
    scene.tracks.resize(config.tracks);
    for (std::size_t track_id = 0; track_id < config.tracks; ++track_id) {
        const Vec3 point{
            x_distribution(random), y_distribution(random),
            z_distribution(random)};
        Track& track = scene.tracks[track_id];
        track.observations.reserve(config.views);
        for (std::size_t view = 0; view < config.views; ++view) {
            Image& image = scene.images[view];
            const Vec2 pixel = camera.project(
                image.pose.transform_world_to_camera(point));
            image.features.keypoints.push_back(
                {static_cast<float>(pixel.x() + noise(random)),
                 static_cast<float>(pixel.y() + noise(random)),
                 1.0F, 0.0F, 1.0F});
            track.observations.push_back(
                {static_cast<Index>(view), static_cast<Index>(track_id)});
        }
    }
    return scene;
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        const Config config = parse_arguments(argc, argv);
        Scene scene;
        if (config.checkpoint.empty()) scene = make_scene(config);
        else {
            const auto stem = config.checkpoint.stem().string();
            const auto separator = stem.find('-');
            const auto prefix = stem.substr(0, separator);
            if (separator == std::string::npos || (prefix != "tracks" && prefix != "reconstruction"))
                throw std::invalid_argument("Expected tracks-HEX.bin or reconstruction-HEX.bin");
            const auto hex = std::string_view(stem).substr(separator + 1);
            std::uint64_t key{};
            const auto parsed = std::from_chars(hex.data(), hex.data() + hex.size(), key, 16);
            if (parsed.ec != std::errc{} || parsed.ptr != hex.data() + hex.size())
                throw std::invalid_argument("Invalid checkpoint fingerprint");
            CheckpointOptions cache; cache.directory = config.checkpoint.parent_path(); cache.write = false;
            if (!CheckpointStore(cache).load_scene(prefix == "tracks" ? CheckpointStage::tracks : CheckpointStage::reconstruction, key, scene))
                throw std::runtime_error("Cannot load checkpoint");
            scene.thread_count = config.threads;
        }
        const std::vector<Track> input_tracks = scene.tracks;
        TriangulationOptions options;
        options.reproj_threshold_px = 4.0F;
        options.min_angle_deg = 0.1F;
        options.min_inliers = 2;
        options.min_observations_for_ransac =
            static_cast<unsigned>(config.views + 1);
        options.use_lo_ransac = false;
        options.refine_nonlinear = false;
        options.split_tracks = false;
        if (!config.checkpoint.empty()) {
            options = TriangulationOptions{};
            options.reproj_threshold_px = 2.F;
            options.min_angle_deg = 1.F;
        }

        double total_ms = 0.0;
        unsigned triangulated = 0;
        for (std::size_t iteration = 0; iteration < config.iterations;
             ++iteration) {
            scene.tracks = input_tracks;
            const auto started = std::chrono::steady_clock::now();
            triangulated = triangulate_tracks(scene, false, options);
            total_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started)
                            .count();
        }

        const double average_ms =
            total_ms / static_cast<double>(config.iterations);
        std::size_t observations = 0;
        for (const auto& track : input_tracks) observations += track.observations.size();
        const double tracks_per_second =
            static_cast<double>(input_tracks.size()) * 1000.0 / average_ms;
        const double observations_per_second =
            static_cast<double>(observations) * 1000.0 / average_ms;
        std::cout << std::fixed << std::setprecision(2)
                  << "triangulation tracks=" << input_tracks.size()
                  << " observations=" << observations
                  << " iterations=" << config.iterations
                  << " triangulated=" << triangulated
                  << " average_ms=" << average_ms
                  << " tracks_per_s=" << tracks_per_second
                  << " observations_per_s=" << observations_per_second
                  << '\n';
        std::uint64_t digest = 14695981039346656037ULL;
        const auto mix = [&](std::uint64_t value) {
            for (unsigned byte = 0; byte < 8; ++byte) {
                digest = (digest ^ (value & 255)) * 1099511628211ULL;
                value >>= 8;
            }
        };
        mix(scene.tracks.size());
        for (const auto& track : scene.tracks) {
            for (unsigned axis = 0; axis < 3; ++axis) mix(std::bit_cast<std::uint64_t>(track.position[axis]));
            mix(track.num_inliers); mix(track.split_generation); mix(track.observations.size());
            for (const auto& observation : track.observations) {
                mix(observation.image_id); mix(observation.feature_id);
            }
        }
        std::cout << "output_tracks=" << scene.tracks.size() << " ordered_geometry_digest=" << digest << '\n';
        return !config.checkpoint.empty() || triangulated == config.tracks ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "triangulation benchmark error: " << error.what() << '\n';
        return 1;
    }
}
