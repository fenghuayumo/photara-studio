#include "sfm/triangulation.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace aetherscan::sfm;

struct Config {
    std::size_t tracks{100'000};
    std::size_t views{6};
    std::size_t iterations{5};
    unsigned threads{0};
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
                << "Usage: aetherscan_triangulation_benchmark [options]\n"
                << "  --tracks N      Track count (default 100000)\n"
                << "  --views N       Observations per track (default 6)\n"
                << "  --iterations N  Timed iterations (default 5)\n"
                << "  --threads N     Worker threads, 0=hardware (default 0)\n";
            std::exit(0);
        }
        if (i + 1 >= argc)
            throw std::invalid_argument(
                "Missing value after " + std::string(argument));
        const std::string_view value = argv[++i];
        if (argument == "--tracks")
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
        Scene scene = make_scene(config);
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
        const double tracks_per_second =
            static_cast<double>(config.tracks) * 1000.0 / average_ms;
        const double observations_per_second =
            tracks_per_second * static_cast<double>(config.views);
        std::cout << std::fixed << std::setprecision(2)
                  << "triangulation tracks=" << config.tracks
                  << " views=" << config.views
                  << " iterations=" << config.iterations
                  << " triangulated=" << triangulated
                  << " average_ms=" << average_ms
                  << " tracks_per_s=" << tracks_per_second
                  << " observations_per_s=" << observations_per_second
                  << '\n';
        return triangulated == config.tracks ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "triangulation benchmark error: " << error.what() << '\n';
        return 1;
    }
}
