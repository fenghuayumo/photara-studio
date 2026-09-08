#include "sfm/appearance.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace aetherscan::sfm {
namespace {

bool sample_image_rgb(
    const io::RgbImage& image, const float x, const float y,
    double& r, double& g, double& b) {
    if (image.width == 0 || image.height == 0 || image.pixels.size() < 3)
        return false;
    const int xi = static_cast<int>(std::lround(static_cast<double>(x)));
    const int yi = static_cast<int>(std::lround(static_cast<double>(y)));
    if (xi < 0 || yi < 0 || xi >= static_cast<int>(image.width) ||
        yi >= static_cast<int>(image.height))
        return false;
    const std::size_t offset =
        (static_cast<std::size_t>(yi) * image.width +
         static_cast<std::size_t>(xi)) *
        3;
    if (offset + 2 >= image.pixels.size()) return false;
    r = image.pixels[offset];
    g = image.pixels[offset + 1];
    b = image.pixels[offset + 2];
    return true;
}

}  // namespace

bool triangulated_tracks_have_color(const Scene& scene) {
    bool any = false;
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        any = true;
        if (!track.has_color) return false;
    }
    return any;
}

std::size_t colour_triangulated_tracks(Scene& scene) {
    std::vector<std::size_t> pending;
    pending.reserve(scene.tracks.size());
    for (std::size_t index = 0; index < scene.tracks.size(); ++index) {
        const Track& track = scene.tracks[index];
        if (!track.is_triangulated() || track.has_color) continue;
        pending.push_back(index);
    }
    if (pending.empty()) return 0;

    std::vector<std::vector<std::pair<std::size_t, Index>>>
        observations_by_image(scene.images.size());
    for (const std::size_t track_index : pending) {
        const Track& track = scene.tracks[track_index];
        const std::size_t inliers = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < inliers; ++i) {
            const Observation& observation = track.observations[i];
            if (observation.image_id >= scene.images.size()) continue;
            observations_by_image[observation.image_id].emplace_back(
                track_index, observation.feature_id);
        }
    }

    std::vector<double> sum_r(scene.tracks.size());
    std::vector<double> sum_g(scene.tracks.size());
    std::vector<double> sum_b(scene.tracks.size());
    std::vector<std::uint32_t> samples(scene.tracks.size());

    std::size_t images_used = 0;
    for (std::size_t image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (observations_by_image[image_id].empty()) continue;
        const Image& image = scene.images[image_id];
        if (image.path.empty()) continue;
        io::RgbImage rgb;
        try {
            rgb = io::load_rgb(image.path);
        } catch (...) {
            continue;
        }
        ++images_used;
        const float scale_x =
            image.features.image_width > 0
                ? static_cast<float>(rgb.width) /
                      static_cast<float>(image.features.image_width)
                : 1.F;
        const float scale_y =
            image.features.image_height > 0
                ? static_cast<float>(rgb.height) /
                      static_cast<float>(image.features.image_height)
                : 1.F;
        for (const auto& [track_index, feature_id] :
             observations_by_image[image_id]) {
            if (feature_id >= image.features.keypoints.size()) continue;
            const auto& keypoint = image.features.keypoints[feature_id];
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            if (!sample_image_rgb(
                    rgb, keypoint.x * scale_x, keypoint.y * scale_y, r, g, b))
                continue;
            sum_r[track_index] += r;
            sum_g[track_index] += g;
            sum_b[track_index] += b;
            ++samples[track_index];
        }
    }

    std::size_t colored = 0;
    for (const std::size_t track_index : pending) {
        if (samples[track_index] == 0) continue;
        const auto channel = [count = samples[track_index]](const double sum) {
            return static_cast<std::uint8_t>(
                std::clamp(std::lround(sum / count), 0L, 255L));
        };
        Track& track = scene.tracks[track_index];
        track.color_r = channel(sum_r[track_index]);
        track.color_g = channel(sum_g[track_index]);
        track.color_b = channel(sum_b[track_index]);
        track.has_color = true;
        ++colored;
    }
    core::Logger::instance().info(
        "sfm_track_colors=", colored, "/", pending.size(),
        " photos=", images_used);
    return colored;
}

}  // namespace aetherscan::sfm
