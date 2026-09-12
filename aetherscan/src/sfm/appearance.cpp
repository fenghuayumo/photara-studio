#include "sfm/appearance.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "parallel/thread_pool.hpp"

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
    core::StageScope stage("sfm.colour_tracks");

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

    // Retain only sampled pixels; bound concurrent full-resolution decodes.
    std::vector<std::vector<std::uint32_t>> pixels_by_image(scene.images.size());
    std::vector<std::uint8_t> image_loaded(scene.images.size());
    const unsigned decode_threads=std::min(8U,parallel::resolve_thread_count(scene.thread_count));
    parallel::parallel_for(scene.images.size(),decode_threads,[&](const std::size_t image_id) {
        if (observations_by_image[image_id].empty()) return;
        const Image& image = scene.images[image_id];
        if (image.path.empty()) return;
        io::RgbImage rgb;
        try {
            rgb = io::load_rgb(image.path);
        } catch (...) {
            return;
        }
        image_loaded[image_id]=1;
        auto& pixels=pixels_by_image[image_id];
        pixels.resize(observations_by_image[image_id].size());
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
        for (std::size_t i=0;i<observations_by_image[image_id].size();++i) {
            const Index feature_id=observations_by_image[image_id][i].second;
            if (feature_id >= image.features.keypoints.size()) continue;
            const auto& keypoint = image.features.keypoints[feature_id];
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            if (!sample_image_rgb(
                    rgb, keypoint.x * scale_x, keypoint.y * scale_y, r, g, b))
                continue;
            pixels[i]=0x01000000U | static_cast<std::uint32_t>(r) |
                (static_cast<std::uint32_t>(g)<<8) | (static_cast<std::uint32_t>(b)<<16);
        }
    });
    std::size_t images_used=0;
    // Preserve image order and exact channel sums independently of scheduling.
    for (std::size_t image_id=0;image_id<scene.images.size();++image_id) {
        images_used+=image_loaded[image_id];
        const auto& pixels=pixels_by_image[image_id];
        for (std::size_t i=0;i<pixels.size();++i) {
            const std::uint32_t pixel=pixels[i];
            if (!(pixel & 0x01000000U)) continue;
            const std::size_t track_index=observations_by_image[image_id][i].first;
            sum_r[track_index]+=pixel & 255U;
            sum_g[track_index]+=(pixel>>8) & 255U;
            sum_b[track_index]+=(pixel>>16) & 255U;
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
