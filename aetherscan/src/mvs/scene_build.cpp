#include "mvs/densify.hpp"
#include "mvs/internal.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace aetherscan::mvs {
namespace {

[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> working_size(
    const std::uint32_t width, const std::uint32_t height,
    const unsigned resolution_level, const unsigned min_resolution) {
    const unsigned div = 1U << resolution_level;
    std::uint32_t w = std::max(1U, width / div);
    std::uint32_t h = std::max(1U, height / div);
    const unsigned long_side = std::max(w, h);
    if (long_side < min_resolution && long_side > 0) {
        const float scale =
            static_cast<float>(min_resolution) / static_cast<float>(long_side);
        w = std::max(1U, static_cast<std::uint32_t>(std::lround(w * scale)));
        h = std::max(1U, static_cast<std::uint32_t>(std::lround(h * scale)));
    }
    return {w, h};
}

[[nodiscard]] sfm::Vec2 project_distorted(
    const MvsView& view, const double xn, const double yn) {
    const double r2 = xn * xn + yn * yn;
    const double radial = 1.0 + view.k1 * r2 + view.k2 * r2 * r2;
    const double xd =
        xn * radial + 2.0 * view.p1 * xn * yn + view.p2 * (r2 + 2.0 * xn * xn);
    const double yd =
        yn * radial + view.p1 * (r2 + 2.0 * yn * yn) + 2.0 * view.p2 * xn * yn;
    return {view.src_fx * xd + view.src_cx, view.src_fy * yd + view.src_cy};
}

void erode_mask(
    std::vector<std::uint8_t>& mask, const std::uint32_t width,
    const std::uint32_t height, const unsigned radius) {
    if (mask.empty() || radius == 0 || width == 0 || height == 0) return;
    const std::uint32_t window = radius * 2U + 1U;
    std::vector<std::uint8_t> horizontal(mask.size(), 0);
    std::vector<std::uint32_t> prefix(
        static_cast<std::size_t>(std::max(width, height)) + 1U, 0U);

    for (std::uint32_t y = 0; y < height; ++y) {
        prefix[0] = 0;
        const std::size_t row = static_cast<std::size_t>(y) * width;
        for (std::uint32_t x = 0; x < width; ++x)
            prefix[x + 1U] = prefix[x] + (mask[row + x] != 0 ? 1U : 0U);
        for (std::uint32_t x = radius; x + radius < width; ++x) {
            const std::uint32_t sum =
                prefix[x + radius + 1U] - prefix[x - radius];
            horizontal[row + x] = sum == window ? 1U : 0U;
        }
    }

    std::fill(mask.begin(), mask.end(), 0U);
    for (std::uint32_t x = 0; x < width; ++x) {
        prefix[0] = 0;
        for (std::uint32_t y = 0; y < height; ++y)
            prefix[y + 1U] = prefix[y] +
                             (horizontal[static_cast<std::size_t>(y) * width + x]
                                      != 0
                                  ? 1U
                                  : 0U);
        for (std::uint32_t y = radius; y + radius < height; ++y) {
            const std::uint32_t sum =
                prefix[y + radius + 1U] - prefix[y - radius];
            mask[static_cast<std::size_t>(y) * width + x] =
                sum == window ? 1U : 0U;
        }
    }
}

}  // namespace

namespace detail {

std::vector<ViewImage> load_view_images(
    const MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.load_images");
    std::vector<ViewImage> images(scene.views.size());
    core::ProgressReporter progress("mvs.load_images", scene.views.size());

    const unsigned threads = parallel::resolve_thread_count(scene.thread_count);
    parallel::parallel_for(
        scene.views.size(), threads, [&](const std::size_t i) {
            const MvsView& view = scene.views[i];
            const io::GrayImage source = io::load_gray(view.path);
            io::GrayImage source_mask;
            if (!options.mask_dir.empty()) {
                const std::filesystem::path mask_path =
                    options.mask_dir / view.path.filename();
                if (std::filesystem::exists(mask_path))
                    source_mask = io::load_gray(mask_path);
            }
            const bool has_distortion =
                view.k1 != 0.F || view.k2 != 0.F || view.p1 != 0.F ||
                view.p2 != 0.F;

            auto sample_source = [&](float x, float y) -> float {
                if (x < 0.F || y < 0.F ||
                    x > static_cast<float>(source.width - 1) ||
                    y > static_cast<float>(source.height - 1))
                    return 0.F;
                const int x0 = static_cast<int>(x);
                const int y0 = static_cast<int>(y);
                const int x1 =
                    std::min(x0 + 1, static_cast<int>(source.width) - 1);
                const int y1 =
                    std::min(y0 + 1, static_cast<int>(source.height) - 1);
                const float tx = x - static_cast<float>(x0);
                const float ty = y - static_cast<float>(y0);
                const auto at = [&](int px, int py) {
                    return static_cast<float>(
                        source.pixels
                            [static_cast<std::size_t>(py) * source.width +
                             static_cast<std::size_t>(px)]);
                };
                return (at(x0, y0) * (1.F - tx) + at(x1, y0) * tx) *
                           (1.F - ty) +
                       (at(x0, y1) * (1.F - tx) + at(x1, y1) * tx) * ty;
            };

            images[i].gray.resize(
                static_cast<std::size_t>(view.width) * view.height);
            if (!source_mask.pixels.empty())
                images[i].mask.resize(
                    static_cast<std::size_t>(view.width) * view.height);
            for (std::uint32_t y = 0; y < view.height; ++y) {
                for (std::uint32_t x = 0; x < view.width; ++x) {
                    float sx = 0.F;
                    float sy = 0.F;
                    if (!has_distortion) {
                        const float scale_x =
                            static_cast<float>(view.width) /
                            static_cast<float>(source.width);
                        const float scale_y =
                            static_cast<float>(view.height) /
                            static_cast<float>(source.height);
                        sx = (static_cast<float>(x) + 0.5F) / scale_x - 0.5F;
                        sy = (static_cast<float>(y) + 0.5F) / scale_y - 0.5F;
                    } else {
                        const double xn =
                            (static_cast<double>(x) - view.cx) /
                            static_cast<double>(view.fx);
                        const double yn =
                            (static_cast<double>(y) - view.cy) /
                            static_cast<double>(view.fy);
                        const sfm::Vec2 distorted =
                            project_distorted(view, xn, yn);
                        sx = static_cast<float>(distorted.x());
                        sy = static_cast<float>(distorted.y());
                    }
                    images[i].gray
                        [static_cast<std::size_t>(y) * view.width + x] =
                        sample_source(sx, sy) / 255.F;
                    if (!images[i].mask.empty()) {
                        const float mx = sx * static_cast<float>(source_mask.width) /
                                         static_cast<float>(source.width);
                        const float my = sy * static_cast<float>(source_mask.height) /
                                         static_cast<float>(source.height);
                        const int mask_x = std::clamp(
                            static_cast<int>(std::lround(mx)), 0,
                            static_cast<int>(source_mask.width) - 1);
                        const int mask_y = std::clamp(
                            static_cast<int>(std::lround(my)), 0,
                            static_cast<int>(source_mask.height) - 1);
                        images[i].mask
                            [static_cast<std::size_t>(y) * view.width + x] =
                            source_mask.pixels
                                [static_cast<std::size_t>(mask_y) * source_mask.width +
                                 static_cast<std::size_t>(mask_x)] >= 128
                                ? 1
                                : 0;
                    }
                }
            }
            erode_mask(
                images[i].mask, view.width, view.height,
                options.mask_border_px);
            progress.advance();
        });
    stage.finish();
    return images;
}

}  // namespace detail

MvsScene build_mvs_scene(
    const sfm::Scene& sfm_scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.build_scene");
    MvsScene scene;
    scene.thread_count = options.thread_count != 0 ? options.thread_count
                                                   : sfm_scene.thread_count;

    std::vector<Index> registered;
    registered.reserve(sfm_scene.images.size());
    for (const auto& image : sfm_scene.images) {
        if (image.registered) registered.push_back(image.id);
    }
    if (registered.size() < 2)
        throw std::runtime_error("MVS requires at least two registered views");

    scene.views.resize(registered.size());
    std::vector<Index> sfm_to_mvs(sfm_scene.images.size(), k_invalid);

    for (std::size_t i = 0; i < registered.size(); ++i) {
        const Index sfm_id = registered[i];
        sfm_to_mvs[sfm_id] = static_cast<Index>(i);
        const sfm::Image& image = sfm_scene.images[sfm_id];
        const sfm::PinholeCamera& camera = sfm_scene.camera_of(image);
        const auto [w, h] = working_size(
            camera.width, camera.height, options.resolution_level,
            options.min_resolution);
        const float scale_x =
            static_cast<float>(w) / static_cast<float>(camera.width);
        const float scale_y =
            static_cast<float>(h) / static_cast<float>(camera.height);

        MvsView& view = scene.views[i];
        view.id = static_cast<Index>(i);
        view.sfm_image_id = sfm_id;
        view.path = image.path;
        view.pose = image.pose;
        view.width = w;
        view.height = h;
        view.fx = static_cast<float>(camera.fx * scale_x);
        view.fy = static_cast<float>(camera.fy * scale_y);
        view.cx = static_cast<float>(camera.cx * scale_x);
        view.cy = static_cast<float>(camera.cy * scale_y);
        view.src_fx = static_cast<float>(camera.fx);
        view.src_fy = static_cast<float>(camera.fy);
        view.src_cx = static_cast<float>(camera.cx);
        view.src_cy = static_cast<float>(camera.cy);
        view.k1 = static_cast<float>(camera.k1);
        view.k2 = static_cast<float>(camera.k2);
        view.p1 = static_cast<float>(camera.p1);
        view.p2 = static_cast<float>(camera.p2);
        view.src_width = camera.width;
        view.src_height = camera.height;
    }

    for (const auto& track : sfm_scene.tracks) {
        if (!track.is_triangulated()) continue;
        SparsePoint point;
        point.position = track.position.cast<float>();
        point.view_ids.reserve(track.num_inliers);
        for (std::uint8_t o = 0; o < track.num_inliers; ++o) {
            const Index sfm_id = track.observations[o].image_id;
            if (sfm_id >= sfm_to_mvs.size()) continue;
            const Index mvs_id = sfm_to_mvs[sfm_id];
            if (mvs_id == k_invalid) continue;
            point.view_ids.push_back(mvs_id);
        }
        if (point.view_ids.size() >= 2)
            scene.sparse_points.push_back(std::move(point));
    }

    core::Logger::instance().info(
        "mvs scene: views=", scene.views.size(),
        " sparse_points=", scene.sparse_points.size());
    stage.finish();
    return scene;
}

}  // namespace aetherscan::mvs
