#include "aetherscan/sfm/frontend.hpp"

#include "aetherscan/features/registry.hpp"
#include "aetherscan/parallel/thread_pool.hpp"
#include "aetherscan/sfm/tracks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace aetherscan::sfm {
namespace {

struct PairCandidate {
    Index id1{};
    Index id2{};
};

std::vector<PairCandidate> build_pair_list(
    const std::size_t image_count, const std::size_t neighbor_window) {
    std::vector<PairCandidate> pairs;
    if (image_count < 2) return pairs;
    if (neighbor_window == 0 || neighbor_window + 1 >= image_count) {
        pairs.reserve(image_count * (image_count - 1) / 2);
        for (Index i = 0; i < image_count; ++i) {
            for (Index j = i + 1; j < image_count; ++j) pairs.push_back({i, j});
        }
        return pairs;
    }
    for (Index i = 0; i < image_count; ++i) {
        for (std::size_t d = 1; d <= neighbor_window && i + d < image_count; ++d) {
            pairs.push_back({i, static_cast<Index>(i + d)});
        }
    }
    return pairs;
}

}  // namespace

FrontEndResult run_frontend(
    const std::vector<std::filesystem::path>& image_paths,
    const FrontEndOptions& options) {
    FrontEndResult result;
    if (image_paths.size() < 2) return result;

    features::ensure_builtin_feature_backends();
    auto extractor = features::create_extractor(options.extractor);
    auto matcher = features::create_matcher(options.matcher);
    if (!extractor || !matcher) {
        throw std::runtime_error("Failed to create feature extractor/matcher");
    }

    Scene& scene = result.scene;
    scene.thread_count = parallel::resolve_thread_count(options.thread_count);
    scene.images.resize(image_paths.size());
    scene.cameras.reserve(image_paths.size());

    const auto extract_started = std::chrono::steady_clock::now();
    const unsigned threads = scene.thread_count;
    result.timing.threads_used = threads;

    std::vector<std::unique_ptr<features::FeatureExtractor>> workers(threads);
    for (unsigned t = 0; t < threads; ++t) {
        workers[t] = extractor->info().thread_safe ? nullptr : extractor->clone();
    }

    parallel::parallel_for(image_paths.size(), threads, [&](const std::size_t i) {
        const unsigned tid = static_cast<unsigned>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % threads);
        features::FeatureExtractor* local =
            workers[tid] ? workers[tid].get() : extractor.get();
        features::FeatureSet features = local->extract_file(image_paths[i]);
        if (options.max_features > 0 && features.keypoints.size() > options.max_features) {
            std::vector<Index> order(features.keypoints.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(
                order.begin(),
                order.begin() + static_cast<std::ptrdiff_t>(options.max_features),
                order.end(),
                [&](Index a, Index b) {
                    return features.keypoints[a].response > features.keypoints[b].response;
                });
            order.resize(options.max_features);
            std::sort(order.begin(), order.end());
            features::FeatureSet trimmed = features;
            trimmed.keypoints.clear();
            trimmed.descriptors.clear();
            trimmed.keypoints.reserve(order.size());
            trimmed.descriptors.reserve(order.size() * features.descriptor_dimension);
            for (Index idx : order) {
                trimmed.keypoints.push_back(features.keypoints[idx]);
                const float* row =
                    features.descriptors.data() + idx * features.descriptor_dimension;
                trimmed.descriptors.insert(
                    trimmed.descriptors.end(), row, row + features.descriptor_dimension);
            }
            features = std::move(trimmed);
        }

        Image image;
        image.id = static_cast<Index>(i);
        image.path = image_paths[i];
        image.features = std::move(features);
        image.camera_id = static_cast<Index>(i);
        scene.images[i] = std::move(image);
    });

    for (Index i = 0; i < scene.images.size(); ++i) {
        const auto& feats = scene.images[i].features;
        PinholeCamera camera;
        camera.id = i;
        camera.width = feats.image_width;
        camera.height = feats.image_height;
        const double focal =
            options.focal_pixels > 0
                ? options.focal_pixels
                : 1.2 * std::max(camera.width, camera.height);
        camera.fx = focal;
        camera.fy = focal;
        camera.cx = 0.5 * camera.width;
        camera.cy = 0.5 * camera.height;
        scene.cameras.push_back(camera);
        scene.images[i].camera_id = i;
    }
    result.timing.extract_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - extract_started)
            .count();

    const auto match_started = std::chrono::steady_clock::now();
    const auto candidates = build_pair_list(scene.images.size(), options.neighbor_window);
    std::vector<ImagePair> pairs(candidates.size());
    std::vector<std::unique_ptr<features::FeatureMatcher>> match_workers(threads);
    for (unsigned t = 0; t < threads; ++t) match_workers[t] = matcher->clone();

    parallel::parallel_for(candidates.size(), threads, [&](const std::size_t ci) {
        const PairCandidate cand = candidates[ci];
        const Image& img1 = scene.images[cand.id1];
        const Image& img2 = scene.images[cand.id2];
        features::FeatureMatcher* local = match_workers[ci % threads].get();
        features::MatchSet raw = local->match(img1.features, img2.features);
        if (raw.matches.size() < options.relative.min_inliers) return;

        std::vector<Vec2> p1, p2;
        p1.reserve(raw.matches.size());
        p2.reserve(raw.matches.size());
        for (const auto& m : raw.matches) {
            const auto& k1 = img1.features.keypoints[m.query];
            const auto& k2 = img2.features.keypoints[m.train];
            p1.emplace_back(k1.x, k1.y);
            p2.emplace_back(k2.x, k2.y);
        }

        RelativePoseResult geo = estimate_relative_pose(
            p1, p2, scene.cameras[img1.camera_id], scene.cameras[img2.camera_id],
            options.relative);
        if (!geo.success) return;

        ImagePair pair(cand.id1, cand.id2);
        pair.relative_pose = geo.pose;
        pair.E = geo.E;
        pair.F = geo.F;
        pair.mean_ray_angle = geo.mean_ray_angle;
        pair.weight_spatial = geo.weight_spatial;
        for (std::size_t i = 0; i < geo.inlier_mask.size(); ++i) {
            if (!geo.inlier_mask[i]) continue;
            pair.matches.push_back(
                {raw.matches[i].query, raw.matches[i].train});
        }
        if (pair.matches.size() < options.relative.min_inliers) return;
        pairs[ci] = std::move(pair);
    });

    scene.pairs.reserve(pairs.size());
    for (auto& pair : pairs) {
        if (pair.matches.empty()) continue;
        scene.pairs.push_back(std::move(pair));
    }
    result.timing.match_verify_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - match_started)
            .count();

    const auto tracks_started = std::chrono::steady_clock::now();
    build_tracks(scene, options.min_pair_weight);
    result.timing.tracks_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tracks_started)
            .count();

    std::cout << "frontend: images=" << scene.images.size()
              << " pairs=" << scene.pairs.size() << " tracks=" << scene.tracks.size()
              << '\n';
    return result;
}

}  // namespace aetherscan::sfm
