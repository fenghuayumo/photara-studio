#include "sfm/frontend.hpp"

#include "features/features.hpp"
#include "features/registry.hpp"
#include "parallel/thread_pool.hpp"
#include "sfm/tracks.hpp"

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

struct PairDiagnostics {
    std::size_t raw_matches{0};
    unsigned ransac_inliers{0};
    unsigned filtered_inliers{0};
    bool attempted_geometry{false};
    bool accepted{false};
};

void initialize_cameras(Scene& scene, const double focal_pixels) {
    scene.cameras.clear();
    scene.cameras.reserve(scene.images.size());
    for (Index i = 0; i < scene.images.size(); ++i) {
        const auto& features = scene.images[i].features;
        PinholeCamera camera;
        camera.width = features.image_width;
        camera.height = features.image_height;
        const double focal =
            focal_pixels > 0
                ? focal_pixels
                : 1.2 * std::max(camera.width, camera.height);
        camera.fx = focal;
        camera.fy = focal;
        camera.cx = 0.5 * camera.width;
        camera.cy = 0.5 * camera.height;
        const auto existing = std::find_if(
            scene.cameras.begin(), scene.cameras.end(),
            [&](const PinholeCamera& candidate) {
                return candidate.width == camera.width &&
                       candidate.height == camera.height &&
                       std::abs(candidate.fx - camera.fx) < 1e-9 &&
                       std::abs(candidate.fy - camera.fy) < 1e-9;
            });
        if (existing == scene.cameras.end()) {
            camera.id = static_cast<Index>(scene.cameras.size());
            scene.images[i].camera_id = camera.id;
            scene.cameras.push_back(camera);
        } else {
            scene.images[i].camera_id =
                static_cast<Index>(existing - scene.cameras.begin());
        }
    }
}

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

std::vector<PairCandidate> build_pair_candidates(
    const Scene& scene, const FrontEndOptions& options) {
    const bool use_retrieval =
        options.retrieval.top_k > 0 &&
        scene.images.size() >= options.retrieval_min_images &&
        (options.neighbor_window == 0 ||
         options.augment_sequential_with_retrieval);
    if (!use_retrieval)
        return build_pair_list(scene.images.size(), options.neighbor_window);

    std::vector<PairCandidate> pairs;
    if (options.neighbor_window > 0)
        pairs = build_pair_list(scene.images.size(), options.neighbor_window);
    const auto retrieved =
        retrieve_image_pairs(scene.images, options.retrieval);
    pairs.reserve(pairs.size() + retrieved.size());
    for (const RetrievedPair& pair : retrieved)
        pairs.push_back({pair.first, pair.second});
    std::sort(
        pairs.begin(), pairs.end(),
        [](const PairCandidate& left, const PairCandidate& right) {
            return left.id1 < right.id1 ||
                   (left.id1 == right.id1 && left.id2 < right.id2);
        });
    pairs.erase(
        std::unique(
            pairs.begin(), pairs.end(),
            [](const PairCandidate& left, const PairCandidate& right) {
                return left.id1 == right.id1 && left.id2 == right.id2;
            }),
        pairs.end());
    if (pairs.empty())
        return build_pair_list(scene.images.size(), options.neighbor_window);
    return pairs;
}

}  // namespace

FrontEndResult run_frontend(
    const std::vector<std::filesystem::path>& image_paths,
    const FrontEndOptions& options) {
    FrontEndResult result;
    if (image_paths.size() < 2) return result;

    features::ensure_builtin_feature_backends();
    std::unique_ptr<features::FeatureExtractor> extractor;
    if (options.extractor == "sift") {
        features::SiftOptions sift_options;
        sift_options.contrast_threshold = options.sift_contrast_threshold;
        if (options.max_features > 0) {
            sift_options.maximum_features = options.max_features;
            sift_options.max_features_per_cell =
                (std::max)(
                    std::size_t{1}, static_cast<std::size_t>(options.max_features) / 9);
            sift_options.min_features_per_cell =
                (std::min)(sift_options.min_features_per_cell,
                           sift_options.max_features_per_cell);
        }
        extractor = std::make_unique<features::SiftExtractor>(sift_options);
    } else {
        extractor = features::create_extractor(options.extractor);
    }
    std::unique_ptr<features::FeatureMatcher> matcher;
    if (options.matcher == "mutual_ratio") {
        features::DescriptorMatcherOptions matcher_options;
        matcher_options.ratio_threshold = options.match_ratio;
        matcher_options.mutual_check = options.mutual_check;
        matcher = std::make_unique<features::MutualRatioMatcher>(matcher_options);
    } else {
        matcher = features::create_matcher(options.matcher);
    }
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

    parallel::parallel_for(
        image_paths.size(), threads,
        [&](const std::size_t i, const unsigned tid) {
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
        scene.images[i] = std::move(image);
        });

    initialize_cameras(scene, options.focal_pixels);
    result.timing.extract_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - extract_started)
            .count();

    const auto match_started = std::chrono::steady_clock::now();
    const auto candidates = build_pair_candidates(scene, options);
    std::vector<ImagePair> pairs(candidates.size());
    std::vector<PairDiagnostics> diagnostics(candidates.size());
    std::vector<std::unique_ptr<features::FeatureMatcher>> match_workers(threads);
    for (unsigned t = 0; t < threads; ++t) match_workers[t] = matcher->clone();
    parallel::parallel_for(
        scene.images.size(), threads,
        [&](const std::size_t image_id, const unsigned tid) {
            match_workers[tid]->prepare(scene.images[image_id].features);
        });

    parallel::parallel_for(
        candidates.size(), threads,
        [&](const std::size_t ci, const unsigned tid) {
        const PairCandidate cand = candidates[ci];
        const Image& img1 = scene.images[cand.id1];
        const Image& img2 = scene.images[cand.id2];
        features::FeatureMatcher* local = match_workers[tid].get();
        features::MatchSet raw = local->match(img1.features, img2.features);
        diagnostics[ci].raw_matches = raw.matches.size();
        if (raw.matches.size() < options.relative.min_inliers) return;
        diagnostics[ci].attempted_geometry = true;

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
        diagnostics[ci].ransac_inliers = geo.num_ransac_inliers;
        diagnostics[ci].filtered_inliers = geo.num_inliers;
        if (!geo.success) return;

        ImagePair pair(cand.id1, cand.id2);
        pair.relative_pose = geo.pose;
        pair.E = geo.E;
        pair.F = geo.F;
        pair.H = geo.H;
        pair.mean_ray_angle = geo.mean_ray_angle;
        pair.weight_spatial = geo.weight_spatial;
        pair.homography_ratio = geo.homography_ratio;
        pair.degenerate_planar = geo.degenerate_planar;
        pair.weight_geometry =
            geo.degenerate_planar ? options.relative.degenerate_weight_scale : 1.F;
        // Keep pair for track connectivity; star_init skips planar via usable_for_init().
        for (std::size_t i = 0; i < geo.inlier_mask.size(); ++i) {
            if (!geo.inlier_mask[i]) continue;
            pair.matches.push_back(
                {raw.matches[i].query, raw.matches[i].train});
        }
        if (pair.matches.size() < options.relative.min_inliers) return;
        diagnostics[ci].accepted = true;
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

    unsigned planar_pairs = 0;
    std::size_t feature_count = 0;
    std::size_t observation_count = 0;
    for (const ImagePair& pair : scene.pairs) {
        if (pair.degenerate_planar) ++planar_pairs;
    }
    for (const Image& image : scene.images) feature_count += image.features.keypoints.size();
    for (const Track& track : scene.tracks) observation_count += track.observations.size();

    std::size_t raw_matches = 0;
    std::size_t ransac_inliers = 0;
    std::size_t filtered_inliers = 0;
    unsigned geometry_attempts = 0;
    unsigned accepted_pairs = 0;
    for (const PairDiagnostics& pair : diagnostics) {
        raw_matches += pair.raw_matches;
        ransac_inliers += pair.ransac_inliers;
        filtered_inliers += pair.filtered_inliers;
        geometry_attempts += pair.attempted_geometry ? 1U : 0U;
        accepted_pairs += pair.accepted ? 1U : 0U;
    }
    const auto average = [](const std::size_t total, const std::size_t count) {
        return count > 0 ? static_cast<double>(total) / static_cast<double>(count) : 0.0;
    };
    std::cout << "frontend diagnostics: features=" << feature_count
              << " avg_features/image=" << average(feature_count, scene.images.size())
              << " candidates=" << candidates.size()
              << " raw_matches=" << raw_matches
              << " avg_raw/pair=" << average(raw_matches, candidates.size())
              << " geometry_attempts=" << geometry_attempts
              << " ransac_inliers=" << ransac_inliers
              << " filter_inliers=" << filtered_inliers
              << " accepted_pairs=" << accepted_pairs
              << " sift_contrast=" << options.sift_contrast_threshold
              << " ratio=" << options.match_ratio
              << " mutual=" << options.mutual_check << '\n';
    std::cout << "frontend: images=" << scene.images.size()
              << " pairs=" << scene.pairs.size() << " planar=" << planar_pairs
              << " tracks=" << scene.tracks.size()
              << " observations=" << observation_count
              << " avg_views/track=" << average(observation_count, scene.tracks.size())
              << '\n';
    return result;
}

}  // namespace aetherscan::sfm
