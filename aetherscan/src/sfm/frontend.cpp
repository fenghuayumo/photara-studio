#include "sfm/frontend.hpp"

#include "core/logging.hpp"
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

struct FrontEndStageKeys {
    std::uint64_t images{};
    std::uint64_t features{};
    std::uint64_t matches{};
    std::uint64_t geometry{};
    std::uint64_t tracks{};
};

void append_cache_build_identity(FingerprintBuilder& key) {
    key.append_string("aetherscan-cache-abi-20260712-1");
    key.append(static_cast<std::uint64_t>(__cplusplus));
#if defined(_MSC_VER)
    key.append(static_cast<std::uint32_t>(_MSC_VER));
#endif
#if defined(AETHERSCAN_HAS_POSELIB)
    key.append_string("poselib");
#else
    key.append_string("fallback-geometry");
#endif
}

void append_relative_options(
    FingerprintBuilder& key, const RelativePoseOptions& options) {
    key.append(options.max_epipolar_error_px);
    key.append(options.max_reproj_error_px);
    key.append(options.min_ray_angle_deg);
    key.append(options.epipole_filter_px);
    key.append(options.confidence);
    key.append(options.max_iterations);
    key.append(options.min_iterations);
    key.append(options.min_inliers);
    key.append(options.force_fundamental);
    key.append(options.force_shared_focal);
    key.append(options.decompose_fundamental);
    key.append(options.estimate_homography);
    key.append(options.homography_degeneracy_ratio);
    key.append(options.degenerate_weight_scale);
}

FrontEndStageKeys make_stage_keys(
    const FrontEndOptions& options,
    const ImageSetFingerprint& images) {
    FingerprintBuilder features;
    features.append_string("aetherscan-features-v3");
    append_cache_build_identity(features);
    features.append(images.value);
    features.append_string(options.extractor);
    features.append(options.sift_contrast_threshold);
    features.append(options.max_features);

    FingerprintBuilder matches;
    matches.append_string("aetherscan-matches-v3");
    append_cache_build_identity(matches);
    matches.append(features.value());
    matches.append(options.neighbor_window);
    matches.append_string(options.matcher);
    matches.append(options.match_ratio);
    matches.append(options.mutual_check);
    matches.append(options.retrieval_min_images);
    matches.append(options.augment_sequential_with_retrieval);
    matches.append(options.retrieval.top_k);
    matches.append(options.retrieval.max_descriptors_per_image);
    matches.append(options.retrieval.sample_grid);
    matches.append(options.retrieval.stop_word_ratio);
    matches.append(options.retrieval.max_posting_images);
    matches.append(options.retrieval.vocabulary.branching);
    matches.append(options.retrieval.vocabulary.depth);
    matches.append(options.retrieval.vocabulary.max_iterations);
    matches.append(options.retrieval.vocabulary.seed);
    matches.append(options.compress_descriptors_u8);
    matches.append_string(options.retrieval.vocabulary_path.string());

    FingerprintBuilder geometry;
    geometry.append_string("aetherscan-geometry-v3");
    append_cache_build_identity(geometry);
    geometry.append(matches.value());
    geometry.append(options.focal_pixels);
    append_relative_options(geometry, options.relative);

    FingerprintBuilder tracks;
    // Track component membership semantics changed in v4; never reuse tracks
    // produced by the previous edge-count based union bookkeeping.
    tracks.append_string("aetherscan-tracks-v6");
    append_cache_build_identity(tracks);
    tracks.append(geometry.value());
    tracks.append(options.min_pair_weight);
    return {
        images.value, features.value(), matches.value(), geometry.value(),
        tracks.value()};
}

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

void release_descriptors(Scene& scene) {
    for (Image& image : scene.images) image.features.release_descriptors();
}

void compress_descriptors(Scene& scene) {
    for (Image& image : scene.images) image.features.compress_descriptors_u8();
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
    core::StageScope retrieval_stage("sfm.retrieve_image_pairs");
    const auto retrieved = retrieve_image_pairs(scene.images, options.retrieval);
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
    core::StageScope frontend_stage("sfm.frontend");

    const ImageSetFingerprint image_fingerprint =
        fingerprint_image_set(image_paths);
    const FrontEndStageKeys stage_keys =
        make_stage_keys(options, image_fingerprint);
    result.tracks_checkpoint_key = stage_keys.tracks;
    CheckpointStore checkpoints(options.checkpoint);
    Scene& scene = result.scene;
    scene.thread_count = parallel::resolve_thread_count(options.thread_count);
    result.timing.threads_used = scene.thread_count;
    if (checkpoints.load_scene(
            CheckpointStage::tracks, stage_keys.tracks, scene)) {
        scene.thread_count = parallel::resolve_thread_count(options.thread_count);
        verify_image_snapshot(image_paths, image_fingerprint);
        core::Logger::instance().info("checkpoint hit: tracks");
        return result;
    }
    if (checkpoints.load_scene(
            CheckpointStage::geometry, stage_keys.geometry, scene)) {
        scene.thread_count = parallel::resolve_thread_count(options.thread_count);
        const auto started = std::chrono::steady_clock::now();
        build_tracks(scene, options.min_pair_weight);
        result.timing.tracks_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started)
                .count();
        verify_image_snapshot(image_paths, image_fingerprint);
        checkpoints.save_scene(
            CheckpointStage::tracks, stage_keys.tracks, scene);
        core::Logger::instance().info("checkpoint hit: geometry");
        return result;
    }

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

    const auto extract_started = std::chrono::steady_clock::now();
    const unsigned threads = scene.thread_count;
    const bool feature_cache_hit = checkpoints.load_scene(
        CheckpointStage::features, stage_keys.features, scene);
    if (!feature_cache_hit) {
        core::ProgressReporter progress("extract features", image_paths.size());
        scene.images.resize(image_paths.size());
        scene.cameras.reserve(image_paths.size());
        std::vector<std::unique_ptr<features::FeatureExtractor>> workers(threads);
        for (unsigned t = 0; t < threads; ++t) {
            workers[t] =
                extractor->info().thread_safe ? nullptr : extractor->clone();
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
        progress.advance();
        });

        initialize_cameras(scene, options.focal_pixels);
        verify_image_snapshot(image_paths, image_fingerprint);
        checkpoints.save_scene(
            CheckpointStage::features, stage_keys.features, scene);
    } else {
        scene.thread_count = parallel::resolve_thread_count(options.thread_count);
        initialize_cameras(scene, options.focal_pixels);
        core::Logger::instance().info("checkpoint hit: features");
    }
    result.timing.extract_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - extract_started)
            .count();

    FrontEndOptions runtime_options = options;
    if (runtime_options.retrieval.vocabulary_path.empty() &&
        !runtime_options.checkpoint.directory.empty()) {
        runtime_options.retrieval.vocabulary_path =
            runtime_options.checkpoint.directory / "vocabulary-v1.bin";
    }

    const auto match_started = std::chrono::steady_clock::now();
    const auto candidates = build_pair_candidates(scene, runtime_options);
    if (runtime_options.compress_descriptors_u8) compress_descriptors(scene);
    std::vector<RawPairMatches> raw_pairs;
    bool match_cache_hit =
        checkpoints.load_matches(stage_keys.matches, raw_pairs);
    if (match_cache_hit && raw_pairs.size() == candidates.size()) {
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (raw_pairs[i].id1 != candidates[i].id1 ||
                raw_pairs[i].id2 != candidates[i].id2) {
                match_cache_hit = false;
                break;
            }
            const auto& first =
                scene.images[candidates[i].id1].features.keypoints;
            const auto& second =
                scene.images[candidates[i].id2].features.keypoints;
            for (const features::FeatureMatch& match :
                 raw_pairs[i].matches) {
                if (match.query >= first.size() ||
                    match.train >= second.size() ||
                    !std::isfinite(match.score)) {
                    match_cache_hit = false;
                    break;
                }
            }
            if (!match_cache_hit) break;
        }
    } else {
        match_cache_hit = false;
    }
    if (!match_cache_hit) {
        raw_pairs.resize(candidates.size());
        std::vector<std::unique_ptr<features::FeatureMatcher>> match_workers(
            threads);
        for (unsigned t = 0; t < threads; ++t)
            match_workers[t] = matcher->clone();
        auto* ratio_matcher =
            dynamic_cast<features::MutualRatioMatcher*>(match_workers.front().get());
        std::vector<std::uint8_t> active_images(scene.images.size(), 0);
        for (const PairCandidate& candidate : candidates) {
            if (candidate.id1 < active_images.size())
                active_images[candidate.id1] = 1;
            if (candidate.id2 < active_images.size())
                active_images[candidate.id2] = 1;
        }
        std::vector<Index> prepare_ids;
        prepare_ids.reserve(scene.images.size());
        for (Index image_id = 0; image_id < active_images.size(); ++image_id) {
            if (active_images[image_id]) prepare_ids.push_back(image_id);
        }
        if (ratio_matcher) {
            for (const Index image_id : prepare_ids)
                ratio_matcher->pin(scene.images[image_id].features);
        }
        core::ProgressReporter prepare_progress(
            "prepare descriptor indices", prepare_ids.size());
        parallel::parallel_for(
            prepare_ids.size(), threads,
            [&](const std::size_t index, const unsigned tid) {
                match_workers[tid]->prepare(
                    scene.images[prepare_ids[index]].features);
                prepare_progress.advance();
            });
        prepare_progress.finish();
        core::ProgressReporter match_progress(
            "match image pairs", candidates.size());
        parallel::parallel_for(
            candidates.size(), threads,
            [&](const std::size_t ci, const unsigned tid) {
                const PairCandidate candidate = candidates[ci];
                RawPairMatches& cached = raw_pairs[ci];
                cached.id1 = candidate.id1;
                cached.id2 = candidate.id2;
                cached.matches =
                    match_workers[tid]
                        ->match(
                            scene.images[candidate.id1].features,
                            scene.images[candidate.id2].features)
                        .matches;
                match_progress.advance();
            });
        match_progress.finish();
        if (ratio_matcher) {
            for (const Index image_id : prepare_ids)
                ratio_matcher->unpin(scene.images[image_id].features);
        }
        // Shared across matcher clones; release HNSW graphs before geometry.
        match_workers.front()->clear_prepared();
        checkpoints.save_matches(stage_keys.matches, raw_pairs);
    } else {
        core::Logger::instance().info("checkpoint hit: matches");
    }
    // Descriptors are no longer needed after matching; keep keypoints only.
    release_descriptors(scene);

    std::vector<ImagePair> pairs(candidates.size());
    std::vector<PairDiagnostics> diagnostics(candidates.size());
    core::ProgressReporter geometry_progress(
        "verify pair geometry", candidates.size());
    parallel::parallel_for(
        candidates.size(), threads, [&](const std::size_t ci) {
        const PairCandidate cand = candidates[ci];
        const Image& img1 = scene.images[cand.id1];
        const Image& img2 = scene.images[cand.id2];
        const auto& raw = raw_pairs[ci].matches;
        diagnostics[ci].raw_matches = raw.size();
        if (raw.size() < options.relative.min_inliers) {
            geometry_progress.advance();
            return;
        }
        diagnostics[ci].attempted_geometry = true;

        std::vector<Vec2> p1, p2;
        p1.reserve(raw.size());
        p2.reserve(raw.size());
        for (const auto& m : raw) {
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
        if (!geo.success) {
            geometry_progress.advance();
            return;
        }

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
                {raw[i].query, raw[i].train});
        }
        if (pair.matches.size() < options.relative.min_inliers) {
            geometry_progress.advance();
            return;
        }
        diagnostics[ci].accepted = true;
        pairs[ci] = std::move(pair);
        geometry_progress.advance();
        });
    geometry_progress.finish();

    scene.pairs.reserve(pairs.size());
    for (auto& pair : pairs) {
        if (pair.matches.empty()) continue;
        scene.pairs.push_back(std::move(pair));
    }
    verify_image_snapshot(
        image_paths, image_fingerprint, ImageSnapshotCheck::content);
    checkpoints.save_scene(
        CheckpointStage::geometry, stage_keys.geometry, scene);
    result.timing.match_verify_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - match_started)
            .count();

    const auto tracks_started = std::chrono::steady_clock::now();
    build_tracks(scene, options.min_pair_weight);
    checkpoints.save_scene(
        CheckpointStage::tracks, stage_keys.tracks, scene);
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
    core::Logger::instance().info("frontend diagnostics: features=", feature_count,
        " avg_features/image=", average(feature_count, scene.images.size()),
        " candidates=", candidates.size(), " raw_matches=", raw_matches,
        " avg_raw/pair=", average(raw_matches, candidates.size()),
        " geometry_attempts=", geometry_attempts,
        " ransac_inliers=", ransac_inliers,
        " filter_inliers=", filtered_inliers,
        " accepted_pairs=", accepted_pairs,
        " sift_contrast=", options.sift_contrast_threshold,
        " ratio=", options.match_ratio, " mutual=", options.mutual_check);
    core::Logger::instance().info(
        "frontend: images=", scene.images.size(), " pairs=", scene.pairs.size(),
        " planar=", planar_pairs, " tracks=", scene.tracks.size(),
        " observations=", observation_count, " avg_views/track=",
        average(observation_count, scene.tracks.size()));
    return result;
}

}  // namespace aetherscan::sfm
