#include "sfm/frontend.hpp"

#include "core/logging.hpp"
#include "features/compat.hpp"
#include "features/features.hpp"
#include "features/registry.hpp"
#include "io/image.hpp"
#include "parallel/thread_pool.hpp"
#include "sfm/tracks.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace aetherscan::sfm {
namespace {

#if !defined(AETHERSCAN_CACHE_BUILD_ID)
#define AETHERSCAN_CACHE_BUILD_ID "unconfigured"
#endif

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
    key.append_string(AETHERSCAN_CACHE_BUILD_ID);
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

// Canonicalize feature-backend selection without changing the default
// siftgpu × gpu_mutual_ratio path when pipeline is empty / "none".
void normalize_feature_selection(FrontEndOptions& options) {
    if (options.pipeline == "none") options.pipeline.clear();

    // Legacy: --matcher siftgpu meant GPU descriptor mutual-ratio, not "SIFT match".
    if (options.matcher == "siftgpu")
        options.matcher = "gpu_mutual_ratio";

    if (options.pipeline.empty()) {
        if (options.matcher == "lightglue" ||
            options.matcher == "hybrid_lightglue" ||
            options.extractor == "superpoint" ||
            options.extractor == "disk" || options.extractor == "aliked")
            options.compress_descriptors_u8 = false;
        return;
    }

    if (!features::is_pair_pipeline_name(options.pipeline))
        throw std::invalid_argument(
            "Unknown --pipeline '" + options.pipeline +
            "' (supported: none, lightglue_end2end)");

    if (options.matcher == "lightglue" ||
        options.matcher == "hybrid_lightglue")
        throw std::invalid_argument(
            "Use either a LightGlue descriptor matcher or "
            "--pipeline lightglue_end2end (fused), not both");

    // Canonical fingerprint fields for fused LightGlue.
    if (options.pipeline == features::kLightGlueEnd2EndPipeline) {
        options.matcher = "lightglue_end2end";
        options.compress_descriptors_u8 = false;
    }
}

[[nodiscard]] bool uses_pair_feature_pipeline(
    const FrontEndOptions& options) noexcept {
    return features::is_pair_pipeline_name(options.pipeline);
}

FrontEndStageKeys make_stage_keys(
    const FrontEndOptions& options,
    const ImageSetFingerprint& images) {
    FingerprintBuilder features;
    features.append_string("features");
    append_cache_build_identity(features);
    features.append(images.value);
    features.append_string(options.extractor);
    features.append(options.sift_contrast_threshold);
    features.append(options.max_features);
    features.append_string(options.pipeline);
    features.append_string(options.matcher);
    features.append_string(options.extractor_model_path.string());
    features.append(options.extractor_input_width);
    features.append(options.extractor_input_height);
    features.append(options.extractor_min_score);
    features.append(options.extractor_use_cuda);
    features.append_string(options.lightglue_model_path.string());
    features.append_string(options.lightglue_extractor);
    features.append(options.lightglue_input_width);
    features.append(options.lightglue_input_height);
    features.append(options.lightglue_min_score);
    features.append(options.hybrid_lightglue_max_features);
    features.append(options.lightglue_use_cuda);

    FingerprintBuilder matches;
    matches.append_string("matches");
    append_cache_build_identity(matches);
    matches.append(features.value());
    matches.append(options.neighbor_window);
    matches.append_string(options.pipeline);
    matches.append_string(options.matcher);
    matches.append(options.match_ratio);
    matches.append(options.mutual_check);
    matches.append_string(options.extractor_model_path.string());
    matches.append(options.extractor_input_width);
    matches.append(options.extractor_input_height);
    matches.append(options.extractor_min_score);
    matches.append(options.extractor_use_cuda);
    matches.append_string(options.lightglue_model_path.string());
    matches.append_string(options.lightglue_extractor);
    matches.append(options.lightglue_input_width);
    matches.append(options.lightglue_input_height);
    matches.append(options.lightglue_min_score);
    matches.append(options.hybrid_lightglue_max_features);
    matches.append(options.lightglue_use_cuda);
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
    geometry.append_string("geometry");
    append_cache_build_identity(geometry);
    geometry.append(matches.value());
    geometry.append(options.focal_pixels);
    geometry.append(options.trust_focal_pixels);
    append_relative_options(geometry, options.relative);

    FingerprintBuilder tracks;
    tracks.append_string("tracks");
    append_cache_build_identity(tracks);
    tracks.append(geometry.value());
    tracks.append(options.min_pair_weight);
    tracks.append(options.pair_weighting.min_inliers);
    tracks.append(options.pair_weighting.max_triplet_rotation_error_deg);
    tracks.append(options.pair_weighting.triplet_saturation);
    tracks.append(options.pair_weighting.min_triplets_for_penalty);
    tracks.append(options.pair_weighting.max_inconsistent_triplet_ratio);
    tracks.append(options.pair_weighting.inconsistent_triplet_scale);
    return {
        images.value, features.value(), matches.value(), geometry.value(),
        tracks.value()};
}

void initialize_cameras(
    Scene& scene, const double focal_pixels, const bool trust_focal_pixels) {
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
        camera.focal_prior = focal;
        camera.cx = 0.5 * camera.width;
        camera.cy = 0.5 * camera.height;
        camera.trust_intrinsics = trust_focal_pixels;
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

double weighted_median(
    std::vector<std::pair<double, double>> samples) {
    if (samples.empty()) return 0.0;
    std::sort(
        samples.begin(), samples.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    double total = 0.0;
    for (const auto& sample : samples) total += sample.second;
    if (!(total > 0.0)) return samples[samples.size() / 2].first;
    double accumulated = 0.0;
    for (const auto& sample : samples) {
        accumulated += sample.second;
        if (accumulated >= 0.5 * total) return sample.first;
    }
    return samples.back().first;
}

void calibrate_view_graph_focals(Scene& scene) {
    std::vector<std::vector<std::pair<double, double>>> grouped(
        scene.cameras.size());
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.degenerate_planar ||
            !pair.estimated_focal.has_value() ||
            pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size())
            continue;
        const Index first_group = scene.images[pair.id1].camera_id;
        const Index second_group = scene.images[pair.id2].camera_id;
        if (first_group != second_group || first_group >= scene.cameras.size())
            continue;
        const PinholeCamera& camera = scene.cameras[first_group];
        if (camera.trust_intrinsics) continue;
        const double focal = *pair.estimated_focal;
        const double image_scale =
            static_cast<double>(std::max(camera.width, camera.height));
        if (!std::isfinite(focal) || focal < 0.25 * image_scale ||
            focal > 4.0 * image_scale)
            continue;
        const double weight =
            std::max(static_cast<double>(pair.composite_weight()), 1e-3);
        grouped[first_group].push_back({std::log(focal), weight});
    }

    for (std::size_t group = 0; group < grouped.size(); ++group) {
        auto& samples = grouped[group];
        if (samples.size() < 8) continue;
        const double center = weighted_median(samples);
        std::vector<std::pair<double, double>> deviations;
        deviations.reserve(samples.size());
        for (const auto& sample : samples)
            deviations.push_back(
                {std::abs(sample.first - center), sample.second});
        const double mad = weighted_median(std::move(deviations));
        const double cutoff =
            std::max(3.0 * 1.4826 * mad, std::log(1.15));
        std::vector<std::pair<double, double>> inliers;
        inliers.reserve(samples.size());
        for (const auto& sample : samples)
            if (std::abs(sample.first - center) <= cutoff)
                inliers.push_back(sample);
        if (inliers.size() < 8) continue;
        const double focal = std::exp(weighted_median(inliers));
        PinholeCamera& camera = scene.cameras[group];
        const double initial = camera.focal();
        camera.fx = focal;
        camera.fy = focal;
        camera.focal_prior = focal;
        core::Logger::instance().info(
            "view-graph focal: camera=", group,
            " initial=", initial, " consensus=", focal,
            " samples=", samples.size(), " inliers=", inliers.size(),
            " log_mad=", mad);
    }
}

void release_descriptors(Scene& scene) {
    for (Image& image : scene.images) image.features.release_descriptors();
}

void compress_descriptors(Scene& scene) {
    for (Image& image : scene.images) image.features.compress_descriptors_u8();
}

float keypoint_selection_weight(const features::Keypoint& keypoint) {
    if (!(keypoint.response > 0.F)) return 0.F;
    const float response_weight =
        keypoint.response / (keypoint.response + 0.03F);
    const float scale =
        std::clamp(keypoint.scale, 2.F, 20.F);
    const float scale_weight = 0.5F + (scale - 2.F) / 18.F;
    return response_weight * scale_weight;
}

// openMVS SelectTopKeypoints: 3x3 spatial cells + round-robin by quality.
features::FeatureSet select_top_features_grid_3x3(
    features::FeatureSet features, const unsigned max_features) {
    if (max_features == 0 || features.keypoints.size() <= max_features)
        return features;
    if (features.image_width == 0 || features.image_height == 0)
        return features;

    constexpr int k_grid = 3;
    const float cell_width =
        static_cast<float>(features.image_width) / static_cast<float>(k_grid);
    const float cell_height =
        static_cast<float>(features.image_height) / static_cast<float>(k_grid);
    std::array<std::vector<Index>, k_grid * k_grid> cells;
    for (Index index = 0; index < features.keypoints.size(); ++index) {
        const auto& keypoint = features.keypoints[index];
        const int column = std::clamp(
            static_cast<int>(keypoint.x / cell_width), 0, k_grid - 1);
        const int row = std::clamp(
            static_cast<int>(keypoint.y / cell_height), 0, k_grid - 1);
        cells[static_cast<std::size_t>(row * k_grid + column)].push_back(index);
    }
    for (auto& cell : cells) {
        std::sort(cell.begin(), cell.end(), [&](const Index left, const Index right) {
            return keypoint_selection_weight(features.keypoints[left]) >
                   keypoint_selection_weight(features.keypoints[right]);
        });
    }

    std::vector<Index> selected;
    selected.reserve(max_features);
    std::array<std::size_t, k_grid * k_grid> offsets{};
    for (int current = 0; selected.size() < max_features;
         current = (current + 1) % (k_grid * k_grid)) {
        auto& offset = offsets[static_cast<std::size_t>(current)];
        const auto& cell = cells[static_cast<std::size_t>(current)];
        if (offset >= cell.size()) {
            // All cells exhausted? Round-robin still advances; break if stuck.
            bool any_remaining = false;
            for (std::size_t cell_index = 0; cell_index < cells.size();
                 ++cell_index) {
                if (offsets[cell_index] < cells[cell_index].size()) {
                    any_remaining = true;
                    break;
                }
            }
            if (!any_remaining) break;
            continue;
        }
        selected.push_back(cell[offset++]);
    }
    std::sort(selected.begin(), selected.end());

    features::FeatureSet trimmed = features;
    trimmed.keypoints.clear();
    trimmed.descriptors.clear();
    trimmed.descriptors_u8.clear();
    trimmed.keypoints.reserve(selected.size());
    const bool has_float =
        features.storage == features::DescriptorStorage::float32 &&
        !features.descriptors.empty();
    const bool has_u8 =
        features.storage == features::DescriptorStorage::uint8 &&
        !features.descriptors_u8.empty();
    if (has_float)
        trimmed.descriptors.reserve(
            selected.size() * features.descriptor_dimension);
    if (has_u8)
        trimmed.descriptors_u8.reserve(
            selected.size() * features.descriptor_dimension);
    for (const Index index : selected) {
        trimmed.keypoints.push_back(features.keypoints[index]);
        if (has_float) {
            const float* row = features.descriptors.data() +
                               index * features.descriptor_dimension;
            trimmed.descriptors.insert(
                trimmed.descriptors.end(), row,
                row + features.descriptor_dimension);
        }
        if (has_u8) {
            const auto* row = features.descriptors_u8.data() +
                              index * features.descriptor_dimension;
            trimmed.descriptors_u8.insert(
                trimmed.descriptors_u8.end(), row,
                row + features.descriptor_dimension);
        }
    }
    trimmed.mark_descriptors_modified();
    return trimmed;
}

// openMVS OptimizePairsOrder: group by id1 for GPU descriptor cache reuse,
// then prefer harder (larger descriptor product) pairs within a group.
void optimize_pairs_order(
    std::vector<PairCandidate>& pairs, const Scene& scene) {
    if (pairs.size() < 2) return;
    struct PairCost {
        PairCandidate pair{};
        std::size_t cost{0};
    };
    std::vector<PairCost> ranked;
    ranked.reserve(pairs.size());
    for (const PairCandidate& pair : pairs) {
        const std::size_t left =
            pair.id1 < scene.images.size()
                ? scene.images[pair.id1].features.keypoints.size()
                : 0;
        const std::size_t right =
            pair.id2 < scene.images.size()
                ? scene.images[pair.id2].features.keypoints.size()
                : 0;
        ranked.push_back({pair, left * right});
    }
    std::stable_sort(
        ranked.begin(), ranked.end(),
        [](const PairCost& left, const PairCost& right) {
            if (left.pair.id1 != right.pair.id1)
                return left.pair.id1 < right.pair.id1;
            return left.cost > right.cost;
        });
    for (std::size_t index = 0; index < pairs.size(); ++index)
        pairs[index] = ranked[index].pair;
}

Image make_image_from_features(
    const Index image_id, const std::filesystem::path& path,
    features::FeatureSet features) {
    Image image;
    image.id = image_id;
    image.path = path;
    image.features = std::move(features);
    return image;
}

// Owner-thread extract coordinator: main thread owns CUDA/ORT context while
// worker threads overlap CPU post-processing (grid selection / Image packing).
void extract_features_siftgpu_coordinator(
    Scene& scene, const std::vector<std::filesystem::path>& image_paths,
    features::FeatureExtractor& extractor, const unsigned max_features,
    core::ProgressReporter& progress) {
    const std::size_t count = image_paths.size();
    scene.images.resize(count);
    auto& pool = parallel::global_thread_pool();
    parallel::FutureGroup post_tasks;
    post_tasks.reserve(count);

    const bool use_gray_prefetch =
        extractor.info().accepts_gray && !extractor.info().accepts_rgb;

    if (use_gray_prefetch) {
        std::future<io::GrayImage> current_load = std::async(
            std::launch::async,
            [&image_paths] { return io::load_gray(image_paths[0]); });

        for (std::size_t index = 0; index < count; ++index) {
            std::future<io::GrayImage> next_load;
            if (index + 1 < count) {
                const std::filesystem::path next_path = image_paths[index + 1];
                next_load = std::async(
                    std::launch::async,
                    [next_path] { return io::load_gray(next_path); });
            }

            io::GrayImage gray = current_load.get();
            features::FeatureSet features = extractor.extract_gray(
                gray.pixels, gray.width, gray.height);

            post_tasks.submit(
                pool,
                [&, index, path = image_paths[index],
                 features = std::move(features)]() mutable {
                    features = select_top_features_grid_3x3(
                        std::move(features), max_features);
                    scene.images[index] = make_image_from_features(
                        static_cast<Index>(index), path, std::move(features));
                    progress.advance();
                });

            if (index + 1 < count) current_load = std::move(next_load);
        }
    } else {
        // Learned RGB extractors (DISK) / SuperPoint via extract_file.
        // Skip grid selection; models already apply top-k.
        for (std::size_t index = 0; index < count; ++index) {
            features::FeatureSet features =
                extractor.extract_file(image_paths[index]);
            post_tasks.submit(
                pool,
                [&, index, path = image_paths[index],
                 features = std::move(features)]() mutable {
                    scene.images[index] = make_image_from_features(
                        static_cast<Index>(index), path, std::move(features));
                    progress.advance();
                });
        }
    }

    post_tasks.wait();
}

struct GeometryVerifyResult {
    PairDiagnostics diagnostics;
    std::optional<ImagePair> pair;
};

GeometryVerifyResult verify_pair_geometry(
    const Scene& scene, const PairCandidate& candidate,
    const std::vector<features::FeatureMatch>& raw,
    const RelativePoseOptions& relative) {
    GeometryVerifyResult result;
    result.diagnostics.raw_matches = raw.size();
    if (raw.size() < relative.min_inliers) return result;

    const Image& img1 = scene.images[candidate.id1];
    const Image& img2 = scene.images[candidate.id2];
    result.diagnostics.attempted_geometry = true;

    std::vector<Vec2> p1, p2;
    p1.reserve(raw.size());
    p2.reserve(raw.size());
    for (const auto& match : raw) {
        const auto& k1 = img1.features.keypoints[match.query];
        const auto& k2 = img2.features.keypoints[match.train];
        p1.emplace_back(k1.x, k1.y);
        p2.emplace_back(k2.x, k2.y);
    }

    const RelativePoseResult geo = estimate_relative_pose(
        p1, p2, scene.cameras[img1.camera_id], scene.cameras[img2.camera_id],
        relative);
    result.diagnostics.ransac_inliers = geo.num_ransac_inliers;
    result.diagnostics.filtered_inliers = geo.num_inliers;
    if (!geo.success) return result;

    ImagePair pair(candidate.id1, candidate.id2);
    pair.relative_pose = geo.pose;
    pair.E = geo.E;
    pair.F = geo.F;
    pair.estimated_focal = geo.estimated_focal;
    pair.H = geo.H;
    pair.mean_ray_angle = geo.mean_ray_angle;
    pair.weight_spatial = geo.weight_spatial;
    pair.homography_ratio = geo.homography_ratio;
    pair.degenerate_planar = geo.degenerate_planar;
    pair.weight_geometry =
        geo.degenerate_planar ? relative.degenerate_weight_scale : 1.F;
    for (std::size_t i = 0; i < geo.inlier_mask.size(); ++i) {
        if (!geo.inlier_mask[i]) continue;
        pair.matches.push_back({raw[i].query, raw[i].train});
    }
    if (pair.matches.size() < relative.min_inliers) return result;

    result.diagnostics.accepted = true;
    result.pair = std::move(pair);
    return result;
}

// GPU/ORT matching stays on the owner thread while geometric verification is
// continuously consumed by the CPU pool. A bounded in-flight window prevents
// an arbitrarily large raw-match backlog without introducing batch barriers.
void match_and_verify_siftgpu_coordinator(
    Scene& scene, const std::vector<PairCandidate>& candidates,
    features::FeatureMatcher& matcher, const FrontEndOptions& options,
    const unsigned worker_threads, std::vector<RawPairMatches>& raw_pairs,
    std::vector<PairDiagnostics>& diagnostics,
    std::vector<ImagePair>& pair_slots, core::ProgressReporter& match_progress) {
    raw_pairs.resize(candidates.size());
    diagnostics.assign(candidates.size(), {});
    pair_slots.assign(candidates.size(), ImagePair{});

    auto& pool = parallel::global_thread_pool();
    const std::size_t max_in_flight = std::max<std::size_t>(
        static_cast<std::size_t>(worker_threads) * 8U, 64U);
    std::mutex throttle_mutex;
    std::condition_variable throttle_condition;
    std::size_t in_flight = 0;
    parallel::FutureGroup geometry_tasks;
    geometry_tasks.reserve(candidates.size());

    for (std::size_t pair_index = 0; pair_index < candidates.size();
         ++pair_index) {
        const PairCandidate candidate = candidates[pair_index];
        RawPairMatches raw;
        raw.id1 = candidate.id1;
        raw.id2 = candidate.id2;
        raw.matches =
            matcher
                .match(
                    scene.images[candidate.id1].features,
                    scene.images[candidate.id2].features)
                .matches;

        {
            std::unique_lock lock(throttle_mutex);
            throttle_condition.wait(
                lock, [&] { return in_flight < max_in_flight; });
            ++in_flight;
        }

        geometry_tasks.submit(
            pool,
            [&, pair_index, candidate, raw = std::move(raw)]() mutable {
                const auto release_slot = [&] {
                    {
                        std::lock_guard lock(throttle_mutex);
                        --in_flight;
                    }
                    throttle_condition.notify_one();
                };
                try {
                    raw_pairs[pair_index] = std::move(raw);
                    GeometryVerifyResult verified = verify_pair_geometry(
                        scene, candidate, raw_pairs[pair_index].matches,
                        options.relative);
                    diagnostics[pair_index] = verified.diagnostics;
                    if (verified.pair) pair_slots[pair_index] = *verified.pair;
                    match_progress.advance();
                } catch (...) {
                    release_slot();
                    throw;
                }
                release_slot();
            });
    }
    geometry_tasks.wait();
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

features::FeatureIndex merge_lightglue_keypoint(
    features::FeatureSet& features, const float x, const float y,
    const float merge_radius_px = 1.5F) {
    const float radius_sq = merge_radius_px * merge_radius_px;
    for (std::size_t index = 0; index < features.keypoints.size(); ++index) {
        const float dx = features.keypoints[index].x - x;
        const float dy = features.keypoints[index].y - y;
        if (dx * dx + dy * dy <= radius_sq)
            return static_cast<features::FeatureIndex>(index);
    }
    features.keypoints.push_back({x, y, 1.0F, 0.0F, 0.0F});
    return static_cast<features::FeatureIndex>(features.keypoints.size() - 1);
}

features::LightGlueOptions make_lightglue_options(
    const FrontEndOptions& options) {
    features::LightGlueOptions lightglue;
    lightglue.model_path = options.lightglue_model_path;
    if (options.lightglue_extractor == "superpoint")
        lightglue.extractor = features::LightGlueExtractor::superpoint;
    else if (
        options.lightglue_extractor == "disk" ||
        options.lightglue_extractor.empty())
        lightglue.extractor = features::LightGlueExtractor::disk;
    else
        throw std::invalid_argument(
            "lightglue_extractor must be disk or superpoint");
    lightglue.device = options.lightglue_use_cuda
        ? features::InferenceDevice::cuda
        : features::InferenceDevice::cpu;
    lightglue.input_width = options.lightglue_input_width;
    lightglue.input_height = options.lightglue_input_height;
    lightglue.min_score = options.lightglue_min_score;
    return lightglue;
}

std::unique_ptr<features::FeatureExtractor> make_frontend_extractor(
    const FrontEndOptions& options) {
    if (options.extractor == "sift") {
        features::SiftOptions sift_options;
        sift_options.contrast_threshold = options.sift_contrast_threshold;
        if (options.max_features > 0) {
            sift_options.maximum_features = options.max_features;
            sift_options.max_features_per_cell = (std::max)(
                std::size_t{1},
                static_cast<std::size_t>(options.max_features) / 9);
            sift_options.min_features_per_cell = (std::min)(
                sift_options.min_features_per_cell,
                sift_options.max_features_per_cell);
        }
        return std::make_unique<features::SiftExtractor>(sift_options);
    }
    if (options.extractor == "siftgpu") {
        features::SiftGpuOptions siftgpu_options;
        siftgpu_options.peak_threshold =
            static_cast<float>(options.sift_contrast_threshold);
        if (options.max_features > 0)
            siftgpu_options.maximum_features = options.max_features;
        auto extractor =
            std::make_unique<features::SiftGpuExtractor>(siftgpu_options);
        if (!extractor->is_available())
            throw std::runtime_error(
                "SiftGPU extractor requested but CUDA context is unavailable");
        return extractor;
    }
    if (options.extractor == "superpoint") {
        if (options.extractor_model_path.empty())
            throw std::runtime_error(
                "extractor superpoint requires --extractor-model");
        if (!features::SuperPointExtractor::is_built())
            throw std::runtime_error(
                "SuperPoint requires ONNX Runtime (AETHERSCAN_ENABLE_ONNX)");
        features::SuperPointOptions sp;
        sp.model_path = options.extractor_model_path;
        sp.maximum_features = options.max_features;
        sp.input_width = options.extractor_input_width;
        sp.input_height = options.extractor_input_height;
        if (options.extractor_min_score >= 0.F)
            sp.keypoint_threshold = options.extractor_min_score;
        sp.cuda = options.extractor_use_cuda;
        auto extractor = std::make_unique<features::SuperPointExtractor>(sp);
        if (!extractor->is_available())
            throw std::runtime_error("SuperPoint failed to initialize ONNX");
        return extractor;
    }
    if (options.extractor == "disk") {
        if (options.extractor_model_path.empty())
            throw std::runtime_error(
                "extractor disk requires --extractor-model");
        if (!features::DiskExtractor::is_built())
            throw std::runtime_error(
                "DISK requires ONNX Runtime (AETHERSCAN_ENABLE_ONNX)");
        features::DiskOptions disk;
        disk.model_path = options.extractor_model_path;
        disk.maximum_features = options.max_features;
        disk.input_width = options.extractor_input_width;
        disk.input_height = options.extractor_input_height;
        if (options.extractor_min_score >= 0.F)
            disk.keypoint_threshold = options.extractor_min_score;
        disk.cuda = options.extractor_use_cuda;
        auto extractor = std::make_unique<features::DiskExtractor>(disk);
        if (!extractor->is_available())
            throw std::runtime_error("DISK failed to initialize ONNX");
        return extractor;
    }
    if (options.extractor == "aliked") {
        if (options.extractor_model_path.empty())
            throw std::runtime_error(
                "extractor aliked requires --extractor-model");
        if (!features::AlikedExtractor::is_built())
            throw std::runtime_error(
                "ALIKED requires ONNX Runtime (AETHERSCAN_ENABLE_ONNX)");
        features::AlikedOptions aliked;
        aliked.model_path = options.extractor_model_path;
        aliked.maximum_features = options.max_features;
        if (options.extractor_min_score >= 0.F)
            aliked.keypoint_threshold = options.extractor_min_score;
        aliked.cuda = options.extractor_use_cuda;
        auto extractor =
            std::make_unique<features::AlikedExtractor>(aliked);
        if (!extractor->is_available())
            throw std::runtime_error("ALIKED failed to initialize ONNX");
        return extractor;
    }
    return features::create_extractor(options.extractor);
}

std::unique_ptr<features::LightGlueMatcher> make_descriptor_lightglue_matcher(
    const FrontEndOptions& options) {
    if (options.lightglue_model_path.empty())
        throw std::runtime_error(
            "LightGlue requires --lightglue-model "
            "(descriptor matcher ONNX, e.g. *_lightglue_fused.onnx)");
    if (!features::LightGlueMatcher::is_built())
        throw std::runtime_error(
            "LightGlue matcher requires ONNX Runtime "
            "(AETHERSCAN_ENABLE_ONNX)");
    features::LightGlueMatcherOptions lg;
    lg.model_path = options.lightglue_model_path;
    lg.device = options.lightglue_use_cuda
        ? features::InferenceDevice::cuda
        : features::InferenceDevice::cpu;
    lg.min_score = options.lightglue_min_score;
    if (options.matcher == "hybrid_lightglue")
        lg.maximum_features = options.hybrid_lightglue_max_features;
    if (options.extractor == "superpoint")
        lg.descriptor_dimension = 256;
    else if (options.extractor == "disk" || options.extractor == "aliked" ||
             options.extractor == "sift" || options.extractor == "siftgpu")
        lg.descriptor_dimension = 128;
    auto matcher = std::make_unique<features::LightGlueMatcher>(lg);
    if (!matcher->is_available())
        throw std::runtime_error("LightGlue matcher failed to initialize ONNX");
    return matcher;
}

std::unique_ptr<features::FeatureMatcher> make_frontend_matcher(
    const FrontEndOptions& options) {
    if (options.matcher == "mutual_ratio") {
        features::DescriptorMatcherOptions matcher_options;
        matcher_options.ratio_threshold = options.match_ratio;
        matcher_options.mutual_check = options.mutual_check;
        return std::make_unique<features::MutualRatioMatcher>(matcher_options);
    }
    if (options.matcher == "gpu_mutual_ratio" ||
        options.matcher == "hybrid_lightglue") {
        features::SiftGpuMatcherOptions matcher_options;
        matcher_options.ratio_threshold = options.match_ratio;
        matcher_options.mutual_check = options.mutual_check;
        matcher_options.maximum_features =
            std::max<std::size_t>(32768, options.max_features);
        auto matcher =
            std::make_unique<features::SiftGpuMatcher>(matcher_options);
        if (!matcher->is_available())
            throw std::runtime_error(
                "gpu_mutual_ratio matcher requested but CUDA context is "
                "unavailable");
        return matcher;
    }
    if (options.matcher == "lightglue")
        return make_descriptor_lightglue_matcher(options);
    return features::create_matcher(options.matcher);
}

// Fused LightGlue path: each pair re-detects keypoints, so merge detections
// into stable per-image FeatureSets before geometric verification / tracks.
void match_and_verify_lightglue(
    Scene& scene,
    const std::vector<PairCandidate>& candidates,
    features::LightGluePipeline& pipeline,
    const FrontEndOptions& options,
    const unsigned worker_threads,
    std::vector<RawPairMatches>& raw_pairs,
    std::vector<PairDiagnostics>& diagnostics,
    std::vector<ImagePair>& pair_slots,
    core::ProgressReporter& match_progress) {
    raw_pairs.resize(candidates.size());
    diagnostics.assign(candidates.size(), {});
    pair_slots.assign(candidates.size(), ImagePair{});

    std::vector<std::shared_ptr<const io::RgbImage>> rgb_cache(
        scene.images.size());
    auto load_rgb = [&](const Index image_id) {
        auto& slot = rgb_cache[image_id];
        if (!slot) {
            slot = std::make_shared<const io::RgbImage>(
                io::load_rgb(scene.images[image_id].path));
        }
        return slot;
    };

    // Phase 1: sequential ONNX match + keypoint merge (FeatureSets must be
    // stable before parallel geometry verification).
    for (std::size_t pair_index = 0; pair_index < candidates.size();
         ++pair_index) {
        const PairCandidate candidate = candidates[pair_index];
        const auto rgb0 = load_rgb(candidate.id1);
        const auto rgb1 = load_rgb(candidate.id2);
        features::ImagePairFeatures pair_features =
            pipeline.match_rgb(*rgb0, *rgb1);

        auto& features0 = scene.images[candidate.id1].features;
        auto& features1 = scene.images[candidate.id2].features;
        if (features0.image_width == 0) {
            features0.image_width = pair_features.first.image_width;
            features0.image_height = pair_features.first.image_height;
            features0.extractor_name = "lightglue";
            features0.metric = features::DescriptorMetric::inner_product;
        }
        if (features1.image_width == 0) {
            features1.image_width = pair_features.second.image_width;
            features1.image_height = pair_features.second.image_height;
            features1.extractor_name = "lightglue";
            features1.metric = features::DescriptorMetric::inner_product;
        }

        RawPairMatches& raw = raw_pairs[pair_index];
        raw.id1 = candidate.id1;
        raw.id2 = candidate.id2;
        raw.matches.clear();
        raw.matches.reserve(pair_features.matches.matches.size());
        for (const features::FeatureMatch& match :
             pair_features.matches.matches) {
            if (match.query >= pair_features.first.keypoints.size() ||
                match.train >= pair_features.second.keypoints.size())
                throw std::runtime_error(
                    "LightGlue match index out of range");
            const auto& kp0 = pair_features.first.keypoints[match.query];
            const auto& kp1 = pair_features.second.keypoints[match.train];
            raw.matches.push_back(
                {merge_lightglue_keypoint(features0, kp0.x, kp0.y),
                 merge_lightglue_keypoint(features1, kp1.x, kp1.y),
                 match.score});
        }
        match_progress.advance();
    }
    match_progress.finish();

    // Cameras must exist before geometry verify (uses scene.cameras[camera_id]).
    // Image sizes are only known after phase-1 LightGlue merges.
    initialize_cameras(
        scene, options.focal_pixels, options.trust_focal_pixels);

    // Phase 2: sequential geometry verify. Parallel submit previously crashed
    // (0xC0000005) under LightGlue; keep this path simple and robust.
    core::ProgressReporter geometry_progress(
        "verify pair geometry", candidates.size());
    for (std::size_t pair_index = 0; pair_index < candidates.size();
         ++pair_index) {
        const PairCandidate candidate = candidates[pair_index];
        GeometryVerifyResult verified = verify_pair_geometry(
            scene, candidate, raw_pairs[pair_index].matches,
            options.relative);
        diagnostics[pair_index] = verified.diagnostics;
        if (verified.pair) pair_slots[pair_index] = *verified.pair;
        geometry_progress.advance();
    }
    geometry_progress.finish();
    (void)worker_threads;
}

}  // namespace

FrontEndResult run_frontend(
    const std::vector<std::filesystem::path>& image_paths,
    const FrontEndOptions& options) {
    FrontEndResult result;
    if (image_paths.size() < 2) return result;
    core::StageScope frontend_stage("sfm.frontend");

    FrontEndOptions runtime_options = options;
    normalize_feature_selection(runtime_options);
    if (runtime_options.retrieval.vocabulary_path.empty() &&
        !runtime_options.checkpoint.directory.empty()) {
        runtime_options.retrieval.vocabulary_path =
            runtime_options.checkpoint.directory / "vocabulary.bin";
    }
    const ImageSetFingerprint image_fingerprint =
        fingerprint_image_set(image_paths);
    const FrontEndStageKeys stage_keys =
        make_stage_keys(runtime_options, image_fingerprint);
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
        compute_pair_weights(scene, options.pair_weighting);
        calibrate_view_graph_focals(scene);
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
    const unsigned threads = scene.thread_count;

    if (uses_pair_feature_pipeline(runtime_options)) {
        if (runtime_options.lightglue_model_path.empty())
            throw std::runtime_error(
                "pipeline lightglue_end2end requires --lightglue-model / "
                "FrontEndOptions::lightglue_model_path");
        if (!features::LightGluePipeline::is_built())
            throw std::runtime_error(
                "LightGlue requires ONNX Runtime "
                "(AETHERSCAN_ENABLE_ONNX / AETHERSCAN_ONNXRUNTIME_ROOT)");

        const auto extract_started = std::chrono::steady_clock::now();
        const bool feature_cache_hit = checkpoints.load_scene(
            CheckpointStage::features, stage_keys.features, scene);
        if (!feature_cache_hit) {
            scene.images.resize(image_paths.size());
            for (std::size_t i = 0; i < image_paths.size(); ++i) {
                scene.images[i].id = static_cast<Index>(i);
                scene.images[i].path = image_paths[i];
            }
        } else {
            scene.thread_count =
                parallel::resolve_thread_count(options.thread_count);
            core::Logger::instance().info("checkpoint hit: features");
        }
        result.timing.extract_seconds = std::chrono::duration<double>(
                                            std::chrono::steady_clock::now() -
                                            extract_started)
                                            .count();

        const auto match_started = std::chrono::steady_clock::now();
        const bool use_retrieval =
            runtime_options.retrieval.top_k > 0 &&
            image_paths.size() >= runtime_options.retrieval_min_images &&
            (runtime_options.neighbor_window == 0 ||
             runtime_options.augment_sequential_with_retrieval);
        std::vector<PairCandidate> candidates;
        if (use_retrieval) {
            // Same pair proposal as the default SiftGPU frontend: sequential
            // window + BoW. Descriptors here are discarded after retrieval.
            Scene retrieval_scene;
            retrieval_scene.thread_count = scene.thread_count;
            features::SiftGpuOptions siftgpu_options;
            siftgpu_options.peak_threshold =
                static_cast<float>(options.sift_contrast_threshold);
            if (options.max_features > 0)
                siftgpu_options.maximum_features = options.max_features;
            features::SiftGpuExtractor extractor(siftgpu_options);
            if (!extractor.is_available())
                throw std::runtime_error(
                    "LightGlue with retrieval requires SiftGPU/CUDA for "
                    "BoW pair proposals");
            core::ProgressReporter retrieval_extract(
                "extract features (lightglue retrieval)", image_paths.size());
            extract_features_siftgpu_coordinator(
                retrieval_scene, image_paths, extractor, options.max_features,
                retrieval_extract);
            retrieval_extract.finish();
            candidates = build_pair_candidates(retrieval_scene, runtime_options);
        } else {
            candidates = build_pair_list(
                scene.images.size(), runtime_options.neighbor_window);
        }
        std::vector<RawPairMatches> raw_pairs;
        std::vector<PairDiagnostics> diagnostics(candidates.size());
        bool match_cache_hit =
            feature_cache_hit &&
            checkpoints.load_matches(stage_keys.matches, raw_pairs);
        if (match_cache_hit && raw_pairs.size() == candidates.size()) {
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                if (raw_pairs[i].id1 != candidates[i].id1 ||
                    raw_pairs[i].id2 != candidates[i].id2) {
                    match_cache_hit = false;
                    break;
                }
            }
        } else {
            match_cache_hit = false;
        }

        if (!match_cache_hit) {
            features::LightGluePipeline pipeline(
                make_lightglue_options(runtime_options));
            if (!pipeline.is_available())
                throw std::runtime_error(
                    "LightGlue failed to initialize ONNX session");
            std::vector<ImagePair> pair_slots;
            core::ProgressReporter match_progress(
                "lightglue match pairs", candidates.size());
            match_and_verify_lightglue(
                scene, candidates, pipeline, runtime_options, threads,
                raw_pairs, diagnostics, pair_slots, match_progress);
            scene.pairs.clear();
            scene.pairs.reserve(pair_slots.size());
            for (ImagePair& pair : pair_slots) {
                if (pair.matches.empty()) continue;
                scene.pairs.push_back(std::move(pair));
            }
            initialize_cameras(
                scene, options.focal_pixels, options.trust_focal_pixels);
            verify_image_snapshot(image_paths, image_fingerprint);
            checkpoints.save_scene(
                CheckpointStage::features, stage_keys.features, scene);
            checkpoints.save_matches(stage_keys.matches, raw_pairs);
        } else {
            core::Logger::instance().info("checkpoint hit: matches");
            initialize_cameras(
                scene, options.focal_pixels, options.trust_focal_pixels);
            std::vector<ImagePair> pairs(candidates.size());
            core::ProgressReporter geometry_progress(
                "verify pair geometry", candidates.size());
            parallel::parallel_for(
                candidates.size(), threads, [&](const std::size_t ci) {
                    GeometryVerifyResult verified = verify_pair_geometry(
                        scene, candidates[ci], raw_pairs[ci].matches,
                        options.relative);
                    diagnostics[ci] = verified.diagnostics;
                    if (verified.pair) pairs[ci] = std::move(*verified.pair);
                    geometry_progress.advance();
                });
            geometry_progress.finish();
            scene.pairs.clear();
            scene.pairs.reserve(pairs.size());
            for (auto& pair : pairs) {
                if (pair.matches.empty()) continue;
                scene.pairs.push_back(std::move(pair));
            }
        }

        verify_image_snapshot(
            image_paths, image_fingerprint, ImageSnapshotCheck::content);
        checkpoints.save_scene(
            CheckpointStage::geometry, stage_keys.geometry, scene);
        result.timing.match_verify_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - match_started)
                .count();

        const auto tracks_started = std::chrono::steady_clock::now();
        compute_pair_weights(scene, options.pair_weighting);
        calibrate_view_graph_focals(scene);
        build_tracks(scene, options.min_pair_weight);
        checkpoints.save_scene(
            CheckpointStage::tracks, stage_keys.tracks, scene);
        result.timing.tracks_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tracks_started)
                .count();

        std::size_t feature_count = 0;
        for (const Image& image : scene.images)
            feature_count += image.features.keypoints.size();
        core::Logger::instance().info(
            "frontend pipeline=lightglue_end2end: features=", feature_count,
            " candidates=", candidates.size(),
            " accepted_pairs=", scene.pairs.size(),
            " extract_s=", result.timing.extract_seconds,
            " match_s=", result.timing.match_verify_seconds,
            " tracks_s=", result.timing.tracks_seconds);
        return result;
    }

    features::validate_extractor_matcher_combo(
        runtime_options.extractor, runtime_options.matcher);

    std::unique_ptr<features::FeatureExtractor> extractor =
        make_frontend_extractor(runtime_options);
    if (!extractor)
        throw std::runtime_error("Failed to create feature extractor");
    std::unique_ptr<features::FeatureMatcher> matcher =
        make_frontend_matcher(runtime_options);
    if (!matcher)
        throw std::runtime_error("Failed to create feature matcher");

    const auto extract_started = std::chrono::steady_clock::now();
    const bool feature_cache_hit = checkpoints.load_scene(
        CheckpointStage::features, stage_keys.features, scene);
    if (!feature_cache_hit) {
        core::ProgressReporter progress("extract features", image_paths.size());
        scene.images.resize(image_paths.size());
        scene.cameras.reserve(image_paths.size());
        if (extractor->info().thread_affine) {
            extract_features_siftgpu_coordinator(
                scene, image_paths, *extractor, options.max_features, progress);
        } else {
            const unsigned extraction_threads = threads;
            std::vector<std::unique_ptr<features::FeatureExtractor>> workers(
                extraction_threads);
            if (extraction_threads > 1 && !extractor->info().thread_safe) {
                for (unsigned t = 0; t < extraction_threads; ++t)
                    workers[t] = extractor->clone();
            }
            parallel::parallel_for(
                image_paths.size(), extraction_threads,
                [&](const std::size_t i, const unsigned tid) {
                    features::FeatureExtractor* local =
                        workers[tid] ? workers[tid].get() : extractor.get();
                    features::FeatureSet features =
                        local->extract_file(image_paths[i]);
                    features = select_top_features_grid_3x3(
                        std::move(features), options.max_features);
                    scene.images[i] = make_image_from_features(
                        static_cast<Index>(i), image_paths[i],
                        std::move(features));
                    progress.advance();
                });
        }

        initialize_cameras(
            scene, options.focal_pixels, options.trust_focal_pixels);
        verify_image_snapshot(image_paths, image_fingerprint);
        checkpoints.save_scene(
            CheckpointStage::features, stage_keys.features, scene);
    } else {
        scene.thread_count = parallel::resolve_thread_count(options.thread_count);
        initialize_cameras(
            scene, options.focal_pixels, options.trust_focal_pixels);
        core::Logger::instance().info("checkpoint hit: features");
    }
    result.timing.extract_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - extract_started)
            .count();

    const auto match_started = std::chrono::steady_clock::now();
    auto candidates = build_pair_candidates(scene, runtime_options);
    // Group by id1 before GPU matching so slot-0 descriptors stay warm.
    if (matcher->requires_owner_thread())
        optimize_pairs_order(candidates, scene);
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
    std::vector<PairDiagnostics> diagnostics(candidates.size());
    bool geometry_verified_in_pipeline = false;
    if (!match_cache_hit) {
        raw_pairs.resize(candidates.size());
        if (matcher->requires_owner_thread()) {
            std::vector<ImagePair> pair_slots;
            core::ProgressReporter match_progress(
                runtime_options.matcher == "hybrid_lightglue"
                    ? "fast match image pairs"
                    : "match image pairs",
                candidates.size());
            match_and_verify_siftgpu_coordinator(
                scene, candidates, *matcher, options, threads, raw_pairs,
                diagnostics, pair_slots, match_progress);
            match_progress.finish();

            if (runtime_options.matcher == "hybrid_lightglue") {
                std::vector<PairCandidate> rescue_candidates;
                std::vector<std::size_t> rescue_indices;
                std::vector<unsigned> primary_degree(scene.images.size(), 0U);
                for (const ImagePair& pair : pair_slots) {
                    if (pair.matches.empty()) continue;
                    ++primary_degree[pair.id1];
                    ++primary_degree[pair.id2];
                }
                constexpr unsigned k_min_primary_degree = 3U;
                const std::size_t rescue_window =
                    std::max<std::size_t>(runtime_options.neighbor_window, 3U);
                rescue_candidates.reserve(candidates.size());
                rescue_indices.reserve(candidates.size());
                for (std::size_t index = 0; index < candidates.size(); ++index) {
                    if (!pair_slots[index].matches.empty()) continue;
                    const PairCandidate candidate = candidates[index];
                    const std::size_t image_gap = candidate.id1 > candidate.id2
                        ? static_cast<std::size_t>(candidate.id1 - candidate.id2)
                        : static_cast<std::size_t>(candidate.id2 - candidate.id1);
                    const bool local_sequence_edge = image_gap <= rescue_window;
                    const bool strengthens_weak_image =
                        primary_degree[candidate.id1] < k_min_primary_degree ||
                        primary_degree[candidate.id2] < k_min_primary_degree;
                    if (!local_sequence_edge && !strengthens_weak_image)
                        continue;
                    rescue_candidates.push_back(candidates[index]);
                    rescue_indices.push_back(index);
                }

                std::size_t primary_accepted = 0;
                for (const ImagePair& pair : pair_slots)
                    primary_accepted += pair.matches.empty() ? 0U : 1U;
                if (!rescue_candidates.empty()) {
                    auto rescue_matcher =
                        make_descriptor_lightglue_matcher(runtime_options);
                    FrontEndOptions rescue_options = options;
                    // LightGlue is used to add graph connectivity, so demand a
                    // tighter geometric consensus than the primary matcher.
                    rescue_options.relative.max_epipolar_error_px = std::min(
                        rescue_options.relative.max_epipolar_error_px, 2.0);
                    rescue_options.relative.max_reproj_error_px = std::min(
                        rescue_options.relative.max_reproj_error_px, 3.0);
                    rescue_options.relative.min_inliers = std::max(
                        rescue_options.relative.min_inliers, 40U);
                    std::vector<RawPairMatches> rescue_raw_pairs;
                    std::vector<PairDiagnostics> rescue_diagnostics;
                    std::vector<ImagePair> rescue_pair_slots;
                    core::ProgressReporter rescue_progress(
                        "rescue difficult pairs with lightglue",
                        rescue_candidates.size());
                    match_and_verify_siftgpu_coordinator(
                        scene, rescue_candidates, *rescue_matcher,
                        rescue_options,
                        threads, rescue_raw_pairs, rescue_diagnostics,
                        rescue_pair_slots, rescue_progress);
                    rescue_progress.finish();

                    std::size_t rescued = 0;
                    for (std::size_t rescue_index = 0;
                         rescue_index < rescue_indices.size(); ++rescue_index) {
                        const std::size_t original =
                            rescue_indices[rescue_index];
                        raw_pairs[original] =
                            std::move(rescue_raw_pairs[rescue_index]);
                        diagnostics[original] =
                            rescue_diagnostics[rescue_index];
                        if (!rescue_pair_slots[rescue_index].matches.empty()) {
                            pair_slots[original] =
                                std::move(rescue_pair_slots[rescue_index]);
                            ++rescued;
                        }
                    }
                    core::Logger::instance().info(
                        "hybrid matcher: primary_accepted=", primary_accepted,
                        " primary_rejected=",
                        candidates.size() - primary_accepted,
                        " rescue_attempted=", rescue_candidates.size(),
                        " rescued=", rescued,
                        " accepted_total=", primary_accepted + rescued);
                }
            }

            scene.pairs.reserve(pair_slots.size());
            for (ImagePair& pair : pair_slots) {
                if (pair.matches.empty()) continue;
                scene.pairs.push_back(std::move(pair));
            }
            geometry_verified_in_pipeline = true;
        } else {
            const unsigned match_threads = threads;
            std::vector<std::unique_ptr<features::FeatureMatcher>> match_workers(
                match_threads);
            for (unsigned t = 0; t < match_threads; ++t)
                match_workers[t] = matcher->clone();
            auto* ratio_matcher =
                dynamic_cast<features::MutualRatioMatcher*>(
                    match_workers.front().get());
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
                prepare_ids.size(), match_threads,
                [&](const std::size_t index, const unsigned tid) {
                    match_workers[tid]->prepare(
                        scene.images[prepare_ids[index]].features);
                    prepare_progress.advance();
                });
            prepare_progress.finish();
            core::ProgressReporter match_progress(
                "match image pairs", candidates.size());
            parallel::parallel_for(
                candidates.size(), match_threads,
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
            match_workers.front()->clear_prepared();
        }
        checkpoints.save_matches(stage_keys.matches, raw_pairs);
    } else {
        core::Logger::instance().info("checkpoint hit: matches");
    }
    // Descriptors are no longer needed after matching; keep keypoints only.
    release_descriptors(scene);

    if (!geometry_verified_in_pipeline) {
        std::vector<ImagePair> pairs(candidates.size());
        core::ProgressReporter geometry_progress(
            "verify pair geometry", candidates.size());
        parallel::parallel_for(
            candidates.size(), threads, [&](const std::size_t ci) {
                const PairCandidate cand = candidates[ci];
                GeometryVerifyResult verified = verify_pair_geometry(
                    scene, cand, raw_pairs[ci].matches, options.relative);
                diagnostics[ci] = verified.diagnostics;
                if (verified.pair) pairs[ci] = std::move(*verified.pair);
                geometry_progress.advance();
            });
        geometry_progress.finish();

        scene.pairs.reserve(pairs.size());
        for (auto& pair : pairs) {
            if (pair.matches.empty()) continue;
            scene.pairs.push_back(std::move(pair));
        }
    }
    verify_image_snapshot(
        image_paths, image_fingerprint, ImageSnapshotCheck::content);
    checkpoints.save_scene(
        CheckpointStage::geometry, stage_keys.geometry, scene);
    result.timing.match_verify_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - match_started)
            .count();

    const auto tracks_started = std::chrono::steady_clock::now();
    compute_pair_weights(scene, options.pair_weighting);
    calibrate_view_graph_focals(scene);
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
