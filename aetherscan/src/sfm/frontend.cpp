#include "sfm/frontend.hpp"

#include "core/logging.hpp"
#include "features/compat.hpp"
#include "features/features.hpp"
#include "features/registry.hpp"
#include "io/image.hpp"
#include "parallel/thread_pool.hpp"
#include "sfm/tracks.hpp"
#include "sfm/camera_selection.hpp"
#include <cctype>
#include "sfm/align_live.hpp"
#include "sfm/submap_recovery.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::sfm {
namespace {

#if !defined(AETHERSCAN_FRONTEND_CACHE_BUILD_ID)
#define AETHERSCAN_FRONTEND_CACHE_BUILD_ID "unconfigured"
#endif

struct PairCandidate {
    Index id1{};
    Index id2{};
    bool zero_baseline{false};
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
	key.append_string(AETHERSCAN_FRONTEND_CACHE_BUILD_ID);
	// Camera metadata now feeds a late intrinsic split; older geometry and
	// track caches lack those fields.
	key.append_string("camera-late-exif-split-v3");
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
    features.append_string("rootsift-u8-scale512-v1");
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
    matches.append_string("zero-baseline-identity-v1");
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
    geometry.append_string("defer-unknown-focal-filter-v1");
    geometry.append_string("zero-baseline-inactive-v2");
    append_cache_build_identity(geometry);
    geometry.append(matches.value());
    geometry.append(options.focal_pixels);
    geometry.append(static_cast<std::uint32_t>(options.camera_model));
    geometry.append_string("validated-camera-calibration-v2");
    geometry.append(options.trust_focal_pixels);
    append_relative_options(geometry, options.relative);
    geometry.append(options.progressive_pair_expansion);
    geometry.append(options.structural_pair_expansion);
    geometry.append(options.progressive_min_verified_degree);
    geometry.append(options.progressive_rescue_match_ratio);
    geometry.append(options.progressive_rescue_min_inliers);
    geometry.append(options.progressive_max_images);
    geometry.append(options.progressive_rescue_neighbor_window);
    geometry.append(options.progressive_rescue_retrieval_top_k);
    geometry.append(options.progressive_rescue_max_pairs_per_image);
    geometry.append(options.progressive_rescue_max_features);
    geometry.append_string("progressive_pair_expansion_v3");

    FingerprintBuilder tracks;
    tracks.append_string("tracks");
    tracks.append_string("always-finalize-relative-poses-v1");
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
    Scene& scene, const double focal_pixels, const bool trust_focal_pixels,
    const CameraModel requested_model) {
    const CameraModel model = requested_model == CameraModel::automatic
        ? CameraModel::pinhole : requested_model;
    scene.cameras.clear();
    scene.cameras.reserve(scene.images.size());
    std::vector<double> exif_focals;
    exif_focals.reserve(scene.images.size());
    for (Index i = 0; i < scene.images.size(); ++i) {
        Image& image = scene.images[i];
        // Capture EXIF identity for late grouped refinement, but keep the
        // view graph on the legacy shared group. Early hard grouping can turn
        // sparse zoom-transition edges into disconnected registration
        // components; the late split gets independent focal/distortion once a
        // connected layout has already been established.
        image.camera_identity = trust_focal_pixels
            ? std::string{}
            : io::load_camera_identity(
                  image.path,
                  &image.focal_length_mm,
                  image.features.image_width,
                  image.features.image_height,
                  &image.exif_focal_px);
        if (image.exif_focal_px > 0.0)
            exif_focals.push_back(image.exif_focal_px);
    }
    const bool use_exif_prior =
        focal_pixels <= 0.0 && exif_focals.size() * 4 >= scene.images.size();
    double exif_focal_px = 0.0;
    if (use_exif_prior) {
        const std::size_t middle = exif_focals.size() / 2;
        std::nth_element(
            exif_focals.begin(), exif_focals.begin() + middle,
            exif_focals.end());
        exif_focal_px = exif_focals[middle];
        core::Logger::instance().info(
            "camera exif focal prior: pixels=", exif_focal_px,
            " images=", exif_focals.size(), '/', scene.images.size());
    }
    for (Index i = 0; i < scene.images.size(); ++i) {
        const auto& features = scene.images[i].features;
        PinholeCamera camera;
        camera.model = model;
        camera.width = features.image_width;
        camera.height = features.image_height;
        if (model == CameraModel::equirectangular) {
            // The panorama chart is fully determined by the image size: no focal
            // search and no EXIF focal prior (a 360 export has no meaningful
            // 35 mm equivalent). Repeated below for the shared-group test.
            camera.set_equirectangular_intrinsics();
            const auto existing = std::find_if(
                scene.cameras.begin(), scene.cameras.end(),
                [&](const PinholeCamera& candidate) {
                    return candidate.is_equirectangular() &&
                           candidate.width == camera.width &&
                           candidate.height == camera.height;
                });
            if (existing == scene.cameras.end()) {
                camera.id = static_cast<Index>(scene.cameras.size());
                scene.images[i].camera_id = camera.id;
                scene.cameras.push_back(camera);
            } else {
                scene.images[i].camera_id =
                    static_cast<Index>(existing - scene.cameras.begin());
            }
            continue;
        }
        const double geometric_focal =
            (model == CameraModel::opencv_fisheye ? 0.5 : 1.2) *
            std::max(camera.width, camera.height);
        const double focal = focal_pixels > 0
            ? focal_pixels
            : (use_exif_prior && exif_focal_px > 0.0
                   ? exif_focal_px
                   : geometric_focal);
        camera.fx = focal;
        camera.fy = focal;
        camera.focal_prior = focal;
        camera.cx = 0.5 * camera.width;
        camera.cy = 0.5 * camera.height;
        camera.trust_intrinsics = trust_focal_pixels;
        const auto existing = std::find_if(
            scene.cameras.begin(), scene.cameras.end(),
            [&](const PinholeCamera& candidate) {
                return candidate.focal_prior == camera.focal_prior &&
                       candidate.width == camera.width &&
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

void select_scene_camera_models(
    Scene& scene, const std::vector<RawPairMatches>& raw_pairs,
    const std::vector<PairCandidate>& candidates, const FrontEndOptions& options) {
    for (auto& camera : scene.cameras) {
        std::vector<std::size_t> eligible;
        for (std::size_t i=0; i<raw_pairs.size(); ++i) {
            const auto& pair = raw_pairs[i];
            if (i >= candidates.size() || candidates[i].zero_baseline || pair.matches.size()<40 ||
                pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size()) continue;
            if (scene.images[pair.id1].camera_id == camera.id &&
                scene.images[pair.id2].camera_id == camera.id) eligible.push_back(i);
        }
        std::vector<CameraModelProbe> probes;
        const std::size_t count = std::min<std::size_t>(16, eligible.size());
        for (std::size_t i=0; i<count; ++i) {
            const auto& pair = raw_pairs[eligible[i*eligible.size()/count]];
            CameraModelProbe probe;
            const auto& first = scene.images[pair.id1].features.keypoints;
            const auto& second = scene.images[pair.id2].features.keypoints;
            const std::size_t samples = std::min<std::size_t>(400,pair.matches.size());
            for (std::size_t j=0; j<samples; ++j) {
                const auto& match = pair.matches[j*pair.matches.size()/samples];
                if (match.query>=first.size() || match.train>=second.size()) continue;
                probe.first.emplace_back(first[match.query].x,first[match.query].y);
                probe.second.emplace_back(second[match.train].x,second[match.train].y);
            }
            probes.push_back(std::move(probe));
        }
        bool lens_hint = false;
        unsigned inspected = 0;
        for (const auto& image : scene.images) {
            if (image.camera_id != camera.id) continue;
            auto lens = io::load_lens_description(image.path);
            std::transform(lens.begin(),lens.end(),lens.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            lens_hint = lens_hint || lens.find("fisheye") != std::string::npos ||
                lens.find("fish-eye") != std::string::npos;
            if (++inspected == 3) break;
        }
        core::Logger::instance().info("auto camera: evaluating group=",camera.id," pairs=",probes.size());
        // A 2:1 image size is the only admissible evidence that a still image
        // covers the full sphere; any other aspect ratio keeps the perspective
        // hypotheses only.
        const bool panorama_hint =
            is_equirectangular_image_size(camera.width, camera.height);
        const auto selected = select_camera_model(
            camera,probes,options.focal_pixels,lens_hint,options.camera_model,
            panorama_hint);
        if (selected.model == CameraModel::equirectangular) {
            camera.set_equirectangular_intrinsics();
        } else {
            camera.model = selected.model;
            camera.k1=selected.distortion[0]; camera.k2=selected.distortion[1];
            camera.p1=selected.distortion[2]; camera.p2=selected.distortion[3];
            camera.fx = camera.fy = camera.focal_prior = selected.focal_pixels;
        }
        core::Logger::instance().info("auto camera: group=",camera.id,
            " selected=", selected.model == CameraModel::equirectangular ? "equirectangular"
                : (selected.model == CameraModel::opencv_fisheye ? "opencv_fisheye" : "pinhole"),
            " focal=",camera.focal()," pinhole_score=",selected.pinhole_score,
            " fisheye_score=",selected.fisheye_score,
            " equirect_score=",selected.equirect_score,
            " informative_pairs=",selected.informative_pairs,
            " confident=",selected.confident," reason=",selected.reason);
    }
}

struct FetzerSameCameraCost {
    Eigen::Vector4d d01{Eigen::Vector4d::Zero()};
    Eigen::Vector4d d12{Eigen::Vector4d::Zero()};
};

Eigen::Vector4d fetzer_cross_terms(
    const Vec3& ai, const Vec3& bi, const Vec3& aj, const Vec3& bj,
    const int u, const int v) {
    return {
        ai(u) * aj(v) - ai(v) * aj(u),
        ai(u) * bj(v) - ai(v) * bj(u),
        bi(u) * aj(v) - bi(v) * aj(u),
        bi(u) * bj(v) - bi(v) * bj(u)};
}

std::optional<FetzerSameCameraCost> make_fetzer_same_camera_cost(
    const Mat3& fundamental, const PinholeCamera& camera) {
    Mat3 principal = Mat3::Identity();
    principal(0, 2) = camera.cx;
    principal(1, 2) = camera.cy;
    const Mat3 G = principal.transpose() * fundamental * principal;
    const Eigen::JacobiSVD<Mat3> svd(
        G, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.info() != Eigen::Success) return std::nullopt;
    const Vec3 singular = svd.singularValues();
    if (!singular.allFinite() || singular(1) <= 1e-15)
        return std::nullopt;
    const Vec3 v0 = svd.matrixV().col(0);
    const Vec3 v1 = svd.matrixV().col(1);
    const Vec3 u0 = svd.matrixU().col(0);
    const Vec3 u1 = svd.matrixU().col(1);
    const Vec3 ai(
        singular(0) * singular(0) *
            (v0(0) * v0(0) + v0(1) * v0(1)),
        singular(0) * singular(1) *
            (v0(0) * v1(0) + v0(1) * v1(1)),
        singular(1) * singular(1) *
            (v1(0) * v1(0) + v1(1) * v1(1)));
    const Vec3 aj(
        u1(0) * u1(0) + u1(1) * u1(1),
        -(u0(0) * u1(0) + u0(1) * u1(1)),
        u0(0) * u0(0) + u0(1) * u0(1));
    const Vec3 bi(
        singular(0) * singular(0) * v0(2) * v0(2),
        singular(0) * singular(1) * v0(2) * v1(2),
        singular(1) * singular(1) * v1(2) * v1(2));
    const Vec3 bj(
        u1(2) * u1(2), -u0(2) * u1(2), u0(2) * u0(2));
    FetzerSameCameraCost cost;
    cost.d01 = fetzer_cross_terms(ai, bi, aj, bj, 1, 0);
    cost.d12 = fetzer_cross_terms(ai, bi, aj, bj, 2, 1);
    if (!cost.d01.allFinite() || !cost.d12.allFinite())
        return std::nullopt;
    return cost;
}

double fetzer_residual_squared(
    const FetzerSameCameraCost& cost, const double focal) {
    const double f2 = focal * focal;
    const double denominator0 = f2 * cost.d01(0) + cost.d01(1);
    const double denominator1 = f2 * cost.d12(0) + cost.d12(2);
    if (std::abs(denominator0) <= 1e-18 ||
        std::abs(denominator1) <= 1e-18)
        return std::numeric_limits<double>::infinity();
    const double k0 =
        (f2 * cost.d01(2) + cost.d01(3)) / denominator0;
    const double k1 =
        (f2 * cost.d12(1) + cost.d12(3)) / denominator1;
    const double residual0 = (f2 + k0) / f2;
    const double residual1 = (f2 + k1) / f2;
    return residual0 * residual0 + residual1 * residual1;
}

double robust_fetzer_objective(
    const std::vector<FetzerSameCameraCost>& costs, const double log_focal) {
    constexpr double loss_scale = 0.1;
    constexpr double loss_scale_squared = loss_scale * loss_scale;
    const double focal = std::exp(log_focal);
    double objective = 0.0;
    for (const FetzerSameCameraCost& cost : costs) {
        const double residual = fetzer_residual_squared(cost, focal);
        objective += std::isfinite(residual)
            ? loss_scale_squared * std::atan(residual / loss_scale_squared)
            : 0.5 * std::numbers::pi * loss_scale_squared;
    }
    return objective;
}

double solve_fetzer_focal(
    const std::vector<FetzerSameCameraCost>& costs,
    const PinholeCamera& camera) {
    const double image_scale =
        static_cast<double>(std::max(camera.width, camera.height));
    const double log_min = std::log(std::max(0.25 * image_scale, 1.0));
    const double log_max = std::log(std::max(4.0 * image_scale, 2.0));
    constexpr int samples = 160;
    int best = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int index = 0; index < samples; ++index) {
        const double alpha =
            static_cast<double>(index) / static_cast<double>(samples - 1);
        const double candidate = log_min + alpha * (log_max - log_min);
        const double objective = robust_fetzer_objective(costs, candidate);
        if (objective < best_cost) {
            best_cost = objective;
            best = index;
        }
    }
    // An optimum on the physical search boundary is the characteristic
    // weak-parallax/self-calibration degeneracy, not a measured focal. Keep
    // the camera prior in that case and let multi-view BA refine it later.
    if (best <= 1 || best >= samples - 2)
        return std::numeric_limits<double>::quiet_NaN();
    const double step = (log_max - log_min) / (samples - 1);
    double left = std::max(log_min, log_min + (best - 1) * step);
    double right = std::min(log_max, log_min + (best + 1) * step);
    for (int iteration = 0; iteration < 48; ++iteration) {
        const double first = (2.0 * left + right) / 3.0;
        const double second = (left + 2.0 * right) / 3.0;
        if (robust_fetzer_objective(costs, first) <=
            robust_fetzer_objective(costs, second))
            right = second;
        else
            left = first;
    }
    return std::exp(0.5 * (left + right));
}

std::optional<double> robust_pair_focal_fallback(
    const std::vector<double>& candidates, const PinholeCamera& camera) {
    const double image_scale =
        static_cast<double>(std::max(camera.width, camera.height));
    const double minimum = std::max(0.25 * image_scale, 1.0);
    const double maximum = std::max(4.0 * image_scale, 2.0);
    std::vector<double> log_focals;
    log_focals.reserve(candidates.size());
    for (const double focal : candidates) {
        if (std::isfinite(focal) && focal >= minimum && focal <= maximum)
            log_focals.push_back(std::log(focal));
    }
    if (log_focals.size() < 16) return std::nullopt;
    std::sort(log_focals.begin(), log_focals.end());

    // Pairwise self-calibration is systematically low-biased on ordered
    // low-parallax orbits. A high robust quantile avoids that collapsed mode,
    // while remaining far below the generic 1.2*image-size initialization.
    const std::size_t index = static_cast<std::size_t>(
        0.75 * static_cast<double>(log_focals.size() - 1));
    const double estimate = log_focals[index];
    const double log_support_radius = std::log(1.5);
    const std::size_t support = static_cast<std::size_t>(std::count_if(
        log_focals.begin(), log_focals.end(), [&](const double value) {
            return std::abs(value - estimate) <= log_support_radius;
        }));
    if (support * 4 < log_focals.size()) return std::nullopt;
    return std::exp(estimate);
}

bool calibrate_view_graph_focals(Scene& scene) {
    bool updated = false;
    std::vector<std::vector<FetzerSameCameraCost>> grouped(
        scene.cameras.size());
    std::vector<std::vector<double>> pair_focals(scene.cameras.size());
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.degenerate_planar || !pair.F.has_value() ||
            pair.num_inliers() < 15 ||
            pair.composite_weight() < 3.F ||
            pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size())
            continue;
        const Index first_group = scene.images[pair.id1].camera_id;
        const Index second_group = scene.images[pair.id2].camera_id;
        if (first_group != second_group || first_group >= scene.cameras.size())
            continue;
        const PinholeCamera& camera = scene.cameras[first_group];
        if (camera.trust_intrinsics || camera.model == CameraModel::opencv_fisheye ||
            camera.model == CameraModel::equirectangular) continue;
        if (const auto cost =
                make_fetzer_same_camera_cost(*pair.F, camera))
            grouped[first_group].push_back(*cost);
        if (pair.estimated_focal.has_value())
            pair_focals[first_group].push_back(*pair.estimated_focal);
    }

    for (std::size_t group = 0; group < grouped.size(); ++group) {
        if (scene.cameras[group].model == CameraModel::opencv_fisheye ||
            scene.cameras[group].model == CameraModel::equirectangular) continue;
        const auto& costs = grouped[group];
        // openMVS requires a meaningful view-graph consensus rather than
        // trusting a handful of independently degenerate pairs.
        if (costs.size() < 16) {
            if (!scene.cameras[group].trust_intrinsics)
                core::Logger::instance().warning(
                    "view-graph focal unresolved: camera=", group,
                    " reason=insufficient_pairs pairs=", costs.size(),
                    " using_initial_focal=", scene.cameras[group].focal());
            continue;
        }
        PinholeCamera& camera = scene.cameras[group];
        const double initial = camera.focal();
        double focal = solve_fetzer_focal(costs, camera);
        const auto pair_consensus = robust_pair_focal_fallback(pair_focals[group], camera);
        core::Logger::instance().info(
            "view-graph focal cross-check: camera=", group,
            " fetzer=", focal, " pair_q75=", pair_consensus.value_or(0.0));
        if (std::isfinite(focal) && pair_consensus.has_value() &&
            std::abs(std::log(focal / *pair_consensus)) > std::log(1.5)) {
            // An interior minimum is not sufficient evidence of observable
            // self-calibration. Independent pair estimates can expose a
            // collapsed graph minimum even when it is away from the bounds.
            // Do not choose either conflicting estimate as a measurement:
            // retain the adjustable prior and still run strict pose filtering.
            core::Logger::instance().warning(
                "view-graph focal unresolved: camera=", group,
                " reason=conflicting_estimators fetzer=", focal,
                " pair_q75=", *pair_consensus,
                " using_initial_focal=", initial);
            continue;
        }
        bool used_pair_fallback = false;
        if (!std::isfinite(focal)) {
            const auto fallback = robust_pair_focal_fallback(
                pair_focals[group], camera);
            if (fallback.has_value()) {
                focal = *fallback;
                used_pair_fallback = true;
            }
        }
        const double ratio = focal / std::max(initial, 1.0);
        // Same broad degeneracy check as openMVS ViewGraphCalibrator. The
        // tighter absolute bounds are the physical search interval above.
        if (!std::isfinite(focal) || ratio < 0.1 || ratio > 10.0) {
            core::Logger::instance().warning(
                "view-graph focal unresolved: camera=", group,
                " reason=degenerate_consensus pairs=", costs.size(),
                " using_initial_focal=", initial);
            continue;
        }
        camera.fx = focal;
        camera.fy = focal;
        camera.focal_prior = focal;
        updated = true;
        core::Logger::instance().info(
            "view-graph focal: camera=", group,
            " initial=", initial, " consensus=", focal,
            " ratio=", ratio, " residual_pairs=", costs.size(),
            " pair_candidates=", pair_focals[group].size(),
            " fallback=", used_pair_fallback ? "pair_q75" : "fetzer");
    }
    return updated;
}

bool calibrate_exif_view_graph_focals(Scene& scene) {
    struct ExifFocalGroup {
        std::vector<FetzerSameCameraCost> costs;
        std::vector<double> pair_focals;
        Index first_image{k_invalid};
        unsigned images{0};
    };

    const auto exif_key = [](const Image& image) {
        return image.camera_identity.empty()
            ? std::string{}
            : "camera-" + std::to_string(image.camera_id) + "|" +
                  image.camera_identity + "|focal-mm-" +
                  std::to_string(image.focal_length_mm);
    };

    std::unordered_map<std::string, ExifFocalGroup> groups;
    for (const Image& image : scene.images) {
        const std::string key = exif_key(image);
        if (key.empty()) continue;
        ExifFocalGroup& group = groups[key];
        ++group.images;
        if (group.first_image == k_invalid) group.first_image = image.id;
    }
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.degenerate_planar || !pair.F.has_value() ||
            pair.num_inliers() < 15 || pair.composite_weight() < 3.F ||
            pair.id1 >= scene.images.size() ||
            pair.id2 >= scene.images.size())
            continue;
        const std::string first_key = exif_key(scene.images[pair.id1]);
        if (first_key.empty() || first_key != exif_key(scene.images[pair.id2]))
            continue;
        const Image& image = scene.images[pair.id1];
        if (image.camera_id >= scene.cameras.size()) continue;
        const PinholeCamera& camera = scene.cameras[image.camera_id];
        if (camera.trust_intrinsics ||
            camera.model == CameraModel::opencv_fisheye ||
            camera.model == CameraModel::equirectangular)
            continue;
        ExifFocalGroup& group = groups[first_key];
        if (const auto cost = make_fetzer_same_camera_cost(*pair.F, camera))
            group.costs.push_back(*cost);
        if (pair.estimated_focal.has_value())
            group.pair_focals.push_back(*pair.estimated_focal);
    }

    bool updated = false;
    for (auto& [key, group] : groups) {
        if (group.first_image >= scene.images.size() ||
            group.costs.size() < 16)
            continue;
        const Image& image = scene.images[group.first_image];
        const PinholeCamera& camera = scene.camera_of(image);
        const double fetzer = solve_fetzer_focal(group.costs, camera);
        const auto pair_consensus =
            robust_pair_focal_fallback(group.pair_focals, camera);
        double focal = fetzer;
        bool used_pair_fallback = false;
        if (!std::isfinite(focal) && pair_consensus.has_value()) {
            focal = *pair_consensus;
            used_pair_fallback = true;
        }
        const double initial = camera.focal();
        if (!std::isfinite(focal) || focal < 0.1 * initial ||
            focal > 10.0 * initial) {
            core::Logger::instance().warning(
                "EXIF view-graph focal unresolved: key=", key,
                " pairs=", group.costs.size(),
                " pair_candidates=", group.pair_focals.size());
            continue;
        }
        if (std::isfinite(fetzer) && pair_consensus.has_value() &&
            std::abs(std::log(fetzer / *pair_consensus)) >
                std::log(1.5)) {
            core::Logger::instance().warning(
                "EXIF view-graph focal conflicting: key=", key,
                " fetzer=", fetzer,
                " pair_q75=", *pair_consensus,
                " using_shared_initial=", initial);
            continue;
        }
        for (Image& destination : scene.images) {
            if (exif_key(destination) != key) continue;
            destination.exif_focal_px = focal;
        }
        updated = true;
        core::Logger::instance().info(
            "EXIF view-graph focal: key=", key,
            " images=", group.images,
            " pairs=", group.costs.size(),
            " focal=", focal,
            " fallback=", used_pair_fallback ? "pair_q75" : "fetzer");
    }
    return updated;
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

struct KeypointIdentity {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t scale{};
    std::uint32_t orientation{};

    bool operator==(const KeypointIdentity&) const = default;
};

struct KeypointIdentityHash {
    std::size_t operator()(const KeypointIdentity& key) const noexcept {
        std::size_t hash = key.x;
        hash ^= static_cast<std::size_t>(key.y) + 0x9e3779b9U +
                (hash << 6U) + (hash >> 2U);
        hash ^= static_cast<std::size_t>(key.scale) + 0x9e3779b9U +
                (hash << 6U) + (hash >> 2U);
        hash ^= static_cast<std::size_t>(key.orientation) + 0x9e3779b9U +
                (hash << 6U) + (hash >> 2U);
        return hash;
    }
};

KeypointIdentity keypoint_identity(const features::Keypoint& keypoint) {
    return {
        std::bit_cast<std::uint32_t>(keypoint.x),
        std::bit_cast<std::uint32_t>(keypoint.y),
        std::bit_cast<std::uint32_t>(keypoint.scale),
        std::bit_cast<std::uint32_t>(keypoint.orientation)};
}

std::size_t append_new_features_preserving_indices(
    features::FeatureSet& base, features::FeatureSet additional,
    const std::size_t maximum_total) {
    if (base.descriptor_dimension != additional.descriptor_dimension ||
        base.image_width != additional.image_width ||
        base.image_height != additional.image_height)
        throw std::runtime_error("Incompatible weak-view feature augmentation");
    base.compress_descriptors_u8();
    additional.compress_descriptors_u8();

    std::unordered_set<KeypointIdentity, KeypointIdentityHash> existing;
    existing.reserve(base.keypoints.size() * 2);
    for (const auto& keypoint : base.keypoints)
        existing.insert(keypoint_identity(keypoint));

    const std::size_t dimension = base.descriptor_dimension;
    const std::size_t original_size = base.keypoints.size();
    const std::size_t reserve_total = maximum_total > 0
        ? std::min(
              original_size + additional.keypoints.size(), maximum_total)
        : original_size + additional.keypoints.size();
    base.keypoints.reserve(reserve_total);
    base.descriptors_u8.reserve(
        reserve_total * dimension);
    for (std::size_t index = 0; index < additional.keypoints.size(); ++index) {
        if (maximum_total > 0 && base.keypoints.size() >= maximum_total) break;
        if (!existing.insert(
                keypoint_identity(additional.keypoints[index])).second)
            continue;
        base.keypoints.push_back(additional.keypoints[index]);
        const auto* descriptor =
            additional.descriptors_u8.data() + index * dimension;
        base.descriptors_u8.insert(
            base.descriptors_u8.end(), descriptor, descriptor + dimension);
    }
    base.mark_descriptors_modified();
    return base.keypoints.size() - original_size;
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

std::vector<AlignLiveIndexMatch> live_matches_from_raw(
    const std::vector<features::FeatureMatch>& matches) {
    std::vector<AlignLiveIndexMatch> out;
    out.reserve(matches.size());
    for (const features::FeatureMatch& match : matches)
        out.push_back({match.query, match.train, match.score});
    return out;
}

std::vector<AlignLiveIndexMatch> live_matches_from_inliers(
    const std::vector<FeatureMatch>& matches) {
    std::vector<AlignLiveIndexMatch> out;
    out.reserve(matches.size());
    for (const FeatureMatch& match : matches)
        out.push_back({match.query, match.train, 1.F});
    return out;
}

void emit_live_features(
    AlignLivePreview* live, const std::size_t index,
    const std::filesystem::path& path, const features::FeatureSet& features) {
    if (live == nullptr) return;
    live->publish_features(index, path, features);
}

void emit_live_raw_matches(
    AlignLivePreview* live, const Scene& scene, const Index id1,
    const Index id2, const std::vector<features::FeatureMatch>& matches) {
    if (live == nullptr || matches.empty()) return;
    if (id1 >= scene.images.size() || id2 >= scene.images.size()) return;
    const auto packed = live_matches_from_raw(matches);
    live->publish_matches(
        id1, scene.images[id1].path, scene.images[id1].features, id2,
        scene.images[id2].path, scene.images[id2].features, packed);
}

void emit_live_inliers(
    AlignLivePreview* live, const Scene& scene, const ImagePair& pair) {
    if (live == nullptr) return;
    if (pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size())
        return;
    const auto packed = live_matches_from_inliers(pair.matches);
    live->publish_matches(
        pair.id1, scene.images[pair.id1].path, scene.images[pair.id1].features,
        pair.id2, scene.images[pair.id2].path, scene.images[pair.id2].features,
        packed, AlignLiveKind::inliers);
}

// Owner-thread extract coordinator: main thread owns CUDA/ORT context while
// worker threads overlap CPU post-processing (grid selection / Image packing).
void extract_features_siftgpu_coordinator(
    Scene& scene, const std::vector<std::filesystem::path>& image_paths,
    features::FeatureExtractor& extractor, const unsigned max_features,
    core::ProgressReporter& progress, const bool compress_u8 = false,
    AlignLivePreview* live = nullptr) {
    const std::size_t count = image_paths.size();
    scene.images.resize(count);
    if (count == 0) return;
    auto& pool = parallel::global_thread_pool();
    parallel::FutureGroup post_tasks;
    post_tasks.reserve(count);

    const bool use_gray_prefetch =
        extractor.info().accepts_gray && !extractor.info().accepts_rgb;

    if (use_gray_prefetch) {
        constexpr std::size_t prefetch_depth = 4;
        std::array<std::future<io::GrayImage>, prefetch_depth> loads;
        const auto schedule_load = [&](std::size_t index) {
            auto task = std::make_shared<std::packaged_task<io::GrayImage()>>(
                [path = image_paths[index]] { return io::load_gray(path); });
            auto future = task->get_future();
            pool.submit([task] { (*task)(); });
            return future;
        };
        for (std::size_t i = 0; i < std::min(count, prefetch_depth); ++i)
            loads[i] = schedule_load(i);
        const auto* sift_gpu = dynamic_cast<const features::SiftGpuExtractor*>(&extractor);
        double decode_wait_seconds = 0, owner_extract_seconds = 0;
        for (std::size_t index = 0; index < count; ++index) {
            const auto wait_start = std::chrono::steady_clock::now();
            io::GrayImage gray = loads[index % prefetch_depth].get();
            decode_wait_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wait_start).count();
            if (index + prefetch_depth < count)
                loads[index % prefetch_depth] = schedule_load(index + prefetch_depth);
            const auto extract_start = std::chrono::steady_clock::now();
            features::FeatureSet features = sift_gpu
                ? sift_gpu->extract_gray_deferred(gray.pixels, gray.width, gray.height)
                : extractor.extract_gray(gray.pixels, gray.width, gray.height);
            owner_extract_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - extract_start).count();

            post_tasks.submit(
                pool,
                [&, index, sift_gpu, path = image_paths[index],
                 features = std::move(features)]() mutable {
                    if (sift_gpu) sift_gpu->finalize_descriptors(features);
                    features = select_top_features_grid_3x3(
                        std::move(features), max_features);
                    if (compress_u8) features.compress_descriptors_u8();
                    scene.images[index] = make_image_from_features(
                        static_cast<Index>(index), path, std::move(features));
                    emit_live_features(
                        live, index, path, scene.images[index].features);
                    progress.advance();
                });

            // Bound the float32 descriptors retained by queued work, even
            // when extraction outruns CPU grid selection / quantization.
            if ((index + 1) % 8 == 0) post_tasks.wait();
        }
        core::Logger::instance().info("feature extraction pipeline: prefetch=", prefetch_depth,
            " decode_wait_s=", decode_wait_seconds, " owner_extract_s=", owner_extract_seconds);
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
                    if (compress_u8) features.compress_descriptors_u8();
                    scene.images[index] = make_image_from_features(
                        static_cast<Index>(index), path, std::move(features));
                    emit_live_features(
                        live, index, path, scene.images[index].features);
                    progress.advance();
                });
            if ((index + 1) % 8 == 0) post_tasks.wait();
        }
    }

    post_tasks.wait();
    if (live != nullptr) live->flush();
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

    if (candidate.zero_baseline) {
        ImagePair pair(candidate.id1, candidate.id2);
        pair.relative_pose = Pose3D::identity();
        pair.matches.reserve(raw.size());
        for (const features::FeatureMatch& match : raw) {
            if (match.query >= img1.features.keypoints.size() ||
                match.train >= img2.features.keypoints.size())
                continue;
            pair.matches.push_back({match.query, match.train});
        }
        if (pair.matches.size() < relative.min_inliers) return result;
        pair.weight_spatial = 0.5F;
        pair.weight_geometry = 0.5F;
        pair.mean_ray_angle = 0.F;
        pair.homography_ratio = 1.F;
        pair.degenerate_planar = true;
        pair.zero_baseline = true;
        // Keep the edge as an observed grouping constraint, but exclude its
        // identical pixels from track construction and relative-pose averaging.
        // A free BA can otherwise split copies even after an identity init.
        pair.active = false;
        result.diagnostics.ransac_inliers =
            static_cast<unsigned>(pair.matches.size());
        result.diagnostics.filtered_inliers =
            static_cast<unsigned>(pair.matches.size());
        result.diagnostics.accepted = true;
        result.pair = std::move(pair);
        return result;
    }

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

unsigned recompute_relative_poses_with_calibrated_focals(
    Scene& scene, const RelativePoseOptions& relative) {
    std::atomic<unsigned> updated{0};
    std::atomic<unsigned> rejected{0};
    RelativePoseOptions calibrated = relative;
    calibrated.force_fundamental = false;
    calibrated.force_shared_focal = false;

    parallel::parallel_for(
        scene.pairs.size(), scene.thread_count,
        [&](const std::size_t pair_index) {
            ImagePair& pair = scene.pairs[pair_index];
            if (pair.zero_baseline || !pair.active ||
                pair.id1 >= scene.images.size() ||
                pair.id2 >= scene.images.size() || pair.matches.empty())
                return;
            const Image& first = scene.images[pair.id1];
            const Image& second = scene.images[pair.id2];
            PinholeCamera camera1 = scene.camera_of(first);
            PinholeCamera camera2 = scene.camera_of(second);
            camera1.trust_intrinsics = true;
            camera2.trust_intrinsics = true;

            std::vector<Vec2> pixels1;
            std::vector<Vec2> pixels2;
            pixels1.reserve(pair.matches.size());
            pixels2.reserve(pair.matches.size());
            for (const FeatureMatch& match : pair.matches) {
                if (match.query >= first.features.keypoints.size() ||
                    match.train >= second.features.keypoints.size())
                    continue;
                const auto& point1 = first.features.keypoints[match.query];
                const auto& point2 = second.features.keypoints[match.train];
                pixels1.emplace_back(point1.x, point1.y);
                pixels2.emplace_back(point2.x, point2.y);
            }
            if (pixels1.size() < calibrated.min_inliers) {
                pair.active = false;
                ++rejected;
                return;
            }

            const RelativePoseResult geometry = estimate_relative_pose(
                pixels1, pixels2, camera1, camera2, calibrated);
            if (!geometry.success ||
                geometry.inlier_mask.size() != pair.matches.size()) {
                pair.active = false;
                pair.relative_pose.reset();
                ++rejected;
                return;
            }

            std::vector<FeatureMatch> inlier_matches;
            inlier_matches.reserve(geometry.num_inliers);
            for (std::size_t index = 0; index < pair.matches.size(); ++index)
                if (geometry.inlier_mask[index])
                    inlier_matches.push_back(pair.matches[index]);
            if (inlier_matches.size() < calibrated.min_inliers) {
                pair.active = false;
                pair.relative_pose.reset();
                ++rejected;
                return;
            }

            pair.matches = std::move(inlier_matches);
            pair.relative_pose = geometry.pose;
            pair.E = geometry.E;
            pair.F = geometry.F;
            pair.estimated_focal.reset();
            pair.H = geometry.H;
            pair.mean_ray_angle = geometry.mean_ray_angle;
            pair.weight_spatial = geometry.weight_spatial;
            pair.homography_ratio = geometry.homography_ratio;
            pair.degenerate_planar = geometry.degenerate_planar;
            pair.weight_geometry = geometry.degenerate_planar
                ? calibrated.degenerate_weight_scale
                : 1.F;
            ++updated;
        });

    core::Logger::instance().info(
        "frontend calibrated relative poses: updated=", updated.load(),
        " rejected=", rejected.load(), " total=", scene.pairs.size());
    return updated.load();
}

void finalize_view_graph(
    Scene& scene, const FrontEndOptions& options) {
    compute_pair_weights(scene, options.pair_weighting);
    const bool focal_updated = calibrate_view_graph_focals(scene);
    const bool exif_focal_updated = calibrate_exif_view_graph_focals(scene);
    const bool has_untrusted_intrinsics = std::any_of(
        scene.cameras.begin(), scene.cameras.end(),
        [](const PinholeCamera& camera) { return !camera.trust_intrinsics; });
    if (focal_updated || exif_focal_updated || has_untrusted_intrinsics) {
        // Unknown-focal F-RANSAC deliberately defers cheirality and strict
        // reprojection filtering. Complete that pass even when calibration
        // keeps the initial focal; otherwise unchecked F support reaches
        // track building and mapping precisely in the degenerate case.
        recompute_relative_poses_with_calibrated_focals(
            scene, options.relative);
        // Relative-pose filtering changes inlier counts, spatial support and
        // rotation cycles. This mirrors openMVS ComputeRelativePoses(), which
        // refreshes pair weights after view-graph calibration.
        compute_pair_weights(scene, options.pair_weighting);
    }
    build_tracks(scene, options.min_pair_weight);
}

// GPU/ORT matching stays on the owner thread while geometric verification is
// continuously consumed by the CPU pool. A bounded in-flight window prevents
// an arbitrarily large raw-match backlog without introducing batch barriers.
void match_and_verify_siftgpu_coordinator(
    Scene& scene, const std::vector<PairCandidate>& candidates,
    features::FeatureMatcher& matcher, const FrontEndOptions& options,
    const unsigned worker_threads, std::vector<RawPairMatches>& raw_pairs,
    std::vector<PairDiagnostics>& diagnostics,
    std::vector<ImagePair>& pair_slots, core::ProgressReporter& match_progress,
    const bool verify_geometry = true, AlignLivePreview* live = nullptr) {
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

    constexpr std::size_t batch_size = 8;
    for (std::size_t begin = 0; begin < candidates.size(); begin += batch_size) {
      const std::size_t end = std::min(begin + batch_size, candidates.size());
      std::vector<features::FeatureMatcher::Pair> batch;
      batch.reserve(end - begin);
      for (std::size_t i = begin; i < end; ++i)
          batch.emplace_back(&scene.images[candidates[i].id1].features,
                             &scene.images[candidates[i].id2].features);
      auto matched = matcher.match_batch(batch);
      for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
        const PairCandidate candidate = candidates[pair_index];
        RawPairMatches raw;
        raw.id1 = candidate.id1;
        raw.id2 = candidate.id2;
        raw.matches = std::move(matched[pair_index - begin].matches);
        if (!verify_geometry) {
            emit_live_raw_matches(
                live, scene, candidate.id1, candidate.id2, raw.matches);
            raw_pairs[pair_index] = std::move(raw);
            match_progress.advance();
            continue;
        }

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
                    if (verified.pair) {
                        pair_slots[pair_index] = *verified.pair;
                        emit_live_inliers(live, scene, pair_slots[pair_index]);
                    }
                    match_progress.advance();
                } catch (...) {
                    release_slot();
                    throw;
                }
                release_slot();
            });
      }
    }
    geometry_tasks.wait();
    if (live != nullptr) live->flush();
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

std::size_t mark_content_duplicate_pairs(
    std::vector<PairCandidate>& pairs,
    const ImageSetFingerprint& image_fingerprint) {
    std::size_t marked = 0;
    for (PairCandidate& pair : pairs) {
        if (pair.id1 >= image_fingerprint.files.size() ||
            pair.id2 >= image_fingerprint.files.size())
            continue;
        const ImageFileFingerprint& first = image_fingerprint.files[pair.id1];
        const ImageFileFingerprint& second = image_fingerprint.files[pair.id2];
        // Exact file equality is itself a measurement: the two views came from
        // the same capture. Estimate an identity pose instead of letting F/E
        // RANSAC invent a baseline for identical pixels.
        pair.zero_baseline = first.size == second.size &&
                             first.digest == second.digest;
        if (pair.zero_baseline) ++marked;
    }
    if (marked > 0)
        core::Logger::instance().info(
            "frontend marked zero-baseline duplicate pairs: ", marked);
    return marked;
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
    core::ProgressReporter& match_progress,
    AlignLivePreview* live = nullptr) {
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
        emit_live_raw_matches(
            live, scene, candidate.id1, candidate.id2, raw.matches);
        match_progress.advance();
    }
    match_progress.finish();

    // Cameras must exist before geometry verify (uses scene.cameras[camera_id]).
    // Image sizes are only known after phase-1 LightGlue merges.
    initialize_cameras(
        scene, options.focal_pixels, options.trust_focal_pixels, options.camera_model);

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
        if (verified.pair) {
            pair_slots[pair_index] = *verified.pair;
            emit_live_inliers(live, scene, pair_slots[pair_index]);
        }
        geometry_progress.advance();
    }
    geometry_progress.finish();
    if (live != nullptr) live->flush();
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
        finalize_view_graph(scene, options);
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
                retrieval_extract, true);
            retrieval_extract.finish();
            candidates = build_pair_candidates(retrieval_scene, runtime_options);
        } else {
            candidates = build_pair_list(
                scene.images.size(), runtime_options.neighbor_window);
        }
        mark_content_duplicate_pairs(candidates, image_fingerprint);
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
                raw_pairs, diagnostics, pair_slots, match_progress,
                runtime_options.live_preview);
            scene.pairs.clear();
            scene.pairs.reserve(pair_slots.size());
            for (ImagePair& pair : pair_slots) {
                if (pair.matches.empty()) continue;
                scene.pairs.push_back(std::move(pair));
            }
            initialize_cameras(
                scene, options.focal_pixels, options.trust_focal_pixels, options.camera_model);
            verify_image_snapshot(image_paths, image_fingerprint);
            checkpoints.save_scene(
                CheckpointStage::features, stage_keys.features, scene);
            checkpoints.save_matches(stage_keys.matches, raw_pairs);
        } else {
            core::Logger::instance().info("checkpoint hit: matches");
            initialize_cameras(
                scene, options.focal_pixels, options.trust_focal_pixels, options.camera_model);
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

        if (options.camera_model == CameraModel::automatic ||
            (options.camera_model == CameraModel::opencv_fisheye && options.focal_pixels <= 0)) {
            select_scene_camera_models(scene, raw_pairs, candidates, options);
            std::vector<ImagePair> pairs(candidates.size());
            parallel::parallel_for(candidates.size(), threads, [&](const std::size_t i) {
                auto verified = verify_pair_geometry(scene,candidates[i],raw_pairs[i].matches,options.relative);
                diagnostics[i] = verified.diagnostics;
                if (verified.pair) pairs[i] = std::move(*verified.pair);
            });
            scene.pairs.clear();
            for (auto& pair : pairs)
                if (!pair.matches.empty()) scene.pairs.push_back(std::move(pair));
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
        finalize_view_graph(scene, options);
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
                scene, image_paths, *extractor, options.max_features, progress,
                runtime_options.compress_descriptors_u8,
                runtime_options.live_preview);
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
                    if (runtime_options.compress_descriptors_u8)
                        features.compress_descriptors_u8();
                    scene.images[i] = make_image_from_features(
                        static_cast<Index>(i), image_paths[i],
                        std::move(features));
                    emit_live_features(
                        runtime_options.live_preview, i, image_paths[i],
                        scene.images[i].features);
                    progress.advance();
                });
        }
        if (runtime_options.live_preview != nullptr)
            runtime_options.live_preview->flush();

        initialize_cameras(
            scene, options.focal_pixels, options.trust_focal_pixels, options.camera_model);
        // Keep the scene in compact form from this point onward. Retrieval and
        // SiftGPU matching consume uint8 descriptors without permanently
        // expanding the full dataset back to float32.
        if (runtime_options.compress_descriptors_u8)
            compress_descriptors(scene);
        verify_image_snapshot(image_paths, image_fingerprint);
        checkpoints.save_scene(
            CheckpointStage::features, stage_keys.features, scene);
    } else {
        scene.thread_count = parallel::resolve_thread_count(options.thread_count);
        initialize_cameras(
            scene, options.focal_pixels, options.trust_focal_pixels, options.camera_model);
        core::Logger::instance().info("checkpoint hit: features");
        if (runtime_options.live_preview != nullptr && !scene.images.empty()) {
            const std::size_t n = scene.images.size();
            const std::size_t stride = n <= 8 ? 1 : std::max<std::size_t>(1, n / 8);
            for (std::size_t i = 0; i < n; ++i) {
                if (i + 1 != n && i % stride != 0) continue;
                emit_live_features(
                    runtime_options.live_preview, i, scene.images[i].path,
                    scene.images[i].features);
                runtime_options.live_preview->flush();
            }
        }
    }
    result.timing.extract_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - extract_started)
            .count();

    const auto match_started = std::chrono::steady_clock::now();
    auto candidates = build_pair_candidates(scene, runtime_options);
    mark_content_duplicate_pairs(candidates, image_fingerprint);
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
            // Automatic calibration consumes raw matches, then invalidates
            // every preliminary geometry estimate. Hybrid rescue still needs
            // preliminary geometry to choose which pairs to send to LightGlue.
            const bool verify_now = runtime_options.matcher == "hybrid_lightglue" ||
                !(options.camera_model == CameraModel::automatic ||
                  (options.camera_model == CameraModel::opencv_fisheye && options.focal_pixels <= 0));
            core::ProgressReporter match_progress(
                runtime_options.matcher == "hybrid_lightglue"
                    ? "fast match image pairs"
                    : "match image pairs",
                candidates.size());
            match_and_verify_siftgpu_coordinator(
                scene, candidates, *matcher, options, threads, raw_pairs,
                diagnostics, pair_slots, match_progress, verify_now,
                runtime_options.live_preview);
            match_progress.finish();

            if (runtime_options.matcher == "hybrid_lightglue") {
                std::vector<PairCandidate> rescue_candidates;
                std::vector<std::size_t> rescue_indices;
                std::vector<unsigned> primary_degree(scene.images.size(), 0U);
                for (const ImagePair& pair : pair_slots) {
                    if (!pair.active || pair.matches.empty()) continue;
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
                    primary_accepted +=
                        pair.active && !pair.matches.empty() ? 1U : 0U;
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
                        rescue_pair_slots, rescue_progress, true,
                        runtime_options.live_preview);
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

            geometry_verified_in_pipeline = verify_now;
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
    if (options.camera_model == CameraModel::automatic ||
            (options.camera_model == CameraModel::opencv_fisheye && options.focal_pixels <= 0)) {
        select_scene_camera_models(scene, raw_pairs, candidates, options);
        scene.pairs.clear();
        geometry_verified_in_pipeline = false;
    }
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
                if (verified.pair) {
                    pairs[ci] = std::move(*verified.pair);
                    emit_live_inliers(
                        runtime_options.live_preview, scene, pairs[ci]);
                }
                geometry_progress.advance();
            });
        geometry_progress.finish();
        if (runtime_options.live_preview != nullptr)
            runtime_options.live_preview->flush();

        scene.pairs.reserve(pairs.size());
        for (auto& pair : pairs) {
            if (pair.matches.empty()) continue;
            scene.pairs.push_back(std::move(pair));
        }
    }

    const bool can_expand_progressively =
        runtime_options.progressive_pair_expansion &&
        runtime_options.matcher == "gpu_mutual_ratio" &&
        runtime_options.neighbor_window + 1 < scene.images.size();
    if (can_expand_progressively) {
        std::vector<unsigned> verified_degree(scene.images.size(), 0U);
        std::vector<unsigned> nonplanar_degree(scene.images.size(), 0U);
        for (const ImagePair& pair : scene.pairs) {
            if (!pair.active) continue;
            ++verified_degree[pair.id1];
            ++verified_degree[pair.id2];
            // A few small accidental edges do not establish usable multi-view
            // depth. Require three well-supported, non-planar neighbors before
            // skipping the bounded high-feature rescue pass.
            if (pair.relative_pose && !pair.degenerate_planar && !pair.zero_baseline &&
                pair.matches.size() >= 100) {
                ++nonplanar_degree[pair.id1];
                ++nonplanar_degree[pair.id2];
            }
        }
        std::vector<std::uint8_t> weak(scene.images.size(), 0);
        const auto structural_risk = runtime_options.structural_pair_expansion
            ? find_structural_pair_risks(scene)
            : std::vector<std::uint8_t>(scene.images.size(), 0);
        unsigned structural_weak_images = 0;
        unsigned weak_images = 0;
        for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
            if (structural_risk[image_id]) ++structural_weak_images;
            if (verified_degree[image_id] >=
                runtime_options.progressive_min_verified_degree &&
                // Stable resection validates depths from at least three
                // anchors. Two neighbors can only form a locally connected
                // pocket and must not suppress feature augmentation.
                nonplanar_degree[image_id] >= 3 &&
                !structural_risk[image_id])
                continue;
            weak[image_id] = 1;
            ++weak_images;
        }

        std::vector<std::uint8_t> augmented(scene.images.size(), 0);
        if (weak_images > 0 && runtime_options.extractor == "siftgpu" &&
            runtime_options.progressive_rescue_max_features >
                runtime_options.max_features) {
            features::SiftGpuOptions extraction_options;
            extraction_options.peak_threshold = static_cast<float>(
                runtime_options.sift_contrast_threshold);
            extraction_options.maximum_features =
                runtime_options.progressive_rescue_max_features;
            features::SiftGpuExtractor rescue_extractor(extraction_options);
            if (!rescue_extractor.is_available())
                throw std::runtime_error(
                    "Weak-view SiftGPU augmentation is unavailable");
            std::size_t appended_features = 0;
            core::ProgressReporter augmentation_progress(
                "augment weak image features", weak_images);
            for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
                if (!weak[image_id]) continue;
                const io::GrayImage gray = io::load_gray(image_paths[image_id]);
                features::FeatureSet additional =
                    rescue_extractor.extract_gray(
                        gray.pixels, gray.width, gray.height);
                additional = select_top_features_grid_3x3(
                    std::move(additional),
                    runtime_options.progressive_rescue_max_features);
                const auto appended = append_new_features_preserving_indices(
                    scene.images[image_id].features, std::move(additional),
                    runtime_options.progressive_rescue_max_features);
                augmented[image_id] = appended != 0;
                appended_features += appended;
                augmentation_progress.advance();
            }
            augmentation_progress.finish();
            core::Logger::instance().info(
                "weak-view feature augmentation: images=", weak_images,
                " target=",
                runtime_options.progressive_rescue_max_features,
                " appended=", appended_features);
        }

        std::unordered_set<std::uint64_t> primary_pairs;
        primary_pairs.reserve(candidates.size() * 2);
        const auto candidate_key = [](const Index first, const Index second) {
            const Index low = std::min(first, second);
            const Index high = std::max(first, second);
            return (static_cast<std::uint64_t>(low) << 32U) | high;
        };
        for (const PairCandidate& candidate : candidates)
            primary_pairs.insert(
                candidate_key(candidate.id1, candidate.id2));
        std::unordered_set<std::uint64_t> verified_pairs;
        for (const auto& pair : scene.pairs)
            if ((pair.active || pair.zero_baseline) && !pair.matches.empty())
                verified_pairs.insert(candidate_key(pair.id1, pair.id2));

        std::vector<PairCandidate> rescue_candidates;
        std::unordered_set<std::uint64_t> rescue_pair_keys;
        const bool has_low_degree_views = std::any_of(
            verified_degree.begin(), verified_degree.end(),
            [&](unsigned degree) {
                return degree < runtime_options.progressive_min_verified_degree;
            });
        // Well-connected but low-parallax views need better anchors, not an
        // exhaustive second pass over a whole video. Reserve that cost for
        // genuinely disconnected/structurally weak small view graphs.
        const bool exhaustive_rescue =
            scene.images.size() <= runtime_options.progressive_max_images &&
            (has_low_degree_views || structural_weak_images > 0);
        rescue_pair_keys.reserve(
            weak_images *
            std::max<std::size_t>(
                1, runtime_options.progressive_rescue_max_pairs_per_image));
        std::vector<std::size_t> rescue_degree(scene.images.size(), 0);
        const auto add_rescue_candidate = [&](Index first, Index second) {
            if (first == second) return false;
            if (first > second) std::swap(first, second);
            const std::uint64_t key = candidate_key(first, second);
            // A failed primary match is not evidence that two images cannot
            // connect. Retry weak-view pairs with the rescue matcher;
            // never duplicate already verified edges.
            const bool retry_failed =
                (weak[first] || weak[second]) &&
                !verified_pairs.count(key);
            const bool retry_augmented = augmented[first] || augmented[second];
            if ((primary_pairs.count(key) && !retry_failed && !retry_augmented) || rescue_pair_keys.count(key))
                return false;
            const std::size_t budget =
                runtime_options.progressive_rescue_max_pairs_per_image;
            if (!exhaustive_rescue && budget > 0 &&
                ((weak[first] && rescue_degree[first] >= budget) ||
                 (weak[second] && rescue_degree[second] >= budget)))
                return false;
            rescue_pair_keys.insert(key);
            rescue_candidates.push_back({first, second});
            if (weak[first]) ++rescue_degree[first];
            if (weak[second]) ++rescue_degree[second];
            return true;
        };

        // Newly detected keypoints must also reach existing neighbors. More
        // edges alone do not help when their old tracks have too little depth.
        for (const auto& pair : scene.pairs)
            if (!pair.zero_baseline && (augmented[pair.id1] || augmented[pair.id2]))
                add_rescue_candidate(pair.id1, pair.id2);

        if (exhaustive_rescue) {
            for (Index first = 0; first < scene.images.size(); ++first) {
                for (Index second = first + 1; second < scene.images.size();
                     ++second) {
                    if (!weak[first] && !weak[second]) continue;
                    add_rescue_candidate(first, second);
                }
            }
        } else {
            // Prefer temporal continuity first. This bridges short runs of
            // low-texture/video frames without an O(weak_views * images) pass.
            const std::size_t radius =
                runtime_options.progressive_rescue_neighbor_window;
            std::vector<PairCandidate> deferred_internal;
            const auto propose = [&](Index first, Index second) {
                // Spend the bounded budget on potential main-map anchors
                // before adding more edges wholly inside a flagged branch.
                if (structural_risk[first] && structural_risk[second])
                    deferred_internal.push_back({first, second});
                else
                    add_rescue_candidate(first, second);
            };
            for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
                if (!weak[image_id]) continue;
                const Index begin = static_cast<Index>(
                    image_id > radius ? image_id - radius : 0);
                const Index end = static_cast<Index>(std::min<std::size_t>(
                    scene.images.size() - 1,
                    static_cast<std::size_t>(image_id) + radius));
                for (Index other = begin; other <= end; ++other)
                    propose(image_id, other);
            }

            if (runtime_options.progressive_rescue_retrieval_top_k > 0 &&
                scene.images.size() >= runtime_options.retrieval_min_images) {
                RetrievalOptions rescue_retrieval = runtime_options.retrieval;
                rescue_retrieval.top_k =
                    runtime_options.progressive_rescue_retrieval_top_k;
                auto retrieved =
                    retrieve_image_pairs(scene.images, rescue_retrieval);
                std::sort(
                    retrieved.begin(), retrieved.end(),
                    [](const RetrievedPair& left, const RetrievedPair& right) {
                        return left.score > right.score;
                    });
                for (const RetrievedPair& pair : retrieved) {
                    if (!weak[pair.first] && !weak[pair.second]) continue;
                    propose(pair.first, pair.second);
                }
            }
            for (const auto& pair : deferred_internal)
                add_rescue_candidate(pair.id1, pair.id2);
        }
        if (!rescue_candidates.empty()) {
            optimize_pairs_order(rescue_candidates, scene);
            FrontEndOptions rescue_options = runtime_options;
            rescue_options.max_features = std::max(
                runtime_options.max_features,
                runtime_options.progressive_rescue_max_features);
            rescue_options.match_ratio = std::max(
                runtime_options.match_ratio,
                runtime_options.progressive_rescue_match_ratio);
            rescue_options.relative.min_inliers = std::max(
                8U, std::min(
                        runtime_options.relative.min_inliers,
                        runtime_options.progressive_rescue_min_inliers));
            auto rescue_matcher = make_frontend_matcher(rescue_options);
            std::vector<RawPairMatches> rescue_raw_pairs;
            std::vector<PairDiagnostics> rescue_diagnostics;
            std::vector<ImagePair> rescue_pair_slots;
            core::ProgressReporter rescue_progress(
                "expand weak image pairs", rescue_candidates.size());
            match_and_verify_siftgpu_coordinator(
                scene, rescue_candidates, *rescue_matcher, rescue_options,
                threads, rescue_raw_pairs, rescue_diagnostics,
                rescue_pair_slots, rescue_progress, true,
                runtime_options.live_preview);
            rescue_progress.finish();

            unsigned rescued_pairs = 0, strengthened_pairs = 0;
            std::unordered_map<std::uint64_t, std::size_t> existing_pairs;
            for (std::size_t i = 0; i < scene.pairs.size(); ++i)
                existing_pairs.emplace(candidate_key(scene.pairs[i].id1, scene.pairs[i].id2), i);
            for (ImagePair& pair : rescue_pair_slots) {
                if (pair.matches.empty()) continue;
                const auto existing = existing_pairs.find(candidate_key(pair.id1, pair.id2));
                if (existing != existing_pairs.end()) {
                    auto& previous = scene.pairs[existing->second];
                    if (!previous.zero_baseline && pair.matches.size() > previous.matches.size() &&
                        (!pair.degenerate_planar || previous.degenerate_planar)) {
                        previous = std::move(pair);
                        ++strengthened_pairs;
                    }
                    continue;
                }
                scene.pairs.push_back(std::move(pair));
                ++rescued_pairs;
            }
            core::Logger::instance().info(
                "progressive pair expansion: weak_images=", weak_images,
                " structural_weak_images=", structural_weak_images,
                " strategy=", exhaustive_rescue ? "exhaustive" : "bounded",
                " attempted=", rescue_candidates.size(),
                " accepted=", rescued_pairs,
                " ratio=", rescue_options.match_ratio,
                " min_inliers=", rescue_options.relative.min_inliers,
                " strengthened=", strengthened_pairs,
                " pairs_total=", scene.pairs.size());
        }
    }
    // Descriptors are no longer needed after matching and weak-view rescue;
    // keep keypoints only for track construction and mapping.
    release_descriptors(scene);
    matcher->clear_prepared();

    verify_image_snapshot(
        image_paths, image_fingerprint, ImageSnapshotCheck::content);
    checkpoints.save_scene(
        CheckpointStage::geometry, stage_keys.geometry, scene);
    result.timing.match_verify_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - match_started)
            .count();

    const auto tracks_started = std::chrono::steady_clock::now();
    finalize_view_graph(scene, options);
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
