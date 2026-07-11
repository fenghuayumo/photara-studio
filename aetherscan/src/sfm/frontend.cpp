#include "aetherscan/sfm/frontend.hpp"

#include "aetherscan/features/registry.hpp"
#include "aetherscan/parallel/thread_pool.hpp"

#include <chrono>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace aetherscan::sfm {
namespace {

struct PairCandidate {
    std::size_t first{};
    std::size_t second{};
};

struct PairOutcome {
    bool valid{false};
    VerifiedPair pair;
    RelativePoseEdge edge;
    double seed_score{-1.0};
};

std::shared_ptr<features::FeatureExtractor> resolve_extractor(const FrontEndOptions& options) {
    if (options.extractor) return options.extractor;
    if (options.extractor_name == "sift")
        return std::make_shared<features::SiftExtractor>(options.sift);
    return std::shared_ptr<features::FeatureExtractor>(
        features::create_extractor(options.extractor_name).release());
}

std::shared_ptr<features::FeatureMatcher> resolve_matcher(const FrontEndOptions& options) {
    if (options.matcher) return options.matcher;
    if (options.matcher_name == "mutual_ratio")
        return std::make_shared<features::MutualRatioMatcher>(options.matcher_options);
    return std::shared_ptr<features::FeatureMatcher>(
        features::create_matcher(options.matcher_name).release());
}

}  // namespace

FrontEndResult run_frontend(
    const std::vector<std::filesystem::path>& image_files, const FrontEndOptions& options) {
    if (image_files.size() < 2) throw std::invalid_argument("Need at least two images");
    if (options.focal_pixels <= 0.0 || options.neighbor_window == 0)
        throw std::invalid_argument("Invalid front-end focal length or neighbor window");

    features::ensure_builtin_feature_backends();
    const auto extractor_proto = resolve_extractor(options);
    const auto matcher_proto = resolve_matcher(options);
    if (!extractor_proto || !matcher_proto)
        throw std::runtime_error("Front-end requires both an extractor and a matcher");

    const unsigned threads = parallel::resolve_thread_count(options.thread_count);
    FrontEndResult result;
    result.timing.threads_used = threads;
    result.scene.views.resize(image_files.size());
    result.scene.cameras.resize(image_files.size());

    {
        const auto extract_started = std::chrono::steady_clock::now();
        std::mutex progress_mutex;
        std::size_t completed = 0;

        parallel::parallel_for(image_files.size(), threads, [&](const std::size_t index) {
            thread_local std::unique_ptr<features::FeatureExtractor> local;
            if (!local || local->name() != extractor_proto->name())
                local = extractor_proto->clone();
            auto features = local->extract_file(image_files[index]);

            Camera camera;
            camera.id = static_cast<Id>(index);
            camera.width = features.image_width;
            camera.height = features.image_height;
            camera.fx = options.focal_pixels;
            camera.fy = options.focal_pixels;
            camera.cx = 0.5 * features.image_width;
            camera.cy = 0.5 * features.image_height;

            View view;
            view.id = static_cast<Id>(index);
            view.camera_id = view.id;
            view.image_path = image_files[index];
            view.features = std::move(features);

            result.scene.cameras[index] = camera;
            result.scene.views[index] = std::move(view);

            std::size_t done = 0;
            {
                std::lock_guard lock(progress_mutex);
                done = ++completed;
            }
            if (done % 10 == 0 || done == image_files.size())
                std::cout << "extracted " << done << '/' << image_files.size()
                          << " [" << extractor_proto->name() << "]\n";
        });
        result.timing.extract_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - extract_started)
                .count();
    }

    std::vector<PairCandidate> pairs;
    pairs.reserve(image_files.size() * options.neighbor_window);
    for (std::size_t first = 0; first < image_files.size(); ++first)
        for (std::size_t second = first + 1;
             second < std::min(image_files.size(), first + options.neighbor_window + 1);
             ++second)
            pairs.push_back({first, second});

    std::vector<PairOutcome> outcomes(pairs.size());
    {
        const auto match_started = std::chrono::steady_clock::now();
        parallel::parallel_for(pairs.size(), threads, [&](const std::size_t index) {
            thread_local std::unique_ptr<features::FeatureMatcher> local;
            if (!local || local->name() != matcher_proto->name())
                local = matcher_proto->clone();

            const auto& candidate = pairs[index];
            const auto& first_view = result.scene.views[candidate.first];
            const auto& second_view = result.scene.views[candidate.second];
            const auto matches = local->match(first_view.features, second_view.features);
            if (matches.matches.size() < options.two_view.minimum_inliers) return;

            const auto geometry = verify_two_view(
                result.scene.cameras[candidate.first], result.scene.cameras[candidate.second],
                first_view.features, second_view.features, matches, options.two_view);
            if (!geometry.valid) return;

            PairOutcome outcome;
            outcome.valid = true;
            outcome.pair = {
                static_cast<Id>(candidate.first), static_cast<Id>(candidate.second),
                geometry.inliers};
            outcome.edge = {
                static_cast<Id>(candidate.first), static_cast<Id>(candidate.second),
                geometry.rotation, geometry.translation_direction,
                static_cast<double>(geometry.inliers.matches.size()),
                geometry.inliers.matches.size()};
            outcome.seed_score =
                static_cast<double>(geometry.inliers.matches.size()) -
                0.5 * static_cast<double>(geometry.homography_inliers);
            outcomes[index] = std::move(outcome);
        });
        result.timing.match_verify_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - match_started)
                .count();
    }

    double best_seed_score = -1.0;
    result.seed_edge_index = 0;
    for (std::size_t index = 0; index < outcomes.size(); ++index) {
        auto& outcome = outcomes[index];
        if (!outcome.valid) continue;
        const auto first = pairs[index].first;
        const auto second = pairs[index].second;
        result.scene.verified_pairs.push_back(std::move(outcome.pair));
        result.edges.push_back(std::move(outcome.edge));
        if (second > first + 1 && outcome.seed_score > best_seed_score) {
            best_seed_score = outcome.seed_score;
            result.seed_edge_index = result.edges.size() - 1;
        }
    }
    if (result.edges.empty()) throw std::runtime_error("No valid view-graph edges");

    const auto tracks_started = std::chrono::steady_clock::now();
    result.scene.tracks = build_tracks(result.scene.views, result.scene.verified_pairs, 2);
    result.timing.tracks_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tracks_started).count();

    for (auto& view : result.scene.views) {
        view.features.descriptors.clear();
        view.features.descriptors.shrink_to_fit();
        view.features.descriptor_dimension = 0;
    }
    return result;
}

}  // namespace aetherscan::sfm
