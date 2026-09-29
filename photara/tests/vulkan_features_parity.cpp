// Manual parity/performance harness for the Vulkan feature backends (not a
// ctest entry, mirroring src/tools/splat_vulkan_parity.cpp). It runs SiftGPU
// (the CUDA reference path) and the Vulkan translation on the same images and
// reports feature counts, per-level distributions, geometric agreement and
// per-image timings, plus matcher agreement and timings.
//
//   photara_vulkan_features_parity [width height runs] [scene|noise]
#include "features/features.hpp"
#include "features/vulkan_features.hpp"
#include "io/image.hpp"

#include <atomic>
#include <memory>
#include <thread>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace photara::features;

// Deterministic multi-scale texture: smooth low-frequency structure, blobs,
// edges and a little noise. Far closer to photographs than uniform noise.
std::vector<std::uint8_t> make_scene(std::uint32_t width, std::uint32_t height,
                                     std::uint32_t seed) {
    std::mt19937 random(seed);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    struct Blob {
        float x, y, radius, amplitude;
    };
    std::vector<Blob> blobs;
    for (int i = 0; i < 220; ++i) {
        Blob blob;
        blob.x = unit(random) * width;
        blob.y = unit(random) * height;
        blob.radius = 4.0F + unit(random) * 40.0F;
        blob.amplitude = (unit(random) - 0.5F) * 1.4F;
        blobs.push_back(blob);
    }
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width) * height);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const float fx = static_cast<float>(x);
            const float fy = static_cast<float>(y);
            float value = 0.5F + 0.10F * std::sin(fx * 0.021F) * std::cos(fy * 0.017F) +
                          0.06F * std::sin((fx + fy) * 0.045F);
            for (const auto& blob : blobs) {
                const float dx = fx - blob.x, dy = fy - blob.y;
                value += blob.amplitude *
                         std::exp(-(dx * dx + dy * dy) /
                                  (2.0F * blob.radius * blob.radius));
            }
            value += ((static_cast<int>(x / 37) + static_cast<int>(y / 41)) & 1)
                         ? 0.05F
                         : -0.05F;
            value += (unit(random) - 0.5F) * 0.02F;
            value = std::min(1.0F, std::max(0.0F, value));
            image[static_cast<std::size_t>(y) * width + x] =
                static_cast<std::uint8_t>(value * 255.0F + 0.5F);
        }
    }
    return image;
}

std::vector<std::uint8_t> make_noise(std::uint32_t width, std::uint32_t height,
                                     std::uint32_t seed) {
    std::mt19937 random(seed);
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width) * height);
    for (auto& pixel : image) pixel = static_cast<std::uint8_t>(random() & 0xffU);
    return image;
}

std::vector<std::uint8_t> shift_image(const std::vector<std::uint8_t>& source,
                                      std::uint32_t width, std::uint32_t height,
                                      std::uint32_t shift_x,
                                      std::uint32_t shift_y) {
    std::vector<std::uint8_t> shifted(source.size(), 0);
    for (std::uint32_t y = shift_y; y < height; ++y)
        for (std::uint32_t x = shift_x; x < width; ++x)
            shifted[static_cast<std::size_t>(y) * width + x] =
                source[static_cast<std::size_t>(y - shift_y) * width + x - shift_x];
    return shifted;
}

// Nearest-keypoint agreement: fraction of `a` keypoints with a `b` keypoint
// within `tolerance` pixels and `scale_tolerance` relative scale.
void report_agreement(const FeatureSet& a, const FeatureSet& b, double tolerance,
                      double scale_tolerance, const char* what) {
    std::size_t matched = 0;
    double distance_sum = 0.0;
    for (const auto& key : a.keypoints) {
        double best = 1e9;
        for (const auto& other : b.keypoints) {
            if (std::abs(double(other.scale) - double(key.scale)) >
                scale_tolerance * double(key.scale))
                continue;
            const double dx = double(other.x) - double(key.x);
            const double dy = double(other.y) - double(key.y);
            best = std::min(best, std::sqrt(dx * dx + dy * dy));
        }
        if (best <= tolerance) {
            ++matched;
            distance_sum += best;
        }
    }
    std::printf("%s: %zu/%zu (%.1f%%, mean offset %.4f px)\n", what, matched,
                a.keypoints.size(),
                a.keypoints.empty()
                    ? 0.0
                    : 100.0 * double(matched) / double(a.keypoints.size()),
                matched ? distance_sum / double(matched) : 0.0);
}

// Per-(octave, level) feature distribution derived from the keypoint scales.
void report_levels(const FeatureSet& set, const char* label) {
    constexpr double kSigma0 = 1.6 * 1.2599210498948732;  // 1.6 * 2^(1/3)
    std::vector<std::pair<int, long long>> buckets;
    for (const auto& key : set.keypoints) {
        const int bucket = static_cast<int>(
            std::llround(3.0 * std::log2(double(key.scale) / kSigma0)));
        auto found = std::find_if(
            buckets.begin(), buckets.end(),
            [&](const auto& entry) { return entry.first == bucket; });
        if (found == buckets.end()) buckets.emplace_back(bucket, 1);
        else ++found->second;
    }
    std::sort(buckets.begin(), buckets.end());
    std::printf("%s: per-level (bucket = 3*octave + level):", label);
    for (const auto& entry : buckets)
        std::printf(" %d:%lld", entry.first, entry.second);
    std::printf("\n");
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
               .count();
}

int run_dataset(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
            extension == ".tif" || extension == ".tiff" || extension == ".bmp")
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.size() < 2) {
        std::fprintf(stderr, "dataset needs at least two images: %s\n",
                     directory.string().c_str());
        return 2;
    }

    SiftGpuExtractor cuda{SiftGpuOptions{}};
    SiftVulkanExtractor vulkan{SiftVulkanOptions{}};
    if (!cuda.is_available() || !vulkan.is_available()) {
        std::fprintf(stderr, "both siftgpu and vulkan_sift are required\n");
        return 2;
    }

    // Decode each image once and feed the same pixels to both backends. This
    // keeps JPEG I/O and the OS file cache out of the GPU comparison.
    std::vector<FeatureSet> vulkan_features;
    vulkan_features.reserve(paths.size());
    double decode_ms = 0.0, cuda_ms = 0.0, vulkan_ms = 0.0;
    std::size_t cuda_feature_count = 0, vulkan_feature_count = 0;
    std::size_t count_mismatches = 0;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        auto start = std::chrono::steady_clock::now();
        const auto image = photara::io::load_gray(paths[i]);
        decode_ms += seconds_since(start);

        // Alternate launch order to avoid consistently assigning the warmer
        // GPU/cache state to one backend.
        FeatureSet cuda_set, vulkan_set;
        if ((i & 1U) == 0U) {
            start = std::chrono::steady_clock::now();
            cuda_set = cuda.extract_gray(image.pixels, image.width, image.height);
            cuda_ms += seconds_since(start);
            start = std::chrono::steady_clock::now();
            vulkan_set = vulkan.extract_gray(image.pixels, image.width, image.height);
            vulkan_ms += seconds_since(start);
        } else {
            start = std::chrono::steady_clock::now();
            vulkan_set = vulkan.extract_gray(image.pixels, image.width, image.height);
            vulkan_ms += seconds_since(start);
            start = std::chrono::steady_clock::now();
            cuda_set = cuda.extract_gray(image.pixels, image.width, image.height);
            cuda_ms += seconds_since(start);
        }
        cuda_feature_count += cuda_set.keypoints.size();
        vulkan_feature_count += vulkan_set.keypoints.size();
        count_mismatches += cuda_set.keypoints.size() != vulkan_set.keypoints.size();
        vulkan_set.compress_descriptors_u8();
        vulkan_features.push_back(std::move(vulkan_set));
    }

    std::vector<FeatureMatcher::Pair> pairs;
    pairs.reserve(vulkan_features.size() - 1);
    for (std::size_t i = 1; i < vulkan_features.size(); ++i)
        pairs.push_back({&vulkan_features[i - 1], &vulkan_features[i]});

    VulkanMutualRatioMatcher vulkan_matcher{VulkanMutualRatioMatcherOptions{}};
    SiftGpuMatcher cuda_matcher{SiftGpuMatcherOptions{}};
    const std::span<const FeatureMatcher::Pair> warm_pair(pairs.data(), 1);
    (void)vulkan_matcher.match_batch(warm_pair);
    (void)cuda_matcher.match_batch(warm_pair);
    vulkan_matcher.clear_prepared();
    cuda_matcher.clear_prepared();

    auto start = std::chrono::steady_clock::now();
    const auto vulkan_matches = vulkan_matcher.match_batch(pairs);
    const double vulkan_match_ms = seconds_since(start);
    start = std::chrono::steady_clock::now();
    const auto cuda_matches = cuda_matcher.match_batch(pairs);
    const double cuda_match_ms = seconds_since(start);
    start = std::chrono::steady_clock::now();
    (void)vulkan_matcher.match_batch(pairs);
    const double vulkan_cached_match_ms = seconds_since(start);
    start = std::chrono::steady_clock::now();
    (void)cuda_matcher.match_batch(pairs);
    const double cuda_cached_match_ms = seconds_since(start);

    std::size_t identical_pairs = 0, vulkan_match_count = 0, cuda_match_count = 0;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        vulkan_match_count += vulkan_matches[i].matches.size();
        cuda_match_count += cuda_matches[i].matches.size();
        if (vulkan_matches[i].matches.size() != cuda_matches[i].matches.size()) continue;
        bool identical = true;
        for (std::size_t j = 0; j < vulkan_matches[i].matches.size(); ++j) {
            identical &= vulkan_matches[i].matches[j].query ==
                             cuda_matches[i].matches[j].query &&
                         vulkan_matches[i].matches[j].train ==
                             cuda_matches[i].matches[j].train;
        }
        identical_pairs += identical;
    }

    const double image_count = static_cast<double>(paths.size());
    std::printf("dataset: %s (%zu images, %u x %u)\n",
                directory.string().c_str(), paths.size(),
                vulkan_features.front().image_width,
                vulkan_features.front().image_height);
    std::printf("decode: %.1f ms total (%.2f ms/image)\n", decode_ms,
                decode_ms / image_count);
    std::printf(
        "extraction: vulkan %.1f ms (%.2f/image, %zu features), siftgpu %.1f ms "
        "(%.2f/image, %zu features), vulkan/cuda %.2fx, count mismatches %zu/%zu\n",
        vulkan_ms, vulkan_ms / image_count, vulkan_feature_count, cuda_ms,
        cuda_ms / image_count, cuda_feature_count, vulkan_ms / cuda_ms,
        count_mismatches, paths.size());
    std::printf(
        "adjacent matching: %zu pairs, vulkan %.1f ms (%.2f/pair), cuda %.1f ms "
        "(%.2f/pair), vulkan/cuda %.2fx, matches %zu/%zu, identical %zu/%zu\n",
        pairs.size(), vulkan_match_ms, vulkan_match_ms / pairs.size(),
        cuda_match_ms, cuda_match_ms / pairs.size(),
        vulkan_match_ms / cuda_match_ms, vulkan_match_count, cuda_match_count,
        identical_pairs, pairs.size());
    std::printf(
        "adjacent matching (descriptor cache warm): vulkan %.1f ms (%.2f/pair), "
        "cuda %.1f ms (%.2f/pair), vulkan/cuda %.2fx\n",
        vulkan_cached_match_ms, vulkan_cached_match_ms / pairs.size(),
        cuda_cached_match_ms, cuda_cached_match_ms / pairs.size(),
        vulkan_cached_match_ms / cuda_cached_match_ms);
    return identical_pairs == pairs.size() ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::filesystem::is_directory(argv[1]))
        return run_dataset(argv[1]);
    std::uint32_t width = 640, height = 480;
    int runs = 3;
    std::string kind = "scene";
    if (argc > 1) width = static_cast<std::uint32_t>(std::atoi(argv[1]));
    if (argc > 2) height = static_cast<std::uint32_t>(std::atoi(argv[2]));
    if (argc > 3) runs = std::atoi(argv[3]);
    if (argc > 4) kind = argv[4];

    const auto first = kind == "noise" ? make_noise(width, height, 1234)
                                       : make_scene(width, height, 20260929);
    const auto second = shift_image(first, width, height, 7, 5);

    SiftGpuExtractor cuda{SiftGpuOptions{}};
    if (!cuda.is_available()) {
        std::printf("siftgpu reference backend is unavailable\n");
        return 1;
    }
    SiftVulkanExtractor vulkan{SiftVulkanOptions{}};
    if (!vulkan.is_available()) {
        std::printf("vulkan_sift backend is unavailable\n");
        return 1;
    }
    if (kind == "mixed") {
        // Pre-condition both backends with an unrelated extraction so that any
        // dependence on recycled device buffers shows up in the comparison.
        const auto scratch = make_noise(width, height, 777);
        (void)cuda.extract_gray(scratch, width, height);
        (void)vulkan.extract_gray(scratch, width, height);
        kind = "scene";
    }

    const FeatureSet cuda_first = cuda.extract_gray(first, width, height);
    const FeatureSet vulkan_first = vulkan.extract_gray(first, width, height);
    const FeatureSet cuda_second = cuda.extract_gray(second, width, height);
    const FeatureSet vulkan_second = vulkan.extract_gray(second, width, height);

    std::printf("%s image %ux%u\n", kind.c_str(), width, height);
    std::printf("features: siftgpu %zu/%zu, vulkan %zu/%zu\n",
                cuda_first.keypoints.size(), cuda_second.keypoints.size(),
                vulkan_first.keypoints.size(), vulkan_second.keypoints.size());
    report_levels(cuda_first, "siftgpu");
    report_levels(vulkan_first, "vulkan ");
    report_agreement(vulkan_first, cuda_first, 0.5, 0.02,
                     "vulkan keypoints reproduced by siftgpu");
    report_agreement(cuda_first, vulkan_first, 0.5, 0.02,
                     "siftgpu keypoints reproduced by vulkan");

    // Descriptors: cross-backend matching must recover the known 7 px / 5 px
    // shift, both for vulkan -> vulkan and vulkan -> siftgpu features.
    auto descriptor_agreement = [&](const FeatureSet& query,
                                    const FeatureSet& train, const char* label) {
        FeatureSet q = query, t = train;
        q.compress_descriptors_u8();
        t.compress_descriptors_u8();
        VulkanMutualRatioMatcher matcher{VulkanMutualRatioMatcherOptions{}};
        const auto matches = matcher.match(q, t);
        std::size_t inliers = 0;
        for (const auto& match : matches.matches) {
            const auto& a = q.keypoints[match.query];
            const auto& b = t.keypoints[match.train];
            if (std::abs((b.x - a.x) - 7.0F) < 2.0F &&
                std::abs((b.y - a.y) - 5.0F) < 2.0F)
                ++inliers;
        }
        std::printf("%s: matches %zu, shift inliers %zu (%.1f%%)\n", label,
                    matches.matches.size(), inliers,
                    matches.matches.empty()
                        ? 0.0
                        : 100.0 * double(inliers) / double(matches.matches.size()));
    };
    descriptor_agreement(vulkan_first, vulkan_second, "vulkan descriptors");
    descriptor_agreement(vulkan_first, cuda_second, "vulkan vs siftgpu descriptors");

    double cuda_ms = 0.0, vulkan_ms = 0.0;
    double cuda_deferred_ms = 0.0, cuda_finalize_ms = 0.0;
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::steady_clock::now();
        const auto warm_cuda = cuda.extract_gray(first, width, height);
        cuda_ms += seconds_since(start);
        // Split the SiftGPU path so the host-side descriptor post-processing it
        // shares with the caller can be attributed separately.
        start = std::chrono::steady_clock::now();
        auto deferred = cuda.extract_gray_deferred(first, width, height);
        cuda_deferred_ms += seconds_since(start);
        start = std::chrono::steady_clock::now();
        cuda.finalize_descriptors(deferred);
        cuda_finalize_ms += seconds_since(start);
        start = std::chrono::steady_clock::now();
        const auto warm_vulkan = vulkan.extract_gray(first, width, height);
        vulkan_ms += seconds_since(start);
        if (warm_cuda.keypoints.size() != cuda_first.keypoints.size() ||
            warm_vulkan.keypoints.size() != vulkan_first.keypoints.size())
            std::printf("warning: repeated extraction changed the feature count\n");
        if (warm_vulkan.keypoints.size() != vulkan_first.keypoints.size()) {
            auto report_delta = [&](const FeatureSet& a, const FeatureSet& b,
                                    const char* what) {
                int printed = 0;
                for (const auto& key : a.keypoints) {
                    bool found = false;
                    for (const auto& other : b.keypoints) {
                        if (std::abs(other.scale - key.scale) > 0.02F * key.scale)
                            continue;
                        if (std::abs(other.x - key.x) <= 0.05F &&
                            std::abs(other.y - key.y) <= 0.05F) {
                            found = true;
                            break;
                        }
                    }
                    if (!found && printed < 6) {
                        ++printed;
                        std::printf("  %s only: (%.4f, %.4f) scale %.4f orientation "
                                    "%.4f\n",
                                    what, double(key.x), double(key.y),
                                    double(key.scale), double(key.orientation));
                    }
                }
            };
            report_delta(vulkan_first, warm_vulkan, "first run");
            report_delta(warm_vulkan, vulkan_first, "later run");
        }
    }
    std::printf("extraction over %d runs: siftgpu %.2f ms, vulkan %.2f ms (%.2fx)\n",
                runs, cuda_ms / runs, vulkan_ms / runs,
                cuda_ms > 0.0 ? vulkan_ms / cuda_ms : 0.0);
    std::printf("siftgpu split: SiftGPU pipeline + download %.2f ms, host RootSIFT "
                "%.2f ms\n",
                cuda_deferred_ms / runs, cuda_finalize_ms / runs);

    FeatureSet cuda_u8 = cuda_first, vulkan_u8 = vulkan_first;
    cuda_u8.compress_descriptors_u8();
    vulkan_u8.compress_descriptors_u8();
    constexpr int kPairCount = 16;
    std::vector<FeatureMatcher::Pair> pairs;
    pairs.reserve(kPairCount);
    for (int i = 0; i < kPairCount; ++i) pairs.push_back({&vulkan_u8, &cuda_u8});
    auto same_matches = [](const MatchSet& a, const MatchSet& b) {
        if (a.matches.size() != b.matches.size()) return false;
        for (std::size_t i = 0; i < a.matches.size(); ++i)
            if (a.matches[i].query != b.matches[i].query ||
                a.matches[i].train != b.matches[i].train)
                return false;
        return true;
    };
    VulkanMutualRatioMatcher matcher{VulkanMutualRatioMatcherOptions{}};
    // The SiftGPU matcher keeps its CUDA context on the constructing thread
    // (the frontend runs it on the owner thread), so the fan-out below reports
    // rejected calls for the SiftGPU side instead of a parallel throughput.
    SiftGpuMatcher reference{SiftGpuMatcherOptions{}};
    if (reference.is_available()) {
        // First-call cost: descriptor caches, pinned output and pipelines are
        // created lazily, so this is what a short run used to pay for.
        SiftGpuMatcher cold_reference{SiftGpuMatcherOptions{}};
        VulkanMutualRatioMatcher cold_matcher{VulkanMutualRatioMatcherOptions{}};
        auto cold_start = std::chrono::steady_clock::now();
        (void)cold_reference.match_batch(pairs);
        const double cold_cuda_ms = seconds_since(cold_start);
        cold_start = std::chrono::steady_clock::now();
        (void)cold_matcher.match_batch(pairs);
        const double cold_vulkan_ms = seconds_since(cold_start);
        std::printf("first batch (cold): vulkan %.3f ms, siftgpu %.3f ms\n",
                    cold_vulkan_ms, cold_cuda_ms);
        // Warm both matchers before timing the steady state.
        const auto warm_vulkan = matcher.match_batch(pairs);
        const auto warm_cuda = reference.match_batch(pairs);
        auto start = std::chrono::steady_clock::now();
        const auto matches = matcher.match_batch(pairs);
        const double vulkan_match_ms = seconds_since(start);
        start = std::chrono::steady_clock::now();
        const auto reference_matches = reference.match_batch(pairs);
        const double cuda_match_ms = seconds_since(start);
        std::size_t identical = 0;
        for (std::size_t i = 0; i < matches.size(); ++i)
            identical += same_matches(matches[i], reference_matches[i]) ? 1 : 0;
        std::printf(
            "matching %d pairs of %zu x %zu descriptors (one batch): vulkan %.3f ms, "
            "siftgpu %.3f ms (%.2fx), identical result sets %zu/%d\n",
            kPairCount, vulkan_u8.keypoints.size(), cuda_u8.keypoints.size(),
            vulkan_match_ms, cuda_match_ms,
            cuda_match_ms > 0.0 ? vulkan_match_ms / cuda_match_ms : 0.0, identical,
            kPairCount);
        // Per-call latency: the frontend matches pairs from a worker pool, so
        // the single-pair round trip is what parallel matching pays.
        constexpr int kSingleCalls = 32;
        const std::vector<FeatureMatcher::Pair> single{pairs[0]};
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < kSingleCalls; ++i) (void)matcher.match_batch(single);
        const double vulkan_single_ms = seconds_since(start) / kSingleCalls;
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < kSingleCalls; ++i) (void)reference.match_batch(single);
        const double cuda_single_ms = seconds_since(start) / kSingleCalls;
        std::printf(
            "single-pair calls (%zu x %zu): vulkan %.3f ms, siftgpu %.3f ms "
            "(%.2fx)\n",
            vulkan_u8.keypoints.size(), cuda_u8.keypoints.size(), vulkan_single_ms,
            cuda_single_ms,
            cuda_single_ms > 0.0 ? vulkan_single_ms / cuda_single_ms : 0.0);
        (void)warm_vulkan;
        (void)warm_cuda;
    }
    {
        // Frontend pattern: one matcher clone per worker thread, each used from
        // its own thread. This is where per-clone setup costs show up.
        constexpr unsigned kWorkers = 8;
        constexpr int kPairsPerWorker = 4;
        auto fan_out = [&](FeatureMatcher& base, std::vector<std::unique_ptr<FeatureMatcher>>& clones,
                           std::atomic<long long>& micros,
                           std::atomic<long long>& failures) {
            clones.reserve(kWorkers);
            for (unsigned t = 0; t < kWorkers; ++t) clones.push_back(base.clone());
            // Warm each clone so the measurement shows steady-state throughput
            // instead of per-clone initialization.
            for (auto& clone : clones) {
                const std::vector<FeatureMatcher::Pair> one{pairs[0]};
                (void)clone->match_batch(one);
            }
            std::vector<std::thread> workers;
            workers.reserve(kWorkers);
            const auto start = std::chrono::steady_clock::now();
            for (unsigned t = 0; t < kWorkers; ++t)
                workers.emplace_back([&, t] {
                    for (int i = 0; i < kPairsPerWorker; ++i) {
                        const std::vector<FeatureMatcher::Pair> one{pairs[0]};
                        try {
                            (void)clones[t]->match_batch(one);
                        } catch (const std::exception&) {
                            ++failures;
                        }
                    }
                });
            for (auto& worker : workers) worker.join();
            micros.store(static_cast<long long>(seconds_since(start) * 1000.0));
        };
        std::vector<std::unique_ptr<FeatureMatcher>> vulkan_clones, cuda_clones;
        std::atomic<long long> vulkan_us{0}, cuda_us{0}, vulkan_failures{0},
            cuda_failures{0};
        fan_out(matcher, vulkan_clones, vulkan_us, vulkan_failures);
        fan_out(reference, cuda_clones, cuda_us, cuda_failures);
        std::printf(
            "clone fan-out (%u threads x %d pairs): vulkan %.0f us/pair (%lld "
            "failures), siftgpu %.0f us/pair (%lld failures; non-zero means the "
            "matcher is owner-thread bound)\n",
            kWorkers, kPairsPerWorker,
            double(vulkan_us.load()) / (kWorkers * kPairsPerWorker),
            static_cast<long long>(vulkan_failures.load()),
            double(cuda_us.load()) / (kWorkers * kPairsPerWorker),
            static_cast<long long>(cuda_failures.load()));
    }
    {
        // Large-feature isolation: the dataset matches ~12k x 12k descriptor
        // pairs, which is where the two matchers diverge. Synthetic descriptors
        // with the same shape keep this independent of the extractor.
        constexpr std::size_t kLarge = 12000;
        std::mt19937 random(7);
        auto make = [&]() {
            FeatureSet set;
            set.image_width = 1000;
            set.image_height = 1000;
            set.descriptor_dimension = 128;
            set.metric = DescriptorMetric::l2_root;
            set.storage = DescriptorStorage::uint8;
            set.keypoints.resize(kLarge);
            set.descriptors_u8.resize(kLarge * 128);
            for (std::size_t i = 0; i < set.descriptors_u8.size(); ++i)
                set.descriptors_u8[i] = static_cast<std::uint8_t>(random() & 0xffU);
            set.mark_descriptors_modified();
            return set;
        };
        FeatureSet large_a, large_b;
        large_a = make();  // move-assignment gives each set a fresh identity
        large_b = make();
        const std::vector<FeatureMatcher::Pair> large_pair{{&large_a, &large_b}};
        VulkanMutualRatioMatcher large_vulkan{VulkanMutualRatioMatcherOptions{}};
        SiftGpuMatcher large_cuda{SiftGpuMatcherOptions{}};
        auto time_one = [&](FeatureMatcher& matcher) {
            const auto start = std::chrono::steady_clock::now();
            const auto matches = matcher.match_batch(large_pair);
            const double ms = seconds_since(start);
            return std::make_pair(ms, matches.front().matches.size());
        };
        (void)time_one(large_vulkan);  // warm
        (void)time_one(large_cuda);
        const auto vulkan_large = time_one(large_vulkan);
        const auto cuda_large = time_one(large_cuda);
        std::printf(
            "large pair (%zu x %zu descriptors): vulkan %.1f ms (%zu matches), "
            "siftgpu %.1f ms (%zu matches)\n",
            kLarge, kLarge, vulkan_large.first, vulkan_large.second,
            cuda_large.first, cuda_large.second);
    }
    return 0;
}
