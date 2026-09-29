// Manual parity/performance harness for the Vulkan feature backends (not a
// ctest entry, mirroring src/tools/splat_vulkan_parity.cpp). It runs SiftGPU
// (the CUDA reference path) and the Vulkan translation on the same images and
// reports feature counts, per-level distributions, geometric agreement and
// per-image timings, plus matcher agreement and timings.
//
//   photara_vulkan_features_parity [width height runs] [scene|noise]
#include "features/features.hpp"
#include "features/vulkan_features.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

}  // namespace

int main(int argc, char** argv) {
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
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::steady_clock::now();
        const auto warm_cuda = cuda.extract_gray(first, width, height);
        cuda_ms += seconds_since(start);
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
    auto start = std::chrono::steady_clock::now();
    const auto matches = matcher.match_batch(pairs);
    const double vulkan_match_ms = seconds_since(start);
    SiftGpuMatcher reference{SiftGpuMatcherOptions{}};
    if (reference.is_available()) {
        start = std::chrono::steady_clock::now();
        const auto reference_matches = reference.match_batch(pairs);
        const double cuda_match_ms = seconds_since(start);
        std::size_t identical = 0;
        for (std::size_t i = 0; i < matches.size(); ++i)
            identical += same_matches(matches[i], reference_matches[i]) ? 1 : 0;
        std::printf(
            "matching %d pairs of %zu x %zu descriptors: vulkan %.2f ms, "
            "siftgpu %.2f ms (%.2fx), identical result sets %zu/%d\n",
            kPairCount, vulkan_u8.keypoints.size(), cuda_u8.keypoints.size(),
            vulkan_match_ms, cuda_match_ms,
            cuda_match_ms > 0.0 ? vulkan_match_ms / cuda_match_ms : 0.0, identical,
            kPairCount);
    }
    return 0;
}
