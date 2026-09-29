#include "features/features.hpp"
#include "features/vulkan_features.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace photara::features;

std::vector<std::uint8_t> make_image(
    std::uint32_t width, std::uint32_t height, std::uint32_t seed,
    std::uint32_t shift_x, std::uint32_t shift_y,
    std::vector<std::uint8_t>& reference) {
    std::mt19937 random(seed);
    reference.assign(static_cast<std::size_t>(width) * height, 0);
    for (auto& pixel : reference) pixel = static_cast<std::uint8_t>(random() & 0xffU);
    std::vector<std::uint8_t> shifted(reference.size(), 0);
    for (std::uint32_t y = shift_y; y < height; ++y) {
        for (std::uint32_t x = shift_x; x < width; ++x) {
            shifted[static_cast<std::size_t>(y) * width + x] =
                reference[static_cast<std::size_t>(y - shift_y) * width + x - shift_x];
        }
    }
    return shifted;
}

// Deterministic textured image (smooth structure + blobs + a little noise),
// matching the generator used by photara_vulkan_features_parity. Uniform noise
// is a poor parity target: its DoG is nearly flat at the detection threshold,
// so SiftGPU's own undefined pyramid-border reads make its keypoint set vary
// by ~10% between runs.
std::vector<std::uint8_t> make_textured_image(std::uint32_t width,
                                              std::uint32_t height,
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

MatchSet oracle(const FeatureSet& query, const FeatureSet& train, bool mutual) {
    auto direction = [](const FeatureSet& a, const FeatureSet& b) {
        std::vector<int> out(a.keypoints.size(), -1);
        for (std::size_t i = 0; i < a.keypoints.size(); ++i) {
            int best = 0, second = 0, index = -1;
            for (std::size_t j = 0; j < b.keypoints.size(); ++j) {
                int dot = 0;
                for (int k = 0; k < 128; ++k)
                    dot += int(a.descriptors_u8[i * 128 + k]) *
                           int(b.descriptors_u8[j * 128 + k]);
                if (dot > best) {
                    second = best;
                    best = dot;
                    index = static_cast<int>(j);
                } else {
                    second = std::max(second, dot);
                }
            }
            const float d = static_cast<float>(std::acos(std::min(best / 262144.0, 1.0)));
            const float d2 =
                static_cast<float>(std::acos(std::min(second / 262144.0, 1.0)));
            if (d < 0.7F && d < 0.8F * d2) out[i] = index;
        }
        return out;
    };
    const auto rows = direction(query, train);
    const auto cols = direction(train, query);
    MatchSet out;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= 0 && (!mutual || cols[rows[i]] == int(i)))
            out.matches.push_back({FeatureIndex(i), FeatureIndex(rows[i]), 1.0F});
    }
    return out;
}

void require_same_matches(const MatchSet& a, const MatchSet& b, const char* what) {
    auto keys = [](const MatchSet& m) {
        std::vector<std::pair<FeatureIndex, FeatureIndex>> v;
        for (const auto& x : m.matches) v.emplace_back(x.query, x.train);
        std::sort(v.begin(), v.end());
        return v;
    };
    if (keys(a) != keys(b)) {
        std::cerr << what << ": correspondence mismatch ("
                  << a.matches.size() << " vs " << b.matches.size() << ")\n";
        throw std::runtime_error(what);
    }
}

void test_matcher() {
    if (!VulkanMutualRatioMatcher::is_built()) {
        std::cout << "vulkan matcher: backend not built\n";
        return;
    }
    constexpr std::uint32_t width = 512, height = 384;
    std::vector<std::uint8_t> first;
    const auto second =
        make_image(width, height, 1234, 7, 5, first);

    FeatureSet query, train;
    if (SiftGpuExtractor::is_built()) {
        SiftGpuExtractor extractor;
        if (extractor.is_available()) {
            query = extractor.extract_gray(first, width, height);
            train = extractor.extract_gray(second, width, height);
        }
    }
    if (query.keypoints.empty()) {
        SiftOptions options;
        options.maximum_features = 3000;
        SiftExtractor extractor(options);
        query = extractor.extract_gray(first, width, height);
        train = extractor.extract_gray(second, width, height);
    }
    query.compress_descriptors_u8();
    train.compress_descriptors_u8();

    for (const bool mutual : {false, true}) {
        VulkanMutualRatioMatcherOptions options;
        options.mutual_check = mutual;
        VulkanMutualRatioMatcher matcher(options);
        require_same_matches(matcher.match(query, train),
                             oracle(query, train, mutual),
                             "vulkan matcher oracle parity");
    }

    VulkanMutualRatioMatcherOptions options;
    VulkanMutualRatioMatcher matcher(options);
    FeatureSet small = query, tail = train, empty = query;
    small.keypoints.resize(33);
    small.descriptors_u8.resize(33 * 128);
    small.mark_descriptors_modified();
    tail.keypoints.resize(65);
    tail.descriptors_u8.resize(65 * 128);
    tail.mark_descriptors_modified();
    empty.keypoints.clear();
    empty.descriptors_u8.clear();
    empty.mark_descriptors_modified();
    std::copy_n(small.descriptors_u8.begin(), 128, tail.descriptors_u8.begin());
    std::copy_n(small.descriptors_u8.begin(), 128, tail.descriptors_u8.begin() + 128);
    std::vector<FeatureMatcher::Pair> batch{
        {&small, &tail}, {&tail, &small}, {&empty, &small}, {&small, &empty}};
    for (int i = 0; i < 9; ++i) batch.emplace_back(&small, &tail);
    const auto out = matcher.match_batch(batch);
    for (std::size_t i = 0; i < batch.size(); ++i)
        require_same_matches(out[i], oracle(*batch[i].first, *batch[i].second, true),
                             "vulkan matcher batch parity");
    std::fill(small.descriptors_u8.begin(), small.descriptors_u8.end(), 0);
    small.mark_descriptors_modified();
    require_same_matches(matcher.match(small, tail), oracle(small, tail, true),
                         "vulkan matcher zero descriptors");
    std::cout << "vulkan matcher parity: ok (query=" << query.keypoints.size()
              << " train=" << train.keypoints.size() << ")\n";
}

void test_extractor() {
    if (!SiftVulkanExtractor::is_built()) {
        std::cout << "vulkan sift: backend not built\n";
        return;
    }
    SiftVulkanOptions options;
    options.maximum_features = 3000;
    SiftVulkanExtractor extractor(options);
    if (!extractor.is_available())
        throw std::runtime_error("vulkan_sift extractor unavailable");

    constexpr std::uint32_t width = 640, height = 480;
    std::vector<std::uint8_t> first;
    const auto second = make_image(width, height, 4321, 11, 8, first);
    {
        // Upright-descriptor control experiment: with orientation disabled,
        // corresponding descriptors must agree; otherwise the fault is in the
        // gradient/descriptor chain rather than orientation.
        SiftVulkanOptions upright_options;
        upright_options.maximum_features = 3000;
        upright_options.maximum_orientations = 0;
        SiftVulkanExtractor upright(upright_options);
        const auto u0 = upright.extract_gray(first, width, height);
        const auto u1 = upright.extract_gray(second, width, height);
        int pairs = 0;
        double similarity_sum = 0.0;
        for (std::size_t i = 0; i < u0.keypoints.size() && pairs < 200; ++i) {
            for (std::size_t j = 0; j < u1.keypoints.size(); ++j) {
                const auto& a = u0.keypoints[i];
                const auto& b = u1.keypoints[j];
                if (std::abs((b.x - a.x) - 11.0F) > 1.5F ||
                    std::abs((b.y - a.y) - 8.0F) > 1.5F)
                    continue;
                double dot = 0.0;
                for (int k = 0; k < 128; ++k)
                    dot += double(u0.descriptors[i * 128 + k]) *
                           double(u1.descriptors[j * 128 + k]);
                similarity_sum += dot;  // RootSIFT rows are L2 normalized.
                ++pairs;
                break;
            }
        }
        std::cout << "vulkan sift upright: pairs=" << pairs << " mean cosine="
                  << (pairs ? similarity_sum / pairs : 0.0) << '\n';
    }
    const auto begin = std::chrono::steady_clock::now();
    const auto features0 = extractor.extract_gray(first, width, height);
    const auto features1 = extractor.extract_gray(second, width, height);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    std::cout << "vulkan sift: features=" << features0.keypoints.size() << ","
              << features1.keypoints.size() << " (" << elapsed << " ms)\n";
    if (features0.keypoints.size() < 100 || features1.keypoints.size() < 100)
        throw std::runtime_error("vulkan sift produced too few features");
    if (features0.descriptor_dimension != 128 ||
        features0.descriptors.size() != features0.keypoints.size() * 128)
        throw std::runtime_error("vulkan sift descriptor layout mismatch");

    for (const auto& keypoint : features0.keypoints) {
        if (!(keypoint.x >= 0.F && keypoint.x < float(width)) ||
            !(keypoint.y >= 0.F && keypoint.y < float(height)) ||
            !(keypoint.scale > 0.F) ||
            !(keypoint.orientation >= 0.F && keypoint.orientation < 6.2831854F))
            throw std::runtime_error("vulkan sift keypoint out of range");
    }

    // Match the shifted pair with the Vulkan matcher; the displacement must
    // agree with the known shift.
    auto q = features0;
    auto t = features1;
    q.compress_descriptors_u8();
    t.compress_descriptors_u8();
    VulkanMutualRatioMatcher matcher;
    const auto self_matches = matcher.match(q, q);
    std::cout << "vulkan sift: self matches=" << self_matches.matches.size()
              << "/" << q.keypoints.size() << '\n';
    {
        // Descriptor-space diagnostic: for keypoint pairs at the known
        // displacement, compare descriptor cosine similarity.
        int pairs = 0;
        double similarity_sum = 0.0;
        double best = -1.0;
        double orientation_error = 0.0;
        double scale_ratio_sum = 0.0;
        for (std::size_t i = 0; i < q.keypoints.size() && pairs < 200; ++i) {
            for (std::size_t j = 0; j < t.keypoints.size(); ++j) {
                const auto& a = q.keypoints[i];
                const auto& b = t.keypoints[j];
                if (std::abs((b.x - a.x) - 11.0F) > 1.5F ||
                    std::abs((b.y - a.y) - 8.0F) > 1.5F ||
                    std::abs(b.scale - a.scale) > 0.15F * a.scale)
                    continue;
                double dot = 0.0;
                double na = 0.0;
                double nb = 0.0;
                for (int k = 0; k < 128; ++k) {
                    const double va = features0.descriptors[i * 128 + k];
                    const double vb = features1.descriptors[j * 128 + k];
                    dot += va * vb;
                    na += va * va;
                    nb += vb * vb;
                }
                const double similarity = dot / std::sqrt(std::max(na * nb, 1e-12));
                similarity_sum += similarity;
                best = std::max(best, similarity);
                double orientation_delta = std::abs(a.orientation - b.orientation);
                orientation_delta = std::min(orientation_delta,
                                             6.2831853 - orientation_delta);
                orientation_error += orientation_delta;
                scale_ratio_sum += b.scale / a.scale;
                ++pairs;
                break;
            }
        }
        std::cout << "vulkan sift: correspondence pairs=" << pairs
                  << " mean cosine=" << (pairs ? similarity_sum / pairs : 0.0)
                  << " best=" << best
                  << " mean orientation error=" << (pairs ? orientation_error / pairs : 0.0)
                  << " mean scale ratio=" << (pairs ? scale_ratio_sum / pairs : 0.0)
                  << '\n';
    }
    const auto matches = matcher.match(q, t);
    std::cout << "vulkan sift: matches=" << matches.matches.size() << '\n';
    if (matches.matches.size() < 60)
        throw std::runtime_error("vulkan sift shift matching failed");
    int inliers = 0;
    for (const auto& match : matches.matches) {
        const auto& a = q.keypoints[match.query];
        const auto& b = t.keypoints[match.train];
        if (std::abs((b.x - a.x) - 11.0F) < 2.5F &&
            std::abs((b.y - a.y) - 8.0F) < 2.5F)
            ++inliers;
    }
    std::cout << "vulkan sift: shift inliers=" << inliers << "/"
              << matches.matches.size() << '\n';
    if (inliers < 50)
        throw std::runtime_error("vulkan sift shift geometry mismatch");
}

void test_against_siftgpu() {
    if (!SiftGpuExtractor::is_built() || !SiftVulkanExtractor::is_built()) return;
    SiftGpuExtractor cuda_extractor;
    if (!cuda_extractor.is_available()) return;
    SiftVulkanOptions options;
    SiftVulkanExtractor vulkan_extractor(options);
    if (!vulkan_extractor.is_available()) return;

    constexpr std::uint32_t width = 640, height = 480;
    std::vector<std::uint8_t> first;
    const auto second = make_image(width, height, 4321, 11, 8, first);
    const auto cuda0 = cuda_extractor.extract_gray(first, width, height);
    const auto vk0 = vulkan_extractor.extract_gray(first, width, height);
    const auto cuda1 = cuda_extractor.extract_gray(second, width, height);
    const auto vk1 = vulkan_extractor.extract_gray(second, width, height);
    std::cout << "siftgpu vs vulkan: " << cuda0.keypoints.size() << " vs "
              << vk0.keypoints.size() << " features\n";

    // Feature-level parity: the Vulkan translation must reproduce the SiftGPU
    // keypoint set (the same DoG extrema, sub-pixel offsets and orientations).
    // SiftGPU itself is not run-to-run reproducible for the keypoints sitting
    // on the pyramid border, where its tex1Dfetch reads run past the texture,
    // so the check allows a small margin and uses a textured image whose
    // keypoints are stable across runs.
    {
        const auto counts = [](const FeatureSet& a, const FeatureSet& b) {
            return double(std::min(a.keypoints.size(), b.keypoints.size())) /
                   double((std::max)(std::size_t{1},
                                     (std::max)(a.keypoints.size(),
                                                b.keypoints.size())));
        };
        const auto textured = make_textured_image(width, height, 20260929);
        const auto cuda_textured = cuda_extractor.extract_gray(textured, width, height);
        const auto vulkan_textured =
            vulkan_extractor.extract_gray(textured, width, height);
        std::size_t reproduced = 0;
        for (const auto& key : vulkan_textured.keypoints) {
            for (const auto& other : cuda_textured.keypoints) {
                if (std::abs(other.scale - key.scale) > 0.02F * key.scale) continue;
                if (std::abs(other.x - key.x) <= 0.5F &&
                    std::abs(other.y - key.y) <= 0.5F) {
                    ++reproduced;
                    break;
                }
            }
        }
        const double count_ratio = counts(vulkan_textured, cuda_textured);
        const double position_ratio =
            double(reproduced) / double((std::max)(std::size_t{1},
                                                   vulkan_textured.keypoints.size()));
        std::cout << "keypoint parity (textured): siftgpu "
                  << cuda_textured.keypoints.size() << " vulkan "
                  << vulkan_textured.keypoints.size() << ", count ratio "
                  << count_ratio << ", position agreement " << position_ratio
                  << '\n';
        if (count_ratio < 0.95)
            throw std::runtime_error("vulkan_sift feature count differs from SiftGPU");
        // SiftGPU's keypoint set also depends on its call history (pyramid and
        // scratch reuse), which moves a few percent of the border keypoints
        // around, so this guards against real divergence (the ported pipeline
        // mismatches at <= 10%) rather than exact set equality.
        if (position_ratio < 0.8)
            throw std::runtime_error(
                "vulkan_sift keypoint positions differ from SiftGPU");
    }

    // Cross-backend matching: vulkan features of the reference against siftgpu
    // features of the shifted image. Both describe the same content, so the
    // matches must recover the shift.
    auto q = vk0;
    auto t = cuda1;
    q.compress_descriptors_u8();
    t.compress_descriptors_u8();
    VulkanMutualRatioMatcher matcher;
    const auto matches = matcher.match(q, t);
    int inliers = 0;
    for (const auto& match : matches.matches) {
        const auto& a = q.keypoints[match.query];
        const auto& b = t.keypoints[match.train];
        if (std::abs((b.x - a.x) - 11.0F) < 3.0F &&
            std::abs((b.y - a.y) - 8.0F) < 3.0F)
            ++inliers;
    }
    std::cout << "cross-backend matches=" << matches.matches.size()
              << " shift inliers=" << inliers << '\n';
    if (inliers < 40)
        throw std::runtime_error("vulkan/cuda cross-backend descriptor mismatch");
}

}  // namespace

int main() {
    try {
        test_matcher();
        test_extractor();
        test_against_siftgpu();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "vulkan features test failed: " << error.what() << '\n';
        return 1;
    }
}
