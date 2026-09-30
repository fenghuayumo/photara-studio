#include "sfm/checkpoint.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint64_t checkpoint_key(const std::filesystem::path& path) {
    const std::string stem = path.stem().string();
    const auto separator = stem.find('-');
    if (separator == std::string::npos)
        throw std::invalid_argument("Checkpoint filename has no hexadecimal key");
    return std::stoull(stem.substr(separator + 1), nullptr, 16);
}

photara::sfm::Scene load(const std::filesystem::path& path) {
    photara::sfm::CheckpointOptions options;
    options.directory = path.parent_path();
    options.write = false;
    photara::sfm::Scene scene;
    if (!photara::sfm::CheckpointStore(options).load_scene(
            photara::sfm::CheckpointStage::features,
            checkpoint_key(path), scene))
        throw std::runtime_error("Cannot load feature checkpoint: " + path.string());
    return scene;
}

double percentile(std::vector<double> values, const double fraction) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t index = static_cast<std::size_t>(
        std::clamp(fraction, 0.0, 1.0) * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

double mean(const std::vector<double>& values) {
    return values.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : std::accumulate(values.begin(), values.end(), 0.0) /
              static_cast<double>(values.size());
}

struct Entry {
    // Coarse location/scale identity tolerates backend floating-point noise;
    // scale/orientation remain sorting keys so duplicate SIFT locations pair.
    std::array<std::int32_t, 2> identity{};
    float scale{};
    float orientation{};
    std::size_t index{};
};

std::vector<Entry> entries(const photara::features::FeatureSet& features) {
    std::vector<Entry> result;
    result.reserve(features.keypoints.size());
    for (std::size_t i = 0; i < features.keypoints.size(); ++i) {
        const auto& key = features.keypoints[i];
        result.push_back({
            {static_cast<std::int32_t>(std::lround(key.x * 20.0F)),
             static_cast<std::int32_t>(std::lround(key.y * 20.0F))},
            key.scale, key.orientation, i});
    }
    std::sort(result.begin(), result.end(), [](const Entry& left, const Entry& right) {
        if (left.identity != right.identity) return left.identity < right.identity;
        if (left.scale != right.scale) return left.scale < right.scale;
        return left.orientation < right.orientation;
    });
    return result;
}

double descriptor_cosine(
    const photara::features::FeatureSet& first, const std::size_t first_index,
    const photara::features::FeatureSet& second, const std::size_t second_index) {
    if (first.descriptor_dimension == 0 ||
        first.descriptor_dimension != second.descriptor_dimension ||
        first.descriptors_u8.size() <
            (first_index + 1) * first.descriptor_dimension ||
        second.descriptors_u8.size() <
            (second_index + 1) * second.descriptor_dimension)
        return std::numeric_limits<double>::quiet_NaN();
    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    const auto* a = first.descriptors_u8.data() +
                    first_index * first.descriptor_dimension;
    const auto* b = second.descriptors_u8.data() +
                    second_index * second.descriptor_dimension;
    for (std::size_t i = 0; i < first.descriptor_dimension; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        norm1 += static_cast<double>(a[i]) * a[i];
        norm2 += static_cast<double>(b[i]) * b[i];
    }
    return norm1 > 0.0 && norm2 > 0.0
        ? dot / std::sqrt(norm1 * norm2)
        : 0.0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument(
                "Usage: photara_sfm_feature_compare REFERENCE_FEATURES.bin "
                "CANDIDATE_FEATURES.bin");
        const auto reference = load(argv[1]);
        const auto candidate = load(argv[2]);
        if (reference.images.size() != candidate.images.size())
            throw std::runtime_error("Feature checkpoints have different image counts");

        std::size_t reference_total = 0;
        std::size_t candidate_total = 0;
        std::size_t matched = 0;
        std::size_t count_difference_images = 0;
        std::vector<double> position_errors;
        std::vector<double> scale_errors;
        std::vector<double> orientation_errors;
        std::vector<double> descriptor_cosines;
        for (std::size_t image = 0; image < reference.images.size(); ++image) {
            const auto& first = reference.images[image].features;
            const auto& second = candidate.images[image].features;
            reference_total += first.keypoints.size();
            candidate_total += second.keypoints.size();
            if (first.keypoints.size() != second.keypoints.size())
                ++count_difference_images;

            const auto a = entries(first);
            const auto b = entries(second);
            std::size_t ai = 0, bi = 0;
            while (ai < a.size() && bi < b.size()) {
                if (a[ai].identity < b[bi].identity) { ++ai; continue; }
                if (b[bi].identity < a[ai].identity) { ++bi; continue; }
                const auto identity = a[ai].identity;
                const std::size_t a_begin = ai;
                const std::size_t b_begin = bi;
                while (ai < a.size() && a[ai].identity == identity) ++ai;
                while (bi < b.size() && b[bi].identity == identity) ++bi;
                const std::size_t group = std::min(ai - a_begin, bi - b_begin);
                for (std::size_t offset = 0; offset < group; ++offset) {
                    const std::size_t first_index = a[a_begin + offset].index;
                    const std::size_t second_index = b[b_begin + offset].index;
                    const auto& ka = first.keypoints[first_index];
                    const auto& kb = second.keypoints[second_index];
                    position_errors.push_back(std::hypot(
                        static_cast<double>(ka.x - kb.x),
                        static_cast<double>(ka.y - kb.y)));
                    scale_errors.push_back(std::abs(
                        static_cast<double>(ka.scale - kb.scale)));
                    orientation_errors.push_back(std::abs(std::remainder(
                        static_cast<double>(ka.orientation - kb.orientation),
                        6.2831853071795864769)));
                    const double cosine = descriptor_cosine(
                        first, first_index, second, second_index);
                    if (std::isfinite(cosine)) descriptor_cosines.push_back(cosine);
                    ++matched;
                }
            }
        }

        std::cout << std::setprecision(12)
                  << "images=" << reference.images.size() << '\n'
                  << "reference_features=" << reference_total << '\n'
                  << "candidate_features=" << candidate_total << '\n'
                  << "feature_count_relative_delta="
                  << (static_cast<double>(candidate_total) /
                          std::max<std::size_t>(1, reference_total) - 1.0) << '\n'
                  << "count_difference_images=" << count_difference_images << '\n'
                  << "matched_features=" << matched << '\n'
                  << "reference_coverage="
                  << static_cast<double>(matched) /
                         std::max<std::size_t>(1, reference_total) << '\n'
                  << "candidate_coverage="
                  << static_cast<double>(matched) /
                         std::max<std::size_t>(1, candidate_total) << '\n'
                  << "position_error_px_mean=" << mean(position_errors) << '\n'
                  << "position_error_px_p95=" << percentile(position_errors, 0.95) << '\n'
                  << "position_error_px_max=" << percentile(position_errors, 1.0) << '\n'
                  << "scale_error_mean=" << mean(scale_errors) << '\n'
                  << "scale_error_p95=" << percentile(scale_errors, 0.95) << '\n'
                  << "orientation_error_rad_mean=" << mean(orientation_errors) << '\n'
                  << "orientation_error_rad_p95=" << percentile(orientation_errors, 0.95) << '\n'
                  << "descriptor_cosine_mean=" << mean(descriptor_cosines) << '\n'
                  << "descriptor_cosine_p01=" << percentile(descriptor_cosines, 0.01) << '\n'
                  << "descriptor_cosine_p05=" << percentile(descriptor_cosines, 0.05) << '\n'
                  << "descriptor_cosine_min=" << percentile(descriptor_cosines, 0.0) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "feature comparison failed: " << error.what() << '\n';
        return 1;
    }
}
