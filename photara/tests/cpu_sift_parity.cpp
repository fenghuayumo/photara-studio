// Manual CPU vs CUDA SIFT parity harness.
//
//   photara_cpu_sift_parity IMAGE_DIRECTORY [MAXIMUM_FEATURES=8192]
#include "features/features.hpp"
#include "io/image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using namespace photara::features;

double milliseconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

struct Agreement {
    std::size_t count_a{};
    std::size_t count_b{};
    std::size_t close{};
    std::size_t compared{};
    double position_sum{};
    double cosine_sum{};
    double worst_position{};
};

Agreement compare_sets(const FeatureSet& cpu, const FeatureSet& gpu) {
    Agreement out;
    out.count_a = cpu.keypoints.size();
    out.count_b = gpu.keypoints.size();
    out.compared = (std::min)(cpu.keypoints.size(), gpu.keypoints.size());
    for (std::size_t i = 0; i < out.compared; ++i) {
        const auto& a = cpu.keypoints[i];
        const auto& b = gpu.keypoints[i];
        const double dx = std::abs(double(a.x) - double(b.x));
        const double dy = std::abs(double(a.y) - double(b.y));
        const double dist = (std::max)(dx, dy);
        out.worst_position = (std::max)(out.worst_position, dist);
        out.position_sum += std::sqrt(dx * dx + dy * dy);
        const bool scale_ok =
            std::abs(a.scale - b.scale) <= 0.02F * std::abs(a.scale);
        if (dx <= 0.05 && dy <= 0.05 && scale_ok) ++out.close;
        double dot = 0.0, na = 0.0, nb = 0.0;
        const float* da = cpu.descriptors.data() + i * 128;
        const float* db = gpu.descriptors.data() + i * 128;
        for (int column = 0; column < 128; ++column) {
            dot += double(da[column]) * double(db[column]);
            na += double(da[column]) * double(da[column]);
            nb += double(db[column]) * double(db[column]);
        }
        out.cosine_sum += dot / std::sqrt((std::max)(na * nb, 1e-24));
    }
    return out;
}

std::size_t nearest_reproduced(const FeatureSet& query, const FeatureSet& train) {
    std::size_t matched = 0;
    for (const auto& key : query.keypoints) {
        for (const auto& other : train.keypoints) {
            if (std::abs(other.scale - key.scale) > 0.02F * key.scale) continue;
            if (std::abs(other.x - key.x) <= 0.5F &&
                std::abs(other.y - key.y) <= 0.5F) {
                ++matched;
                break;
            }
        }
    }
    return matched;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "Usage: photara_cpu_sift_parity IMAGE_DIRECTORY "
                             "[MAXIMUM_FEATURES=8192]\n");
        return 2;
    }
    const std::filesystem::path directory(argv[1]);
    const std::size_t maximum_features =
        argc >= 3 ? static_cast<std::size_t>(std::stoull(argv[2])) : 8192U;

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

    SiftGpuOptions gpu_options;
    gpu_options.maximum_features = maximum_features;
    SiftOptions cpu_options;
    cpu_options.maximum_features = gpu_options.maximum_features;
    cpu_options.contrast_threshold = gpu_options.peak_threshold;
    cpu_options.edge_threshold = gpu_options.edge_threshold;
    cpu_options.first_octave = gpu_options.first_octave;
    cpu_options.octave_layers = gpu_options.octave_layers;
    cpu_options.maximum_orientations = gpu_options.maximum_orientations;
    cpu_options.maximum_image_dimension = gpu_options.maximum_image_dimension;
    cpu_options.root_sift = gpu_options.root_sift;

    SiftExtractor cpu(cpu_options);
    SiftGpuExtractor gpu(gpu_options);
    if (!gpu.is_available()) {
        std::fprintf(stderr, "CUDA SIFT extractor is not available\n");
        return 1;
    }

    std::vector<FeatureSet> cpu_features;
    std::vector<FeatureSet> gpu_features;
    cpu_features.reserve(paths.size());
    gpu_features.reserve(paths.size());

    double decode_ms = 0.0, cpu_ms = 0.0, gpu_ms = 0.0;
    std::size_t cpu_count = 0, gpu_count = 0, count_mismatches = 0;
    std::size_t index_close = 0, index_compared = 0;
    std::size_t nn_cpu_in_gpu = 0, nn_gpu_in_cpu = 0, nn_cpu_total = 0, nn_gpu_total = 0;
    double cosine_sum = 0.0, position_sum = 0.0, worst_position = 0.0;
    std::uint32_t width = 0, height = 0;

    for (std::size_t i = 0; i < paths.size(); ++i) {
        auto start = std::chrono::steady_clock::now();
        const auto image = photara::io::load_gray(paths[i]);
        decode_ms += milliseconds_since(start);
        width = image.width;
        height = image.height;

        FeatureSet cpu_set, gpu_set;
        if ((i & 1U) == 0U) {
            start = std::chrono::steady_clock::now();
            cpu_set = cpu.extract_gray(image.pixels, image.width, image.height);
            cpu_ms += milliseconds_since(start);
            start = std::chrono::steady_clock::now();
            gpu_set = gpu.extract_gray(image.pixels, image.width, image.height);
            gpu_ms += milliseconds_since(start);
        } else {
            start = std::chrono::steady_clock::now();
            gpu_set = gpu.extract_gray(image.pixels, image.width, image.height);
            gpu_ms += milliseconds_since(start);
            start = std::chrono::steady_clock::now();
            cpu_set = cpu.extract_gray(image.pixels, image.width, image.height);
            cpu_ms += milliseconds_since(start);
        }

        const Agreement agree = compare_sets(cpu_set, gpu_set);
        const std::size_t cpu_nn = nearest_reproduced(cpu_set, gpu_set);
        const std::size_t gpu_nn = nearest_reproduced(gpu_set, cpu_set);
        cpu_count += agree.count_a;
        gpu_count += agree.count_b;
        count_mismatches += agree.count_a != agree.count_b;
        index_close += agree.close;
        index_compared += agree.compared;
        cosine_sum += agree.cosine_sum;
        position_sum += agree.position_sum;
        worst_position = (std::max)(worst_position, agree.worst_position);
        nn_cpu_in_gpu += cpu_nn;
        nn_gpu_in_cpu += gpu_nn;
        nn_cpu_total += agree.count_a;
        nn_gpu_total += agree.count_b;

        std::printf(
            "[%zu/%zu] %s  %ux%u  cpu=%zu gpu=%zu  pos=%.4f cosine=%.6f  "
            "nn=%zu/%zu %zu/%zu\n",
            i + 1, paths.size(), paths[i].filename().string().c_str(),
            image.width, image.height, agree.count_a, agree.count_b,
            agree.compared ? double(agree.close) / double(agree.compared) : 0.0,
            agree.compared ? agree.cosine_sum / double(agree.compared) : 0.0,
            cpu_nn, agree.count_a, gpu_nn, agree.count_b);
        if (agree.count_a != agree.count_b) {
            auto extras = [](const FeatureSet& query, const FeatureSet& train,
                             const char* label) {
                int printed = 0;
                for (std::size_t qi = 0; qi < query.keypoints.size(); ++qi) {
                    const auto& key = query.keypoints[qi];
                    bool found = false;
                    for (const auto& other : train.keypoints) {
                        if (std::abs(other.scale - key.scale) > 0.02F * key.scale)
                            continue;
                        if (std::abs(other.x - key.x) <= 0.05F &&
                            std::abs(other.y - key.y) <= 0.05F) {
                            found = true;
                            break;
                        }
                    }
                    if (!found && printed < 8) {
                        ++printed;
                        std::printf(
                            "  %s only #%zu: (%.4f, %.4f) scale %.4f ori %.4f\n",
                            label, qi, double(key.x), double(key.y),
                            double(key.scale), double(key.orientation));
                    }
                }
            };
            extras(cpu_set, gpu_set, "cpu");
            extras(gpu_set, cpu_set, "cuda");
        }
        std::fflush(stdout);

        cpu_set.compress_descriptors_u8();
        gpu_set.compress_descriptors_u8();
        cpu_features.push_back(std::move(cpu_set));
        gpu_features.push_back(std::move(gpu_set));
    }

    std::vector<FeatureMatcher::Pair> cpu_pairs, gpu_pairs;
    cpu_pairs.reserve(paths.size() - 1);
    gpu_pairs.reserve(paths.size() - 1);
    for (std::size_t i = 1; i < paths.size(); ++i) {
        cpu_pairs.push_back({&cpu_features[i - 1], &cpu_features[i]});
        gpu_pairs.push_back({&gpu_features[i - 1], &gpu_features[i]});
    }

    SiftGpuMatcher matcher;
    auto start = std::chrono::steady_clock::now();
    const auto cpu_matches = matcher.match_batch(cpu_pairs);
    const double cpu_match_ms = milliseconds_since(start);
    matcher.clear_prepared();
    start = std::chrono::steady_clock::now();
    const auto gpu_matches = matcher.match_batch(gpu_pairs);
    const double gpu_match_ms = milliseconds_since(start);

    std::size_t cpu_match_count = 0, gpu_match_count = 0, identical_pairs = 0;
    for (std::size_t i = 0; i < cpu_pairs.size(); ++i) {
        cpu_match_count += cpu_matches[i].matches.size();
        gpu_match_count += gpu_matches[i].matches.size();
        if (cpu_matches[i].matches.size() != gpu_matches[i].matches.size()) continue;
        bool identical = true;
        for (std::size_t j = 0; j < cpu_matches[i].matches.size(); ++j) {
            identical &= cpu_matches[i].matches[j].query ==
                             gpu_matches[i].matches[j].query &&
                         cpu_matches[i].matches[j].train ==
                             gpu_matches[i].matches[j].train;
        }
        identical_pairs += identical;
    }

    const double n = static_cast<double>(paths.size());
    std::printf("dataset: %s (%zu images, %u x %u, max_features=%zu)\n",
                directory.string().c_str(), paths.size(), width, height,
                maximum_features);
    std::printf("decode: %.1f ms total (%.2f ms/image)\n", decode_ms, decode_ms / n);
    std::printf(
        "extraction: cpu %.1f ms (%.2f/image, %zu features), cuda %.1f ms "
        "(%.2f/image, %zu features), cpu/cuda %.2fx, count mismatches %zu/%zu\n",
        cpu_ms, cpu_ms / n, cpu_count, gpu_ms, gpu_ms / n, gpu_count,
        gpu_ms > 0.0 ? cpu_ms / gpu_ms : 0.0, count_mismatches, paths.size());
    std::printf(
        "index agreement: position %.4f, mean cosine %.6f, mean offset %.4f px, "
        "worst %.4f px (%zu compared)\n",
        index_compared ? double(index_close) / double(index_compared) : 0.0,
        index_compared ? cosine_sum / double(index_compared) : 0.0,
        index_compared ? position_sum / double(index_compared) : 0.0,
        worst_position, index_compared);
    std::printf("nearest 0.5px: cpu in cuda %zu/%zu (%.1f%%), cuda in cpu %zu/%zu "
                "(%.1f%%)\n",
                nn_cpu_in_gpu, nn_cpu_total,
                nn_cpu_total ? 100.0 * double(nn_cpu_in_gpu) / double(nn_cpu_total)
                             : 0.0,
                nn_gpu_in_cpu, nn_gpu_total,
                nn_gpu_total ? 100.0 * double(nn_gpu_in_cpu) / double(nn_gpu_total)
                             : 0.0);
    std::printf(
        "adjacent matching: %zu pairs, cpu %.1f ms, cuda %.1f ms, matches "
        "%zu/%zu, identical tables %zu/%zu\n",
        cpu_pairs.size(), cpu_match_ms, gpu_match_ms, cpu_match_count,
        gpu_match_count, identical_pairs, cpu_pairs.size());

    const bool counts_ok = count_mismatches == 0;
    const bool position_ok =
        index_compared > 0 && double(index_close) / double(index_compared) >= 0.95;
    const bool cosine_ok =
        index_compared > 0 && cosine_sum / double(index_compared) >= 0.99;
    return counts_ok && position_ok && cosine_ok ? 0 : 3;
}
