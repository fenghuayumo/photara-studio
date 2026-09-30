#include "features/features.hpp"
#include "io/image.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint64_t hash_bytes(std::uint64_t hash, const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t feature_hash(const photara::features::FeatureSet& features) {
    std::uint64_t hash = 14695981039346656037ULL;
    hash = hash_bytes(hash, features.keypoints.data(),
                      features.keypoints.size() * sizeof(features.keypoints[0]));
    hash = hash_bytes(hash, features.descriptors.data(),
                      features.descriptors.size() * sizeof(features.descriptors[0]));
    return hash;
}

std::size_t used_cuda_bytes() {
    std::size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess)
        throw std::runtime_error("cudaMemGetInfo failed");
    return total - free;
}

bool supported_image(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
           extension == ".tif" || extension == ".tiff" || extension == ".bmp";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4)
            throw std::invalid_argument(
                "Usage: photara_cuda_sift_benchmark IMAGE_DIRECTORY "
                "[ROUNDS=2] [MAXIMUM_FEATURES=27000]");
        const std::filesystem::path directory(argv[1]);
        const unsigned rounds = argc >= 3 ? std::stoul(argv[2]) : 2U;
        const std::size_t maximum_features =
            argc >= 4 ? std::stoull(argv[3]) : 27000U;
        if (rounds == 0 || maximum_features == 0)
            throw std::invalid_argument("ROUNDS and MAXIMUM_FEATURES must be positive");

        std::vector<std::filesystem::path> paths;
        for (const auto& entry : std::filesystem::directory_iterator(directory))
            if (entry.is_regular_file() && supported_image(entry.path()))
                paths.push_back(entry.path());
        std::sort(paths.begin(), paths.end());
        if (paths.empty()) throw std::runtime_error("Dataset has no supported images");

        photara::features::SiftGpuOptions options;
        options.maximum_features = maximum_features;
        photara::features::SiftGpuExtractor extractor(options);
        if (!extractor.is_available())
            throw std::runtime_error("Native CUDA SIFT is unavailable");
        const std::size_t baseline_bytes = used_cuda_bytes();
        std::size_t peak_bytes = baseline_bytes;
        std::vector<std::uint64_t> reference_hashes(paths.size());
        std::vector<std::size_t> reference_counts(paths.size());
        std::size_t total_features = 0;
        std::size_t determinism_failures = 0;
        double decode_seconds = 0.0;
        double extract_seconds = 0.0;
        using Clock = std::chrono::steady_clock;

        for (unsigned round = 0; round < rounds; ++round) {
            const auto round_start = Clock::now();
            std::size_t round_features = 0;
            for (std::size_t i = 0; i < paths.size(); ++i) {
                auto start = Clock::now();
                const auto image = photara::io::load_gray(paths[i]);
                decode_seconds += std::chrono::duration<double>(Clock::now() - start).count();
                start = Clock::now();
                const auto features = extractor.extract_gray(
                    image.pixels, image.width, image.height);
                extract_seconds += std::chrono::duration<double>(Clock::now() - start).count();
                peak_bytes = std::max(peak_bytes, used_cuda_bytes());
                round_features += features.keypoints.size();
                const std::uint64_t hash = feature_hash(features);
                if (round == 0) {
                    reference_hashes[i] = hash;
                    reference_counts[i] = features.keypoints.size();
                } else if (reference_hashes[i] != hash ||
                           reference_counts[i] != features.keypoints.size()) {
                    ++determinism_failures;
                }
            }
            total_features = round_features;
            const double elapsed =
                std::chrono::duration<double>(Clock::now() - round_start).count();
            std::cout << "round=" << round << " elapsed_s=" << elapsed
                      << " features=" << round_features << '\n';
        }

        constexpr double mib = 1024.0 * 1024.0;
        const double samples = static_cast<double>(paths.size()) * rounds;
        std::cout << std::setprecision(10)
                  << "images=" << paths.size() << '\n'
                  << "rounds=" << rounds << '\n'
                  << "maximum_features=" << maximum_features << '\n'
                  << "features_per_round=" << total_features << '\n'
                  << "decode_ms_per_image=" << decode_seconds * 1000.0 / samples << '\n'
                  << "extract_ms_per_image=" << extract_seconds * 1000.0 / samples << '\n'
                  << "extract_images_per_second=" << samples / extract_seconds << '\n'
                  << "cuda_baseline_mib=" << baseline_bytes / mib << '\n'
                  << "cuda_peak_mib=" << peak_bytes / mib << '\n'
                  << "cuda_extractor_delta_mib="
                  << static_cast<double>(peak_bytes - baseline_bytes) / mib << '\n'
                  << "determinism_failures=" << determinism_failures << '\n';
        return determinism_failures == 0 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "CUDA SIFT benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
