#include "features/features.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
    try {
        aetherscan::features::ensure_builtin_feature_backends();
        if (argc >= 2 && std::string_view(argv[1]) == "--list") {
            std::cout << "extractors:";
            for (const auto& name : aetherscan::features::list_extractors())
                std::cout << ' ' << name;
            std::cout << "\nmatchers:";
            for (const auto& name : aetherscan::features::list_matchers())
                std::cout << ' ' << name;
            std::cout << '\n';
            return 0;
        }
        if (argc < 3) {
            std::cout << "Usage: aetherscan_features_match image0 image1 "
                         "[--extractor sift|siftgpu|superpoint] [--matcher mutual_ratio]\n"
                         "       aetherscan_features_match image0 image1 "
                         "[--lightglue model.onnx] [--superpoint] [--cpu]\n"
                         "       aetherscan_features_match --list\n";
            return argc == 1 ? 0 : 1;
        }
        const std::filesystem::path first = argv[1], second = argv[2];
        std::filesystem::path model;
        std::string extractor_name = "sift";
        std::string matcher_name = "mutual_ratio";
        bool superpoint = false, cpu = false;
        for (int i = 3; i < argc; ++i) {
            const std::string_view argument = argv[i];
            if (argument == "--lightglue" && i + 1 < argc) model = argv[++i];
            else if (argument == "--extractor" && i + 1 < argc) extractor_name = argv[++i];
            else if (argument == "--matcher" && i + 1 < argc) matcher_name = argv[++i];
            else if (argument == "--superpoint") {
                superpoint = true;
                extractor_name = "superpoint";
            } else if (argument == "--cpu") cpu = true;
            else if (argument == "--siftgpu") extractor_name = "siftgpu";
            else throw std::invalid_argument("Unknown or incomplete option: " + std::string(argument));
        }
        const auto started = std::chrono::steady_clock::now();
        if (!model.empty()) {
            aetherscan::features::LightGlueOptions options;
            options.model_path = model;
            options.extractor = superpoint ? aetherscan::features::LightGlueExtractor::superpoint
                                           : aetherscan::features::LightGlueExtractor::disk;
            options.device = cpu ? aetherscan::features::InferenceDevice::cpu
                                 : aetherscan::features::InferenceDevice::cuda;
            aetherscan::features::LightGluePipeline pipeline(options);
            const auto result = pipeline.match_files(first, second);
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            std::cout << "lightglue: " << result.first.keypoints.size() << " + "
                      << result.second.keypoints.size() << " features, "
                      << result.matches.matches.size() << " matches, " << milliseconds << " ms\n";
        } else {
            auto extractor = aetherscan::features::create_extractor(extractor_name);
            auto matcher = aetherscan::features::create_matcher(matcher_name);
            const auto first_features = extractor->extract_file(first);
            const auto second_features = extractor->extract_file(second);
            const auto matches = matcher->match(first_features, second_features);
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            std::cout << extractor->name() << "+" << matcher->name() << ": "
                      << first_features.keypoints.size() << " + "
                      << second_features.keypoints.size() << " features, "
                      << matches.matches.size() << " matches, " << milliseconds << " ms\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
