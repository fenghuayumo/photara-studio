#include "features/registry.hpp"
#include "sfm/geometry.hpp"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cout << "Usage: aetherscan_pair_verify image0 image1 focal_pixels\n";
            return argc == 1 ? 0 : 1;
        }
        double focal = 0.0;
        const std::string focal_text = argv[3];
        const auto parsed =
            std::from_chars(focal_text.data(), focal_text.data() + focal_text.size(), focal);
        if (parsed.ec != std::errc{} || focal <= 0.0)
            throw std::invalid_argument("Invalid focal length");

        aetherscan::features::ensure_builtin_feature_backends();
        auto extractor = aetherscan::features::create_extractor("sift");
        auto matcher = aetherscan::features::create_matcher("sift");
        if (!extractor || !matcher) throw std::runtime_error("Missing sift backends");

        const auto started = std::chrono::steady_clock::now();
        const auto first = extractor->extract_file(std::filesystem::path(argv[1]));
        const auto second = extractor->extract_file(std::filesystem::path(argv[2]));
        const auto matches = matcher->match(first, second);

        aetherscan::sfm::PinholeCamera camera0;
        camera0.width = first.image_width;
        camera0.height = first.image_height;
        camera0.fx = camera0.fy = focal;
        camera0.cx = 0.5 * first.image_width;
        camera0.cy = 0.5 * first.image_height;
        aetherscan::sfm::PinholeCamera camera1 = camera0;
        camera1.width = second.image_width;
        camera1.height = second.image_height;
        camera1.cx = 0.5 * second.image_width;
        camera1.cy = 0.5 * second.image_height;

        std::vector<aetherscan::sfm::Vec2> p1, p2;
        p1.reserve(matches.matches.size());
        p2.reserve(matches.matches.size());
        for (const auto& m : matches.matches) {
            p1.emplace_back(first.keypoints[m.query].x, first.keypoints[m.query].y);
            p2.emplace_back(second.keypoints[m.train].x, second.keypoints[m.train].y);
        }

        aetherscan::sfm::RelativePoseOptions options;
        options.min_inliers = 30;
        const auto geometry =
            aetherscan::sfm::estimate_relative_pose(p1, p2, camera0, camera1, options);
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "features=" << first.keypoints.size() << "+" << second.keypoints.size()
                  << " raw_matches=" << matches.matches.size()
                  << " essential_inliers=" << geometry.num_inliers
                  << " valid=" << geometry.success
                  << " time_ms=" << milliseconds << '\n';
        return geometry.success ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
