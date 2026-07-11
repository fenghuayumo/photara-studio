#include "aetherscan/sfm/two_view.hpp"

#include <charconv>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cout << "Usage: aetherscan_pair_verify image0 image1 focal_pixels\n";
            return argc == 1 ? 0 : 1;
        }
        double focal = 0.0;
        const std::string focal_text = argv[3];
        const auto parsed = std::from_chars(focal_text.data(), focal_text.data() + focal_text.size(), focal);
        if (parsed.ec != std::errc{} || focal <= 0.0) throw std::invalid_argument("Invalid focal length");
        const auto started = std::chrono::steady_clock::now();
        aetherscan::features::SiftExtractor extractor;
        const auto first = extractor.extract_file(std::filesystem::path(argv[1]));
        const auto second = extractor.extract_file(std::filesystem::path(argv[2]));
        const auto matches = aetherscan::features::match_descriptors(first, second);
        aetherscan::sfm::Camera camera0{0, first.image_width, first.image_height,
            focal, focal, 0.5 * first.image_width, 0.5 * first.image_height};
        aetherscan::sfm::Camera camera1{1, second.image_width, second.image_height,
            focal, focal, 0.5 * second.image_width, 0.5 * second.image_height};
        const auto geometry = aetherscan::sfm::verify_two_view(
            camera0, camera1, first, second, matches);
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "features=" << first.keypoints.size() << "+" << second.keypoints.size()
                  << " raw_matches=" << matches.matches.size()
                  << " essential_inliers=" << geometry.inliers.matches.size()
                  << " homography_inliers=" << geometry.homography_inliers
                  << " valid=" << geometry.valid
                  << " time_ms=" << milliseconds << '\n';
        return geometry.valid ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
