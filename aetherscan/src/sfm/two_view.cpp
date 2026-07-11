#include "aetherscan/sfm/two_view.hpp"

#include "aetherscan/geometry/pose.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace aetherscan::sfm {
namespace {

std::array<double, 9> to_array(const geometry::Mat3& matrix) {
    std::array<double, 9> result{};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            result[static_cast<std::size_t>(row * 3 + column)] = matrix(row, column);
    return result;
}

geometry::Vec2 normalize(const features::Keypoint& point, const Camera& camera) {
    return {(point.x - camera.cx) / camera.fx, (point.y - camera.cy) / camera.fy};
}

}  // namespace

TwoViewGeometry verify_two_view(
    const Camera& first_camera, const Camera& second_camera,
    const features::FeatureSet& first_features, const features::FeatureSet& second_features,
    const features::MatchSet& matches, const TwoViewOptions& options) {
    if (first_camera.fx <= 0.0 || first_camera.fy <= 0.0 || second_camera.fx <= 0.0 ||
        second_camera.fy <= 0.0 || options.ransac_threshold_pixels <= 0.0 ||
        options.minimum_inliers < 5)
        throw std::invalid_argument("Invalid two-view geometry options or calibration");

    std::vector<geometry::Vec2> normalized_first, normalized_second;
    std::vector<geometry::Vec2> pixels_first, pixels_second;
    normalized_first.reserve(matches.matches.size());
    normalized_second.reserve(matches.matches.size());
    pixels_first.reserve(matches.matches.size());
    pixels_second.reserve(matches.matches.size());
    for (const auto& match : matches.matches) {
        if (match.query >= first_features.keypoints.size() ||
            match.train >= second_features.keypoints.size())
            throw std::out_of_range("Two-view match references an invalid feature");
        const auto& first = first_features.keypoints[match.query];
        const auto& second = second_features.keypoints[match.train];
        normalized_first.push_back(normalize(first, first_camera));
        normalized_second.push_back(normalize(second, second_camera));
        pixels_first.emplace_back(first.x, first.y);
        pixels_second.emplace_back(second.x, second.y);
    }

    TwoViewGeometry result;
    if (matches.matches.size() < 5) return result;

    const double mean_focal =
        0.25 * (first_camera.fx + first_camera.fy + second_camera.fx + second_camera.fy);
    const double essential_threshold = options.ransac_threshold_pixels / mean_focal;
    auto essential = geometry::estimate_essential_ransac(
        normalized_first, normalized_second, essential_threshold, options.confidence,
        options.maximum_iterations);
    if (!essential.valid) return result;

    auto pose = geometry::recover_pose(
        essential.model, normalized_first, normalized_second, essential.inliers);
    if (!pose.valid) return result;

    const auto fundamental = geometry::estimate_fundamental_ransac(
        pixels_first, pixels_second, options.ransac_threshold_pixels, options.confidence,
        options.maximum_iterations);
    const auto homography = geometry::estimate_homography_ransac(
        pixels_first, pixels_second, options.ransac_threshold_pixels, options.confidence,
        options.maximum_iterations);

    result.essential = to_array(essential.model);
    result.fundamental = to_array(fundamental.model);
    result.homography = to_array(homography.model);
    result.rotation = to_array(pose.R);
    result.translation_direction = {pose.t.x(), pose.t.y(), pose.t.z()};
    result.homography_inliers = homography.inlier_count;
    for (std::size_t i = 0; i < matches.matches.size(); ++i)
        if (i < essential.inliers.size() && essential.inliers[i] != 0)
            result.inliers.matches.push_back(matches.matches[i]);
    result.valid = result.inliers.matches.size() >= options.minimum_inliers;
    return result;
}

}  // namespace aetherscan::sfm
