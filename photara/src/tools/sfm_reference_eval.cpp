#include "sfm/asfm.hpp"
#include "sfm/hierarchical.hpp"
#include "splat/colmap.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using photara::sfm::Mat3;
using photara::sfm::Similarity3;
using photara::sfm::Vec3;

std::string image_key(const std::filesystem::path& path) {
    std::string key = path.filename().generic_string();
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return key;
}

double percentile(std::vector<double> values, const double fraction) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = std::clamp(fraction, 0.0, 1.0) *
                            static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double blend = position - static_cast<double>(lower);
    return values[lower] * (1.0 - blend) + values[upper] * blend;
}

double mean(const std::vector<double>& values) {
    return values.empty()
        ? std::numeric_limits<double>::quiet_NaN()
        : std::accumulate(values.begin(), values.end(), 0.0) /
              static_cast<double>(values.size());
}

double rotation_error_deg(const Mat3& measured, const Mat3& expected) {
    const Mat3 delta = measured * expected.transpose();
    constexpr double radians_to_degrees = 57.2957795130823208768;
    return std::acos(std::clamp(0.5 * (delta.trace() - 1.0), -1.0, 1.0)) *
           radians_to_degrees;
}

struct Correspondence {
    std::string name;
    Vec3 source_center;
    Vec3 reference_center;
    Mat3 source_rotation;
    Mat3 reference_rotation;
    double source_focal{};
    double reference_focal{};
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::runtime_error(
                "Usage: photara_sfm_reference_eval REFERENCE_SPARSE "
                "IMAGE_DIRECTORY RECONSTRUCTION.asfm");

        const auto reference = photara::splat::load_colmap_scene(argv[1], argv[2]);
        const auto reconstruction = photara::sfm::load_asfm(argv[3]);

        std::unordered_map<std::string, std::size_t> reference_by_name;
        for (std::size_t i = 0; i < reference.scene.views.size(); ++i)
            reference_by_name.emplace(image_key(reference.scene.views[i].path), i);

        std::unordered_set<std::string> reconstruction_names;
        std::vector<std::string> reference_missing;
        std::vector<std::string> reconstruction_missing;
        std::vector<std::string> reconstruction_missing_from_reference;
        for (const auto& image : reconstruction.images) {
            const std::string key = image_key(image.path);
            reconstruction_names.insert(key);
            if (!image.registered) {
                reconstruction_missing.push_back(image.path.filename().string());
                if (reference_by_name.contains(key))
                    reconstruction_missing_from_reference.push_back(
                        image.path.filename().string());
            }
        }
        for (const auto& entry : std::filesystem::directory_iterator(argv[2])) {
            if (!entry.is_regular_file()) continue;
            const std::string key = image_key(entry.path());
            if (reconstruction_names.contains(key) &&
                !reference_by_name.contains(key))
                reference_missing.push_back(entry.path().filename().string());
        }
        std::sort(reference_missing.begin(), reference_missing.end());
        std::sort(reconstruction_missing.begin(), reconstruction_missing.end());
        std::sort(
            reconstruction_missing_from_reference.begin(),
            reconstruction_missing_from_reference.end());

        std::vector<Correspondence> common;
        for (const auto& image : reconstruction.images) {
            if (!image.registered || image.camera_id >= reconstruction.cameras.size())
                continue;
            const auto found = reference_by_name.find(image_key(image.path));
            if (found == reference_by_name.end()) continue;
            const auto& view = reference.scene.views[found->second];
            const auto& camera = reconstruction.cameras[image.camera_id];
            common.push_back({
                image.path.filename().string(),
                image.pose.C,
                view.pose.C,
                image.pose.R,
                view.pose.R,
                camera.focal(),
                0.5 * (static_cast<double>(view.src_fx) + view.src_fy)});
        }
        if (common.size() < 3)
            throw std::runtime_error("Fewer than three registered image names match");

        std::vector<Vec3> source;
        std::vector<Vec3> destination;
        source.reserve(common.size());
        destination.reserve(common.size());
        for (const auto& item : common) {
            source.push_back(item.source_center);
            destination.push_back(item.reference_center);
        }

        std::vector<double> reference_steps;
        reference_steps.reserve(destination.size() - 1);
        for (std::size_t i = 1; i < destination.size(); ++i) {
            const double step = (destination[i] - destination[i - 1]).norm();
            if (std::isfinite(step) && step > 1e-12) reference_steps.push_back(step);
        }
        const double step_median = percentile(reference_steps, 0.5);
        if (!std::isfinite(step_median) || step_median <= 1e-12)
            throw std::runtime_error("Reference camera trajectory is degenerate");

        // First fit establishes scale; a second robust fit excludes gross pose
        // failures with a threshold expressed in reference trajectory units.
        Similarity3 initial;
        if (photara::sfm::estimate_similarity_transform(
                source, destination, initial, 0.0) < 3)
            throw std::runtime_error("Initial Sim(3) alignment failed");
        std::vector<double> initial_errors;
        initial_errors.reserve(common.size());
        for (std::size_t i = 0; i < common.size(); ++i)
            initial_errors.push_back((initial.apply(source[i]) - destination[i]).norm());
        const double robust_threshold = std::max(
            0.25 * step_median, 3.0 * percentile(initial_errors, 0.5));

        Similarity3 alignment;
        std::vector<std::size_t> inliers;
        const unsigned inlier_count = photara::sfm::estimate_similarity_transform(
            source, destination, alignment, robust_threshold, 4096,
            0x51F7A11u, &inliers);
        if (inlier_count < 3) throw std::runtime_error("Robust Sim(3) alignment failed");

        std::vector<double> centers;
        std::vector<double> rotations;
        std::vector<double> focal_relative;
        std::vector<std::pair<double, std::string>> worst_centers;
        std::vector<std::pair<double, std::string>> worst_rotations;
        centers.reserve(common.size());
        rotations.reserve(common.size());
        focal_relative.reserve(common.size());
        for (std::size_t i = 0; i < common.size(); ++i) {
            centers.push_back(
                (alignment.apply(common[i].source_center) -
                 common[i].reference_center).norm() /
                step_median);
            worst_centers.emplace_back(centers.back(), common[i].name);
            const Mat3 aligned_rotation =
                common[i].source_rotation * alignment.R.transpose();
            rotations.push_back(rotation_error_deg(
                aligned_rotation, common[i].reference_rotation));
            worst_rotations.emplace_back(
                rotations.back(), common[i].name);
            if (common[i].reference_focal > 1e-12)
                focal_relative.push_back(
                    std::abs(common[i].source_focal - common[i].reference_focal) /
                    common[i].reference_focal);
        }

        std::size_t triangulated = 0;
        std::size_t observations = 0;
        std::size_t reference_observations = 0;
        for (const auto& point : reference.scene.sparse_points)
            reference_observations += point.view_ids.size();
        for (const auto& track : reconstruction.tracks) {
            if (!track.is_triangulated()) continue;
            ++triangulated;
            observations += std::min<std::size_t>(
                track.num_inliers, track.observations.size());
        }

        std::cout << std::setprecision(10)
                  << "reference_views=" << reference.scene.views.size() << '\n'
                  << "reference_points=" << reference.scene.sparse_points.size() << '\n'
                  << "reference_observations=" << reference_observations << '\n'
                  << "registered_views=" << reconstruction.registered_count() << '\n'
                  << "common_views=" << common.size() << '\n'
                  << "alignment_inliers=" << inlier_count << '\n'
                  << "alignment_scale=" << alignment.scale << '\n'
                  << "reference_step_median=" << step_median << '\n'
                  << "center_error_steps_mean=" << mean(centers) << '\n'
                  << "center_error_steps_median=" << percentile(centers, 0.5) << '\n'
                  << "center_error_steps_p95=" << percentile(centers, 0.95) << '\n'
                  << "rotation_error_deg_mean=" << mean(rotations) << '\n'
                  << "rotation_error_deg_median=" << percentile(rotations, 0.5) << '\n'
                  << "rotation_error_deg_p95=" << percentile(rotations, 0.95) << '\n'
                  << "focal_relative_error_mean=" << mean(focal_relative) << '\n'
                  << "focal_relative_error_median=" << percentile(focal_relative, 0.5) << '\n'
                  << "triangulated_tracks=" << triangulated << '\n'
                  << "triangulated_observations=" << observations << '\n';
        std::sort(worst_centers.begin(), worst_centers.end(), std::greater<>());
        for (std::size_t i = 0; i < std::min<std::size_t>(10, worst_centers.size()); ++i)
            std::cout << "worst_center[" << i << "]="
                      << worst_centers[i].second << ',' << worst_centers[i].first
                      << '\n';
        std::sort(worst_rotations.begin(), worst_rotations.end(), std::greater<>());
        for (std::size_t i = 0; i < std::min<std::size_t>(10, worst_rotations.size()); ++i)
            std::cout << "worst_rotation[" << i << "]="
                      << worst_rotations[i].second << ',' << worst_rotations[i].first
                      << '\n';
        for (std::size_t i = 0; i < reference_missing.size(); ++i)
            std::cout << "reference_unregistered[" << i << "]="
                      << reference_missing[i] << '\n';
        for (std::size_t i = 0; i < reconstruction_missing.size(); ++i)
            std::cout << "reconstruction_unregistered[" << i << "]="
                      << reconstruction_missing[i] << '\n';
        for (std::size_t i = 0;
             i < reconstruction_missing_from_reference.size(); ++i)
            std::cout << "missing_relative_to_reference[" << i << "]="
                      << reconstruction_missing_from_reference[i] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sfm reference evaluation failed: " << error.what() << '\n';
        return 1;
    }
}
