#include "sfm/camera_selection.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace aetherscan::sfm {
CameraModelSelection select_camera_model(
    const PinholeCamera& source, const std::vector<CameraModelProbe>& probes,
    const double supplied_focal, const bool fisheye_lens_hint) {
    CameraModelSelection result;
    const double size = static_cast<double>(std::max(source.width, source.height));
    result.focal_pixels = supplied_focal > 0 ? supplied_focal : 1.2*size;
    result.reason = "insufficient geometry; pinhole fallback";
    if (size <= 0 || probes.size() < 2) return result;
    RelativePoseOptions options;
    options.max_iterations = 1000;
    options.min_iterations = 100;
    options.min_inliers = 20;
    options.max_epipolar_error_px = 2.0;
    options.max_reproj_error_px = 3.0;
    options.min_ray_angle_deg = 0.5;
    std::array<double, 2> best_score{};
    std::array<double, 2> best_focal{};
    std::array<std::vector<double>, 2> best_pairs;
    const std::array<double, 8> ratios{0.3, 0.4, 0.5, 0.65, 0.85, 1.2, 1.6, 2.2};
    for (int model = 0; model < 2; ++model) {
        std::vector<double> focals;
        if (supplied_focal > 0) focals.push_back(supplied_focal);
        else for (const double ratio : ratios) focals.push_back(size*ratio);
        for (const double focal : focals) {
            PinholeCamera camera = source;
            camera.model = static_cast<CameraModel>(model);
            camera.fx = camera.fy = focal;
            camera.k1 = camera.k2 = camera.p1 = camera.p2 = 0;
            camera.trust_intrinsics = true;
            std::vector<double> scores(probes.size(), 0.0);
            for (std::size_t i = 0; i < probes.size(); ++i) {
                const auto& probe = probes[i];
                if (probe.first.size() < 30 || probe.first.size() != probe.second.size()) continue;
                // Fisheye pixels outside the supported forward hemisphere
                // cannot contribute to a valid calibration hypothesis.
                std::vector<Vec2> first, second;
                for (std::size_t j=0; j<probe.first.size(); ++j) {
                    if (!camera.unproject(probe.first[j]).allFinite() ||
                        !camera.unproject(probe.second[j]).allFinite()) continue;
                    first.push_back(probe.first[j]); second.push_back(probe.second[j]);
                }
                if (first.size() < 30) continue;
                const auto fit = estimate_relative_pose(first, second, camera, camera, options);
                if (!fit.success || fit.degenerate_planar) continue;
                scores[i] = static_cast<double>(fit.num_inliers)/probe.first.size();
            }
            const double score = std::accumulate(scores.begin(), scores.end(), 0.0)/probes.size();
            if (score > best_score[model]) {
                best_score[model] = score; best_focal[model] = focal;
                best_pairs[model] = std::move(scores);
            }
        }
    }
    result.pinhole_score = best_score[0]; result.fisheye_score = best_score[1];
    const int winner = best_score[1] > best_score[0] ? 1 : 0;
    if (best_pairs[winner].empty()) return result;
    unsigned wins = 0;
    for (std::size_t i = 0; i < probes.size(); ++i) {
        const double score = best_pairs[winner][i];
        if (score >= 0.25) ++result.informative_pairs;
        const double other = best_pairs[1-winner].empty() ? 0 : best_pairs[1-winner][i];
        if (score >= other + 0.08) ++wins;
    }
    result.confident = result.informative_pairs >= 2 && wins >= 2 &&
        best_score[winner] >= 0.3 && best_score[winner]-best_score[1-winner] >= 0.08;
    if (result.confident) {
        result.model = static_cast<CameraModel>(winner);
        result.focal_pixels = best_focal[winner];
        result.reason = "multi-view geometry";
    } else if (fisheye_lens_hint && result.informative_pairs >= 2 &&
               best_score[1] >= 0.3 && best_score[1] + 0.04 >= best_score[0]) {
        result.model = CameraModel::opencv_fisheye;
        result.focal_pixels = best_focal[1];
        result.reason = "EXIF fisheye lens hint with geometric support";
    } else {
        result.reason = "ambiguous geometry; pinhole fallback";
        // Do not replace the original focal prior with an ambiguous hypothesis.
    }
    return result;
}
} // namespace aetherscan::sfm
