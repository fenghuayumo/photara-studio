#pragma once
#include "../src/splat/pam_cuda.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace photara::splat::detail {
// Test-only brute-force oracle, independent of the production KD-tree lookup.
inline std::vector<float> pam_field_gradients_reference(
    const GaussianModel& model, const std::vector<float>& points, unsigned neighbors) {
    if (points.size() % 3 || neighbors == 0)
        throw std::invalid_argument("Invalid PAM reference query");
    const auto values = make_pam_gpu_field(model).values.to_vector();
    std::vector<float> result(points.size(), 0.F);
    std::vector<std::pair<float, std::size_t>> nearest(model.size());
    const auto count = std::min<std::size_t>(neighbors, model.size());
    for (std::size_t query = 0; query < points.size() / 3; ++query) {
        const Eigen::Map<const Eigen::Vector3f> point(points.data() + 3 * query);
        for (std::size_t i = 0; i < model.size(); ++i) {
            const Eigen::Map<const Eigen::Vector3f> mean(values.data() + 19 * i);
            nearest[i] = {(point - mean).squaredNorm(), i};
        }
        std::partial_sort(nearest.begin(), nearest.begin() + count, nearest.end());
        Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
        for (std::size_t j = 0; j < count; ++j) {
            const float* row = values.data() + 19 * nearest[j].second;
            const Eigen::Map<const Eigen::Vector3f> mean(row), normal(row + 3), inverse(row + 6);
            const Eigen::Map<const Eigen::Matrix3f> rotation(row + 9);
            const Eigen::Vector3f delta = point - mean;
            if (normal.dot(delta) < 0.F) continue;
            const Eigen::Vector3f local = rotation.transpose() * delta;
            const float exponent = -.5F * (local.array().square() * inverse.array()).sum();
            const float gaussian = std::clamp(row[18] * std::exp(exponent), 0.F, 1.F - 1e-7F);
            gradient += gaussian / (1.F - gaussian + 1e-8F) *
                (rotation * (inverse.array() * local.array()).matrix());
        }
        Eigen::Map<Eigen::Vector3f>(result.data() + 3 * query) = gradient;
    }
    return result;
}
}  // namespace photara::splat::detail
