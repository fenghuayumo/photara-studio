#include "mvs/densify.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace aetherscan::mvs {
namespace {

constexpr float k_deg = 180.F / 3.14159265358979323846F;

[[nodiscard]] float angle_weight(const float angle_deg, const float optim_deg) {
    const float delta = angle_deg - optim_deg;
    const float sigma = delta < 0.F ? (optim_deg * 0.5F) : (optim_deg * 1.5F);
    const float s = std::max(sigma, 1.F);
    return std::exp(-(delta * delta) / (2.F * s * s));
}

}  // namespace

void select_neighbors(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.select_neighbors");

    // view -> list of (other_view, shared sparse point count + angle accum)
    struct Acc {
        unsigned shared{0};
        float angle_sum{0.F};
        float scale_sum{0.F};
    };
    std::vector<std::unordered_map<Index, Acc>> accum(scene.views.size());

    for (const auto& point : scene.sparse_points) {
        for (std::size_t a = 0; a < point.view_ids.size(); ++a) {
            const Index id_a = point.view_ids[a];
            const MvsView& va = scene.views[id_a];
            const Vec3f xa =
                va.pose.transform_world_to_camera(point.position.cast<double>())
                    .cast<float>();
            if (xa.z() <= 1e-6F) continue;
            const float footprint_a =
                xa.z() / std::max(va.fx, 1.F);  // approx meters/pixel

            for (std::size_t b = 0; b < point.view_ids.size(); ++b) {
                if (a == b) continue;
                const Index id_b = point.view_ids[b];
                const MvsView& vb = scene.views[id_b];
                const Vec3f xb =
                    vb.pose.transform_world_to_camera(point.position.cast<double>())
                        .cast<float>();
                if (xb.z() <= 1e-6F) continue;

                const Vec3f dir_a = (point.position - va.pose.C.cast<float>()).normalized();
                const Vec3f dir_b = (point.position - vb.pose.C.cast<float>()).normalized();
                const float cos_a =
                    std::clamp(dir_a.dot(dir_b), -1.F, 1.F);
                const float angle = std::acos(cos_a) * k_deg;
                const float footprint_b = xb.z() / std::max(vb.fx, 1.F);
                const float scale =
                    footprint_a / std::max(footprint_b, 1e-8F);

                Acc& slot = accum[id_a][id_b];
                slot.shared += 1;
                slot.angle_sum += angle;
                slot.scale_sum += scale;
            }
        }
    }

    for (std::size_t i = 0; i < scene.views.size(); ++i) {
        std::vector<NeighborScore> neighbors;
        neighbors.reserve(accum[i].size());
        for (const auto& [other, acc] : accum[i]) {
            if (acc.shared < options.min_shared_points) continue;
            const float mean_angle = acc.angle_sum / static_cast<float>(acc.shared);
            const float mean_scale = acc.scale_sum / static_cast<float>(acc.shared);
            if (mean_angle < 3.F || mean_angle > 65.F) continue;
            if (mean_scale < 0.2F || mean_scale > 3.2F) continue;

            float scale_w = 1.F;
            if (mean_scale > 1.6F)
                scale_w = 1.6F / mean_scale;
            else if (mean_scale < 0.625F)
                scale_w = mean_scale / 0.625F;

            NeighborScore ns;
            ns.view_id = other;
            ns.angle_deg = mean_angle;
            ns.scale = mean_scale;
            ns.shared_points = acc.shared;
            ns.score = static_cast<float>(acc.shared) *
                       angle_weight(mean_angle, options.optim_angle_deg) *
                       scale_w;
            neighbors.push_back(ns);
        }
        std::sort(
            neighbors.begin(), neighbors.end(),
            [](const NeighborScore& a, const NeighborScore& b) {
                return a.score > b.score;
            });
        if (neighbors.size() > options.max_neighbors)
            neighbors.resize(options.max_neighbors);
        scene.views[i].neighbors = std::move(neighbors);
    }

    std::size_t with_neighbors = 0;
    for (const auto& view : scene.views)
        if (!view.neighbors.empty()) ++with_neighbors;
    core::Logger::instance().info(
        "mvs neighbors: views_with_neighbors=", with_neighbors, '/',
        scene.views.size());
    stage.finish();
}

}  // namespace aetherscan::mvs
