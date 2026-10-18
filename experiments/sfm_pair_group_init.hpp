#pragma once
#include "sfm/resection.hpp"
#include "core/logging.hpp"
#include "sfm/triangulation.hpp"
#include <algorithm>
#include <vector>

// Bounded offline alternative to a >=4-view star: independent two-view seeds,
// then joint incremental PnP/BA. No parent pose, depth or reference is available.
inline bool initialize_missing_group_pairs(photara::sfm::Scene& scene) {
    using namespace photara::sfm;
    std::vector<std::size_t> seeds;
    for (std::size_t p=0;p<scene.pairs.size();++p) {
        const auto& pair=scene.pairs[p];
        if (pair.usable_for_init() && !pair.zero_baseline && pair.matches.size()>=30 &&
            pair.relative_pose->C.allFinite() && pair.relative_pose->C.norm()>1e-8) seeds.push_back(p);
    }
    std::stable_sort(seeds.begin(),seeds.end(),[&](auto a,auto b) {
        return scene.pairs[a].composite_weight()>scene.pairs[b].composite_weight();
    });
    photara::core::Logger::instance().info("group pair initializer: views=",scene.images.size()," eligible_seeds=",seeds.size());
    if (seeds.size()>8) seeds.resize(8);
    Scene best; unsigned best_views=0, best_points=0;
    for (auto seed:seeds) {
        Scene trial=scene;
        // Clear all state even if a caller mistakenly passed a partially solved scene.
        for (auto& image:trial.images) { image.registered=false; image.pose=Pose3D::identity(); }
        for (auto& t:trial.tracks) { t.num_inliers=0; t.position.setZero(); }
        const auto& pair=trial.pairs[seed];
        trial.images[pair.id1].registered=trial.images[pair.id2].registered=true;
        trial.images[pair.id2].pose=*pair.relative_pose;
        triangulate_tracks(trial,false,2.F,1.F);
        unsigned points=0; for (const auto& t:trial.tracks) points+=t.is_triangulated();
        photara::core::Logger::instance().info("group pair seed: first=",pair.id1," second=",pair.id2," points=",points);
        if (points<30) continue;
        ResectionConfig config;
        config.full_ba.optimize_focal=false;
        config.full_ba.optimize_aspect_ratio=false;
        config.full_ba.optimize_distortion=false;
        register_images(trial,config);
        triangulate_tracks(trial,false,2.F,1.F);
        points=0; for (const auto& t:trial.tracks) points+=t.is_triangulated();
        photara::core::Logger::instance().info("group pair growth: views=",trial.registered_count()," points=",points);
        if (trial.registered_count()>best_views || (trial.registered_count()==best_views && points>best_points)) {
            best_views=trial.registered_count(); best_points=points; best=std::move(trial);
        }
        if (best_views==scene.images.size()) break;
    }
    if (best_views<4) return false;
    scene=std::move(best); return true;
}
