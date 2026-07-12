#include "sfm/star_init.hpp"

#include "sfm/bundle.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::sfm {
namespace {

bool triangulate_midpoint(
    const Mat3& R,
    const Vec3& C,
    const Vec3& b1,
    const Vec3& b2,
    Vec3& X) {
    const Vec3 d1 = b1.normalized();
    const Vec3 d2 = (R.transpose() * b2).normalized();
    const Mat3 A =
        Mat3::Identity() - d1 * d1.transpose() + Mat3::Identity() - d2 * d2.transpose();
    const Vec3 rhs = (Mat3::Identity() - d2 * d2.transpose()) * C;
    Eigen::LDLT<Mat3> ldlt(A);
    if (ldlt.info() != Eigen::Success) return false;
    X = ldlt.solve(rhs);
    return X.allFinite();
}

}  // namespace

Index select_reference_view(const Scene& scene) {
    std::vector<unsigned> degree(scene.images.size(), 0);
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.usable_for_init()) continue;
        degree[pair.id1] += pair.num_inliers();
        degree[pair.id2] += pair.num_inliers();
    }
    Index best = k_invalid;
    unsigned max_degree = 0;
    for (Index i = 0; i < degree.size(); ++i) {
        if (degree[i] > max_degree) {
            max_degree = degree[i];
            best = i;
        }
    }
    return best;
}

bool star_initialize(Scene& scene, const StarInitConfig& config) {
    for (auto& image : scene.images) {
        image.registered = false;
        image.pose = Pose3D::identity();
    }
    for (auto& track : scene.tracks) track.num_inliers = 0;

    const Index ref = select_reference_view(scene);
    if (ref == k_invalid) {
        std::cerr << "star_init: no reference view\n";
        return false;
    }

    struct Neighbor {
        Index image_id{};
        Index pair_index{};
        unsigned inliers{};
        float weight{};
    };
    std::vector<Neighbor> neighbors;
    for (Index pi = 0; pi < scene.pairs.size(); ++pi) {
        const ImagePair& pair = scene.pairs[pi];
        // Exclude H-dominant / low-parallax pairs from star seeds (still in view graph).
        if (!pair.usable_for_init()) continue;
        Index other = k_invalid;
        if (pair.id1 == ref) other = pair.id2;
        else if (pair.id2 == ref) other = pair.id1;
        else continue;
        neighbors.push_back(
            {other, pi, pair.num_inliers(), pair.composite_weight()});
    }
    std::sort(neighbors.begin(), neighbors.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.weight > b.weight || (a.weight == b.weight && a.inliers > b.inliers);
    });
    if (neighbors.size() > config.max_views)
        neighbors.resize(config.max_views);
    if (neighbors.size() + 1 < config.min_views) {
        std::cerr << "star_init: insufficient neighbors (" << neighbors.size() << ")\n";
        return false;
    }

    scene.images[ref].registered = true;
    scene.images[ref].pose = Pose3D::identity();

    // Seed absolute poses from relative poses (unit translation); then scale.
    std::vector<double> pair_scales(scene.pairs.size(), 1.0);
    std::unordered_set<Index> star_views{ref};
    for (const Neighbor& n : neighbors) star_views.insert(n.image_id);

    // Scale averaging: for each star pair, collect depths; constrain scales of
    // pairs that share a (view, feature) via median log-ratio (openMVS-inspired).
    struct FeatDepth {
        Index pair_index{};
        double depth{};
    };
    std::unordered_map<Index, std::unordered_map<Index, std::vector<FeatDepth>>> view_feat_depths;

    for (Index pi = 0; pi < scene.pairs.size(); ++pi) {
        const ImagePair& pair = scene.pairs[pi];
        if (!pair.relative_pose.has_value()) continue;
        if (!star_views.count(pair.id1) || !star_views.count(pair.id2)) continue;
        const Image& img1 = scene.images[pair.id1];
        const Image& img2 = scene.images[pair.id2];
        const PinholeCamera& cam1 = scene.camera_of(img1);
        const PinholeCamera& cam2 = scene.camera_of(img2);
        const Pose3D& rel = *pair.relative_pose;
        const double cos_reproj = std::cos(
            0.5 * (cam1.pixel_error_to_angular(config.max_reproj_error) +
                   cam2.pixel_error_to_angular(config.max_reproj_error)));
        const double max_cos_angle = std::cos(0.5 * 3.14159265358979323846 / 180.0);

        for (const FeatureMatch& match : pair.matches) {
            const auto& kp1 = img1.features.keypoints[match.query];
            const auto& kp2 = img2.features.keypoints[match.train];
            const Vec3 b1 = cam1.unproject_normalized({kp1.x, kp1.y});
            const Vec3 b2 = cam2.unproject_normalized({kp2.x, kp2.y});
            Vec3 X;
            if (!triangulate_midpoint(rel.R, rel.C, b1, b2, X)) continue;
            const double n1 = X.norm();
            if (n1 < 1e-12) continue;
            if (b1.dot(X / n1) < cos_reproj) continue;
            const Vec3 X2 = rel.transform_world_to_camera(X);
            const double n2 = X2.norm();
            if (n2 < 1e-12) continue;
            if (b2.dot(X2 / n2) < cos_reproj) continue;
            const double cos_a = X.normalized().dot((X - rel.C).normalized());
            if (cos_a > max_cos_angle) continue;
            view_feat_depths[pair.id1][match.query].push_back({pi, n1});
            view_feat_depths[pair.id2][match.train].push_back({pi, (X - rel.C).norm()});
        }
    }

    // Collect pairwise scale ratios
    std::unordered_map<Index, std::unordered_map<Index, std::vector<double>>> ratios;
    for (const auto& [view_id, feats] : view_feat_depths) {
        (void)view_id;
        for (const auto& [feat_id, depths] : feats) {
            (void)feat_id;
            if (depths.size() < 2) continue;
            for (std::size_t i = 0; i + 1 < depths.size(); ++i) {
                for (std::size_t j = i + 1; j < depths.size(); ++j) {
                    Index p1 = depths[i].pair_index;
                    Index p2 = depths[j].pair_index;
                    double d1 = depths[i].depth;
                    double d2 = depths[j].depth;
                    if (p1 > p2) {
                        std::swap(p1, p2);
                        std::swap(d1, d2);
                    }
                    if (d1 < 1e-12) continue;
                    ratios[p1][p2].push_back(d2 / d1);
                }
            }
        }
    }

    // Initialize scales from spanning constraints with median ratios
    // Fix scale of strongest ref-connected pair to 1.
    Index anchor_pair = neighbors.empty() ? k_invalid : neighbors.front().pair_index;
    if (anchor_pair != k_invalid) pair_scales[anchor_pair] = 1.0;

    for (const auto& [p1, map] : ratios) {
        for (const auto& [p2, vals] : map) {
            if (vals.size() < 8) continue;
            std::vector<double> sorted = vals;
            std::sort(sorted.begin(), sorted.end());
            const double median = sorted[sorted.size() / 2];
            // S1/S2 = median => if one known, set the other
            if (anchor_pair == p1) {
                pair_scales[p2] = pair_scales[p1] / median;
            } else if (anchor_pair == p2) {
                pair_scales[p1] = pair_scales[p2] * median;
            } else if (pair_scales[p1] != 1.0 || p1 == anchor_pair) {
                pair_scales[p2] = pair_scales[p1] / median;
            } else if (pair_scales[p2] != 1.0 || p2 == anchor_pair) {
                pair_scales[p1] = pair_scales[p2] * median;
            }
        }
    }

    // relative_pose is the pose of id2 in id1's frame (id1 at identity).
    for (const Neighbor& n : neighbors) {
        ImagePair& pair = scene.pairs[n.pair_index];
        Pose3D rel = *pair.relative_pose;
        rel.C *= pair_scales[n.pair_index];
        pair.relative_pose = rel;
        if (pair.id1 == ref) {
            scene.images[pair.id2].pose = rel;
            scene.images[pair.id2].registered = true;
        } else {
            scene.images[pair.id1].pose = rel.inverse();
            scene.images[pair.id1].registered = true;
        }
    }

    unsigned n_tracks = triangulate_tracks(
        scene, false, config.max_reproj_error, config.min_angle_deg);
    if (n_tracks < config.min_initial_tracks) {
        std::cerr << "star_init: too few tracks after triangulation (" << n_tracks << ")\n";
        return false;
    }

    BundleOptions ba;
    ba.optimizer.maximum_iterations = 10;
    ba.optimizer.huber_delta = 2.0;
    ba.optimizer.optimize_focal = true;
    if (!run_bundle_adjustment(scene, ba).success) {
        std::cerr << "star_init: initial BA failed\n";
        return false;
    }

    triangulate_tracks(
        scene, true, std::max(config.max_reproj_error - 1.F, 1.F),
        std::min(config.min_angle_deg + 0.5F, 3.F));
    filter_tracks(
        scene, std::max(config.max_reproj_error - 1.F, 1.F),
        std::min(config.min_angle_deg + 0.5F, 3.F));

    n_tracks = 0;
    for (const Track& t : scene.tracks) {
        if (t.is_triangulated()) ++n_tracks;
    }
    if (n_tracks < config.min_initial_tracks) {
        std::cerr << "star_init: too few tracks after filter (" << n_tracks << ")\n";
        return false;
    }

    ba.optimizer.maximum_iterations = 25;
    ba.optimizer.optimize_focal = true;
    ba.optimizer.optimize_distortion = true;
    run_bundle_adjustment(scene, ba);

    triangulate_tracks(
        scene, true, std::max(config.max_reproj_error - 2.F, 1.F),
        std::min(config.min_angle_deg + 1.F, 3.F));
    filter_tracks(
        scene, std::max(config.max_reproj_error - 2.F, 1.F),
        std::min(config.min_angle_deg + 1.F, 3.F));

    std::cout << "star_init: ref=" << ref << " views=" << scene.registered_count()
              << " tracks=" << n_tracks << '\n';
    return scene.registered_count() >= config.min_views;
}

}  // namespace aetherscan::sfm
