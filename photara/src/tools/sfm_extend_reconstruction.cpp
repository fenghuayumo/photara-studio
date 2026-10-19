// Extend a retained full-feature ASFM with newly verified correspondences.
// Candidate poses/points never enter the solver; retained cameras stay frozen.
#include "sfm/asfm.hpp"
#include "sfm/appearance.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/submap_recovery.hpp"
#include "sfm/triangulation.hpp"
#include "sfm/tracks.hpp"
#include "core/logging.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <bit>
#include <map>
#include <array>

int main(int argc, char** argv) {
    using namespace photara::sfm;
    try {
        bool fresh_tracks = false, merge_features = false;
        if (argc < 4) throw std::runtime_error(
            "baseline-full.asfm {tracks,reconstruction}-HEX.bin output.asfm [--fresh-tracks] [--merge-features]");
        for (int arg = 4; arg < argc; ++arg) {
            if (std::string_view(argv[arg]) == "--fresh-tracks") fresh_tracks = true;
            else if (std::string_view(argv[arg]) == "--merge-features") merge_features = true;
            else throw std::runtime_error("unknown extension option");
        }
        const std::filesystem::path input(argv[2]), output(argv[3]);
        auto csv_path = output;
        csv_path.replace_extension();
        csv_path += "_sfm_diagnostics.csv";
        if (std::filesystem::exists(output) || std::filesystem::exists(csv_path))
            throw std::runtime_error("output already exists");
        photara::core::Logger::instance().configure(output.parent_path(), "frozen-extension");
        Scene baseline = load_asfm(argv[1]), correspondences;
        const auto stem = input.stem().string();
        const auto stage = stem.starts_with("tracks-") ? CheckpointStage::tracks
            : CheckpointStage::reconstruction;
        const std::string prefix = stage == CheckpointStage::tracks ? "tracks-" : "reconstruction-";
        if (!stem.starts_with(prefix)) throw std::runtime_error("expected tracks or reconstruction checkpoint");
        const auto key_text = stem.substr(prefix.size());
        if (key_text.empty() || key_text.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
            throw std::runtime_error("invalid checkpoint key");
        const auto key = std::stoull(key_text, nullptr, 16);
        CheckpointOptions options;
        options.directory = input.parent_path();
        options.write = false;
        if (!CheckpointStore(options).load_scene(stage, key, correspondences))
            throw std::runtime_error("cannot read correspondence checkpoint");
        if (baseline.images.size() != correspondences.images.size())
            throw std::runtime_error("image sets differ");
        std::vector<std::uint8_t> stable(baseline.images.size());
        std::vector<Pose3D> frozen;
        std::vector<std::vector<Index>> feature_map(merge_features ? baseline.images.size() : 0);
        for (Index i = 0; i < baseline.images.size(); ++i) {
            auto& old = baseline.images[i];
            const auto& fresh = correspondences.images[i];
            if (old.path.lexically_normal() != fresh.path.lexically_normal() ||
                old.features.image_width != fresh.features.image_width ||
                old.features.image_height != fresh.features.image_height)
                throw std::runtime_error("incompatible image/features: " + old.path.string());
            if (merge_features) {
                // Exact keypoint identity only, never nearest-pixel matching.
                // Keep every retained index; remap candidate associations and
                // append new features. Distinct duplicate occurrences remain
                // distinct indices instead of collapsing track identities.
                const auto signature = [](const auto& p) {
                    return std::array<std::uint32_t, 5>{std::bit_cast<std::uint32_t>(p.x),
                        std::bit_cast<std::uint32_t>(p.y), std::bit_cast<std::uint32_t>(p.scale),
                        std::bit_cast<std::uint32_t>(p.orientation), std::bit_cast<std::uint32_t>(p.response)};
                };
                std::map<std::array<std::uint32_t, 5>, std::vector<Index>> available;
                for (std::size_t j = old.features.keypoints.size(); j > 0; --j)
                    available[signature(old.features.keypoints[j - 1])].push_back(static_cast<Index>(j - 1));
                auto& remap = feature_map[i];
                remap.reserve(fresh.features.keypoints.size());
                for (const auto& point : fresh.features.keypoints) {
                    auto& indices = available[signature(point)];
                    if (!indices.empty()) {
                        remap.push_back(indices.back());
                        indices.pop_back();
                    } else {
                        remap.push_back(static_cast<Index>(old.features.keypoints.size()));
                        old.features.keypoints.push_back(point);
                    }
                }
                stable[i] = old.registered;
                frozen.push_back(old.pose);
                continue;
            }
            // Without explicit merging, reject reordered/extracted features.
            const auto shared_features = std::min(old.features.keypoints.size(), fresh.features.keypoints.size());
            for (std::size_t j = 0; j < shared_features; ++j) {
                const auto& a = old.features.keypoints[j];
                const auto& b = fresh.features.keypoints[j];
                if (a.x != b.x || a.y != b.y || a.scale != b.scale ||
                    a.orientation != b.orientation || a.response != b.response)
                    throw std::runtime_error("feature indices differ: " + old.path.string());
            }
            stable[i] = old.registered;
            frozen.push_back(old.pose);
            // Missing-view searches may have appended a longer suffix to the
            // retained map. Keep it: candidate indices in the checked common
            // prefix already refer to exactly the same keypoints.
            if ((!stable[i] || fresh_tracks) && fresh.features.keypoints.size() > old.features.keypoints.size())
                old.features = fresh.features;
        }
        if (merge_features) {
            for (auto& pair : correspondences.pairs) {
                for (auto& match : pair.matches) {
                    match.query = feature_map.at(pair.id1).at(match.query);
                    match.train = feature_map.at(pair.id2).at(match.train);
                }
            }
            for (auto& track : correspondences.tracks)
                for (auto& observation : track.observations)
                    observation.feature_id = feature_map.at(observation.image_id).at(observation.feature_id);
        }
        if (fresh_tracks) {
            // Import associations only. Re-estimate every depth from the
            // retained poses; candidate coordinates and inlier labels must
            // never influence either fitting or held-out validation.
            baseline.tracks = std::move(correspondences.tracks);
            for (auto& track : baseline.tracks) {
                track.position.setZero();
                track.num_inliers = 0;
                track.has_color = false;
            }
            rebuild_track_index(baseline);
        }
        // Import boundary geometry, never candidate registration, poses or
        // intrinsics. Track associations are opt-in above.
        for (auto& pair : correspondences.pairs) {
            if (!pair.active || pair.zero_baseline || pair.id1 >= stable.size() ||
                pair.id2 >= stable.size() || stable[pair.id1] == stable[pair.id2]) continue;
            const auto& a = baseline.images[pair.id1].features.keypoints;
            const auto& b = baseline.images[pair.id2].features.keypoints;
            std::erase_if(pair.matches, [&](const auto& m) { return m.query >= a.size() || m.train >= b.size(); });
            if (!pair.matches.empty()) baseline.pairs.push_back(std::move(pair));
        }
        correspondences.clear();
        const auto before = baseline.registered_count();
        const auto recovered = recover_stable_resections(baseline, stable);
        if (fresh_tracks) {
            TriangulationOptions tri;
            tri.reproj_threshold_px = 2;
            tri.min_angle_deg = 1;
            tri.split_tracks = false;
            triangulate_tracks(baseline, false, tri);
            rebuild_track_index(baseline);
            colour_triangulated_tracks(baseline);
        }
        for (Index i = 0; i < stable.size(); ++i) {
            if (!stable[i]) continue;
            if (!baseline.images[i].registered ||
                !(baseline.images[i].pose.R.array() == frozen[i].R.array()).all() ||
                !(baseline.images[i].pose.C.array() == frozen[i].C.array()).all())
                throw std::runtime_error("frozen pose changed");
        }
        for (Index i : recovered) baseline.images[i].position_audit = PositionAudit::unverified;
        std::ofstream csv(csv_path);
        csv << std::setprecision(17)
            << "image_id,name,registered,camera_id,width,height,fx,fy,cx,cy,k1,k2,p1,p2,center_x,center_y,center_z,qw,qx,qy,qz,camera_model\n";
        for (const auto& im : baseline.images) {
            const auto& c = baseline.camera_of(im);
            const auto q = im.pose.quaternion();
            csv << im.id << ',' << std::quoted(im.path.filename().string()) << ',' << im.registered << ','
                << im.camera_id << ',' << c.width << ',' << c.height << ',' << c.fx << ',' << c.fy << ','
                << c.cx << ',' << c.cy << ',' << c.k1 << ',' << c.k2 << ',' << c.p1 << ',' << c.p2 << ','
                << im.pose.C.x() << ',' << im.pose.C.y() << ',' << im.pose.C.z() << ','
                << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
                << (c.model == photara::CameraModel::opencv_fisheye ? "opencv_fisheye" :
                    c.model == photara::CameraModel::equirectangular ? "equirectangular" : "pinhole") << '\n';
        }
        if (!csv) throw std::runtime_error("cannot write diagnostics");
        save_asfm(baseline, output);
        std::cout << "before=" << before << " added=" << recovered.size()
                  << " after=" << baseline.registered_count() << " frozen_poses_equal=1\n";
        for (Index i : recovered) std::cout << "added_image=" << baseline.images[i].path.filename() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
