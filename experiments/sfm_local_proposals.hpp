#pragma once

#include "sfm/frontend.hpp"
#include <map>
#include <set>

// Experimental local frontend only. Descriptor proposals are not registrations.
struct LocalProposalSummary {
    unsigned raw_pairs{}, accepted_pairs{}, boundary_pairs{}, weak_boundary_pairs{};
    unsigned cycle_supported_matches{}, rejected_weak_pairs{};
};

inline photara::sfm::Scene load_local_proposal_snapshot(
    const std::filesystem::path& directory, std::vector<photara::sfm::RawPairMatches>& raw) {
    using namespace photara::sfm;
    std::filesystem::path tracks, matches;
    for (const auto& file : std::filesystem::directory_iterator(directory)) {
        if (file.path().extension() != ".bin") continue;
        auto assign=[&](std::filesystem::path& destination) {
            if (!destination.empty()) throw std::runtime_error("Ambiguous local frontend snapshot");
            destination=file.path();
        };
        const auto stem=file.path().stem().string();
        if (stem.starts_with("tracks-")) assign(tracks);
        if (stem.starts_with("matches-")) assign(matches);
    }
    if (tracks.empty() || matches.empty()) throw std::runtime_error("Incomplete local frontend snapshot");
    CheckpointOptions options; options.directory=directory; options.write=false;
    const CheckpointStore store(options);
    Scene scene;
    if (!store.load_scene(CheckpointStage::tracks,std::stoull(tracks.stem().string().substr(7),nullptr,16),scene) ||
        !store.load_matches(std::stoull(matches.stem().string().substr(8),nullptr,16),raw))
        throw std::runtime_error("Cannot read local frontend snapshot");
    return scene;
}

inline LocalProposalSummary verify_local_raw_proposals(
    photara::sfm::Scene& scene, const std::vector<std::uint8_t>& stable,
    const std::vector<photara::sfm::RawPairMatches>& raw, bool require_weak_cycles=true) {
    using namespace photara::sfm;
    if (stable.size()!=scene.images.size()) throw std::invalid_argument("Invalid local stable mask");
    std::vector<ImagePair> verified;
    LocalProposalSummary summary; summary.raw_pairs=static_cast<unsigned>(raw.size());
    std::set<std::pair<Index,Index>> pairs;
    for (const auto& proposal : raw) {
        const auto a=proposal.id1,b=proposal.id2;
        if (a>=scene.images.size() || b>=scene.images.size() || a>=b || !pairs.emplace(a,b).second)
            throw std::invalid_argument("Invalid/duplicate local pair IDs");
        const auto& first=scene.images[a]; const auto& second=scene.images[b];
        std::vector<Vec2> x,y;
        for (const auto& match:proposal.matches) {
            if (match.query>=first.features.keypoints.size() || match.train>=second.features.keypoints.size())
                throw std::invalid_argument("Invalid local feature index");
            const auto& p=first.features.keypoints[match.query];
            const auto& q=second.features.keypoints[match.train];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(q.x) || !std::isfinite(q.y))
                throw std::invalid_argument("Nonfinite local feature");
            x.emplace_back(p.x,p.y); y.emplace_back(q.x,q.y);
        }
        ImagePair pair(a,b);
        if (stable[a] && stable[b]) {
            const Vec3 baseline=second.pose.C-first.pose.C;
            if (!first.registered || !second.registered || baseline.norm()<1e-10) continue;
            const double tolerance=std::sin(std::max(scene.camera_of(first).pixel_error_to_angular(4),
                scene.camera_of(second).pixel_error_to_angular(4)));
            for (std::size_t i=0;i<x.size();++i) {
                const Vec3 u=first.pose.R.transpose()*scene.camera_of(first).unproject_normalized(x[i]);
                const Vec3 v=second.pose.R.transpose()*scene.camera_of(second).unproject_normalized(y[i]);
                const Vec3 n1=baseline.cross(u),n2=baseline.cross(v);
                if (n1.norm()>1e-10 && n2.norm()>1e-10 && std::abs(n1.normalized().dot(v))<=tolerance &&
                    std::abs(n2.normalized().dot(u))<=tolerance)
                    pair.matches.push_back({proposal.matches[i].query,proposal.matches[i].train});
            }
            if (pair.matches.size()<8) continue;
            pair.relative_pose=second.pose/first.pose;
            pair.weight_spatial=compute_spatial_weight(x,scene.camera_of(first));
        } else {
            RelativePoseOptions options; options.min_inliers=15;
            const auto result=estimate_relative_pose(x,y,scene.camera_of(first),scene.camera_of(second),options);
            if (!result.success) continue;
            for (std::size_t i=0;i<proposal.matches.size();++i)
                if (i<result.inlier_mask.size() && result.inlier_mask[i])
                    pair.matches.push_back({proposal.matches[i].query,proposal.matches[i].train});
            pair.relative_pose=result.pose; pair.degenerate_planar=result.degenerate_planar;
            pair.weight_geometry=result.degenerate_planar ? 0.05F : 1.F;
            pair.weight_spatial=result.weight_spatial; pair.mean_ray_angle=result.mean_ray_angle;
        }
        verified.push_back(std::move(pair));
    }
    // Exact feature cycles, never candidate poses, corroborate weak boundary
    // matches through a different missing camera. Strong pairs stay proposals
    // too and still face the independent submap validator.
    using Key=std::pair<Index,Index>;
    std::map<Key,std::set<Key>> adjacency;
    for (const auto& pair:verified) for (const auto& match:pair.matches) {
        adjacency[{pair.id1,match.query}].insert({pair.id2,match.train});
        adjacency[{pair.id2,match.train}].insert({pair.id1,match.query});
    }
    for (auto& pair:verified) {
        if (stable[pair.id1]==stable[pair.id2]) continue;
        const bool weak=pair.matches.size()<30;
        if (weak && require_weak_cycles) {
            std::erase_if(pair.matches,[&](const FeatureMatch& match) {
                const auto& left=adjacency.at({pair.id1,match.query});
                const auto& right=adjacency.at({pair.id2,match.train});
                for (const auto& observation:left)
                    if (!stable[observation.first] && observation.first!=pair.id1 && observation.first!=pair.id2 &&
                        right.contains(observation)) { ++summary.cycle_supported_matches; return false; }
                return true;
            });
            if (pair.matches.size()<15) { pair.active=false; ++summary.rejected_weak_pairs; continue; }
        }
        ++summary.boundary_pairs;
        summary.weak_boundary_pairs+=weak;
    }
    std::erase_if(verified,[](const auto& pair) { return !pair.active; });
    summary.accepted_pairs=static_cast<unsigned>(verified.size());
    scene.pairs=std::move(verified);
    return summary;
}
