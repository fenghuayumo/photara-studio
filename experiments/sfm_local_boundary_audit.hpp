#pragma once

#include "sfm/checkpoint.hpp"
#include "sfm/geometry.hpp"
#include <iomanip>
#include <map>
#include <fstream>

// Read-only counterfactual: reverify the SAME raw descriptor proposals under
// native calibration, including pairs discarded before the old local probe.
inline void audit_local_boundary(const photara::sfm::Scene& parent,
    const std::filesystem::path& cache_root, std::ostream& out) {
    using namespace photara::sfm;
    std::map<std::filesystem::path, Index> ids;
    for (Index i=0;i<parent.images.size();++i) ids.emplace(parent.images[i].path,i);
    out << "group,first,second,kind,raw,frontend_kept,native_raw_success,native_raw_ransac,native_raw_kept,native_subset_success,native_subset_kept,approx_raw15_success,approx_raw15_kept,native_raw30_success,native_raw30_kept\n";
    for (const auto& directory : std::filesystem::directory_iterator(cache_root)) {
        if (!directory.is_directory()) continue;
        CheckpointOptions options; options.directory=directory.path(); options.write=false;
        CheckpointStore store(options);
        Scene local; std::vector<RawPairMatches> raw;
        unsigned scenes=0,matches=0;
        for (const auto& file : std::filesystem::directory_iterator(directory.path())) {
            const auto stem=file.path().stem().string();
            if (file.path().extension()!=".bin") continue;
            if (stem.starts_with("tracks-")) {
                if (!store.load_scene(CheckpointStage::tracks,std::stoull(stem.substr(7),nullptr,16),local))
                    throw std::runtime_error("Cannot read local track snapshot");
                ++scenes;
            } else if (stem.starts_with("matches-")) {
                if (!store.load_matches(std::stoull(stem.substr(8),nullptr,16),raw))
                    throw std::runtime_error("Cannot read local raw matches");
                ++matches;
            }
        }
        if (scenes!=1 || matches!=1) throw std::runtime_error("Ambiguous local snapshots");
        std::vector<Index> global;
        for (const auto& image : local.images) global.push_back(ids.at(image.path));
        std::map<std::pair<Index,Index>, const ImagePair*> verified;
        for (const auto& pair : local.pairs) verified.emplace(std::pair{pair.id1,pair.id2},&pair);
        for (const auto& pair : raw) {
            const auto a=pair.id1,b=pair.id2;
            const auto& first=parent.images.at(global.at(a));
            const auto& second=parent.images.at(global.at(b));
            if (first.registered && second.registered) continue;
            const auto solve=[&](const auto& proposals, bool native=true, unsigned minimum=15) {
                std::vector<Vec2> x,y;
                for (const auto& m : proposals) {
                    const auto& p=local.images[a].features.keypoints.at(m.query);
                    const auto& q=local.images[b].features.keypoints.at(m.train);
                    x.emplace_back(p.x,p.y); y.emplace_back(q.x,q.y);
                }
                RelativePoseOptions verification; verification.min_inliers=minimum;
                auto camera1=native ? parent.camera_of(first) : local.camera_of(local.images[a]);
                auto camera2=native ? parent.camera_of(second) : local.camera_of(local.images[b]);
                // Compare calibration values with the SAME calibrated solver.
                // The saved native camera can retain an untrusted frontend flag;
                // leaving it unchanged would switch E to shared-focal/F instead.
                camera1.trust_intrinsics=camera2.trust_intrinsics=true;
                return estimate_relative_pose(x,y,camera1,camera2,verification);
            };
            const auto result=solve(pair.matches);
            const auto approximate=solve(pair.matches,false);
            const auto native30=solve(pair.matches,true,30);
            const auto found=verified.find({a,b});
            RelativePoseResult subset; unsigned kept=0;
            if (found!=verified.end()) { kept=found->second->matches.size(); subset=solve(found->second->matches); }
            out << directory.path().filename().string() << ',' << std::quoted(first.path.filename().string()) << ','
                << std::quoted(second.path.filename().string()) << ','
                << (first.registered!=second.registered ? "boundary" : "internal") << ','
                << pair.matches.size() << ',' << kept << ',' << result.success << ',' << result.num_ransac_inliers << ','
                << result.num_inliers << ',' << subset.success << ',' << subset.num_inliers << ','
                << approximate.success << ',' << approximate.num_inliers << ',' << native30.success << ',' << native30.num_inliers << '\n';
        }
        out.flush();
    }
    if (!out) throw std::runtime_error("Cannot write local boundary audit");
}
