#pragma once
#include "sfm_local_proposals.hpp"
#include "sfm/submap_recovery.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"
#include <iomanip>
#include <array>
#include <limits>
#include <queue>

namespace photara::sfm::experiment {
using PrecisionKey=std::pair<Index,Index>;
struct PrecisionSample { Index target{},feature{}; Vec3 point; unsigned anchors{}; };
struct PrecisionPartition {
    std::set<PrecisionKey> reserved;
    std::vector<PrecisionSample> samples;
    // Counts are whole boundary components, before depth selection.
    std::map<Index,std::array<unsigned,4>> support;
    unsigned ambiguous{},depth_rejected{};
};

// Partition connected components of the full verified correspondence graph,
// including conflicting edges that build_tracks may decline to union. No pixel
// of a reserved component can subsequently enter pose fitting through an alias.
inline PrecisionPartition partition_precision(const Scene& scene,const std::vector<std::uint8_t>& stable) {
    if(stable.size()!=scene.images.size())throw std::invalid_argument("Invalid precision stable mask");
    std::map<PrecisionKey,std::set<PrecisionKey>> graph;
    for(const auto& pair:scene.pairs) {
        if(!pair.active || !pair.has_geometry() || pair.zero_baseline)continue;
        for(const auto& m:pair.matches) {
            const PrecisionKey a{pair.id1,m.query},b{pair.id2,m.train};
            for(const auto& k:{a,b})
                if(k.first>=scene.images.size() || k.second>=scene.images[k.first].features.keypoints.size())
                    throw std::invalid_argument("Invalid precision observation");
            graph[a].insert(b);graph[b].insert(a);
        }
    }
    PrecisionPartition result;std::set<PrecisionKey> visited;unsigned ordinal=0;
    for(const auto& [start,edges]:graph) {
        if(!visited.insert(start).second)continue;
        std::queue<PrecisionKey> pending;pending.push(start);std::vector<PrecisionKey> component;
        std::map<Index,Index> observations;bool conflict=false;unsigned anchors=0,targets=0;
        while(!pending.empty()) {
            const auto key=pending.front();pending.pop();component.push_back(key);
            const bool inserted=observations.emplace(key.first,key.second).second;
            conflict |= !inserted;
            if(inserted){if(stable[key.first])++anchors;else ++targets;}
            for(const auto& next:graph.at(key))if(visited.insert(next).second)pending.push(next);
        }
        if(!targets || !anchors)continue;
        for(const auto& [id,f]:observations)if(!stable[id])++result.support[id][std::min(anchors,3u)];
        if(ordinal++%5!=0)continue;
        result.reserved.insert(component.begin(),component.end());
        if(conflict){++result.ambiguous;continue;}
        if(anchors<3)continue;
        Track depth;for(const auto& [id,f]:observations)if(stable[id])depth.observations.push_back({id,f});
        // Leave one stable anchor out of each depth fit. Check ALL stable
        // observations, never target pixels, before selecting evaluation points.
        const auto check=depth.observations.back();depth.observations.pop_back();
        TriangulationOptions options;options.reproj_threshold_px=2;options.min_angle_deg=1;
        options.split_tracks=false;options.min_inliers=2;
        const auto expected=depth.observations.size();
        bool valid=triangulate_track(depth,scene,options)==expected;
        for(const auto& obs: [&] {auto all=depth.observations;all.push_back(check);return all;}()) {
            const auto& image=scene.images[obs.image_id];const auto& p=image.features.keypoints[obs.feature_id];
            const auto error=scene.camera_of(image).checked_reprojection_error(image.pose.transform_world_to_camera(depth.position),Vec2(p.x,p.y));
            valid &=std::isfinite(error) && error<=2;
        }
        if(!valid){++result.depth_rejected;continue;}
        for(const auto& [id,f]:observations)if(!stable[id])result.samples.push_back({id,f,depth.position,anchors});
    }
    return result;
}

inline void exclude_precision_partition(Scene& scene,const PrecisionPartition& split) {
    for(auto& pair:scene.pairs)std::erase_if(pair.matches,[&](const auto& m) {
        return split.reserved.contains({pair.id1,m.query}) || split.reserved.contains({pair.id2,m.train});
    });
    // Rebuild rather than trusting old 3D points with reserved ancestry.
    build_tracks(scene);
}

struct PrecisionScore {
    unsigned count{},inliers{},nonfinite{},cells{};double mean{},median{},p95{};
    bool accepted{};
};
inline PrecisionScore score_precision(const Scene& scene,Index target,const Pose3D& pose,const PrecisionPartition& split) {
    PrecisionScore result;std::vector<double> errors;std::set<std::pair<int,int>> cells;
    const auto& image=scene.images.at(target);const auto& camera=scene.camera_of(image);
    for(const auto& sample:split.samples)if(sample.target==target) {
        const auto& p=image.features.keypoints[sample.feature];
        double e=camera.checked_reprojection_error(pose.transform_world_to_camera(sample.point),Vec2(p.x,p.y));
        if(!std::isfinite(e)){++result.nonfinite;e=std::numeric_limits<double>::infinity();}
        errors.push_back(e);result.mean+=e;result.inliers+=e<=2;
        // Coverage belongs to the entire preselected set, not surviving inliers.
        cells.emplace(std::clamp(int(4*p.x/std::max(1u,camera.width)),0,3),std::clamp(int(4*p.y/std::max(1u,camera.height)),0,3));
    }
    result.count=static_cast<unsigned>(errors.size());result.cells=static_cast<unsigned>(cells.size());
    if(errors.empty())return result;
    std::sort(errors.begin(),errors.end());result.mean/=errors.size();
    result.median=errors[(errors.size()-1)/2];result.p95=errors[static_cast<std::size_t>(std::ceil(.95*errors.size()))-1];
    // Conservative experimental gate; not a calibrated production guarantee.
    result.accepted=result.count>=30 && !result.nonfinite && result.cells>=6 && result.mean<=2 && result.p95<=2;
    return result;
}

inline void probe_cross_anchor_precision(const Scene& parent,const std::filesystem::path& cache,std::ostream& out) {
    out<<"group,target,one_anchor_components,two_anchor_components,multi_anchor_components,raw_anchor_pairs,absent_anchor_pairs,reserved_observations,eligible_reserved_features,ambiguous_components,depth_rejected,variant,resection_accepted,evaluated,inliers_2px,nonfinite,cells,mean_px,median_px,p95_px,precision_accepted\n";
    std::map<std::filesystem::path,Index> ids;for(Index i=0;i<parent.images.size();++i)ids.emplace(parent.images[i].path,i);
    for(const auto& directory:std::filesystem::directory_iterator(cache)) {
        if(!directory.is_directory())continue;
        std::vector<RawPairMatches> raw;auto local=load_local_proposal_snapshot(directory.path(),raw);
        std::vector<std::uint8_t> stable;
        for(auto& image:local.images) {
            const auto& original=parent.images.at(ids.at(image.path));image.camera_id=original.camera_id;
            image.registered=original.registered;image.pose=original.registered?original.pose:Pose3D{};stable.push_back(original.registered);
        }
        local.cameras=parent.cameras;for(auto& camera:local.cameras)camera.trust_intrinsics=true;
        verify_local_raw_proposals(local,stable,raw,false);
        const auto split=partition_precision(local,stable);
        for(bool independent:{false,true}) {
            auto trial=local;
            if(independent) {
                // Pair rotations/directions also feed the resection checks.
                // Re-estimate them without reserved pixels, rather than retaining
                // relative poses that were fitted using validation measurements.
                auto fit_raw=raw;
                for(auto& pair:fit_raw)std::erase_if(pair.matches,[&](const auto& m) {
                    return split.reserved.contains({pair.id1,m.query}) || split.reserved.contains({pair.id2,m.train});
                });
                verify_local_raw_proposals(trial,stable,fit_raw,false);
                exclude_precision_partition(trial,split);
            } else build_tracks(trial);
            const auto recovered=recover_stable_resections(trial,stable);
            const std::set<Index> selected(recovered.begin(),recovered.end());
            for(Index id=0;id<local.images.size();++id)if(!stable[id]) {
                const auto found=split.support.find(id);const std::array<unsigned,4> support=found==split.support.end()?std::array<unsigned,4>{}:found->second;
                std::set<Index> raw_anchors;
                for(const auto& pair:raw) {
                    if(pair.id1==id && stable[pair.id2])raw_anchors.insert(pair.id2);
                    if(pair.id2==id && stable[pair.id1])raw_anchors.insert(pair.id1);
                }
                const auto total=static_cast<unsigned>(std::count(stable.begin(),stable.end(),std::uint8_t{1}));
                const auto eligible=std::count_if(split.samples.begin(),split.samples.end(),[&](const auto& sample){return sample.target==id;});
                PrecisionScore score;if(selected.contains(id))score=score_precision(trial,id,trial.images[id].pose,split);
                out<<directory.path().filename().string()<<','<<std::quoted(local.images[id].path.filename().string())<<','<<support[1]<<','<<support[2]<<','<<support[3]<<','
                    <<raw_anchors.size()<<','<<total-raw_anchors.size()<<','<<split.reserved.size()<<','<<eligible<<','<<split.ambiguous<<','<<split.depth_rejected<<','
                    <<(independent?"component_disjoint":"control_overlapping")<<','<<selected.contains(id)<<','<<score.count<<','<<score.inliers<<','<<score.nonfinite<<','<<score.cells<<','
                    <<score.mean<<','<<score.median<<','<<score.p95<<','<<(independent && score.accepted)<<'\n';
            }
            out.flush();
        }
    }
    if(!out)throw std::runtime_error("Cannot write precision report");
}
} // namespace photara::sfm::experiment
