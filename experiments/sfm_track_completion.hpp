#pragma once
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"
#include <map>
#include <set>
#include <queue>

namespace photara::sfm::experiment {
struct CompletionReport {unsigned added{},merged{},retriangulated{},conflicts{},projection_rejected{};};
using CompletionKey=std::pair<Index,Index>;

inline bool completion_projects(const Scene& scene,const Observation& obs,const Vec3& point) {
    if(obs.image_id>=scene.images.size())return false;
    const auto& image=scene.images[obs.image_id];
    if(!image.registered || obs.feature_id>=image.features.keypoints.size())return false;
    const auto& kp=image.features.keypoints[obs.feature_id];
    return scene.camera_of(image).checked_reprojection_error(image.pose.transform_world_to_camera(point),Vec2(kp.x,kp.y))<=2;
}

// Proposal graph and pose support remain distinct: only current registered
// cameras can contribute depth. No target pose is inferred by graph traversal.
inline CompletionReport complete_stable_tracks(Scene& scene,unsigned max_depth=5) {
    CompletionReport report;
    std::map<CompletionKey,std::vector<CompletionKey>> graph;
    for(const auto& pair:scene.pairs) {
        if(!pair.active || !pair.has_geometry() || pair.zero_baseline)continue;
        if(pair.id1>=scene.images.size() || pair.id2>=scene.images.size())throw std::invalid_argument("Invalid completion pair");
        for(const auto& m:pair.matches) {
            if(m.query>=scene.images[pair.id1].features.keypoints.size() || m.train>=scene.images[pair.id2].features.keypoints.size())
                throw std::invalid_argument("Invalid completion feature");
            const CompletionKey a{pair.id1,m.query},b{pair.id2,m.train};
            graph[a].push_back(b);graph[b].push_back(a);
        }
    }
    std::map<CompletionKey,Index> owner;
    for(Index t=0;t<scene.tracks.size();++t) {
        const auto& track=scene.tracks[t];if(!track.is_triangulated())continue;
        for(unsigned o=0;o<std::min<std::size_t>(track.num_inliers,track.observations.size());++o) {
            const auto& obs=track.observations[o];
            if(!completion_projects(scene,obs,track.position))continue;
            auto [it,added]=owner.emplace(CompletionKey{obs.image_id,obs.feature_id},t);
            if(!added && it->second!=t)it->second=k_invalid;
        }
    }
    for(Index t=0;t<scene.tracks.size();++t) {
        auto& track=scene.tracks[t];if(!track.is_triangulated())continue;
        std::queue<std::pair<CompletionKey,unsigned>> pending;std::set<CompletionKey> visited;
        for(unsigned o=0;o<track.num_inliers;++o) {
            const auto& obs=track.observations[o];CompletionKey key{obs.image_id,obs.feature_id};
            const auto found=owner.find(key);
            if(found!=owner.end() && found->second==t && visited.insert(key).second)pending.push({key,0});
        }
        while(!pending.empty()) {
            const auto [key,depth]=pending.front();pending.pop();if(depth>=max_depth)continue;
            const auto found=graph.find(key);if(found==graph.end())continue;
            for(const auto& next:found->second) {
                if(!visited.insert(next).second)continue;
                const Observation obs{next.first,next.second};
                if(!scene.images[obs.image_id].registered)continue;
                const auto occupied=owner.find(next);
                if(occupied!=owner.end() && occupied->second!=t) {
                    const Index other_id=occupied->second;
                    if(other_id==k_invalid){++report.conflicts;continue;}
                    auto& other=scene.tracks[other_id];if(!other.is_triangulated())continue;
                    const Vec3 point=(double(track.num_inliers)*track.position+double(other.num_inliers)*other.position)/double(track.num_inliers+other.num_inliers);
                    std::set<Index> images;bool valid=true;
                    // Preserve tails too: never create two observations in one image.
                    for(const auto& o:track.observations)valid &=images.insert(o.image_id).second;
                    for(const auto& o:other.observations)valid &=images.insert(o.image_id).second;
                    if(!valid || unsigned(track.num_inliers)+other.num_inliers>255){++report.conflicts;continue;}
                    for(unsigned o=0;o<track.num_inliers;++o)valid &=completion_projects(scene,track.observations[o],point);
                    for(unsigned o=0;o<other.num_inliers;++o)valid &=completion_projects(scene,other.observations[o],point);
                    if(!valid){++report.projection_rejected;continue;}
                    const auto old=track.num_inliers;
                    track.observations.insert(track.observations.begin()+old,other.observations.begin(),other.observations.begin()+other.num_inliers);
                    track.num_inliers=static_cast<std::uint8_t>(old+other.num_inliers);track.position=point;
                    track.observations.insert(track.observations.end(),other.observations.begin()+other.num_inliers,other.observations.end());
                    for(unsigned o=old;o<track.num_inliers;++o) {
                        const auto& ob=track.observations[o];const CompletionKey nk{ob.image_id,ob.feature_id};
                        owner[nk]=t;if(visited.insert(nk).second)pending.push({nk,depth+1});
                    }
                    other.observations.clear();other.num_inliers=0;++report.merged;
                    pending.push({next,depth+1});continue;
                }
                if(occupied!=owner.end()){pending.push({next,depth+1});continue;}
                const auto same=std::find_if(track.observations.begin(),track.observations.end(),[&](const auto& o){return o.image_id==obs.image_id;});
                if(same!=track.observations.end() && same->feature_id!=obs.feature_id){++report.conflicts;continue;}
                if(!completion_projects(scene,obs,track.position)){++report.projection_rejected;continue;}
                if(track.num_inliers==255)continue;
                if(same!=track.observations.end())track.observations.erase(same);
                track.observations.insert(track.observations.begin()+track.num_inliers,obs);++track.num_inliers;
                owner[next]=t;++report.added;pending.push({next,depth+1});
            }
        }
    }
    // Revisit only untriangulated candidate tracks. Existing stable points are
    // preserved except for the explicitly checked merges above.
    TriangulationOptions options;options.reproj_threshold_px=2;options.min_angle_deg=1;
    options.min_inliers=3;options.split_tracks=false;
    for(Index t=0;t<scene.tracks.size();++t) {
        auto& track=scene.tracks[t];if(track.is_triangulated() || track.observations.size()<3)continue;
        Track candidate=track;
        std::erase_if(candidate.observations,[&](const auto& obs) {
            const auto it=owner.find({obs.image_id,obs.feature_id});return it!=owner.end();
        });
        if(candidate.observations.size()<3 || !triangulate_track(candidate,scene,options))continue;
        bool free=true;for(unsigned o=0;o<candidate.num_inliers;++o)
            free &=!owner.contains({candidate.observations[o].image_id,candidate.observations[o].feature_id});
        if(!free)continue;
        track=std::move(candidate);++report.retriangulated;
        for(unsigned o=0;o<track.num_inliers;++o)owner[{track.observations[o].image_id,track.observations[o].feature_id}]=t;
    }
    rebuild_track_index(scene);return report;
}

inline unsigned stable_depth_support(const Scene& scene,Index target) {
    std::map<CompletionKey,std::set<Index>> landmarks;
    for(Index t=0;t<scene.tracks.size();++t) {
        const auto& track=scene.tracks[t];if(!track.is_triangulated())continue;
        for(unsigned o=0;o<track.num_inliers;++o) {
            const auto& obs=track.observations[o];
            if(completion_projects(scene,obs,track.position))landmarks[{obs.image_id,obs.feature_id}].insert(t);
        }
    }
    std::map<Index,std::set<Index>> candidates;
    for(const auto& pair:scene.pairs) {
        if(!pair.active || !pair.has_geometry() || (pair.id1!=target && pair.id2!=target))continue;
        const bool first=pair.id1==target;const Index anchor=first?pair.id2:pair.id1;
        if(!scene.images[anchor].registered)continue;
        for(const auto& m:pair.matches) {
            const auto found=landmarks.find({anchor,first?m.train:m.query});
            if(found!=landmarks.end())candidates[first?m.query:m.train].insert(found->second.begin(),found->second.end());
        }
    }
    std::map<Index,unsigned> unique;
    for(const auto& [f,tracks]:candidates)if(tracks.size()==1)++unique[*tracks.begin()];
    unsigned count=0;for(const auto& [t,n]:unique)count+=n==1;return count;
}
} // namespace photara::sfm::experiment
