#pragma once
#include "sfm/scene.hpp"
#include <fstream>
#include <map>
#include <set>
#include <unordered_map>

// Read-only topology audit. Counts potential links, never certifies a pose.
inline void probe_boundary_support(const photara::sfm::Scene& scene, std::ostream& out) {
    using namespace photara::sfm;
    struct Counts {
        std::set<Index> active, inactive, internal;
        std::size_t matches{};
        std::map<Index,std::map<Index,std::set<Index>>> links;
    };
    std::vector<Counts> counts(scene.images.size());
    const auto key = [](Index image, Index feature) { return (std::uint64_t{image}<<32)|feature; };
    std::unordered_map<std::uint64_t,Index> needed;
    for (const auto& pair : scene.pairs) {
        if (!pair.relative_pose || pair.matches.empty()) continue;
        for (const auto [target,anchor] : {std::pair{pair.id1,pair.id2},std::pair{pair.id2,pair.id1}}) {
            if (target>=scene.images.size() || anchor>=scene.images.size() || scene.images[target].registered) continue;
            if (!scene.images[anchor].registered) { counts[target].internal.insert(anchor); continue; }
            (pair.active ? counts[target].active : counts[target].inactive).insert(anchor);
            counts[target].matches += pair.matches.size();
            for (const auto& match : pair.matches)
                needed.emplace(key(anchor,target==pair.id1 ? match.train : match.query),k_invalid);
        }
    }
    std::vector<unsigned> depth_views(scene.tracks.size());
    for (Index t=0;t<scene.tracks.size();++t) {
        const auto& track=scene.tracks[t];
        if (!track.is_triangulated()) continue;
        std::set<Index> anchors;
        for (std::size_t o=0;o<std::min<std::size_t>(track.num_inliers,track.observations.size());++o) {
            const auto& obs=track.observations[o];
            if (obs.image_id>=scene.images.size() || !scene.images[obs.image_id].registered) continue;
            anchors.insert(obs.image_id);
            if (const auto it=needed.find(key(obs.image_id,obs.feature_id));it!=needed.end()) {
                if (it->second==k_invalid) it->second=t;
                else if (it->second!=t) it->second=k_invalid-1; // ambiguous duplicate landmark
            }
        }
        depth_views[t]=static_cast<unsigned>(anchors.size());
    }
    for (const auto& pair:scene.pairs) {
        if (!pair.relative_pose || pair.matches.empty() || pair.id1>=scene.images.size() || pair.id2>=scene.images.size()) continue;
        const bool forward=!scene.images[pair.id1].registered && scene.images[pair.id2].registered;
        const bool reverse=scene.images[pair.id1].registered && !scene.images[pair.id2].registered;
        if (!forward && !reverse) continue;
        const Index target=forward?pair.id1:pair.id2, anchor=forward?pair.id2:pair.id1;
        for (const auto& match:pair.matches) {
            const Index feature=forward?match.query:match.train;
            const auto it=needed.find(key(anchor,forward?match.train:match.query));
            if (it!=needed.end() && it->second<scene.tracks.size())
                counts[target].links[feature][it->second].insert(anchor);
        }
    }
    out << "image,name,active_anchor_neighbors,inactive_anchor_neighbors,missing_neighbors,boundary_matches,features_linked_to_map,unambiguous_features,features_with_two_anchor_links,features_with_three_view_depth\n";
    for (Index id=0;id<scene.images.size();++id) {
        if (scene.images[id].registered) continue;
        const auto& c=counts[id];
        unsigned unique=0,cycles=0,depth=0;
        for (const auto& [feature,tracks]:c.links) {
            if (tracks.size()!=1) continue;
            ++unique;
            const auto& [track,anchors]=*tracks.begin();
            cycles+=anchors.size()>=2;
            depth+=depth_views[track]>=3;
        }
        out << id << ',' << scene.images[id].path.filename().string() << ',' << c.active.size() << ','
            << c.inactive.size() << ',' << c.internal.size() << ',' << c.matches << ',' << c.links.size()
            << ',' << unique << ',' << cycles << ',' << depth << '\n';
    }
}
