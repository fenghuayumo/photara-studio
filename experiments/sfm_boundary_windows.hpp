#pragma once
#include "sfm_ray_group_alignment.hpp"
#include <fstream>

namespace photara::sfm::experiment {
inline void probe_boundary_windows(const Scene& scene,const std::vector<std::uint8_t>& stable,
    const std::filesystem::path& directory) {
    if(!std::filesystem::create_directory(directory))throw std::invalid_argument("Window diagnostic directory already exists");
    std::vector<Index> targets;for(Index i=0;i<stable.size();++i)if(!stable[i])targets.push_back(i);
    std::ofstream summary(directory/"windows.csv");
    summary<<"window,targets,excluded_anchor,tracks,fit_observations,heldout_observations,anchors,fit_inliers,accepted,reason\n";
    const auto attempt=[&](const std::string& label,const std::vector<Index>& ids,Index excluded=k_invalid) {
        auto trial=scene;std::ofstream residuals(directory/(label+".csv"));
        const auto report=recover_ray_group(trial,ids,stable,false,&residuals,excluded);
        summary<<label<<','<<ids.size()<<','<<std::quoted(excluded==k_invalid?"":scene.images[excluded].path.filename().string())<<','
            <<report.tracks<<','<<report.fit_observations<<','<<report.heldout_observations<<','<<report.anchors<<','<<report.fit_inliers<<','
            <<report.accepted<<','<<report.reason<<'\n';summary.flush();
    };
    attempt("whole",targets);
    if(targets.size()>6)for(unsigned start=0;start+4<=targets.size();start+=3) {
        const auto end=std::min<std::size_t>(start+6,targets.size());
        attempt("window_"+std::to_string(start)+"_"+std::to_string(end),std::vector<Index>(targets.begin()+start,targets.begin()+end));
    }
    std::map<Index,unsigned> support;
    for(const auto& pair:scene.pairs)if(pair.active && pair.relative_pose && stable[pair.id1]!=stable[pair.id2])
        support[stable[pair.id1]?pair.id1:pair.id2]+=static_cast<unsigned>(pair.matches.size());
    std::vector<std::pair<Index,unsigned>> anchors(support.begin(),support.end());
    std::stable_sort(anchors.begin(),anchors.end(),[](const auto& a,const auto& b){return a.second>b.second;});
    if(anchors.size()>12)anchors.resize(12);
    for(const auto& [id,n]:anchors)attempt("without_anchor_"+std::to_string(id),targets,id);
    if(!summary)throw std::runtime_error("Cannot write boundary-window summary");
}
} // namespace photara::sfm::experiment
