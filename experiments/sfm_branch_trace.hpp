#pragma once
#include "sfm/scene.hpp"
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

// Fixed target identities from a completed control; no pose/reference enters the solve.
struct BranchTrace {
    std::set<std::string> names;
    std::filesystem::path output_prefix;
    std::ofstream views, edges;
    BranchTrace(const std::filesystem::path& targets, const std::filesystem::path& prefix) {
        output_prefix=prefix;
        if (std::filesystem::exists(prefix.string()+".views.csv") ||
            std::filesystem::exists(prefix.string()+".edges.csv")) throw std::runtime_error("Trace output exists");
        std::ifstream input(targets); std::string line;
        if (!input) throw std::runtime_error("Cannot read trace target audit");
        std::getline(input,line);
        while (std::getline(input,line)) {
            std::istringstream row(line); std::string id, name;
            std::getline(row,id,','); std::getline(row,name,',');
            if (!name.empty()) names.insert(name);
        }
        if (names.empty()) throw std::runtime_error("Empty trace targets");
        views.open(prefix.string()+".views.csv"); edges.open(prefix.string()+".edges.csv");
        if (!views || !edges) throw std::runtime_error("Cannot open branch trace");
        views << "stage,image,name,registered,boundary_active,boundary_inactive,boundary_matches,tracks,shared_tracks,triangulated,shared_inliers\n";
        edges << "stage,image,anchor,active,matches,weight,planar,angle_radians,cycle,connectivity\n";
    }
    void selection_audit(const photara::sfm::Scene& scene,
                         const std::vector<photara::sfm::Index>& selected, unsigned minimum_views) {
        using namespace photara::sfm;
        const auto path=output_prefix.string()+".selection.csv";
        if (std::filesystem::exists(path)) throw std::runtime_error("Selection audit exists");
        std::ofstream out(path); if (!out) throw std::runtime_error("Cannot open selection audit");
        struct C { unsigned all{},shared{},eligible{},eligible_shared{},selected{},selected_shared{},shared2{},shared3{}; };
        std::vector<C> counts(scene.images.size()); std::vector<bool> chosen(scene.tracks.size());
        for (Index id:selected) chosen[id]=true;
        for (Index id=0;id<scene.tracks.size();++id) {
            const auto& track=scene.tracks[id]; unsigned registered=0; bool outside=false;
            for (const auto& obs:track.observations) if (obs.image_id<scene.images.size() && scene.images[obs.image_id].registered) {
                ++registered; outside=outside || !names.contains(scene.images[obs.image_id].path.filename().string());
            }
            const bool eligible=track.observations.size()>=minimum_views && registered>=minimum_views;
            for (const auto& obs:track.observations) if (obs.image_id<scene.images.size() && scene.images[obs.image_id].registered) {
                auto& c=counts[obs.image_id]; ++c.all; c.shared+=outside; c.eligible+=eligible;
                c.eligible_shared+=eligible && outside; c.selected+=chosen[id]; c.selected_shared+=chosen[id] && outside;
                c.shared2+=outside && registered==2; c.shared3+=outside && registered==3;
            }
        }
        out << "image,name,registered,all,shared,eligible,eligible_shared,selected,selected_shared,shared_two_registered,shared_three_registered\n";
        for (Index id=0;id<scene.images.size();++id) if (names.contains(scene.images[id].path.filename().string())) {
            const auto& c=counts[id]; out << id << ',' << scene.images[id].path.filename().string() << ','
                << scene.images[id].registered << ',' << c.all << ',' << c.shared << ',' << c.eligible << ','
                << c.eligible_shared << ',' << c.selected << ',' << c.selected_shared << ',' << c.shared2 << ',' << c.shared3 << '\n';
        }
        if (!out) throw std::runtime_error("Selection audit write failed");
    }
    void selection(const photara::sfm::Scene& scene, const std::vector<photara::sfm::Index>& selected) {
        using namespace photara::sfm;
        std::vector<unsigned> counts(scene.images.size()), shared(scene.images.size());
        for (Index t:selected) {
            const auto& track=scene.tracks[t]; bool outside=false;
            for (const auto& obs:track.observations)
                if (scene.images[obs.image_id].registered && !names.contains(scene.images[obs.image_id].path.filename().string())) outside=true;
            for (const auto& obs:track.observations) if (scene.images[obs.image_id].registered) { ++counts[obs.image_id]; shared[obs.image_id]+=outside; }
        }
        for (Index id=0;id<scene.images.size();++id) if (names.contains(scene.images[id].path.filename().string()))
            views << "positioning_selection," << id << ',' << scene.images[id].path.filename().string() << ','
                << scene.images[id].registered << ",0,0,0," << counts[id] << ',' << shared[id] << ",0,0\n";
        views.flush();
    }
    void operator()(const char* stage, const photara::sfm::Scene& scene) {
        using namespace photara::sfm;
        struct Count { unsigned active{}, inactive{}, matches{}, tracks{}, shared{}, depth{}, inliers{}; };
        std::vector<Count> counts(scene.images.size());
        std::vector<bool> target(scene.images.size());
        for (Index id=0;id<scene.images.size();++id) target[id]=names.contains(scene.images[id].path.filename().string());
        for (const auto& p:scene.pairs) {
            if (!p.relative_pose || p.id1>=target.size() || p.id2>=target.size() || target[p.id1]==target[p.id2]) continue;
            const Index id=target[p.id1]?p.id1:p.id2, anchor=target[p.id1]?p.id2:p.id1;
            auto& c=counts[id]; ++(p.active?c.active:c.inactive); c.matches+=p.matches.size();
            edges << stage << ',' << id << ',' << anchor << ',' << p.active << ',' << p.matches.size() << ','
                << p.composite_weight() << ',' << p.degenerate_planar << ',' << p.mean_ray_angle << ','
                << p.weight_cycle << ',' << p.weight_connectivity << '\n';
        }
        for (const auto& t:scene.tracks) {
            bool outside=false;
            for (const auto& o:t.observations) if (o.image_id<target.size() && !target[o.image_id]) outside=true;
            bool outside_inlier=false;
            for (std::size_t i=0;i<std::min<std::size_t>(t.num_inliers,t.observations.size());++i)
                if (t.observations[i].image_id<target.size() && !target[t.observations[i].image_id] && scene.images[t.observations[i].image_id].registered) outside_inlier=true;
            for (std::size_t i=0;i<t.observations.size();++i) {
                const auto& o=t.observations[i]; if (o.image_id>=target.size() || !target[o.image_id]) continue;
                auto& c=counts[o.image_id]; ++c.tracks; c.shared+=outside;
                c.depth+=t.is_triangulated();
                c.inliers+=t.is_triangulated() && outside_inlier && i<t.num_inliers;
            }
        }
        for (Index id=0;id<target.size();++id) if (target[id]) {
            const auto& c=counts[id];
            views << stage << ',' << id << ',' << scene.images[id].path.filename().string() << ',' << scene.images[id].registered << ','
                << c.active << ',' << c.inactive << ',' << c.matches << ',' << c.tracks << ',' << c.shared << ',' << c.depth << ',' << c.inliers << '\n';
        }
        views.flush(); edges.flush();
        if (!views || !edges) throw std::runtime_error("Branch trace write failed");
    }
};
