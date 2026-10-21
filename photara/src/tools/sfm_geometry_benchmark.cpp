#include "sfm/asfm.hpp"
#include "sfm/gpu_geometry.hpp"
#include "sfm/tracks.hpp"
#include "parallel/thread_pool.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace photara::sfm;
namespace {
double seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
struct Difference { unsigned support{}, order{}; double max_position{}; };
Difference compare(const Scene& a,const Scene& b) {
    Difference d;
    for(std::size_t i=0;i<a.tracks.size();++i) {
        const auto& x=a.tracks[i]; const auto& y=b.tracks[i];
        d.support+=x.num_inliers!=y.num_inliers;
        if(x.is_triangulated()&&y.is_triangulated()) d.max_position=std::max(d.max_position,(x.position-y.position).norm());
        for(std::size_t j=0;j<x.observations.size();++j)
            if(x.observations[j].image_id!=y.observations[j].image_id||x.observations[j].feature_id!=y.observations[j].feature_id) { ++d.order; break; }
    }
    return d;
}
}
struct VerificationPair {
    Index first{}, second{};
    std::vector<Vec2> pixels1, pixels2;
};

// Gather real two-view correspondences from the shared tracks of consecutive
// views. The previous implementation only sampled every 32nd image index and
// produced a few dozen pairs, which did not exercise the batched CUDA filter.
std::uint64_t pair_key(Index first,Index second) {
    return (static_cast<std::uint64_t>(first)<<32)|second;
}
std::vector<VerificationPair> collect_verification_pairs(const Scene& scene,std::size_t limit) {
    std::unordered_map<std::uint64_t,unsigned> counts;
    const auto collect=[&](auto&& add) {
        for(const auto& track:scene.tracks) {
            if(track.observations.size()>255) continue;
            for(std::size_t i=0;i<track.observations.size();++i) {
                const auto first=track.observations[i];
                if(first.image_id>=scene.images.size()||!scene.images[first.image_id].registered||
                    first.feature_id>=scene.images[first.image_id].features.keypoints.size()) continue;
                for(std::size_t j=i+1;j<track.observations.size();++j) {
                    const auto second=track.observations[j];
                    if(second.image_id>=scene.images.size()||!scene.images[second.image_id].registered||
                        second.feature_id>=scene.images[second.image_id].features.keypoints.size()) continue;
                    const Index a=std::min(first.image_id,second.image_id);
                    const Index b=std::max(first.image_id,second.image_id);
                    if(b-a>3) continue;
                    add(a,b,first,second);
                }
            }
        }
    };
    collect([&](Index a,Index b,const Observation&,const Observation&) { ++counts[pair_key(a,b)]; });
    std::vector<std::pair<unsigned,std::uint64_t>> ranked;
    for(const auto& [key,count]:counts) if(count>=64) ranked.push_back({count,key});
    std::sort(ranked.begin(),ranked.end(),[](const auto& left,const auto& right) {
        if(left.first!=right.first) return left.first>right.first;
        return left.second<right.second;
    });
    if(ranked.size()>limit) ranked.resize(limit);
    std::unordered_map<std::uint64_t,VerificationPair> pairs;
    pairs.reserve(ranked.size());
    for(const auto& entry:ranked) {
        const Index first=static_cast<Index>(entry.second>>32);
        const Index second=static_cast<Index>(entry.second&0xffffffffu);
        pairs.emplace(entry.second,VerificationPair{first,second});
    }
    collect([&](Index a,Index b,const Observation& first,const Observation& second) {
        const auto found=pairs.find(pair_key(a,b)); if(found==pairs.end()) return;
        const auto& first_keypoint=scene.images[first.image_id].features.keypoints[first.feature_id];
        const auto& second_keypoint=scene.images[second.image_id].features.keypoints[second.feature_id];
        auto& pair=found->second;
        if(first.image_id==a) {
            pair.pixels1.emplace_back(first_keypoint.x,first_keypoint.y);
            pair.pixels2.emplace_back(second_keypoint.x,second_keypoint.y);
        } else {
            pair.pixels1.emplace_back(second_keypoint.x,second_keypoint.y);
            pair.pixels2.emplace_back(first_keypoint.x,first_keypoint.y);
        }
    });
    std::vector<VerificationPair> result; result.reserve(pairs.size());
    for(auto& [key,pair]:pairs) if(pair.pixels1.size()>=64) result.push_back(std::move(pair));
    return result;
}

int main(int argc,char** argv) {
    if(argc<2||argc>4) { std::cerr<<"Usage: photara_sfm_geometry_benchmark INPUT.asfm [TRACK_LIMIT [PAIR_LIMIT]]\n"; return 1; }
    if(!geometry_cuda_available()) { std::cerr<<"CUDA unavailable\n"; return 77; }
    try {
        auto source=load_asfm(argv[1]);
        if(argc>=3) source.tracks.resize(std::min(source.tracks.size(),static_cast<std::size_t>(std::stoull(argv[2]))));
        const std::size_t pair_limit=argc>=4?static_cast<std::size_t>(std::stoull(argv[3])):512;
        source.thread_count=photara::parallel::resolve_thread_count(0);
        std::cout<<"tracks="<<source.tracks.size()<<" images="<<source.images.size()<<'\n';
        Scene cpu=source,gpu=source;
        TriangulationOptions options; options.split_tracks=false;
        set_geometry_backend(GeometryBackend::cpu); auto start=std::chrono::steady_clock::now();
        auto cpu_count=triangulate_tracks(cpu,false,options); double cpu_tri=seconds(start);
        set_geometry_backend(GeometryBackend::cuda); start=std::chrono::steady_clock::now();
        auto gpu_count=triangulate_tracks(gpu,false,options); double gpu_tri=seconds(start);
        auto triangulation=compare(cpu,gpu);
        std::size_t short_tracks=0;
        for(const auto& track:source.tracks)
            if(track.is_valid()&&track.observations.size()<4) ++short_tracks;
        std::cout<<"triangulate_cpu_s="<<cpu_tri<<" triangulate_cuda_s="<<gpu_tri
            <<" cpu_tracks="<<cpu_count<<" cuda_tracks="<<gpu_count
            <<" short_tracks="<<short_tracks
            <<" support_differences="<<triangulation.support<<" order_differences="<<triangulation.order
            <<" max_position_difference="<<triangulation.max_position<<std::endl;
        cpu=source; gpu=source;
        set_geometry_backend(GeometryBackend::cpu); start=std::chrono::steady_clock::now();
        auto cpu_stats=filter_tracks(cpu,3.f,2.f,0.f,0.f); double cpu_filter=seconds(start);
        set_geometry_backend(GeometryBackend::cuda); start=std::chrono::steady_clock::now();
        auto gpu_stats=filter_tracks(gpu,3.f,2.f,0.f,0.f); double gpu_filter=seconds(start);
        auto filtering=compare(cpu,gpu);
        const double mean_error_difference=std::abs(cpu_stats.first-gpu_stats.first);
        std::cout<<"filter_cpu_s="<<cpu_filter<<" filter_cuda_s="<<gpu_filter
            <<" support_differences="<<filtering.support<<" order_differences="<<filtering.order
            <<" mean_error_difference="<<mean_error_difference<<std::endl;
        auto inputs=collect_verification_pairs(source,pair_limit);
        std::cout<<"verification_candidates="<<inputs.size()<<std::endl;
        std::vector<RelativePoseResult> fits[2]; double pair_times[2]{};
        for(int mode=0;mode<2;++mode) {
            set_geometry_backend(mode==0?GeometryBackend::cpu:GeometryBackend::cuda); fits[mode].resize(inputs.size());
            start=std::chrono::steady_clock::now();
            photara::parallel::parallel_for(inputs.size(),source.thread_count,[&](std::size_t k) {
                const auto& p=inputs[k];
                fits[mode][k]=estimate_relative_pose(p.pixels1,p.pixels2,
                    source.camera_of(source.images[p.first]),
                    source.camera_of(source.images[p.second]));
            });
            pair_times[mode]=seconds(start);
            std::cout<<"verification_mode="<<(mode==0?"cpu":"cuda")
                <<" elapsed_s="<<pair_times[mode]<<std::endl;
        }
        unsigned pair_differences=0;
        for(std::size_t i=0;i<inputs.size();++i)
            pair_differences+=fits[0][i].success!=fits[1][i].success||fits[0][i].inlier_mask!=fits[1][i].inlier_mask;
        std::cout<<"verification_pairs="<<inputs.size()<<" verification_cpu_s="<<pair_times[0]
            <<" verification_cuda_s="<<pair_times[1]<<" mask_differences="<<pair_differences<<'\n';
        const bool geometry_mismatch=triangulation.support||filtering.support||
            triangulation.order||filtering.order||pair_differences||
            triangulation.max_position>1e-5||mean_error_difference>1e-4;
        return geometry_mismatch?2:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
