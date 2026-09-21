#include "features/features.hpp"
#include "sfm/checkpoint.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) throw std::invalid_argument(
            "Usage: photara_sfm_match_benchmark features-HEX.bin matches-HEX.bin native|legacy pair-limit(0=all) [start-index]");
        using namespace photara;
        auto key=[](const std::filesystem::path& p) {
            const auto stem=p.stem().string(); return std::stoull(stem.substr(stem.find('-')+1),nullptr,16);
        };
        sfm::Scene scene;
        std::filesystem::path feature_file(argv[1]),match_file(argv[2]);
        sfm::CheckpointOptions cache; cache.directory=feature_file.parent_path(); cache.write=false;
        if (!sfm::CheckpointStore(cache).load_scene(sfm::CheckpointStage::features,key(feature_file),scene))
            throw std::runtime_error("Cannot read features checkpoint");
        cache.directory=match_file.parent_path();
        std::vector<sfm::RawPairMatches> reference;
        if (!sfm::CheckpointStore(cache).load_matches(key(match_file),reference))
            throw std::runtime_error("Cannot read matches checkpoint");
        const std::string backend(argv[3]);
        if (backend!="native" && backend!="legacy") throw std::invalid_argument("Unknown backend");
        features::SiftGpuMatcherOptions options; options.native_cuda=backend=="native";
        for (const auto& image:scene.images) options.maximum_features=std::max(options.maximum_features,image.features.keypoints.size());
        features::SiftGpuMatcher matcher(options);
        const auto limit=std::stoull(argv[4]);
        const std::size_t first=argc==6 ? std::stoull(argv[5]) : 0;
        if (first>=reference.size()) throw std::invalid_argument("Start index out of range");
        const auto count=limit ? std::min<std::size_t>(limit,reference.size()-first) : reference.size()-first;
        std::vector<std::size_t> selected;
        for (std::size_t i=0;i<count;++i) selected.push_back(argc==6 ? first+i : i*reference.size()/count);
        std::size_t differences=0,matches=0;
        double seconds=0;
        for (std::size_t begin=0;begin<count;begin+=8) {
            std::vector<features::FeatureMatcher::Pair> batch;
            const auto end=std::min(begin+8,count);
            for (std::size_t i=begin;i<end;++i) {
                const auto& p=reference[selected[i]];
                batch.emplace_back(&scene.images.at(p.id1).features,&scene.images.at(p.id2).features);
            }
            const auto start=std::chrono::steady_clock::now();
            auto result=matcher.match_batch(batch);
            seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            for (std::size_t i=begin;i<end;++i) {
                const auto& actual=result[i-begin].matches;
                const auto& expected=reference[selected[i]].matches;
                matches+=actual.size();
                if (actual.size()!=expected.size() || !std::equal(actual.begin(),actual.end(),expected.begin(),
                    [](auto a,auto b){return a.query==b.query && a.train==b.train;})) {
                    ++differences;
                    const auto& p=reference[selected[i]];
                    std::cout<<"difference pair_index="<<selected[i]<<" images="<<p.id1<<","<<p.id2
                             <<" expected="<<expected.size()<<" actual="<<actual.size()<<std::endl;
                    auto show = [&](const auto& a,const auto& b) {
                        for (const auto m:a) if (std::none_of(b.begin(),b.end(),[&](auto n){return n.query==m.query && n.train==m.train;})) {
                            std::cout<<"  differing correspondence="<<m.query<<","<<m.train;
                            for (bool reverse:{false,true}) {
                                const auto& q=scene.images[reverse?p.id2:p.id1].features;
                                const auto& t=scene.images[reverse?p.id1:p.id2].features;
                                const auto qi=reverse?m.train:m.query;
                                int best=0,second=0;
                                for (std::size_t ti=0;ti<t.keypoints.size();++ti) {
                                    int dot=0;
                                    for (int k=0;k<128;++k) dot+=int(q.descriptors_u8[qi*128+k])*t.descriptors_u8[ti*128+k];
                                    if (dot>best) {second=best;best=dot;} else second=std::max(second,dot);
                                }
                                std::cout<<std::setprecision(12)<<" top2="<<best<<","<<second
                                    <<" angle_ratio="<<std::acos(std::min(best/262144.,1.))/std::acos(std::min(second/262144.,1.));
                            }
                            std::cout<<std::endl;
                        }
                    };
                    show(actual,expected); show(expected,actual);
                }
            }
            if (end%1024==0) std::cout<<"progress pairs="<<end<<" seconds="<<seconds<<" differences="<<differences<<std::endl;
        }
        std::cout<<"backend="<<backend<<" pairs="<<count<<" matches="<<matches
                 <<" seconds="<<seconds<<" different_pairs="<<differences<<std::endl;
        return differences ? 2 : 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
}
