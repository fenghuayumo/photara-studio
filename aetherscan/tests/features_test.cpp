#include "features/compat.hpp"
#include "features/features.hpp"
#include "features/registry.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <future>

namespace {
using namespace aetherscan::features;
void same_matches(const MatchSet& a, const MatchSet& b) {
    auto keys = [](const MatchSet& m) {
        std::vector<std::pair<FeatureIndex, FeatureIndex>> v;
        for (auto x : m.matches) v.emplace_back(x.query,x.train);
        std::sort(v.begin(),v.end()); return v;
    };
    if (keys(a) != keys(b)) throw std::runtime_error("CUDA matcher correspondence mismatch");
}
MatchSet oracle(const FeatureSet& q, const FeatureSet& t, bool mutual) {
    auto direction = [](const FeatureSet& a, const FeatureSet& b) {
        std::vector<int> out(a.keypoints.size(),-1);
        for (std::size_t i=0;i<a.keypoints.size();++i) {
            int best=0,second=0,index=-1;
            for (std::size_t j=0;j<b.keypoints.size();++j) {
                int dot=0;
                for (int k=0;k<128;++k) dot+=int(a.descriptors_u8[i*128+k])*b.descriptors_u8[j*128+k];
                if (dot>best) { second=best;best=dot;index=static_cast<int>(j); }
                else second=std::max(second,dot);
            }
            const float d=static_cast<float>(std::acos(std::min(best/262144.,1.)));
            const float d2=static_cast<float>(std::acos(std::min(second/262144.,1.)));
            if (d<.7f && d<.8f*d2) out[i]=index;
        }
        return out;
    };
    auto rows=direction(q,t),cols=direction(t,q);
    MatchSet out;
    for (std::size_t i=0;i<rows.size();++i)
        if (rows[i]>=0 && (!mutual || cols[rows[i]]==int(i)))
            out.matches.push_back({FeatureIndex(i),FeatureIndex(rows[i]),1.f});
    return out;
}
void check_native(FeatureSet q, FeatureSet t) {
    q.compress_descriptors_u8(); t.compress_descriptors_u8();
    for (bool mutual : {false,true}) {
        SiftGpuMatcherOptions options; options.mutual_check=mutual;
        SiftGpuMatcher native(options);
        options.native_cuda=false; SiftGpuMatcher legacy(options);
        same_matches(native.match(q,t),legacy.match(q,t));
        FeatureSet small=q, tail=t, empty=q;
        small.keypoints.resize(33); small.descriptors_u8.resize(33*128); small.mark_descriptors_modified();
        tail.keypoints.resize(65); tail.descriptors_u8.resize(65*128); tail.mark_descriptors_modified();
        empty.keypoints.clear(); empty.descriptors_u8.clear(); empty.mark_descriptors_modified();
        // Exact ties, zero descriptors, and tile tails exercise both top-two reductions.
        std::copy_n(small.descriptors_u8.begin(),128,tail.descriptors_u8.begin());
        std::copy_n(small.descriptors_u8.begin(),128,tail.descriptors_u8.begin()+128);
        std::vector<FeatureMatcher::Pair> batch{{&small,&tail},{&tail,&small},{&empty,&small},{&small,&empty}};
        for (int i=0;i<9;++i) batch.emplace_back(&small,&tail);
        auto out=native.match_batch(batch);
        for (std::size_t i=0;i<batch.size();++i) same_matches(out[i],oracle(*batch[i].first,*batch[i].second,mutual));
        std::fill(small.descriptors_u8.begin(),small.descriptors_u8.end(),0);
        small.mark_descriptors_modified();
        same_matches(native.match(small,tail),oracle(small,tail,mutual));
        native.clear_prepared();
        same_matches(native.match(q,t),legacy.match(q,t));
        // Office pair 816/818 exposed this one-ULP angular-ratio boundary.
        // Reproduce its two exact dot products without requiring the dataset.
        FeatureSet boundary_q, boundary_t;
        boundary_q.image_width=boundary_t.image_width=64;
        boundary_q.image_height=boundary_t.image_height=64;
        boundary_q.storage=boundary_t.storage=DescriptorStorage::uint8;
        boundary_q.descriptor_dimension=boundary_t.descriptor_dimension=128;
        boundary_q.keypoints.resize(1); boundary_t.keypoints.resize(2);
        boundary_q.descriptors_u8.assign(128,45); boundary_q.descriptors_u8[0]=46;
        boundary_t.descriptors_u8.resize(256);
        for (int j=0;j<2;++j) {
            const int dot=j==0 ? 248262 : 240562;
            const int first=dot%45, rest=(dot-first)/45-first;
            boundary_t.descriptors_u8[j*128]=static_cast<std::uint8_t>(first);
            for (int k=0;k<127;++k)
                boundary_t.descriptors_u8[j*128+k+1]=static_cast<std::uint8_t>(rest/127+(k<rest%127));
        }
        const auto boundary=native.match(boundary_q,boundary_t);
        same_matches(boundary,legacy.match(boundary_q,boundary_t));
        if (boundary.matches.size()!=1) throw std::runtime_error("Angular ratio boundary rejected");
    }
}
}

int main() {
    try {
    aetherscan::features::ensure_builtin_feature_backends();
    if (!aetherscan::features::has_extractor("aliked") ||
        !aetherscan::features::extractor_matcher_compatible(
            "aliked", "lightglue") ||
        !aetherscan::features::extractor_matcher_compatible(
            "sift", "lightglue") ||
        !aetherscan::features::extractor_matcher_compatible(
            "siftgpu", "lightglue") ||
        !aetherscan::features::extractor_matcher_compatible(
            "siftgpu", "hybrid_lightglue") ||
        aetherscan::features::extractor_matcher_compatible(
            "sift", "hybrid_lightglue") ||
        aetherscan::features::extractor_matcher_compatible(
            "aliked", "mutual_ratio"))
        return 5;

    constexpr std::uint32_t width = 512, height = 384;
    std::vector<std::uint8_t> first(static_cast<std::size_t>(width) * height);
    std::vector<std::uint8_t> second(first.size(), 0);
    std::mt19937 random(1234);
    for (auto& pixel : first) pixel = static_cast<std::uint8_t>(random() & 0xffU);
    constexpr std::uint32_t shift_x = 7, shift_y = 5;
    for (std::uint32_t y = shift_y; y < height; ++y)
        for (std::uint32_t x = shift_x; x < width; ++x)
            second[static_cast<std::size_t>(y) * width + x] =
                first[static_cast<std::size_t>(y - shift_y) * width + x - shift_x];
    aetherscan::features::SiftOptions options;
    options.maximum_features = 3000;
    aetherscan::features::SiftExtractor extractor(options);
    const auto features0 = extractor.extract_gray(first, width, height);
    const auto features1 = extractor.extract_gray(second, width, height);
    const auto matches = aetherscan::features::match_descriptors(features0, features1);
    std::cout << "features=" << features0.keypoints.size() << ","
              << features1.keypoints.size() << " matches=" << matches.matches.size() << '\n';
    if (features0.keypoints.size() < 150 || features1.keypoints.size() < 150 ||
        matches.matches.size() < 80) return 1;
    for (const auto& match : matches.matches) {
        if (match.query >= features0.keypoints.size() || match.train >= features1.keypoints.size())
            return 2;
    }
    aetherscan::features::DescriptorMatcherOptions ann_options;
    ann_options.ann_min_features = 1;
    aetherscan::features::MutualRatioMatcher prepared_matcher(ann_options);
    prepared_matcher.prepare(features0);
    aetherscan::features::FeatureSet resized_train = features1;
    prepared_matcher.prepare(resized_train);
    resized_train.keypoints.resize(32);
    resized_train.descriptors.resize(
        resized_train.keypoints.size() * resized_train.descriptor_dimension);
    resized_train.mark_descriptors_modified();
    const auto resized_matches =
        prepared_matcher.match(features0, resized_train);
    for (const auto& match : resized_matches.matches) {
        if (match.query >= features0.keypoints.size() ||
            match.train >= resized_train.keypoints.size())
            return 3;
    }
    if (aetherscan::features::SiftGpuExtractor::is_built()) {
        aetherscan::features::SiftGpuOptions gpu_options;
        gpu_options.maximum_features = 3000;
        aetherscan::features::SiftGpuExtractor gpu_extractor(gpu_options);
        aetherscan::features::SiftGpuMatcher gpu_matcher;
        if (gpu_extractor.is_available() && gpu_matcher.is_available()) {
            const auto gpu_features0 =
                gpu_extractor.extract_gray(first, width, height);
            const auto gpu_features1 =
                gpu_extractor.extract_gray(second, width, height);
            auto deferred = gpu_extractor.extract_gray_deferred(first, width, height);
            if (deferred.metric != aetherscan::features::DescriptorMetric::l2) return 11;
            auto synchronous = deferred;
            gpu_extractor.finalize_descriptors(synchronous);
            auto finalized = std::async(std::launch::async, [&] {
                gpu_extractor.finalize_descriptors(deferred);
            });
            const auto concurrent = gpu_extractor.extract_gray(second, width, height);
            finalized.get();
            if (concurrent.keypoints.empty() || deferred.metric != synchronous.metric ||
                deferred.descriptors != synchronous.descriptors ||
                deferred.keypoints.size() != synchronous.keypoints.size()) return 12;
            const auto normalized = deferred.descriptors;
            gpu_extractor.finalize_descriptors(deferred);
            if (deferred.descriptors != normalized) return 13;
            const auto gpu_matches =
                gpu_matcher.match(gpu_features0, gpu_features1);
            std::cout << " gpu_features=" << gpu_features0.keypoints.size()
                      << "," << gpu_features1.keypoints.size()
                      << " gpu_matches=" << gpu_matches.matches.size()
                      << '\n';
            if (gpu_features0.keypoints.size() < 150 ||
                gpu_features1.keypoints.size() < 150 ||
                gpu_matches.matches.size() < 80)
                return 4;
            check_native(gpu_features0,gpu_features1);
        }
    }
    return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 10;
    }
}
