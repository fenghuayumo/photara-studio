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
using namespace photara::features;
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
    photara::features::ensure_builtin_feature_backends();
    if (!photara::features::has_extractor("aliked") ||
        !photara::features::extractor_matcher_compatible(
            "aliked", "lightglue") ||
        !photara::features::extractor_matcher_compatible(
            "sift", "lightglue") ||
        !photara::features::extractor_matcher_compatible(
            "siftgpu", "lightglue") ||
        !photara::features::extractor_matcher_compatible(
            "siftgpu", "hybrid_lightglue") ||
        photara::features::extractor_matcher_compatible(
            "sift", "hybrid_lightglue") ||
        photara::features::extractor_matcher_compatible(
            "aliked", "mutual_ratio"))
        return 5;

    photara::features::FeatureSet mask_probe;
    mask_probe.image_width = mask_probe.image_height = 4;
    mask_probe.descriptor_dimension = 2;
    mask_probe.keypoints = {
        {0.5F, 0.5F}, {2.5F, 0.5F},
        {0.5F, 2.5F}, {2.5F, 2.5F}};
    mask_probe.descriptors = {0.F, 1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F};
    const std::vector<std::uint8_t> valid_mask{255, 0, 0, 255};
    const auto masked_float = photara::features::filter_features_by_mask(
        mask_probe, valid_mask, 2, 2);
    if (masked_float.keypoints.size() != 2 ||
        masked_float.descriptors != std::vector<float>({0.F, 1.F, 6.F, 7.F}))
        return 14;
    mask_probe.compress_descriptors_u8();
    const auto masked_u8 = photara::features::filter_features_by_mask(
        mask_probe, valid_mask, 2, 2);
    if (masked_u8.keypoints.size() != 2 ||
        masked_u8.descriptors_u8.size() != 4)
        return 15;

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
    photara::features::SiftOptions options;
    options.maximum_features = 3000;
    photara::features::SiftExtractor extractor(options);
    const auto features0 = extractor.extract_gray(first, width, height);
    const auto features1 = extractor.extract_gray(second, width, height);
    if (features0.metric != extractor.info().metric ||
        features0.metric != photara::features::DescriptorMetric::l2_root ||
        features1.metric != features0.metric)
        return 20;
    const auto matches = photara::features::match_descriptors(features0, features1);
    std::cout << "features=" << features0.keypoints.size() << ","
              << features1.keypoints.size() << " matches=" << matches.matches.size() << '\n';
    if (features0.keypoints.size() < 150 || features1.keypoints.size() < 150 ||
        matches.matches.size() < 80) return 1;
    int shift_inliers = 0;
    for (const auto& match : matches.matches) {
        if (match.query >= features0.keypoints.size() || match.train >= features1.keypoints.size())
            return 2;
        const auto& a = features0.keypoints[match.query];
        const auto& b = features1.keypoints[match.train];
        if (std::abs((b.x - a.x) - static_cast<float>(shift_x)) < 3.0F &&
            std::abs((b.y - a.y) - static_cast<float>(shift_y)) < 3.0F)
            ++shift_inliers;
    }
    std::cout << " cpu_shift_inliers=" << shift_inliers << "/"
              << matches.matches.size() << '\n';
    if (shift_inliers < 50) return 16;
    {
        const auto again = extractor.extract_gray(first, width, height);
        if (again.keypoints.size() != features0.keypoints.size() ||
            again.descriptors != features0.descriptors)
            return 17;
    }
    photara::features::DescriptorMatcherOptions ann_options;
    ann_options.ann_min_features = 1;
    photara::features::MutualRatioMatcher prepared_matcher(ann_options);
    prepared_matcher.prepare(features0);
    photara::features::FeatureSet resized_train = features1;
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
    if (photara::features::SiftGpuExtractor::is_built()) {
        photara::features::SiftGpuOptions gpu_options;
        gpu_options.maximum_features = 3000;
        photara::features::SiftGpuExtractor gpu_extractor(gpu_options);
        photara::features::SiftGpuMatcher gpu_matcher;
        if (gpu_extractor.is_available() && gpu_matcher.is_available()) {
            const auto gpu_features0 =
                gpu_extractor.extract_gray(first, width, height);
            const auto gpu_features1 =
                gpu_extractor.extract_gray(second, width, height);
            auto deferred = gpu_extractor.extract_gray_deferred(first, width, height);
            if (deferred.metric != photara::features::DescriptorMetric::l2) return 11;
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

            std::vector<std::uint8_t> textured(static_cast<std::size_t>(width) * height);
            std::mt19937 textured_random(20260929);
            std::uniform_real_distribution<float> unit(0.0F, 1.0F);
            for (std::uint32_t y = 0; y < height; ++y) {
                for (std::uint32_t x = 0; x < width; ++x) {
                    const float fx = static_cast<float>(x);
                    const float fy = static_cast<float>(y);
                    float value = 0.5F + 0.12F * std::sin(fx * 0.021F) * std::cos(fy * 0.017F) +
                                  0.08F * std::sin((fx + fy) * 0.045F);
                    value += ((static_cast<int>(x / 37) + static_cast<int>(y / 41)) & 1)
                                 ? 0.08F
                                 : -0.08F;
                    value += (unit(textured_random) - 0.5F) * 0.02F;
                    value = (std::min)(1.0F, (std::max)(0.0F, value));
                    textured[static_cast<std::size_t>(y) * width + x] =
                        static_cast<std::uint8_t>(value * 255.0F + 0.5F);
                }
            }
            photara::features::SiftOptions cpu_options;
            cpu_options.maximum_features = gpu_options.maximum_features;
            cpu_options.contrast_threshold = gpu_options.peak_threshold;
            cpu_options.edge_threshold = gpu_options.edge_threshold;
            cpu_options.first_octave = gpu_options.first_octave;
            cpu_options.octave_layers = gpu_options.octave_layers;
            cpu_options.maximum_orientations = gpu_options.maximum_orientations;
            cpu_options.maximum_image_dimension = gpu_options.maximum_image_dimension;
            cpu_options.root_sift = gpu_options.root_sift;
            photara::features::SiftExtractor cpu_extractor(cpu_options);
            const auto cpu_textured = cpu_extractor.extract_gray(textured, width, height);
            const auto gpu_textured = gpu_extractor.extract_gray(textured, width, height);
            if (cpu_textured.metric != gpu_textured.metric ||
                cpu_textured.metric != cpu_extractor.info().metric)
                return 21;
            std::size_t close = 0;
            double cosine_sum = 0.0;
            const std::size_t compared =
                (std::min)(cpu_textured.keypoints.size(), gpu_textured.keypoints.size());
            for (std::size_t i = 0; i < compared; ++i) {
                const auto& cpu = cpu_textured.keypoints[i];
                const auto& gpu = gpu_textured.keypoints[i];
                const bool scale_ok =
                    std::abs(gpu.scale - cpu.scale) <= 0.02F * std::abs(cpu.scale);
                if (std::abs(gpu.x - cpu.x) <= 0.05F &&
                    std::abs(gpu.y - cpu.y) <= 0.05F && scale_ok)
                    ++close;
                double dot = 0.0, na = 0.0, nb = 0.0;
                const float* da = cpu_textured.descriptors.data() + i * 128;
                const float* db = gpu_textured.descriptors.data() + i * 128;
                for (int column = 0; column < 128; ++column) {
                    dot += static_cast<double>(da[column]) * db[column];
                    na += static_cast<double>(da[column]) * da[column];
                    nb += static_cast<double>(db[column]) * db[column];
                }
                cosine_sum += dot / std::sqrt((std::max)(na * nb, 1e-24));
            }
            const double position_ratio =
                static_cast<double>(close) /
                static_cast<double>((std::max)(std::size_t{1}, compared));
            const double mean_cosine =
                cosine_sum / static_cast<double>((std::max)(std::size_t{1}, compared));
            std::cout << " cpu vs cuda (textured): " << cpu_textured.keypoints.size()
                      << " vs " << gpu_textured.keypoints.size() << " position "
                      << position_ratio << " cosine " << mean_cosine << '\n';
            if (cpu_textured.keypoints.size() != gpu_textured.keypoints.size())
                return 18;
            if (position_ratio < 0.95 || mean_cosine < 0.99) return 19;
        }
    }
    return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 10;
    }
}
