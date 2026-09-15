#pragma once

#include "features/types.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace aetherscan::sfm {

enum class AlignLiveKind : std::uint32_t {
    none = 0,
    features = 1,
    matching = 2,
};

struct AlignLiveKeypoint {
    float u{};
    float v{};
    float scale{0.01F};
};

struct AlignLiveMatch {
    float u0{};
    float v0{};
    float u1{};
    float v1{};
};

struct AlignLiveIndexMatch {
    std::uint32_t query{};
    std::uint32_t train{};
    float score{};
};

// Latest feature or pair snapshot the editor polls during Align Photos.
struct AlignLiveFrame {
    AlignLiveKind kind{AlignLiveKind::none};
    std::uint64_t revision{};
    std::int32_t index_a{-1};
    std::int32_t index_b{-1};
    std::filesystem::path path_a;
    std::filesystem::path path_b;
    std::vector<AlignLiveKeypoint> keypoints_a;
    std::vector<AlignLiveKeypoint> keypoints_b;
    std::vector<AlignLiveMatch> matches;
    std::uint32_t total_keypoints_a{};
    std::uint32_t total_keypoints_b{};
    std::uint32_t total_matches{};
};

class AlignLivePreview {
public:
    AlignLivePreview() = default;
    ~AlignLivePreview() { flush(); }
    AlignLivePreview(const AlignLivePreview&) = delete;
    AlignLivePreview& operator=(const AlignLivePreview&) = delete;

    void set_path(std::filesystem::path path);
    void publish_features(
        std::size_t index, const std::filesystem::path& path,
        const features::FeatureSet& features);
    void publish_matches(
        std::size_t index_a, const std::filesystem::path& path_a,
        const features::FeatureSet& features_a, std::size_t index_b,
        const std::filesystem::path& path_b,
        const features::FeatureSet& features_b,
        std::span<const AlignLiveIndexMatch> matches);
    void flush();

private:
    void queue(AlignLiveFrame frame);
    void write_locked(bool force);

    std::mutex mutex_;
    std::filesystem::path path_;
    AlignLiveFrame pending_;
    bool dirty_{};
    std::uint64_t revision_{};
    std::chrono::steady_clock::time_point last_write_{};
};

[[nodiscard]] bool load_align_live_frame(
    const std::filesystem::path& path, AlignLiveFrame& frame);

}  // namespace aetherscan::sfm
