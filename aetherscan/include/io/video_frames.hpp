#pragma once

#include "io/image.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace aetherscan::io {

// Container extensions treated as video input (case-insensitive). The file
// need not exist; this is a name test so the editor can switch UI as soon as
// the user types or drops a path.
bool is_video_path(const std::filesystem::path& path);

// `<video_parent>/<stem>/images`, same layout extract_frames.py uses.
std::filesystem::path default_video_frames_dir(const std::filesystem::path& video);

struct VideoProbe {
    double duration_seconds{};
    double fps{};
    int width{};
    int height{};
};

struct VideoExtractOptions {
    std::filesystem::path video;
    std::filesystem::path output_dir;
    std::filesystem::path ffmpeg{"ffmpeg"};
    // Kept frames per second of source time. ffmpeg `fps=` filter.
    float fps = 2.0F;
    // Keep the sharpest of N consecutive candidates (1 = every fps sample).
    // Matches spirula DatasetPrep::sharp_window / extract_frames.py --keep.
    int sharp_window = 3;
    // 0 = no cap.
    int max_frames = 0;
    // JPEG quality 0..100. Outside that range writes PNG.
    int quality = 95;
    float scale = 1.0F;
    int rotate = 0;  // 0 / 90 / 180 / 270 clockwise
    bool resume = true;
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(const std::string&)> log;
};

struct VideoExtractResult {
    std::filesystem::path image_dir;
    int frames_written{};
    bool reused{};
};

bool ffmpeg_available(const std::filesystem::path& ffmpeg = "ffmpeg");

bool probe_video(
    const std::filesystem::path& ffmpeg,
    const std::filesystem::path& video,
    VideoProbe& out);

// Laplacian-variance sharpness used by extract_frames.py / FrameSelect:
// 512x512 box-averaged luma, mean-subtracted, variance of the 3x3 Laplacian.
double laplacian_sharpness(const RgbImage& image);
double laplacian_sharpness(const std::filesystem::path& path);

// Keep the sharpest of each consecutive `window` files in `candidate_dir`
// (sorted by name) and write them into `output_dir` as 00000.<ext>, ...
// Returns the number kept, or -1 on cancel.
int select_sharpest_frames(
    const std::filesystem::path& candidate_dir,
    const std::filesystem::path& output_dir,
    int window,
    int max_frames,
    const std::atomic<bool>* cancel = nullptr,
    const std::function<void(const std::string&)>& log = {});

// Decode `options.video` with ffmpeg, optionally pick the least-blurred frame
// of each window, and write stills into `options.output_dir`. Throws on
// failure. A matching previous extract is reused when `resume` is set.
VideoExtractResult extract_video_frames(const VideoExtractOptions& options);

// True when output_dir already holds a complete extract of this job (manifest
// matches and stills exist). Train/Dense use this to avoid rewriting frames
// under an existing alignment.
bool has_matching_video_extract(const VideoExtractOptions& options);

// True when output_dir has an AetherScan extract manifest, matching or not.
bool has_video_extract_manifest(const std::filesystem::path& output_dir);

}  // namespace aetherscan::io
