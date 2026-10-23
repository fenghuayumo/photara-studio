#pragma once

#include "io/image.hpp"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace photara::io {

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

// One capture range in source time. `end_seconds <= start_seconds` means
// "until the end of the source" and is only valid as the final range.
struct VideoSegment {
    double start_seconds{0.0};
    double end_seconds{0.0};
};

bool operator==(const VideoSegment& a, const VideoSegment& b);
bool operator<(const VideoSegment& a, const VideoSegment& b);

// Sorts, clamps, merges overlapping ranges, and drops empty ones. Open-ended
// ranges are moved to the end and at most one is kept.
std::vector<VideoSegment> normalize_video_segments(
    const std::vector<VideoSegment>& segments, double duration_seconds = 0.0);

// `start:end;start:end`, with `-` as the open end (for example
// `0.000:12.500;20.000:-`). Empty text means the whole source.
std::string serialize_video_segments(const std::vector<VideoSegment>& segments);
bool parse_video_segments(
    std::string_view text, std::vector<VideoSegment>& segments,
    std::string* error = nullptr);

struct VideoExtractOptions {
    std::filesystem::path video;
    std::filesystem::path output_dir;
    std::filesystem::path ffmpeg{"ffmpeg"};
    // Kept frames per second of source time. ffmpeg `fps=` filter.
    float fps = 2.0F;
    // Keep the sharpest of N consecutive candidates (1 = every fps sample).
    int sharp_window = 3;
    // 0 = no cap. With segments the cap applies to the merged sequence.
    int max_frames = 0;
    // JPEG quality 0..100. Outside that range writes PNG.
    int quality = 95;
    float scale = 1.0F;
    int rotate = 0;  // 0 / 90 / 180 / 270 clockwise
    bool resume = true;
    // Optional capture ranges in source time. Empty = the whole video.
    // Ranges are extracted in time order into one continuously numbered
    // sequence, so non-contiguous clips can be assembled without renaming.
    std::vector<VideoSegment> segments;
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(const std::string&)> log;
    // 0..1 while decoding, or negative when the remaining work is unknown.
    std::function<void(float)> progress;
};

// Playback stills for the editor, optionally limited to a short seek interval.
// Writes p_NNNNNN.jpg from first_frame progressively into output_dir (existing
// preview stills are removed first), allowing playback before decoding ends.
struct VideoPreviewOptions {
    std::filesystem::path video;
    std::filesystem::path output_dir;
    std::filesystem::path ffmpeg{"ffmpeg"};
    float fps = 10.0F;
    int max_width = 640;
    int max_height = 480;
    // JPEG quality 0..100.
    int quality = 80;
    // Optional bounded preview; seek before decoding and preserve source-time indexing.
    double start_seconds = 0.0;
    double duration_seconds = 0.0;
    int first_frame = 1;
    VideoProbe source_probe{};
    bool keep_existing = false;
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(float)> progress;
};

int generate_video_preview(const VideoPreviewOptions& options);

struct VideoExtractResult {
    std::filesystem::path image_dir;
    int frames_written{};
    bool reused{};
};

bool ffmpeg_available(const std::filesystem::path& ffmpeg = "ffmpeg");

// Locate a runnable ffmpeg. `hint` may be empty (search), a bare name
// (`ffmpeg`), or a full path. Search order: explicit path, PATH, the
// current process directory, then well-known install locations. Empty
// means nothing runnable was found.
std::filesystem::path locate_ffmpeg(const std::filesystem::path& hint = {});

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
    const std::function<void(const std::string&)>& log = {},
    int first_index = 0);

// Decode `options.video` with ffmpeg, optionally pick the least-blurred frame
// of each window, and write stills into `options.output_dir`. Throws on
// failure. A matching previous extract is reused when `resume` is set.
VideoExtractResult extract_video_frames(const VideoExtractOptions& options);

// True when output_dir already holds a complete extract of this job (manifest
// matches and stills exist). Train/Dense use this to avoid rewriting frames
// under an existing alignment.
bool has_matching_video_extract(const VideoExtractOptions& options);

// True when output_dir has an Photara extract manifest, matching or not.
bool has_video_extract_manifest(const std::filesystem::path& output_dir);

}  // namespace photara::io
