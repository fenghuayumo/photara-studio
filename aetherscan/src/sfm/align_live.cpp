#include "sfm/align_live.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace aetherscan::sfm {
namespace {

constexpr char k_magic[4] = {'A', 'S', 'A', 'L'};
constexpr std::uint32_t k_version = 2;
constexpr std::size_t k_max_keypoints = 2'000;
constexpr std::size_t k_max_matches = 400;
constexpr auto k_min_write_interval = std::chrono::milliseconds(90);

void append_bytes(std::string& out, const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const char*>(data);
    out.append(bytes, size);
}

template <class T>
void append_pod(std::string& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    append_bytes(out, &value, sizeof(T));
}

void append_path(std::string& out, const std::filesystem::path& path) {
    const std::u8string text = path.u8string();
    const auto size = static_cast<std::uint32_t>(text.size());
    append_pod(out, size);
    if (size > 0)
        append_bytes(out, text.data(), static_cast<std::size_t>(size));
}

bool read_bytes(
    const std::string& in, std::size_t& offset, void* data,
    const std::size_t size) {
    if (offset + size > in.size()) return false;
    std::memcpy(data, in.data() + offset, size);
    offset += size;
    return true;
}

template <class T>
bool read_pod(const std::string& in, std::size_t& offset, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return read_bytes(in, offset, &value, sizeof(T));
}

bool read_path(
    const std::string& in, std::size_t& offset, std::filesystem::path& path) {
    std::uint32_t size{};
    if (!read_pod(in, offset, size)) return false;
    if (offset + size > in.size()) return false;
    std::u8string text(
        reinterpret_cast<const char8_t*>(in.data() + offset),
        static_cast<std::size_t>(size));
    offset += size;
    path = std::filesystem::path(text);
    return true;
}

std::vector<AlignLiveKeypoint> subsample_keypoints(
    const features::FeatureSet& features) {
    const auto& keypoints = features.keypoints;
    const float width = std::max(1.F, static_cast<float>(features.image_width));
    const float height = std::max(1.F, static_cast<float>(features.image_height));
    auto to_live = [&](const std::size_t i) {
        const auto& keypoint = keypoints[i];
        AlignLiveKeypoint point;
        point.u = keypoint.x / width;
        point.v = keypoint.y / height;
        point.scale = std::clamp(keypoint.scale / width, 0.002F, 0.08F);
        return point;
    };
    if (keypoints.size() <= k_max_keypoints) {
        std::vector<AlignLiveKeypoint> out;
        out.reserve(keypoints.size());
        for (std::size_t i = 0; i < keypoints.size(); ++i)
            out.push_back(to_live(i));
        return out;
    }

    // Cover the frame instead of taking the first/highest-response slice.
    // SiftGPU leaves response at 0 and VLFeat at 1, so a global top-K by
    // response collapses to detection order (top of the image).
    constexpr int k_grid = 16;
    const float cell_w = width / static_cast<float>(k_grid);
    const float cell_h = height / static_cast<float>(k_grid);
    std::array<std::vector<std::size_t>, k_grid * k_grid> cells;
    for (std::size_t i = 0; i < keypoints.size(); ++i) {
        const auto& keypoint = keypoints[i];
        const int column = std::clamp(
            static_cast<int>(keypoint.x / cell_w), 0, k_grid - 1);
        const int row = std::clamp(
            static_cast<int>(keypoint.y / cell_h), 0, k_grid - 1);
        cells[static_cast<std::size_t>(row * k_grid + column)].push_back(i);
    }
    const auto stronger = [&](const std::size_t a, const std::size_t b) {
        const auto& left = keypoints[a];
        const auto& right = keypoints[b];
        if (left.response != right.response)
            return left.response > right.response;
        if (left.scale != right.scale) return left.scale > right.scale;
        return a < b;
    };
    for (auto& cell : cells) {
        if (cell.size() > 1) std::sort(cell.begin(), cell.end(), stronger);
    }

    std::vector<std::size_t> selected;
    selected.reserve(k_max_keypoints);
    std::array<std::size_t, k_grid * k_grid> offsets{};
    for (int current = 0; selected.size() < k_max_keypoints;
         current = (current + 1) % (k_grid * k_grid)) {
        auto& offset = offsets[static_cast<std::size_t>(current)];
        const auto& cell = cells[static_cast<std::size_t>(current)];
        if (offset >= cell.size()) {
            bool remaining = false;
            for (std::size_t i = 0; i < cells.size(); ++i) {
                if (offsets[i] < cells[i].size()) {
                    remaining = true;
                    break;
                }
            }
            if (!remaining) break;
            continue;
        }
        selected.push_back(cell[offset++]);
    }

    std::vector<AlignLiveKeypoint> out;
    out.reserve(selected.size());
    for (const std::size_t i : selected) out.push_back(to_live(i));
    return out;
}

std::vector<AlignLiveMatch> subsample_matches(
    const features::FeatureSet& a, const features::FeatureSet& b,
    const std::span<const AlignLiveIndexMatch> matches) {
    const float aw = std::max(1.F, static_cast<float>(a.image_width));
    const float ah = std::max(1.F, static_cast<float>(a.image_height));
    const float bw = std::max(1.F, static_cast<float>(b.image_width));
    const float bh = std::max(1.F, static_cast<float>(b.image_height));
    std::vector<AlignLiveIndexMatch> ranked(matches.begin(), matches.end());
    if (ranked.size() > k_max_matches) {
        std::partial_sort(
            ranked.begin(),
            ranked.begin() + static_cast<std::ptrdiff_t>(k_max_matches),
            ranked.end(),
            [](const AlignLiveIndexMatch& left, const AlignLiveIndexMatch& right) {
                return left.score > right.score;
            });
        ranked.resize(k_max_matches);
    }
    std::vector<AlignLiveMatch> out;
    out.reserve(ranked.size());
    for (const AlignLiveIndexMatch& match : ranked) {
        if (match.query >= a.keypoints.size() ||
            match.train >= b.keypoints.size())
            continue;
        const auto& p0 = a.keypoints[match.query];
        const auto& p1 = b.keypoints[match.train];
        AlignLiveMatch line;
        line.u0 = p0.x / aw;
        line.v0 = p0.y / ah;
        line.u1 = p1.x / bw;
        line.v1 = p1.y / bh;
        line.score = match.score;
        out.push_back(line);
    }
    return out;
}

void encode_frame(const AlignLiveFrame& frame, std::string& out) {
    out.clear();
    out.reserve(
        64 + frame.keypoints_a.size() * 12 + frame.keypoints_b.size() * 12 +
        frame.matches.size() * 16 + 512);
    append_bytes(out, k_magic, 4);
    append_pod(out, k_version);
    append_pod(out, static_cast<std::uint32_t>(frame.kind));
    append_pod(out, frame.revision);
    append_pod(out, frame.index_a);
    append_pod(out, frame.index_b);
    const auto n_a = static_cast<std::uint32_t>(frame.keypoints_a.size());
    append_pod(out, n_a);
    for (const AlignLiveKeypoint& point : frame.keypoints_a) {
        append_pod(out, point.u);
        append_pod(out, point.v);
        append_pod(out, point.scale);
    }
    const auto n_b = static_cast<std::uint32_t>(frame.keypoints_b.size());
    append_pod(out, n_b);
    for (const AlignLiveKeypoint& point : frame.keypoints_b) {
        append_pod(out, point.u);
        append_pod(out, point.v);
        append_pod(out, point.scale);
    }
    const auto n_m = static_cast<std::uint32_t>(frame.matches.size());
    append_pod(out, n_m);
    for (const AlignLiveMatch& line : frame.matches) {
        append_pod(out, line.u0);
        append_pod(out, line.v0);
        append_pod(out, line.u1);
        append_pod(out, line.v1);
        append_pod(out, line.score);
    }
    append_pod(out, frame.total_keypoints_a);
    append_pod(out, frame.total_keypoints_b);
    append_pod(out, frame.total_matches);
    append_path(out, frame.path_a);
    append_path(out, frame.path_b);
}

bool decode_frame(const std::string& in, AlignLiveFrame& frame) {
    if (in.size() < 8) return false;
    std::size_t offset = 0;
    char magic[4]{};
    if (!read_bytes(in, offset, magic, 4)) return false;
    if (std::memcmp(magic, k_magic, 4) != 0) return false;
    std::uint32_t version{};
    if (!read_pod(in, offset, version) || (version != 1 && version != 2))
        return false;
    std::uint32_t kind{};
    if (!read_pod(in, offset, kind)) return false;
    frame = {};
    frame.kind = static_cast<AlignLiveKind>(kind);
    if (!read_pod(in, offset, frame.revision)) return false;
    if (!read_pod(in, offset, frame.index_a)) return false;
    if (!read_pod(in, offset, frame.index_b)) return false;
    std::uint32_t n_a{};
    if (!read_pod(in, offset, n_a) || n_a > 16'384) return false;
    frame.keypoints_a.resize(n_a);
    for (AlignLiveKeypoint& point : frame.keypoints_a) {
        if (!read_pod(in, offset, point.u) || !read_pod(in, offset, point.v) ||
            !read_pod(in, offset, point.scale))
            return false;
    }
    std::uint32_t n_b{};
    if (!read_pod(in, offset, n_b) || n_b > 16'384) return false;
    frame.keypoints_b.resize(n_b);
    for (AlignLiveKeypoint& point : frame.keypoints_b) {
        if (!read_pod(in, offset, point.u) || !read_pod(in, offset, point.v) ||
            !read_pod(in, offset, point.scale))
            return false;
    }
    std::uint32_t n_m{};
    if (!read_pod(in, offset, n_m) || n_m > 16'384) return false;
    frame.matches.resize(n_m);
    for (AlignLiveMatch& line : frame.matches) {
        if (!read_pod(in, offset, line.u0) || !read_pod(in, offset, line.v0) ||
            !read_pod(in, offset, line.u1) || !read_pod(in, offset, line.v1))
            return false;
        if (version >= 2) {
            if (!read_pod(in, offset, line.score)) return false;
        } else {
            line.score = 1.F;
        }
    }
    if (!read_pod(in, offset, frame.total_keypoints_a) ||
        !read_pod(in, offset, frame.total_keypoints_b) ||
        !read_pod(in, offset, frame.total_matches))
        return false;
    if (!read_path(in, offset, frame.path_a)) return false;
    if (!read_path(in, offset, frame.path_b)) return false;
    return true;
}

void publish_file(const std::filesystem::path& path, const std::string& bytes) {
    if (path.empty() || bytes.empty()) return;
    auto temporary = path;
    temporary += ".tmp";
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("Cannot write alignment live file");
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!output) throw std::runtime_error("Cannot write alignment live file");
    }
#if defined(_WIN32)
    if (!MoveFileExW(
            temporary.c_str(), path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot publish alignment live file");
#else
    std::filesystem::rename(temporary, path);
#endif
}

}  // namespace

void AlignLivePreview::set_path(std::filesystem::path path) {
    std::lock_guard lock(mutex_);
    path_ = std::move(path);
}

void AlignLivePreview::queue(AlignLiveFrame frame) {
    std::lock_guard lock(mutex_);
    if (path_.empty()) return;
    frame.revision = ++revision_;
    pending_ = std::move(frame);
    dirty_ = true;
    write_locked(false);
}

void AlignLivePreview::write_locked(const bool force) {
    if (!dirty_ || path_.empty()) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force && last_write_.time_since_epoch().count() != 0 &&
        now - last_write_ < k_min_write_interval)
        return;
    std::string bytes;
    encode_frame(pending_, bytes);
    try {
        publish_file(path_, bytes);
        dirty_ = false;
        last_write_ = now;
    } catch (...) {
        // The editor keeps the previous snapshot if a write races.
    }
}

void AlignLivePreview::publish_features(
    const std::size_t index, const std::filesystem::path& path,
    const features::FeatureSet& features) {
    AlignLiveFrame frame;
    frame.kind = AlignLiveKind::features;
    frame.index_a = static_cast<std::int32_t>(index);
    frame.path_a = path;
    frame.total_keypoints_a =
        static_cast<std::uint32_t>(features.keypoints.size());
    frame.keypoints_a = subsample_keypoints(features);
    queue(std::move(frame));
}

void AlignLivePreview::publish_matches(
    const std::size_t index_a, const std::filesystem::path& path_a,
    const features::FeatureSet& features_a, const std::size_t index_b,
    const std::filesystem::path& path_b, const features::FeatureSet& features_b,
    const std::span<const AlignLiveIndexMatch> matches,
    const AlignLiveKind kind) {
    if (matches.empty() && kind != AlignLiveKind::inliers) return;
    AlignLiveFrame frame;
    frame.kind = is_live_pair_kind(kind) ? kind : AlignLiveKind::matching;
    frame.index_a = static_cast<std::int32_t>(index_a);
    frame.index_b = static_cast<std::int32_t>(index_b);
    frame.path_a = path_a;
    frame.path_b = path_b;
    frame.total_keypoints_a =
        static_cast<std::uint32_t>(features_a.keypoints.size());
    frame.total_keypoints_b =
        static_cast<std::uint32_t>(features_b.keypoints.size());
    frame.total_matches = static_cast<std::uint32_t>(matches.size());
    frame.keypoints_a = subsample_keypoints(features_a);
    frame.keypoints_b = subsample_keypoints(features_b);
    frame.matches = subsample_matches(features_a, features_b, matches);
    queue(std::move(frame));
}

void AlignLivePreview::flush() {
    std::lock_guard lock(mutex_);
    write_locked(true);
}

bool load_align_live_frame(
    const std::filesystem::path& path, AlignLiveFrame& frame) {
    std::error_code error;
    if (path.empty() || !std::filesystem::is_regular_file(path, error))
        return false;
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    std::string bytes(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    AlignLiveFrame parsed;
    if (!decode_frame(bytes, parsed)) return false;
    frame = std::move(parsed);
    return true;
}

}  // namespace aetherscan::sfm
