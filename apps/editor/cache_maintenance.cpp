#include "cache_maintenance.hpp"

#include <algorithm>
#include <cstdio>
#include <string_view>
#include <system_error>

namespace editor {
namespace {

constexpr std::string_view k_session_prefix = "session-";

bool same_directory(
    const std::filesystem::path& left, const std::filesystem::path& right) {
    if (left.empty() || right.empty()) return false;
    std::error_code error;
    if (std::filesystem::equivalent(left, right, error)) return true;
    return left.lexically_normal() == right.lexically_normal();
}

bool is_session_folder(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    return name.rfind(k_session_prefix, 0) == 0;
}

// Namespaces are named "<project stem>-<fnv1a hash in lower-case hex>". The hash
// prints without padding, so 12..16 digits covers everything we ever write while
// keeping date-like names ("scans-2024") out.
bool looks_like_namespace(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    const std::size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash == 0) return false;
    const std::size_t digits = name.size() - dash - 1;
    if (digits < 12 || digits > 16) return false;
    for (std::size_t i = dash + 1; i < name.size(); ++i) {
        const char value = name[i];
        const bool hex = (value >= '0' && value <= '9') ||
                         (value >= 'a' && value <= 'f');
        if (!hex) return false;
    }
    return true;
}

// Deleting a cache folder is destructive and the cache root is user chosen, so a
// candidate must also hold at least one artifact this editor writes. An empty or
// foreign folder is left alone.
bool holds_cache_artifacts(const std::filesystem::path& dir) {
    constexpr std::string_view k_names[] = {
        "sfm.bin",       "splat.ply",       "mesh.ply",
        "dense.ply",     "textured",        "subject_bounds.txt",
        "preview_view",  "preview_camera",  "preview_vis",
        "preview_ack"};
    std::error_code error;
    for (const std::string_view name : k_names) {
        if (std::filesystem::exists(dir / name, error)) return true;
        error.clear();
    }
    // Checkpoint-only namespaces hold "<stage>-<hex key>.bin" files.
    std::filesystem::directory_iterator entries(dir, error);
    if (error) return false;
    for (const auto& entry : entries) {
        std::error_code entry_error;
        if (!entry.is_regular_file(entry_error) || entry_error) continue;
        const std::string name = entry.path().filename().string();
        if (!name.ends_with(".bin")) continue;
        const std::size_t dot = name.size() - 4;
        const std::size_t dash = name.rfind('-', dot);
        if (dash == std::string::npos || dash == 0 || dot <= dash + 1) continue;
        bool hex = true;
        for (std::size_t i = dash + 1; i < dot && hex; ++i) {
            const char value = name[i];
            hex = (value >= '0' && value <= '9') ||
                  (value >= 'a' && value <= 'f');
        }
        if (hex) return true;
    }
    return false;
}

// Size and newest write of one namespace. Only metadata is touched, so this
// stays fast even for the multi-GB folders.
void measure_folder(
    const std::filesystem::path& dir, std::uintmax_t& bytes,
    std::filesystem::file_time_type& last_used) {
    std::error_code error;
    const auto own_stamp = std::filesystem::last_write_time(dir, error);
    if (!error && own_stamp > last_used) last_used = own_stamp;
    error.clear();
    std::filesystem::recursive_directory_iterator entries(
        dir, std::filesystem::directory_options::skip_permission_denied, error);
    if (error) return;
    for (const auto& entry : entries) {
        std::error_code entry_error;
        if (entry.is_regular_file(entry_error)) {
            const auto size = entry.file_size(entry_error);
            if (!entry_error) bytes += size;
        }
        entry_error.clear();
        const auto stamp = entry.last_write_time(entry_error);
        if (!entry_error && stamp > last_used) last_used = stamp;
    }
}

}  // namespace

std::vector<std::filesystem::path> cache_roots(
    const ProjectSettings& settings) {
    std::vector<std::filesystem::path> roots;
    const std::filesystem::path configured =
        path_from_utf8_field(settings.cache_dir.data());
    if (!configured.empty()) roots.push_back(configured);
    std::error_code error;
    const auto temp = std::filesystem::temp_directory_path(error);
    if (!error) roots.push_back(temp / "Photara");
    return roots;
}

CacheUsage scan_cache_usage(
    const ProjectSettings& settings, const std::filesystem::path& current_dir) {
    CacheUsage usage;
    usage.valid = true;
    for (const auto& root : cache_roots(settings)) {
        std::error_code root_error;
        std::filesystem::directory_iterator entries(root, root_error);
        if (root_error) continue;
        for (const auto& entry : entries) {
            std::error_code entry_error;
            const bool directory = entry.is_directory(entry_error);
            if (entry_error || !directory) continue;
            // Handshake folders are per process and get their own lifecycle.
            if (is_session_folder(entry.path())) continue;
            if (!looks_like_namespace(entry.path())) continue;
            if (!holds_cache_artifacts(entry.path())) continue;
            CacheFolder folder;
            folder.path = entry.path();
            measure_folder(folder.path, folder.bytes, folder.last_used);
            folder.in_use = same_directory(folder.path, current_dir);
            usage.bytes += folder.bytes;
            if (!folder.in_use) {
                usage.unused_bytes += folder.bytes;
                ++usage.unused_folders;
            }
            usage.folders.push_back(std::move(folder));
        }
    }
    if (!current_dir.empty()) {
        std::error_code error;
        if (std::filesystem::is_directory(current_dir, error)) {
            bool listed = false;
            for (const auto& folder : usage.folders) {
                if (same_directory(folder.path, current_dir)) {
                    listed = true;
                    break;
                }
            }
            if (!listed) {
                CacheFolder folder;
                folder.path = current_dir;
                measure_folder(folder.path, folder.bytes, folder.last_used);
                folder.in_use = true;
                usage.bytes += folder.bytes;
                usage.folders.push_back(std::move(folder));
            }
        }
    }
    return usage;
}

std::size_t clean_unused_caches(
    const ProjectSettings& settings, const std::filesystem::path& current_dir,
    const std::chrono::minutes min_idle, std::uintmax_t* freed_bytes) {
    const CacheUsage usage = scan_cache_usage(settings, current_dir);
    const auto roots = cache_roots(settings);
    const auto now = std::filesystem::file_time_type::clock::now();
    std::size_t removed = 0;
    for (const CacheFolder& folder : usage.folders) {
        if (folder.in_use) continue;
        if (folder.last_used != std::filesystem::file_time_type{} &&
            now - folder.last_used < min_idle)
            continue;
        // Defensive: only ever remove a direct child of a known root. The scan
        // above already enumerates nothing else, but deletion is destructive.
        bool under_root = false;
        for (const auto& root : roots) {
            if (same_directory(folder.path.parent_path(), root)) {
                under_root = true;
                break;
            }
        }
        if (!under_root) continue;
        std::error_code error;
        std::filesystem::remove_all(folder.path, error);
        if (error) continue;
        ++removed;
        if (freed_bytes != nullptr) *freed_bytes += folder.bytes;
    }
    return removed;
}

std::string format_cache_bytes(const std::uintmax_t bytes) {
    constexpr std::uintmax_t k_mib = 1024ULL * 1024ULL;
    constexpr std::uintmax_t k_gib = 1024ULL * k_mib;
    char text[32];
    if (bytes >= k_gib)
        std::snprintf(
            text, sizeof(text), "%.2f GB",
            static_cast<double>(bytes) / static_cast<double>(k_gib));
    else if (bytes >= k_mib)
        std::snprintf(
            text, sizeof(text), "%.0f MB",
            static_cast<double>(bytes) / static_cast<double>(k_mib));
    else
        std::snprintf(
            text, sizeof(text), "%.0f KB",
            static_cast<double>(bytes) / 1024.0);
    return text;
}

}  // namespace editor
