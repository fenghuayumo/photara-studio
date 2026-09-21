#pragma once

#include "pipeline.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace editor {

// One cache namespace: a per-project folder holding the runtime working copies
// and, when feature reuse is on, the feature/match checkpoints.
struct CacheFolder {
    std::filesystem::path path;
    std::uintmax_t bytes{};
    // Newest write inside the folder. A namespace whose dataset was renamed,
    // moved or deleted is never touched again, which is what makes it a
    // candidate for eviction.
    std::filesystem::file_time_type last_used{};
    bool in_use{};  // belongs to the project this session has open
};

// What the cache roots currently hold. Pure metadata: no file contents, and in
// particular no checkpoint, are read.
struct CacheUsage {
    std::vector<CacheFolder> folders;
    std::uintmax_t bytes{};
    std::uintmax_t unused_bytes{};
    std::size_t unused_folders{};
    bool valid{};
};

// Cache roots: the configured cache folder (when set) and the %TEMP% namespace
// the editor used before working copies moved next to the project.
std::vector<std::filesystem::path> cache_roots(const ProjectSettings& settings);

CacheUsage scan_cache_usage(
    const ProjectSettings& settings, const std::filesystem::path& current_dir);

// Removes every namespace that is not this project's and whose newest write is
// older than `min_idle`, so a second editor instance that is mid-run keeps its
// own folders. Returns how many folders were removed and, when `freed_bytes` is
// given, adds what they occupied.
std::size_t clean_unused_caches(
    const ProjectSettings& settings, const std::filesystem::path& current_dir,
    std::chrono::minutes min_idle,
    std::uintmax_t* freed_bytes = nullptr);

std::string format_cache_bytes(std::uintmax_t bytes);

}  // namespace editor
