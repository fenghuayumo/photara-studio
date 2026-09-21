#pragma once

#include <string>
#include <filesystem>
#include <fstream>

namespace tinytensor {

// Minimal path utilities
inline std::string join_path(const std::string& a, const std::string& b) {
    std::filesystem::path p(a);
    p /= b;
    return p.string();
}

inline std::string get_parent_path(const std::string& path) {
    return std::filesystem::path(path).parent_path().string();
}

inline std::string get_filename(const std::string& path) {
    return std::filesystem::path(path).filename().string();
}

// Stub for utf8_to_path - converts UTF-8 string to filesystem path
inline std::filesystem::path utf8_to_path(const std::string& utf8_str) {
    return std::filesystem::path(utf8_str);
}

// Stub for open_file_for_write - opens file for writing
inline bool open_file_for_write(const std::filesystem::path& path, std::ofstream& out) {
    out.open(path, std::ios::out | std::ios::binary);
    return out.is_open();
}

} // namespace tinytensor
