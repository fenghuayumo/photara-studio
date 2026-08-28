#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace aetherscan::io {

// Every AetherScan container stores a writer/min-reader pair:
//   can_open = file.min_reader_version <= this_build.current_version
// Additive layout changes bump only the writer version. Breaking changes
// also bump min_reader_version so older builds refuse the file instead of
// silently mis-parsing it. A stored min_reader of 0 is treated as 1 (legacy).
//
// Chunk payloads follow the same contract internally. Extra fields are
// appended, never inserted. Unknown trailing bytes are ignored. Unknown
// .ascan chunks are copied on save unless they set k_chunk_must_understand.

inline constexpr std::uint32_t k_chunk_must_understand = 1u;

inline std::uint32_t normalize_min_reader(
    const std::uint32_t min_reader) noexcept {
    return min_reader == 0 ? 1u : min_reader;
}

inline void require_readable(
    std::uint32_t file_writer,
    std::uint32_t file_min_reader,
    const std::uint32_t our_version,
    const std::string_view format_name) {
    file_min_reader = normalize_min_reader(file_min_reader);
    if (file_writer == 0) {
        throw std::runtime_error(
            std::string(format_name) + " writer version is invalid");
    }
    if (file_min_reader > file_writer) {
        throw std::runtime_error(
            std::string(format_name) + " version fields are inconsistent");
    }
    if (file_min_reader > our_version) {
        throw std::runtime_error(
            std::string(format_name) + " requires reader v" +
            std::to_string(file_min_reader) + ", this build is v" +
            std::to_string(our_version));
    }
}

inline std::string unsupported_payload_version(
    const std::string_view format_name,
    const std::uint32_t file_version,
    const std::uint32_t our_version) {
    return std::string(format_name) + " v" + std::to_string(file_version) +
           " is newer than this build (v" + std::to_string(our_version) + ")";
}

}  // namespace aetherscan::io
