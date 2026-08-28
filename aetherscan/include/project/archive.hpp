#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace aetherscan::project {

inline constexpr std::uint32_t k_ascan_version = 1;
inline constexpr std::uint32_t k_ascan_min_reader = 1;

enum class ChunkType : std::uint32_t {
    settings = 1,
    sfm = 2,
    gaussians = 3,
    mesh = 4,
    texture = 5,
};

[[nodiscard]] inline bool known_chunk_type(const ChunkType type) noexcept {
    const auto id = static_cast<std::uint32_t>(type);
    return id >= static_cast<std::uint32_t>(ChunkType::settings) &&
           id <= static_cast<std::uint32_t>(ChunkType::texture);
}

// Chunked single-file AetherScan project (.ascan). Unread blobs stay on disk
// until save(), so a settings-only rewrite can copy the Gaussian chunk as
// opaque bytes without decoding it. save() always rewrites the current
// container version. Unknown optional chunks are preserved; a chunk with
// io::k_chunk_must_understand set is rejected if this build does not know it.
class Archive {
public:
    static Archive create();
    static Archive open(const std::filesystem::path& path);

    [[nodiscard]] bool has(ChunkType type) const;
    [[nodiscard]] std::vector<std::uint8_t> chunk(ChunkType type) const;
    void set_chunk(
        ChunkType type, std::vector<std::uint8_t> bytes,
        std::uint32_t flags = 0);
    void erase_chunk(ChunkType type);
    void save(const std::filesystem::path& path) const;

    [[nodiscard]] const std::filesystem::path& source_path() const noexcept {
        return source_;
    }
    [[nodiscard]] std::uint32_t writer_version() const noexcept {
        return writer_version_;
    }
    [[nodiscard]] std::uint32_t min_reader_version() const noexcept {
        return min_reader_version_;
    }

private:
    struct Slot {
        ChunkType type{};
        std::uint32_t flags{};
        std::uint64_t source_offset{};
        std::uint64_t size{};
        std::uint64_t checksum{};
        std::optional<std::vector<std::uint8_t>> replacement;
    };

    Slot* find(ChunkType type);
    [[nodiscard]] const Slot* find(ChunkType type) const;

    std::filesystem::path source_;
    std::vector<Slot> slots_;
    std::uint32_t writer_version_{k_ascan_version};
    std::uint32_t min_reader_version_{k_ascan_min_reader};
};

}  // namespace aetherscan::project
