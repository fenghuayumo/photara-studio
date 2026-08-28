#include "project/archive.hpp"

#include "io/format_version.hpp"
#include "../io/binary_codec.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <string>
#include <thread>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace aetherscan::project {
namespace {

constexpr std::array<char, 8> k_magic{
    'A', 'E', 'T', 'H', 'S', 'C', 'A', 'N'};

using io::binary::hash_bytes;
using io::binary::k_fnv_offset;

struct FileHeader {
    std::array<char, 8> magic{};
    std::uint32_t version{};
    std::uint32_t flags{};
    std::uint32_t chunk_count{};
    std::uint32_t min_reader_version{};
};

struct FileEntry {
    std::uint32_t type{};
    std::uint32_t flags{};
    std::uint64_t offset{};
    std::uint64_t size{};
    std::uint64_t checksum{};
};

constexpr std::uint32_t k_header_size = 8 + 4 + 4 + 4 + 4;
constexpr std::uint32_t k_entry_size = 4 + 4 + 8 + 8 + 8;
static_assert(sizeof(FileHeader) == 24);
static_assert(sizeof(FileEntry) == 32);
static_assert(k_header_size == 24);
static_assert(k_entry_size == 32);

std::uint64_t checksum_bytes(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t hash = k_fnv_offset;
    hash_bytes(hash, bytes.data(), bytes.size());
    return hash;
}

void write_pod(std::ostream& stream, const void* data, const std::size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Failed while writing .ascan");
}

void read_pod(std::istream& stream, void* data, const std::size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Truncated .ascan file");
}

std::uint64_t align8(const std::uint64_t value) {
    return (value + 7ULL) & ~7ULL;
}

void atomic_replace(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination) {
#if defined(_WIN32)
    DWORD replace_error = ERROR_SUCCESS;
    for (unsigned attempt = 0; attempt < 6; ++attempt) {
        if (MoveFileExW(
                temporary.c_str(), destination.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            return;
        replace_error = GetLastError();
        if (replace_error != ERROR_SHARING_VIOLATION &&
            replace_error != ERROR_ACCESS_DENIED)
            throw std::system_error(
                static_cast<int>(replace_error), std::system_category(),
                "Failed to replace .ascan file");
        std::this_thread::sleep_for(std::chrono::milliseconds(5U << attempt));
    }
    throw std::system_error(
        static_cast<int>(replace_error), std::system_category(),
        "Failed to replace .ascan file");
#else
    std::filesystem::rename(temporary, destination);
#endif
}

}  // namespace

Archive::Slot* Archive::find(const ChunkType type) {
    for (auto& slot : slots_)
        if (slot.type == type) return &slot;
    return nullptr;
}

const Archive::Slot* Archive::find(const ChunkType type) const {
    for (const auto& slot : slots_)
        if (slot.type == type) return &slot;
    return nullptr;
}

Archive Archive::create() { return {}; }

Archive Archive::open(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open .ascan: " + path.string());
    FileHeader header{};
    read_pod(input, &header, sizeof(header));
    if (header.magic != k_magic)
        throw std::runtime_error("Not an AetherScan .ascan file");
    io::require_readable(
        header.version, header.min_reader_version, k_ascan_version, ".ascan");
    if (header.chunk_count > 64)
        throw std::runtime_error(".ascan chunk directory is invalid");

    Archive archive;
    archive.source_ = path;
    archive.writer_version_ = header.version;
    archive.min_reader_version_ = io::normalize_min_reader(header.min_reader_version);
    archive.slots_.reserve(header.chunk_count);
    const auto file_size = std::filesystem::file_size(path);
    for (std::uint32_t index = 0; index < header.chunk_count; ++index) {
        FileEntry entry{};
        read_pod(input, &entry, sizeof(entry));
        if (entry.size > file_size || entry.offset > file_size ||
            entry.offset + entry.size > file_size)
            throw std::runtime_error(".ascan chunk is outside the file");
        const auto type = static_cast<ChunkType>(entry.type);
        if (!known_chunk_type(type) &&
            (entry.flags & io::k_chunk_must_understand) != 0) {
            throw std::runtime_error(
                ".ascan has an unrecognized required chunk (type " +
                std::to_string(entry.type) + ")");
        }
        Slot slot;
        slot.type = type;
        slot.flags = entry.flags;
        slot.source_offset = entry.offset;
        slot.size = entry.size;
        slot.checksum = entry.checksum;
        archive.slots_.push_back(std::move(slot));
    }
    return archive;
}

bool Archive::has(const ChunkType type) const {
    return find(type) != nullptr;
}

std::vector<std::uint8_t> Archive::chunk(const ChunkType type) const {
    const Slot* slot = find(type);
    if (slot == nullptr)
        throw std::runtime_error("Requested .ascan chunk is missing");
    if (slot->replacement) return *slot->replacement;
    if (source_.empty())
        throw std::runtime_error("In-memory .ascan chunk has no payload");
    std::ifstream input(source_, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to read .ascan chunk");
    input.seekg(static_cast<std::streamoff>(slot->source_offset));
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(slot->size));
    if (!bytes.empty())
        read_pod(input, bytes.data(), bytes.size());
    if (checksum_bytes(bytes) != slot->checksum)
        throw std::runtime_error(".ascan chunk checksum mismatch");
    return bytes;
}

void Archive::set_chunk(
    const ChunkType type, std::vector<std::uint8_t> bytes,
    const std::uint32_t flags) {
    if (Slot* existing = find(type)) {
        existing->size = bytes.size();
        existing->checksum = checksum_bytes(bytes);
        existing->flags = flags;
        existing->replacement = std::move(bytes);
        return;
    }
    Slot slot;
    slot.type = type;
    slot.flags = flags;
    slot.size = bytes.size();
    slot.checksum = checksum_bytes(bytes);
    slot.replacement = std::move(bytes);
    slots_.push_back(std::move(slot));
}

void Archive::erase_chunk(const ChunkType type) {
    slots_.erase(
        std::remove_if(
            slots_.begin(), slots_.end(),
            [type](const Slot& slot) { return slot.type == type; }),
        slots_.end());
}

void Archive::save(const std::filesystem::path& path) const {
    std::vector<Slot> ordered = slots_;
    std::sort(
        ordered.begin(), ordered.end(),
        [](const Slot& a, const Slot& b) {
            return static_cast<std::uint32_t>(a.type) <
                   static_cast<std::uint32_t>(b.type);
        });

    std::vector<std::vector<std::uint8_t>> payloads;
    payloads.reserve(ordered.size());
    for (const Slot& slot : ordered) {
        if (slot.replacement) {
            payloads.push_back(*slot.replacement);
        } else {
            payloads.push_back(chunk(slot.type));
        }
    }

    FileHeader header{k_magic, k_ascan_version, 0,
                      static_cast<std::uint32_t>(ordered.size()),
                      k_ascan_min_reader};
    std::uint64_t cursor = k_header_size +
        static_cast<std::uint64_t>(ordered.size()) * k_entry_size;
    std::vector<FileEntry> entries(ordered.size());
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        cursor = align8(cursor);
        entries[index].type = static_cast<std::uint32_t>(ordered[index].type);
        entries[index].flags = ordered[index].flags;
        entries[index].offset = cursor;
        entries[index].size = payloads[index].size();
        entries[index].checksum = checksum_bytes(payloads[index]);
        cursor += payloads[index].size();
    }

    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("Failed to create .ascan: " + path.string());
        write_pod(output, &header, sizeof(header));
        for (const FileEntry& entry : entries)
            write_pod(output, &entry, sizeof(entry));
        std::uint64_t written = k_header_size +
            static_cast<std::uint64_t>(entries.size()) * k_entry_size;
        for (std::size_t index = 0; index < payloads.size(); ++index) {
            const std::uint64_t aligned = align8(written);
            while (written < aligned) {
                const char zero = 0;
                write_pod(output, &zero, 1);
                ++written;
            }
            if (!payloads[index].empty())
                write_pod(output, payloads[index].data(), payloads[index].size());
            written += payloads[index].size();
        }
        output.flush();
        if (!output)
            throw std::runtime_error("Failed while writing .ascan: " + path.string());
    }
    try {
        atomic_replace(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

}  // namespace aetherscan::project
