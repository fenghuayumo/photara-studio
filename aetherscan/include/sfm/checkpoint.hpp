#pragma once

#include "features/types.hpp"
#include "sfm/scene.hpp"

#include <array>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <string_view>
#include <type_traits>
#include <vector>

namespace aetherscan::sfm {

enum class CheckpointStage : std::uint32_t {
    features = 1,
    matches = 2,
    geometry = 3,
    tracks = 4,
    reconstruction = 5,
};

struct CheckpointOptions {
    std::filesystem::path directory;
    bool read{true};
    bool write{true};
    unsigned reconstruction_interval{1};
    std::size_t max_variants_per_stage{4};

    [[nodiscard]] bool enabled() const noexcept {
        return !directory.empty() && (read || write);
    }
};

struct RawPairMatches {
    Index id1{k_invalid};
    Index id2{k_invalid};
    std::vector<features::FeatureMatch> matches;
};

// Stable FNV-1a builder used for stage dependency and parameter keys.
class FingerprintBuilder {
public:
    FingerprintBuilder() = default;

    void append_bytes(const void* data, std::size_t size) noexcept;
    void append_string(std::string_view value) noexcept;

    template <class T>
    void append(const T& value) noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        append_bytes(&value, sizeof(value));
    }

    [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

private:
    std::uint64_t value_{14695981039346656037ULL};
};

struct ImageFileFingerprint {
    std::filesystem::path normalized_path;
    std::uint64_t size{};
    std::filesystem::file_time_type write_time{};
    std::array<std::uint8_t, 32> digest{};
};

struct ImageSetFingerprint {
    std::uint64_t value{};
    std::vector<ImageFileFingerprint> files;
};

enum class ImageSnapshotCheck {
    identity,  // path / size / mtime only
    content,   // full SHA-256 rehash
};

// Hash every image in parallel and build a stable content fingerprint.
ImageSetFingerprint fingerprint_image_set(
    const std::vector<std::filesystem::path>& image_paths);

std::uint64_t fingerprint_images(
    const std::vector<std::filesystem::path>& image_paths);

// Reuse digests from fingerprint_image_set; identity checks avoid rehashing.
void verify_image_snapshot(
    const std::vector<std::filesystem::path>& image_paths,
    const ImageSetFingerprint& expected,
    ImageSnapshotCheck check = ImageSnapshotCheck::identity);

class CheckpointStore {
public:
    explicit CheckpointStore(CheckpointOptions options);

    [[nodiscard]] bool enabled() const noexcept { return options_.enabled(); }
    [[nodiscard]] bool writable() const noexcept {
        return options_.enabled() && options_.write;
    }

    bool load_scene(
        CheckpointStage stage, std::uint64_t key, Scene& scene) const;
    void save_scene(
        CheckpointStage stage, std::uint64_t key, const Scene& scene) const;

    bool load_matches(
        std::uint64_t key, std::vector<RawPairMatches>& matches) const;
    void save_matches(
        std::uint64_t key,
        const std::vector<RawPairMatches>& matches) const;

private:
    CheckpointOptions options_;
};

}  // namespace aetherscan::sfm
