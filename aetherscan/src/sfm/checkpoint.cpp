#include "sfm/checkpoint.hpp"
#include "sfm/tracks.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#if !defined(O_DIRECTORY)
#define O_DIRECTORY 0
#endif
#endif

namespace aetherscan::sfm {
namespace {

constexpr std::array<char, 8> magic{'A', 'E', 'T', 'H', 'C', 'K', 'P', 'T'};
constexpr std::uint32_t schema_version = 4;
constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

struct FileHeader {
    std::array<char, 8> signature{};
    std::uint32_t version{};
    std::uint32_t stage{};
    std::uint64_t key{};
    std::uint64_t payload_size{};
    std::uint64_t payload_checksum{};
};

void hash_bytes(
    std::uint64_t& hash, const void* data, const std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= fnv_prime;
    }
}

class Sha256 {
public:
    void update(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        total_bytes_ += size;
        while (size > 0) {
            const std::size_t copied =
                (std::min)(size, buffer_.size() - buffered_);
            std::memcpy(buffer_.data() + buffered_, bytes, copied);
            buffered_ += copied;
            bytes += copied;
            size -= copied;
            if (buffered_ == buffer_.size()) {
                transform(buffer_.data());
                buffered_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bit_count = total_bytes_ * 8;
        const std::uint8_t marker = 0x80;
        update(&marker, 1);
        const std::uint8_t zero = 0;
        while (buffered_ != 56) update(&zero, 1);
        std::array<std::uint8_t, 8> length{};
        for (std::size_t i = 0; i < length.size(); ++i)
            length[7 - i] =
                static_cast<std::uint8_t>(bit_count >> (i * 8));
        update(length.data(), length.size());
        std::array<std::uint8_t, 32> digest{};
        for (std::size_t word = 0; word < state_.size(); ++word)
            for (std::size_t byte = 0; byte < 4; ++byte)
                digest[word * 4 + byte] = static_cast<std::uint8_t>(
                    state_[word] >> (24 - byte * 8));
        return digest;
    }

private:
    static std::uint32_t rotate(
        const std::uint32_t value, const unsigned bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    void transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> constants{
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::uint32_t words[64]{};
        for (std::size_t i = 0; i < 16; ++i)
            words[i] =
                (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                block[i * 4 + 3];
        for (std::size_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 =
                rotate(words[i - 15], 7) ^
                rotate(words[i - 15], 18) ^ (words[i - 15] >> 3);
            const std::uint32_t s1 =
                rotate(words[i - 2], 17) ^
                rotate(words[i - 2], 19) ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        auto [a,b,c,d,e,f,g,h] = state_;
        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t s1 =
                rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const std::uint32_t choice = (e & f) ^ (~e & g);
            const std::uint32_t t1 =
                h + s1 + choice + constants[i] + words[i];
            const std::uint32_t s0 =
                rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const std::uint32_t majority =
                (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + majority;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffered_{0};
    std::uint64_t total_bytes_{0};
};

std::array<std::uint8_t, 32> hash_file(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Failed to hash image: " + path.string());
    Sha256 hash;
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), buffer.size());
        const std::streamsize count = input.gcount();
        if (count > 0)
            hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!input.eof())
        throw std::runtime_error("Failed while hashing image: " + path.string());
    return hash.finish();
}

class Writer {
public:
    explicit Writer(std::ostream& stream) : stream_(stream) {}

    void bytes(const void* data, const std::size_t size) {
        if (size == 0) return;
        stream_.write(static_cast<const char*>(data),
                      static_cast<std::streamsize>(size));
        if (!stream_) throw std::runtime_error("Checkpoint write failed");
        hash_bytes(checksum_, data, size);
        size_ += size;
    }

    template <class T>
    void value(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(value));
    }

    void string(const std::string_view value) {
        value_size(value.size());
        bytes(value.data(), value.size());
    }

    void value_size(const std::size_t value) {
        const auto fixed = static_cast<std::uint64_t>(value);
        this->value(fixed);
    }

    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] std::uint64_t checksum() const noexcept { return checksum_; }

private:
    std::ostream& stream_;
    std::uint64_t size_{0};
    std::uint64_t checksum_{fnv_offset};
};

class Reader {
public:
    Reader(std::istream& stream, const std::uint64_t payload_size)
        : stream_(stream), remaining_(payload_size) {}

    void bytes(void* data, const std::size_t size) {
        if (size > remaining_)
            throw std::runtime_error("Checkpoint payload is truncated");
        if (size == 0) return;
        stream_.read(static_cast<char*>(data),
                     static_cast<std::streamsize>(size));
        if (!stream_) throw std::runtime_error("Checkpoint read failed");
        hash_bytes(checksum_, data, size);
        remaining_ -= size;
    }

    template <class T>
    T value() {
        static_assert(std::is_trivially_copyable_v<T>);
        T result{};
        bytes(&result, sizeof(result));
        return result;
    }

    [[nodiscard]] std::size_t size(
        const std::size_t minimum_record_size = 1,
        const std::uint64_t maximum_count = 1'000'000'000ULL) {
        const std::uint64_t count = value<std::uint64_t>();
        if (minimum_record_size == 0 || count > maximum_count ||
            count > remaining_ / minimum_record_size)
            throw std::runtime_error("Checkpoint container size is invalid");
        if (count > static_cast<std::uint64_t>(
                        (std::numeric_limits<std::size_t>::max)()))
            throw std::runtime_error("Checkpoint container is too large");
        return static_cast<std::size_t>(count);
    }

    std::string string() {
        std::string result(size(1, 16'777'216), '\0');
        bytes(result.data(), result.size());
        return result;
    }

    void finish(const std::uint64_t expected_checksum) const {
        if (remaining_ != 0 || checksum_ != expected_checksum)
            throw std::runtime_error("Checkpoint checksum mismatch");
    }

    [[nodiscard]] std::uint64_t remaining() const noexcept {
        return remaining_;
    }

private:
    std::istream& stream_;
    std::uint64_t remaining_;
    std::uint64_t checksum_{fnv_offset};
};

template <class T>
void write_vector(Writer& writer, const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    writer.value_size(values.size());
    writer.bytes(values.data(), values.size() * sizeof(T));
}

template <class T>
void read_vector(Reader& reader, std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::size_t count =
        reader.size(sizeof(T), 1'000'000'000ULL);
    if (count > reader.remaining() / sizeof(T))
        throw std::runtime_error("Checkpoint vector size is invalid");
    values.resize(count);
    reader.bytes(values.data(), values.size() * sizeof(T));
}

void write_matrix(Writer& writer, const Mat3& matrix) {
    for (Eigen::Index row = 0; row < 3; ++row)
        for (Eigen::Index column = 0; column < 3; ++column)
            writer.value(matrix(row, column));
}

Mat3 read_matrix(Reader& reader) {
    Mat3 matrix;
    for (Eigen::Index row = 0; row < 3; ++row)
        for (Eigen::Index column = 0; column < 3; ++column)
            matrix(row, column) = reader.value<double>();
    return matrix;
}

void write_pose(Writer& writer, const Pose3D& pose) {
    write_matrix(writer, pose.R);
    for (Eigen::Index axis = 0; axis < 3; ++axis) writer.value(pose.C[axis]);
}

Pose3D read_pose(Reader& reader) {
    Pose3D pose;
    pose.R = read_matrix(reader);
    for (Eigen::Index axis = 0; axis < 3; ++axis)
        pose.C[axis] = reader.value<double>();
    return pose;
}

template <class T, class WriteValue>
void write_optional(
    Writer& writer, const std::optional<T>& value, WriteValue&& write_value) {
    writer.value(static_cast<std::uint8_t>(value.has_value()));
    if (value) write_value(writer, *value);
}

template <class T, class ReadValue>
std::optional<T> read_optional(Reader& reader, ReadValue&& read_value) {
    if (reader.value<std::uint8_t>() == 0) return std::nullopt;
    return read_value(reader);
}

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string text = path.generic_u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::filesystem::path path_from_utf8(const std::string& text) {
    return std::filesystem::path(
        std::u8string(
            reinterpret_cast<const char8_t*>(text.data()),
            reinterpret_cast<const char8_t*>(text.data() + text.size())));
}

void write_feature_set(
    Writer& writer, const features::FeatureSet& features,
    const bool include_descriptors) {
    writer.value(features.image_width);
    writer.value(features.image_height);
    writer.value_size(features.descriptor_dimension);
    writer.value(static_cast<std::uint32_t>(features.metric));
    writer.string(features.extractor_name);
    writer.value_size(features.keypoints.size());
    for (const features::Keypoint& keypoint : features.keypoints) {
        writer.value(keypoint.x);
        writer.value(keypoint.y);
        writer.value(keypoint.scale);
        writer.value(keypoint.orientation);
        writer.value(keypoint.response);
    }
    writer.value(static_cast<std::uint8_t>(include_descriptors));
    if (include_descriptors)
        write_vector(writer, features.descriptors);
}

features::FeatureSet read_feature_set(Reader& reader) {
    features::FeatureSet features;
    features.image_width = reader.value<std::uint32_t>();
    features.image_height = reader.value<std::uint32_t>();
    features.descriptor_dimension = reader.size(1, 1'048'576);
    features.metric =
        static_cast<features::DescriptorMetric>(reader.value<std::uint32_t>());
    features.extractor_name = reader.string();
    features.keypoints.resize(
        reader.size(5 * sizeof(float), 100'000'000));
    for (features::Keypoint& keypoint : features.keypoints) {
        keypoint.x = reader.value<float>();
        keypoint.y = reader.value<float>();
        keypoint.scale = reader.value<float>();
        keypoint.orientation = reader.value<float>();
        keypoint.response = reader.value<float>();
    }
    const bool has_descriptors = reader.value<std::uint8_t>() != 0;
    if (has_descriptors) {
        read_vector(reader, features.descriptors);
        features.validate();
    } else if (features.descriptor_dimension == 0 &&
               !features.keypoints.empty()) {
        throw std::runtime_error(
            "Checkpoint keypoints have no descriptor dimension");
    }
    return features;
}

void write_scene(
    Writer& writer, const Scene& scene,
    const bool include_descriptors) {
    writer.value(scene.thread_count);
    writer.value(scene.resection_progress.since_full_ba);
    writer.value(scene.resection_progress.bundle_adjustment_stage);
    write_vector(writer, scene.resection_progress.last_registered);
    write_vector(writer, scene.resection_progress.recent_inlier_ratios);
    writer.value_size(scene.cameras.size());
    for (const PinholeCamera& camera : scene.cameras) {
        writer.value(camera.id);
        writer.value(camera.width);
        writer.value(camera.height);
        writer.value(camera.fx);
        writer.value(camera.fy);
        writer.value(camera.cx);
        writer.value(camera.cy);
        writer.value(camera.k1);
        writer.value(camera.k2);
        writer.value(camera.p1);
        writer.value(camera.p2);
        writer.value(static_cast<std::uint8_t>(camera.trust_intrinsics));
    }
    writer.value_size(scene.images.size());
    for (const Image& image : scene.images) {
        writer.value(image.id);
        writer.value(image.camera_id);
        writer.string(path_utf8(image.path));
        write_feature_set(writer, image.features, include_descriptors);
        write_pose(writer, image.pose);
        writer.value(static_cast<std::uint8_t>(image.registered));
    }
    writer.value_size(scene.pairs.size());
    for (const ImagePair& pair : scene.pairs) {
        writer.value(pair.id1);
        writer.value(pair.id2);
        writer.value_size(pair.matches.size());
        for (const FeatureMatch& match : pair.matches) {
            writer.value(match.query);
            writer.value(match.train);
        }
        write_optional(
            writer, pair.relative_pose,
            [](Writer& out, const Pose3D& pose) { write_pose(out, pose); });
        const auto write_mat = [](Writer& out, const Mat3& matrix) {
            write_matrix(out, matrix);
        };
        write_optional(writer, pair.E, write_mat);
        write_optional(writer, pair.F, write_mat);
        write_optional(writer, pair.H, write_mat);
        writer.value(pair.weight_spatial);
        writer.value(pair.weight_geometry);
        writer.value(pair.mean_ray_angle);
        writer.value(pair.homography_ratio);
        writer.value(static_cast<std::uint8_t>(pair.degenerate_planar));
        writer.value(static_cast<std::uint8_t>(pair.active));
    }
    writer.value_size(scene.tracks.size());
    for (const Track& track : scene.tracks) {
        for (Eigen::Index axis = 0; axis < 3; ++axis)
            writer.value(track.position[axis]);
        writer.value_size(track.observations.size());
        for (const Observation& observation : track.observations) {
            writer.value(observation.image_id);
            writer.value(observation.feature_id);
        }
        writer.value(track.num_inliers);
    }
}

void validate_scene(const Scene& scene) {
    if (!scene.images.empty() && scene.cameras.empty())
        throw std::runtime_error("Checkpoint scene has no cameras");
    for (std::size_t image_index = 0;
         image_index < scene.images.size(); ++image_index) {
        const Image& image = scene.images[image_index];
        if (image.id != image_index || image.camera_id >= scene.cameras.size())
            throw std::runtime_error(
                "Checkpoint image references an invalid parameter block");
        const auto metric =
            static_cast<std::uint32_t>(image.features.metric);
        if (metric >
            static_cast<std::uint32_t>(
                features::DescriptorMetric::inner_product))
            throw std::runtime_error("Checkpoint descriptor metric is invalid");
        if (!image.features.descriptors.empty())
            image.features.validate();
    }
    for (const Index image_id : scene.resection_progress.last_registered)
        if (image_id >= scene.images.size() ||
            !scene.images[image_id].registered)
            throw std::runtime_error(
                "Checkpoint resection state is invalid");
    for (const ImagePair& pair : scene.pairs) {
        if (pair.id1 >= scene.images.size() ||
            pair.id2 >= scene.images.size() || pair.id1 >= pair.id2)
            throw std::runtime_error("Checkpoint image pair is invalid");
        const auto& first = scene.images[pair.id1].features.keypoints;
        const auto& second = scene.images[pair.id2].features.keypoints;
        for (const FeatureMatch& match : pair.matches)
            if (match.query >= first.size() || match.train >= second.size())
                throw std::runtime_error(
                    "Checkpoint pair references an invalid feature");
    }
    for (const Track& track : scene.tracks) {
        if (track.num_inliers > track.observations.size())
            throw std::runtime_error(
                "Checkpoint track inlier count is invalid");
        for (const Observation& observation : track.observations)
            if (observation.image_id >= scene.images.size() ||
                observation.feature_id >=
                    scene.images[observation.image_id]
                        .features.keypoints.size())
                throw std::runtime_error(
                    "Checkpoint track references an invalid feature");
    }
}

Scene read_scene(Reader& reader) {
    Scene scene;
    scene.thread_count = reader.value<unsigned>();
    scene.resection_progress.since_full_ba = reader.value<unsigned>();
    scene.resection_progress.bundle_adjustment_stage =
        reader.value<unsigned>();
    read_vector(reader, scene.resection_progress.last_registered);
    read_vector(reader, scene.resection_progress.recent_inlier_ratios);
    if (scene.resection_progress.recent_inlier_ratios.size() > 10)
        throw std::runtime_error(
            "Checkpoint resection history is invalid");
    scene.cameras.resize(reader.size(64, 10'000'000));
    for (PinholeCamera& camera : scene.cameras) {
        camera.id = reader.value<Index>();
        camera.width = reader.value<std::uint32_t>();
        camera.height = reader.value<std::uint32_t>();
        camera.fx = reader.value<double>();
        camera.fy = reader.value<double>();
        camera.cx = reader.value<double>();
        camera.cy = reader.value<double>();
        camera.k1 = reader.value<double>();
        camera.k2 = reader.value<double>();
        camera.p1 = reader.value<double>();
        camera.p2 = reader.value<double>();
        camera.trust_intrinsics = reader.value<std::uint8_t>() != 0;
    }
    scene.images.resize(reader.size(32, 10'000'000));
    for (Image& image : scene.images) {
        image.id = reader.value<Index>();
        image.camera_id = reader.value<Index>();
        image.path = path_from_utf8(reader.string());
        image.features = read_feature_set(reader);
        image.pose = read_pose(reader);
        image.registered = reader.value<std::uint8_t>() != 0;
    }
    scene.pairs.resize(reader.size(32, 100'000'000));
    for (ImagePair& pair : scene.pairs) {
        pair.id1 = reader.value<Index>();
        pair.id2 = reader.value<Index>();
        pair.matches.resize(reader.size(2 * sizeof(Index), 1'000'000'000));
        for (FeatureMatch& match : pair.matches) {
            match.query = reader.value<Index>();
            match.train = reader.value<Index>();
        }
        pair.relative_pose = read_optional<Pose3D>(
            reader, [](Reader& in) { return read_pose(in); });
        const auto read_mat = [](Reader& in) { return read_matrix(in); };
        pair.E = read_optional<Mat3>(reader, read_mat);
        pair.F = read_optional<Mat3>(reader, read_mat);
        pair.H = read_optional<Mat3>(reader, read_mat);
        pair.weight_spatial = reader.value<float>();
        pair.weight_geometry = reader.value<float>();
        pair.mean_ray_angle = reader.value<float>();
        pair.homography_ratio = reader.value<float>();
        pair.degenerate_planar = reader.value<std::uint8_t>() != 0;
        pair.active = reader.value<std::uint8_t>() != 0;
    }
    scene.tracks.resize(reader.size(32, 1'000'000'000));
    for (Track& track : scene.tracks) {
        for (Eigen::Index axis = 0; axis < 3; ++axis)
            track.position[axis] = reader.value<double>();
        track.observations.resize(
            reader.size(2 * sizeof(Index), 1'000'000'000));
        for (Observation& observation : track.observations) {
            observation.image_id = reader.value<Index>();
            observation.feature_id = reader.value<Index>();
        }
        track.num_inliers = reader.value<std::uint8_t>();
    }
    validate_scene(scene);
    rebuild_track_index(scene);
    return scene;
}

void write_matches(
    Writer& writer, const std::vector<RawPairMatches>& matches) {
    writer.value_size(matches.size());
    for (const RawPairMatches& pair : matches) {
        writer.value(pair.id1);
        writer.value(pair.id2);
        writer.value_size(pair.matches.size());
        for (const features::FeatureMatch& match : pair.matches) {
            writer.value(match.query);
            writer.value(match.train);
            writer.value(match.score);
        }
    }
}

std::vector<RawPairMatches> read_matches(Reader& reader) {
    std::vector<RawPairMatches> matches(
        reader.size(16, 100'000'000));
    for (RawPairMatches& pair : matches) {
        pair.id1 = reader.value<Index>();
        pair.id2 = reader.value<Index>();
        pair.matches.resize(
            reader.size(
                2 * sizeof(features::FeatureIndex) + sizeof(float),
                1'000'000'000));
        for (features::FeatureMatch& match : pair.matches) {
            match.query = reader.value<features::FeatureIndex>();
            match.train = reader.value<features::FeatureIndex>();
            match.score = reader.value<float>();
        }
    }
    return matches;
}

std::string stage_prefix(const CheckpointStage stage) {
    switch (stage) {
        case CheckpointStage::features: return "features";
        case CheckpointStage::matches: return "matches";
        case CheckpointStage::geometry: return "geometry";
        case CheckpointStage::tracks: return "tracks";
        case CheckpointStage::reconstruction: return "reconstruction";
    }
    throw std::invalid_argument("Unknown checkpoint stage");
}

std::filesystem::path stage_path(
    const CheckpointOptions& options, const CheckpointStage stage,
    const std::uint64_t key) {
    std::array<char, 16> key_text{};
    const auto converted = std::to_chars(
        key_text.data(), key_text.data() + key_text.size(), key, 16);
    return options.directory /
        (stage_prefix(stage) + "-" +
         std::string(key_text.data(), converted.ptr) + ".bin");
}

void prune_variants(
    const CheckpointOptions& options, const CheckpointStage stage,
    const std::filesystem::path& current) {
    if (options.max_variants_per_stage == 0) return;
    struct Candidate {
        std::filesystem::path path;
        std::filesystem::file_time_type modified;
    };
    std::vector<Candidate> candidates;
    const std::string prefix = stage_prefix(stage) + "-";
    std::error_code error;
    const auto current_modified =
        std::filesystem::last_write_time(current, error);
    if (error) return;
    for (const auto& entry :
         std::filesystem::directory_iterator(options.directory, error)) {
        if (error) return;
        if (!entry.is_regular_file(error) || error) continue;
        const std::string filename = entry.path().filename().string();
        if (!filename.starts_with(prefix) || !filename.ends_with(".bin"))
            continue;
        candidates.push_back({entry.path(), entry.last_write_time(error)});
        if (error) error.clear();
    }
    if (candidates.size() <= options.max_variants_per_stage) return;
    std::sort(
        candidates.begin(), candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            return left.modified < right.modified;
        });
    std::size_t remove_count =
        candidates.size() - options.max_variants_per_stage;
    for (const Candidate& candidate : candidates) {
        if (remove_count == 0) break;
        if (candidate.path == current ||
            candidate.modified >= current_modified)
            continue;
        std::filesystem::remove(candidate.path, error);
        if (!error) --remove_count;
        error.clear();
    }
}

void atomic_replace(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination) {
#if defined(_WIN32)
    for (unsigned attempt = 0; attempt < 6; ++attempt) {
        if (MoveFileExW(
                temporary.c_str(), destination.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            return;
        const DWORD error = GetLastError();
        if (error != ERROR_SHARING_VIOLATION &&
            error != ERROR_ACCESS_DENIED)
            throw std::system_error(
                static_cast<int>(error), std::system_category(),
                "Failed to atomically replace checkpoint");
        std::this_thread::sleep_for(
            std::chrono::milliseconds(5U << attempt));
    }
    throw std::system_error(
        static_cast<int>(GetLastError()), std::system_category(),
        "Failed to atomically replace busy checkpoint");
#else
    std::filesystem::rename(temporary, destination);
    const int directory = ::open(
        destination.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
    if (directory < 0)
        throw std::system_error(
            errno, std::generic_category(),
            "Failed to open checkpoint directory for sync");
    const int synced = ::fsync(directory);
    const int error = synced == 0 ? 0 : errno;
    ::close(directory);
    if (synced != 0)
        throw std::system_error(
            error, std::generic_category(),
            "Failed to sync checkpoint directory");
#endif
}

void flush_file(const std::filesystem::path& path) {
#if defined(_WIN32)
    const HANDLE file = CreateFileW(
        path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::system_error(
            static_cast<int>(GetLastError()), std::system_category(),
            "Failed to reopen checkpoint for flush");
    const BOOL flushed = FlushFileBuffers(file);
    const DWORD error = flushed ? ERROR_SUCCESS : GetLastError();
    CloseHandle(file);
    if (!flushed)
        throw std::system_error(
            static_cast<int>(error), std::system_category(),
            "Failed to flush checkpoint");
#else
    const int file = ::open(path.c_str(), O_RDONLY);
    if (file < 0)
        throw std::system_error(
            errno, std::generic_category(),
            "Failed to open checkpoint for sync");
    const int synced = ::fsync(file);
    const int error = synced == 0 ? 0 : errno;
    ::close(file);
    if (synced != 0)
        throw std::system_error(
            error, std::generic_category(),
            "Failed to sync checkpoint");
#endif
}

template <class WritePayload>
void save_file(
    const CheckpointOptions& options, const CheckpointStage stage,
    const std::uint64_t key, WritePayload&& write_payload) {
    if (!options.enabled() || !options.write) return;
    std::filesystem::create_directories(options.directory);
    const std::filesystem::path destination =
        stage_path(options, stage, key);
    const auto suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    static std::atomic<std::uint64_t> temporary_counter{0};
    const std::uint64_t counter =
        temporary_counter.fetch_add(1, std::memory_order_relaxed);
#if defined(_WIN32)
    const std::wstring temporary_suffix =
        L".tmp." + std::to_wstring(GetCurrentProcessId()) + L"." +
        std::to_wstring(counter) + L"." + std::to_wstring(suffix);
    std::filesystem::path temporary = destination;
    temporary += temporary_suffix;
#else
    std::filesystem::path temporary = destination;
    temporary +=
        ".tmp." + std::to_string(counter) + "." + std::to_string(suffix);
#endif
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error(
                "Failed to create checkpoint: " + temporary.string());
        FileHeader header{magic, schema_version,
                          static_cast<std::uint32_t>(stage), key, 0, 0};
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        Writer writer(output);
        write_payload(writer);
        header.payload_size = writer.size();
        header.payload_checksum = writer.checksum();
        output.seekp(0);
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.flush();
        if (!output) throw std::runtime_error("Failed to finalize checkpoint");
        output.close();
        flush_file(temporary);
        atomic_replace(temporary, destination);
        prune_variants(options, stage, destination);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

template <class ReadPayload>
bool load_file(
    const CheckpointOptions& options, const CheckpointStage stage,
    const std::uint64_t key, ReadPayload&& read_payload) {
    if (!options.enabled() || !options.read) return false;
    const std::filesystem::path path = stage_path(options, stage, key);
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    try {
        FileHeader header{};
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!input || header.signature != magic ||
            header.version != schema_version ||
            header.stage != static_cast<std::uint32_t>(stage) ||
            header.key != key)
            return false;
        const std::uint64_t actual_size = std::filesystem::file_size(path);
        if (actual_size < sizeof(header) ||
            header.payload_size != actual_size - sizeof(header))
            throw std::runtime_error("Checkpoint file size mismatch");
        Reader reader(input, header.payload_size);
        read_payload(reader);
        reader.finish(header.payload_checksum);
        return true;
    } catch (const std::exception& error) {
        std::cerr << "checkpoint ignored: " << path
                  << " (" << error.what() << ")\n";
        return false;
    }
}

}  // namespace

void FingerprintBuilder::append_bytes(
    const void* data, const std::size_t size) noexcept {
    hash_bytes(value_, data, size);
}

void FingerprintBuilder::append_string(const std::string_view value) noexcept {
    const std::uint64_t size = value.size();
    append(size);
    append_bytes(value.data(), value.size());
}

std::uint64_t fingerprint_images(
    const std::vector<std::filesystem::path>& image_paths) {
    FingerprintBuilder fingerprint;
    fingerprint.append_string("aetherscan-images-sha256-v2");
    fingerprint.append(static_cast<std::uint64_t>(image_paths.size()));
    for (const std::filesystem::path& path : image_paths) {
        std::error_code error;
        std::filesystem::path normalized =
            std::filesystem::weakly_canonical(path, error);
        if (error) normalized = std::filesystem::absolute(path, error);
        fingerprint.append_string(path_utf8(normalized));
        const std::uint64_t size = std::filesystem::file_size(path);
        fingerprint.append(size);
        const auto digest = hash_file(path);
        fingerprint.append_bytes(digest.data(), digest.size());
    }
    return fingerprint.value();
}

CheckpointStore::CheckpointStore(CheckpointOptions options)
    : options_(std::move(options)) {}

bool CheckpointStore::load_scene(
    const CheckpointStage stage, const std::uint64_t key, Scene& scene) const {
    Scene loaded;
    const bool hit = load_file(
        options_, stage, key,
        [&](Reader& reader) { loaded = read_scene(reader); });
    if (hit) scene = std::move(loaded);
    return hit;
}

void CheckpointStore::save_scene(
    const CheckpointStage stage, const std::uint64_t key,
    const Scene& scene) const {
    save_file(
        options_, stage, key,
        [&](Writer& writer) {
            write_scene(
                writer, scene, stage == CheckpointStage::features);
        });
}

bool CheckpointStore::load_matches(
    const std::uint64_t key,
    std::vector<RawPairMatches>& matches) const {
    std::vector<RawPairMatches> loaded;
    const bool hit = load_file(
        options_, CheckpointStage::matches, key,
        [&](Reader& reader) { loaded = read_matches(reader); });
    if (hit) matches = std::move(loaded);
    return hit;
}

void CheckpointStore::save_matches(
    const std::uint64_t key,
    const std::vector<RawPairMatches>& matches) const {
    save_file(
        options_, CheckpointStage::matches, key,
        [&](Writer& writer) { write_matches(writer, matches); });
}

}  // namespace aetherscan::sfm
