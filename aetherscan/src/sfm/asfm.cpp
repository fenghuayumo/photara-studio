#include "sfm/asfm.hpp"
#include "sfm/tracks.hpp"

#include "io/format_version.hpp"
#include "../io/binary_codec.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace aetherscan::sfm {
namespace {

constexpr std::array<char, 8> k_magic{
    'A', 'E', 'T', 'H', 'S', 'F', 'M', '\0'};

using io::binary::BufferReader;
using io::binary::BufferWriter;

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

std::string store_path(
    const std::filesystem::path& path, const std::filesystem::path& base) {
    if (base.empty() || path.empty()) return path_utf8(path);
    std::error_code error;
    const auto relative = std::filesystem::relative(path, base, error);
    const std::string relative_text = relative.generic_string();
    if (!error && !relative_text.empty() &&
        relative_text.find("..") == std::string::npos)
        return relative_text;
    return path_utf8(std::filesystem::weakly_canonical(path, error));
}

std::filesystem::path load_path(
    const std::string& stored, const std::filesystem::path& base) {
    std::filesystem::path path = path_from_utf8(stored);
    if (path.empty() || path.is_absolute() || base.empty()) return path;
    return std::filesystem::weakly_canonical(base / path);
}

void write_pose(BufferWriter& writer, const Pose3D& pose) {
    for (Eigen::Index row = 0; row < 3; ++row)
        for (Eigen::Index column = 0; column < 3; ++column)
            writer.value(pose.R(row, column));
    for (Eigen::Index axis = 0; axis < 3; ++axis) writer.value(pose.C[axis]);
}

Pose3D read_pose(BufferReader& reader) {
    Pose3D pose;
    for (Eigen::Index row = 0; row < 3; ++row)
        for (Eigen::Index column = 0; column < 3; ++column)
            pose.R(row, column) = reader.value<double>();
    for (Eigen::Index axis = 0; axis < 3; ++axis)
        pose.C[axis] = reader.value<double>();
    return pose;
}

void write_payload(
    BufferWriter& writer, const Scene& scene, const AsfmOptions& options) {
    writer.value_size(scene.cameras.size());
    for (const PinholeCamera& camera : scene.cameras) {
        writer.record([&](BufferWriter& record) {
            record.value(camera.id);
            record.value(camera.width);
            record.value(camera.height);
            record.value(camera.fx);
            record.value(camera.fy);
            record.value(camera.cx);
            record.value(camera.cy);
            record.value(camera.k1);
            record.value(camera.k2);
            record.value(camera.p1);
            record.value(camera.p2);
            record.value(camera.focal_prior);
            record.value(static_cast<std::uint8_t>(camera.trust_intrinsics));
        });
    }

    writer.value_size(scene.images.size());
    for (const Image& image : scene.images) {
        writer.record([&](BufferWriter& record) {
            record.value(image.id);
            record.value(image.camera_id);
            record.string(store_path(image.path, options.path_base));
            write_pose(record, image.pose);
            record.value(static_cast<std::uint8_t>(image.registered));
            record.value(image.features.image_width);
            record.value(image.features.image_height);
            record.string(image.features.extractor_name);
            record.value_size(image.features.keypoints.size());
            for (const features::Keypoint& keypoint : image.features.keypoints) {
                record.value(keypoint.x);
                record.value(keypoint.y);
                record.value(keypoint.scale);
                record.value(keypoint.orientation);
                record.value(keypoint.response);
            }
        });
    }

    std::uint64_t track_count = 0;
    for (const Track& track : scene.tracks)
        if (track.is_triangulated()) ++track_count;
    writer.value_size(static_cast<std::size_t>(track_count));
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        writer.record([&](BufferWriter& record) {
            for (Eigen::Index axis = 0; axis < 3; ++axis)
                record.value(track.position[axis]);
            record.value_size(track.observations.size());
            for (const Observation& observation : track.observations) {
                record.value(observation.image_id);
                record.value(observation.feature_id);
            }
            record.value(static_cast<std::uint32_t>(track.num_inliers));
        });
    }
}

Scene read_payload(BufferReader& reader, const AsfmOptions& options) {
    Scene scene;
    scene.cameras.resize(reader.size(8, 10'000'000));
    for (PinholeCamera& camera : scene.cameras) {
        reader.record([&](BufferReader& record) {
            camera.id = record.value<Index>();
            camera.width = record.value<std::uint32_t>();
            camera.height = record.value<std::uint32_t>();
            camera.fx = record.value<double>();
            camera.fy = record.value<double>();
            camera.cx = record.value<double>();
            camera.cy = record.value<double>();
            camera.k1 = record.value<double>();
            camera.k2 = record.value<double>();
            camera.p1 = record.value<double>();
            camera.p2 = record.value<double>();
            camera.focal_prior = record.value<double>();
            camera.trust_intrinsics = record.value<std::uint8_t>() != 0;
        });
    }

    scene.images.resize(reader.size(8, 10'000'000));
    for (std::size_t index = 0; index < scene.images.size(); ++index) {
        Image& image = scene.images[index];
        reader.record([&](BufferReader& record) {
            image.id = record.value<Index>();
            image.camera_id = record.value<Index>();
            image.path = load_path(record.string(), options.path_base);
            image.pose = read_pose(record);
            image.registered = record.value<std::uint8_t>() != 0;
            image.features.image_width = record.value<std::uint32_t>();
            image.features.image_height = record.value<std::uint32_t>();
            image.features.extractor_name = record.string();
            image.features.descriptor_dimension = 0;
            image.features.keypoints.resize(
                record.size(sizeof(float) * 5, 10'000'000));
            for (features::Keypoint& keypoint : image.features.keypoints) {
                keypoint.x = record.value<float>();
                keypoint.y = record.value<float>();
                keypoint.scale = record.value<float>();
                keypoint.orientation = record.value<float>();
                keypoint.response = record.value<float>();
            }
        });
        if (image.id != index || image.camera_id >= scene.cameras.size())
            throw std::runtime_error("ASFM image references an invalid camera");
        if ((image.features.image_width != 0 ||
             image.features.image_height != 0) &&
            !image.features.keypoints.empty())
            image.features.validate();
    }

    scene.tracks.resize(reader.size(8, 1'000'000'000));
    for (Track& track : scene.tracks) {
        reader.record([&](BufferReader& record) {
            for (Eigen::Index axis = 0; axis < 3; ++axis)
                track.position[axis] = record.value<double>();
            track.observations.resize(
                record.size(2 * sizeof(Index), 1'000'000'000));
            for (Observation& observation : track.observations) {
                observation.image_id = record.value<Index>();
                observation.feature_id = record.value<Index>();
            }
            const auto inliers = record.value<std::uint32_t>();
            if (inliers > track.observations.size())
                throw std::runtime_error("ASFM track inlier count is invalid");
            track.num_inliers = static_cast<std::uint8_t>(
                (std::min)(inliers, 255U));
        });
        for (const Observation& observation : track.observations) {
            if (observation.image_id >= scene.images.size() ||
                observation.feature_id >=
                    scene.images[observation.image_id].features.keypoints.size())
                throw std::runtime_error(
                    "ASFM track references an invalid feature");
        }
    }

    rebuild_track_index(scene);
    return scene;
}

}  // namespace

std::vector<std::uint8_t> encode_asfm(
    const Scene& scene, const AsfmOptions& options) {
    BufferWriter payload;
    write_payload(payload, scene, options);

    BufferWriter file;
    file.bytes(k_magic.data(), k_magic.size());
    file.value(k_asfm_version);
    file.value(k_asfm_min_reader);
    file.value(payload.size());
    file.value(payload.checksum());
    file.bytes(payload.buffer().data(), payload.buffer().size());
    return file.take();
}

Scene decode_asfm(
    const std::span<const std::uint8_t> bytes, const AsfmOptions& options) {
    if (bytes.size() < 8 + 4 + 4 + 8 + 8)
        throw std::runtime_error("ASFM file is too small");
    BufferReader reader(bytes);
    std::array<char, 8> magic{};
    reader.bytes(magic.data(), magic.size());
    if (magic != k_magic) throw std::runtime_error("Not an AetherScan .asfm file");
    const auto version = reader.value<std::uint32_t>();
    const auto min_reader = reader.value<std::uint32_t>();
    io::require_readable(version, min_reader, k_asfm_version, ".asfm");
    const auto payload_size = reader.value<std::uint64_t>();
    const auto checksum = reader.value<std::uint64_t>();
    if (payload_size != reader.remaining())
        throw std::runtime_error("ASFM payload size mismatch");
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(payload_size));
    reader.bytes(payload.data(), payload.size());
    BufferReader body(payload);
    Scene scene = read_payload(body, options);
    body.finish(checksum);
    return scene;
}

void save_asfm(
    const Scene& scene,
    const std::filesystem::path& path,
    const AsfmOptions& options) {
    const auto bytes = encode_asfm(scene, options);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Failed to create ASFM file: " + path.string());
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size()));
    if (!output) throw std::runtime_error("Failed while writing ASFM file: " + path.string());
}

Scene load_asfm(
    const std::filesystem::path& path, const AsfmOptions& options) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open ASFM file: " + path.string());
    input.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(size);
    if (size > 0)
        input.read(
            reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(size));
    if (!input) throw std::runtime_error("Failed while reading ASFM file: " + path.string());
    AsfmOptions resolved = options;
    if (resolved.path_base.empty()) resolved.path_base = path.parent_path();
    return decode_asfm(bytes, resolved);
}

}  // namespace aetherscan::sfm
