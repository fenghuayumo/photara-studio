#include "aetherscan/sfm/export_mvs.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::uint64_t read_u64(std::ifstream& stream) {
    std::uint64_t value = 0;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

std::uint32_t read_u32(std::ifstream& stream) {
    std::uint32_t value = 0;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

std::string read_string(std::ifstream& stream) {
    const auto size = read_u64(stream);
    std::string value(size, '\0');
    if (size > 0) stream.read(value.data(), static_cast<std::streamsize>(size));
    return value;
}

aetherscan::sfm::Scene make_registered_scene() {
    using namespace aetherscan::sfm;
    Scene scene;
    for (Id i = 0; i < 3; ++i) {
        scene.cameras.push_back({i, 640, 480, 500, 500, 320, 240});
        View view;
        view.id = i;
        view.camera_id = i;
        view.image_path = "img_" + std::to_string(i) + ".jpg";
        view.registered = true;
        view.pose.cx = 0.3 * i;
        scene.views.push_back(std::move(view));
    }
    for (Id t = 0; t < 5; ++t) {
        Track track;
        track.id = t;
        track.observations = {{0, 0}, {1, 0}, {2, 0}};
        scene.tracks.push_back(std::move(track));
        Landmark landmark;
        landmark.id = t;
        landmark.track_id = t;
        landmark.position = {0.1 * t, 0.2, 5.0};
        scene.landmarks.push_back(landmark);
    }
    // One landmark with a single observation should be skipped.
    Track short_track;
    short_track.id = 5;
    short_track.observations = {{0, 0}};
    scene.tracks.push_back(std::move(short_track));
    Landmark short_landmark;
    short_landmark.id = 5;
    short_landmark.track_id = 5;
    short_landmark.position = {9, 9, 9};
    scene.landmarks.push_back(short_landmark);
    return scene;
}

}  // namespace

int main() {
    using namespace aetherscan::sfm;
    const auto scene = make_registered_scene();
    const auto path = std::filesystem::temp_directory_path() / "aetherscan_export_test.mvs";
    ExportMvsOptions options;
    options.image_path_base = std::filesystem::current_path();
    options.sample_colors = false;
    export_openmvs_interface(scene, path, options);

    std::ifstream stream(path, std::ios::binary);
    if (!stream) return 2;
    char magic[4]{};
    stream.read(magic, 4);
    if (std::string(magic, 4) != "MVSI") return 3;
    if (read_u32(stream) != 7) return 4;
    read_u32(stream);  // reserved

    if (read_u64(stream) != 1) return 5;
    read_string(stream);
    if (read_u64(stream) != 1) return 6;
    read_string(stream);
    read_string(stream);
    if (read_u32(stream) != 640 || read_u32(stream) != 480) return 7;
    double k[9]{};
    stream.read(reinterpret_cast<char*>(k), sizeof(k));
    if (std::abs(k[0] - 500.0) > 1e-9) return 8;
    stream.seekg(9 * 8 + 3 * 8, std::ios::cur);  // camera R,C
    if (read_u64(stream) != 3) return 9;
    stream.seekg(3 * (9 + 3) * 8, std::ios::cur);

    if (read_u64(stream) != 3) return 10;
    for (int i = 0; i < 3; ++i) {
        read_string(stream);
        read_string(stream);
        stream.seekg(4 * 4 + 3 * 4, std::ios::cur);
        if (read_u64(stream) != 0) return 11;
    }

    if (read_u64(stream) != 5) return 12;  // short landmark dropped
    for (int i = 0; i < 5; ++i) {
        stream.seekg(3 * 4, std::ios::cur);
        const auto views = read_u64(stream);
        if (views != 3) return 13;
        stream.seekg(static_cast<std::streamoff>(views * 8), std::ios::cur);
    }

    if (read_u64(stream) != 0) return 14;  // normals
    if (read_u64(stream) != 0) return 15;  // colors

    std::cout << "mvs_ok path=" << path << '\n';
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return stream ? 0 : 16;
}
