#include "aetherscan/sfm/export_mvs.hpp"

#include "aetherscan/io/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace aetherscan::sfm {
namespace {

constexpr char k_mvsi_id[4] = {'M', 'V', 'S', 'I'};
constexpr std::uint32_t k_mvsi_version = 7;

class BinaryWriter {
public:
    explicit BinaryWriter(std::ofstream& stream) : stream_(stream) {}

    template <typename T>
    void write_pod(const T& value) {
        stream_.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    void write_bytes(const void* data, const std::size_t size) {
        if (size > 0) stream_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    void write_string(const std::string& value) {
        write_pod(static_cast<std::uint64_t>(value.size()));
        write_bytes(value.data(), value.size());
    }

    template <typename T>
    void write_array(const T* data, const std::size_t count) {
        write_bytes(data, sizeof(T) * count);
    }

    [[nodiscard]] bool good() const { return static_cast<bool>(stream_); }

private:
    std::ofstream& stream_;
};

std::array<double, 9> pose_rotation(const ba::Pose& pose) {
    const double w = pose.qw, x = pose.qx, y = pose.qy, z = pose.qz;
    return {
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    };
}

std::string unify_slash(std::string path) {
    for (char& ch : path) {
        if (ch == '\\') ch = '/';
    }
    return path;
}

std::string image_name_for_export(
    const std::filesystem::path& image_path,
    const std::filesystem::path& base) {
    std::error_code ec;
    if (!base.empty()) {
        const auto relative = std::filesystem::relative(image_path, base, ec);
        const std::string relative_text = relative.generic_string();
        if (!ec && !relative_text.empty() && relative_text.find("..") == std::string::npos) {
            return unify_slash(relative_text);
        }
    }
    return unify_slash(std::filesystem::absolute(image_path).generic_string());
}

const io::RgbImage* cached_rgb(
    std::unordered_map<std::uint32_t, std::unique_ptr<io::RgbImage>>& cache,
    const Scene& scene,
    const std::uint32_t view_id) {
    const auto existing = cache.find(view_id);
    if (existing != cache.end()) return existing->second.get();
    try {
        auto image = std::make_unique<io::RgbImage>(io::load_rgb(scene.views[view_id].image_path));
        const io::RgbImage* pointer = image.get();
        cache.emplace(view_id, std::move(image));
        return pointer;
    } catch (...) {
        cache.emplace(view_id, nullptr);
        return nullptr;
    }
}

bool sample_rgb(
    const io::RgbImage& image, const float x, const float y,
    double& r, double& g, double& b) {
    if (image.width == 0 || image.height == 0 || image.pixels.size() < 3) return false;
    const int xi = static_cast<int>(std::lround(x));
    const int yi = static_cast<int>(std::lround(y));
    if (xi < 0 || yi < 0 ||
        xi >= static_cast<int>(image.width) || yi >= static_cast<int>(image.height)) {
        return false;
    }
    const std::size_t offset =
        (static_cast<std::size_t>(yi) * image.width + static_cast<std::size_t>(xi)) * 3;
    r = image.pixels[offset + 0];
    g = image.pixels[offset + 1];
    b = image.pixels[offset + 2];
    return true;
}

}  // namespace

void export_openmvs_interface(
    const Scene& scene,
    const std::filesystem::path& path,
    const ExportMvsOptions& options) {
    std::vector<std::uint32_t> registered_views;
    registered_views.reserve(scene.views.size());
    for (std::uint32_t i = 0; i < scene.views.size(); ++i) {
        if (scene.views[i].registered) registered_views.push_back(i);
    }
    if (registered_views.size() < 2)
        throw std::runtime_error("OpenMVS export needs at least two registered views");

    std::unordered_map<std::uint32_t, std::uint32_t> view_to_image;
    view_to_image.reserve(registered_views.size());
    for (std::uint32_t image_id = 0; image_id < registered_views.size(); ++image_id)
        view_to_image.emplace(registered_views[image_id], image_id);

    const auto& first_view = scene.views[registered_views.front()];
    if (first_view.camera_id >= scene.cameras.size())
        throw std::runtime_error("Registered view references invalid camera");
    const Camera& shared_camera = scene.cameras[first_view.camera_id];

    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Failed to create MVS file: " + path.string());
    BinaryWriter writer(stream);

    writer.write_bytes(k_mvsi_id, 4);
    writer.write_pod(k_mvsi_version);
    writer.write_pod(std::uint32_t{0});  // reserved

    writer.write_pod(std::uint64_t{1});
    writer.write_string("platform0");

    writer.write_pod(std::uint64_t{1});
    writer.write_string("camera0");
    writer.write_string("RGB");
    writer.write_pod(shared_camera.width);
    writer.write_pod(shared_camera.height);
    const double k[9] = {
        shared_camera.fx, 0.0, shared_camera.cx,
        0.0, shared_camera.fy, shared_camera.cy,
        0.0, 0.0, 1.0};
    writer.write_array(k, 9);
    const double identity_r[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double zero_c[3] = {0, 0, 0};
    writer.write_array(identity_r, 9);
    writer.write_array(zero_c, 3);

    writer.write_pod(static_cast<std::uint64_t>(registered_views.size()));
    for (const std::uint32_t view_id : registered_views) {
        const auto rotation = pose_rotation(scene.views[view_id].pose);
        const double center[3] = {
            scene.views[view_id].pose.cx,
            scene.views[view_id].pose.cy,
            scene.views[view_id].pose.cz};
        writer.write_array(rotation.data(), 9);
        writer.write_array(center, 3);
    }

    writer.write_pod(static_cast<std::uint64_t>(registered_views.size()));
    for (std::uint32_t image_id = 0; image_id < registered_views.size(); ++image_id) {
        const auto& view = scene.views[registered_views[image_id]];
        writer.write_string(image_name_for_export(view.image_path, options.image_path_base));
        writer.write_string("");
        writer.write_pod(std::uint32_t{0});
        writer.write_pod(std::uint32_t{0});
        writer.write_pod(image_id);
        writer.write_pod(registered_views[image_id]);
        writer.write_pod(0.0f);
        writer.write_pod(0.0f);
        writer.write_pod(0.0f);
        writer.write_pod(std::uint64_t{0});
    }

    struct ExportedVertex {
        float x, y, z;
        std::uint8_t b{255}, g{255}, r{255};
        std::vector<std::pair<std::uint32_t, float>> views;
    };
    std::vector<ExportedVertex> vertices;
    vertices.reserve(scene.landmarks.size());
    std::unordered_map<std::uint32_t, std::unique_ptr<io::RgbImage>> rgb_cache;
    bool wrote_any_color = false;

    for (const auto& landmark : scene.landmarks) {
        if (landmark.track_id >= scene.tracks.size()) continue;
        const Track& track = scene.tracks[landmark.track_id];
        ExportedVertex vertex{
            static_cast<float>(landmark.position[0]),
            static_cast<float>(landmark.position[1]),
            static_cast<float>(landmark.position[2]),
            255,
            255,
            255,
            {}};
        vertex.views.reserve(track.observations.size());

        double sum_r = 0, sum_g = 0, sum_b = 0;
        std::size_t color_samples = 0;

        for (const auto& observation : track.observations) {
            const auto found = view_to_image.find(observation.view_id);
            if (found == view_to_image.end()) continue;
            vertex.views.emplace_back(found->second, 0.0f);

            if (!options.sample_colors) continue;
            if (observation.view_id >= scene.views.size()) continue;
            const auto& view = scene.views[observation.view_id];
            if (observation.feature_index >= view.features.keypoints.size()) continue;
            const auto* image = cached_rgb(rgb_cache, scene, observation.view_id);
            if (!image) continue;
            const auto& keypoint = view.features.keypoints[observation.feature_index];
            double r = 0, g = 0, b = 0;
            if (!sample_rgb(*image, keypoint.x, keypoint.y, r, g, b)) continue;
            sum_r += r;
            sum_g += g;
            sum_b += b;
            ++color_samples;
        }

        if (vertex.views.size() < 2) continue;
        std::sort(
            vertex.views.begin(), vertex.views.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        if (color_samples > 0) {
            vertex.r = static_cast<std::uint8_t>(std::lround(sum_r / color_samples));
            vertex.g = static_cast<std::uint8_t>(std::lround(sum_g / color_samples));
            vertex.b = static_cast<std::uint8_t>(std::lround(sum_b / color_samples));
            wrote_any_color = true;
        }
        vertices.push_back(std::move(vertex));
    }

    writer.write_pod(static_cast<std::uint64_t>(vertices.size()));
    for (const auto& vertex : vertices) {
        const float xyz[3] = {vertex.x, vertex.y, vertex.z};
        writer.write_array(xyz, 3);
        writer.write_pod(static_cast<std::uint64_t>(vertex.views.size()));
        for (const auto& [image_id, confidence] : vertex.views) {
            writer.write_pod(image_id);
            writer.write_pod(confidence);
        }
    }

    writer.write_pod(std::uint64_t{0});  // verticesNormal
    if (options.sample_colors && wrote_any_color) {
        writer.write_pod(static_cast<std::uint64_t>(vertices.size()));
        for (const auto& vertex : vertices) {
            // OpenMVS Col3 is BGR.
            const std::uint8_t bgr[3] = {vertex.b, vertex.g, vertex.r};
            writer.write_array(bgr, 3);
        }
    } else {
        writer.write_pod(std::uint64_t{0});
    }

    writer.write_pod(std::uint64_t{0});
    writer.write_pod(std::uint64_t{0});
    writer.write_pod(std::uint64_t{0});

    const double identity4[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1};
    writer.write_array(identity4, 16);
    writer.write_array(identity_r, 9);
    writer.write_array(zero_c, 3);
    writer.write_array(zero_c, 3);

    if (!writer.good())
        throw std::runtime_error("Failed while writing MVS file: " + path.string());
}

}  // namespace aetherscan::sfm
