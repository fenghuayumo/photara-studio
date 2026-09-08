#include "sfm/export_mvs.hpp"
#include "core/logging.hpp"

#include "io/image.hpp"

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

std::array<double, 9> pose_rotation(const Pose3D& pose) {
    const Mat3& R = pose.R;
    return {R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2)};
}

std::string unify_slash(std::string path) {
    for (char& ch : path) {
        if (ch == '\\') ch = '/';
    }
    return path;
}

std::string image_name_for_export(
    const std::filesystem::path& image_path, const std::filesystem::path& base) {
    std::error_code ec;
    if (!base.empty()) {
        const auto relative = std::filesystem::relative(image_path, base, ec);
        const std::string relative_text = relative.generic_string();
        if (!ec && !relative_text.empty() && relative_text.find("..") == std::string::npos)
            return unify_slash(relative_text);
    }
    return unify_slash(std::filesystem::absolute(image_path).generic_string());
}

const io::RgbImage* cached_rgb(
    std::unordered_map<std::uint32_t, std::unique_ptr<io::RgbImage>>& cache,
    const Scene& scene,
    const std::uint32_t image_id) {
    const auto existing = cache.find(image_id);
    if (existing != cache.end()) return existing->second.get();
    try {
        auto image = std::make_unique<io::RgbImage>(io::load_rgb(scene.images[image_id].path));
        const io::RgbImage* pointer = image.get();
        cache.emplace(image_id, std::move(image));
        return pointer;
    } catch (...) {
        cache.emplace(image_id, nullptr);
        return nullptr;
    }
}

bool sample_rgb(
    const io::RgbImage& image,
    const float x,
    const float y,
    double& r,
    double& g,
    double& b) {
    if (image.width == 0 || image.height == 0 || image.pixels.size() < 3) return false;
    const int xi = static_cast<int>(std::lround(x));
    const int yi = static_cast<int>(std::lround(y));
    if (xi < 0 || yi < 0 || xi >= static_cast<int>(image.width) ||
        yi >= static_cast<int>(image.height)) {
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
    core::StageScope stage("sfm.export_openmvs");
    for (const auto& image : scene.images)
        if (image.registered && scene.camera_of(image).model == CameraModel::opencv_fisheye)
            throw std::runtime_error(
                "OpenMVS interface requires rectified pinhole images; use .asfm for fisheye alignment");
    std::vector<std::uint32_t> registered;
    registered.reserve(scene.images.size());
    for (std::uint32_t i = 0; i < scene.images.size(); ++i) {
        if (scene.images[i].registered) registered.push_back(i);
    }
    if (registered.size() < 2)
        throw std::runtime_error("OpenMVS export needs at least two registered views");

    std::unordered_map<std::uint32_t, std::uint32_t> image_to_export;
    for (std::uint32_t export_id = 0; export_id < registered.size(); ++export_id)
        image_to_export.emplace(registered[export_id], export_id);

    const Image& first = scene.images[registered.front()];
    const PinholeCamera& shared_camera = scene.camera_of(first);

    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Failed to create MVS file: " + path.string());
    BinaryWriter writer(stream);

    writer.write_bytes(k_mvsi_id, 4);
    writer.write_pod(k_mvsi_version);
    writer.write_pod(std::uint32_t{0});

    writer.write_pod(std::uint64_t{1});
    writer.write_string("platform0");
    writer.write_pod(std::uint64_t{1});
    writer.write_string("camera0");
    writer.write_string("RGB");
    writer.write_pod(shared_camera.width);
    writer.write_pod(shared_camera.height);
    const double k[9] = {
        shared_camera.fx, 0.0, shared_camera.cx, 0.0, shared_camera.fy, shared_camera.cy, 0.0,
        0.0, 1.0};
    writer.write_array(k, 9);
    const double identity_r[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double zero_c[3] = {0, 0, 0};
    writer.write_array(identity_r, 9);
    writer.write_array(zero_c, 3);

    writer.write_pod(static_cast<std::uint64_t>(registered.size()));
    for (const std::uint32_t image_id : registered) {
        const auto rotation = pose_rotation(scene.images[image_id].pose);
        const double center[3] = {
            scene.images[image_id].pose.C.x(), scene.images[image_id].pose.C.y(),
            scene.images[image_id].pose.C.z()};
        writer.write_array(rotation.data(), 9);
        writer.write_array(center, 3);
    }

    writer.write_pod(static_cast<std::uint64_t>(registered.size()));
    for (std::uint32_t export_id = 0; export_id < registered.size(); ++export_id) {
        const Image& image = scene.images[registered[export_id]];
        writer.write_string(image_name_for_export(image.path, options.image_path_base));
        writer.write_string("");
        writer.write_pod(std::uint32_t{0});
        writer.write_pod(std::uint32_t{0});
        writer.write_pod(export_id);
        writer.write_pod(registered[export_id]);
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
    std::unordered_map<std::uint32_t, std::unique_ptr<io::RgbImage>> rgb_cache;
    bool wrote_any_color = false;

    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        ExportedVertex vertex{
            static_cast<float>(track.position.x()),
            static_cast<float>(track.position.y()),
            static_cast<float>(track.position.z()),
            255,
            255,
            255,
            {}};
        double sum_r = 0, sum_g = 0, sum_b = 0;
        std::size_t color_samples = 0;
        for (unsigned o = 0; o < track.num_inliers; ++o) {
            const Observation& obs = track.observations[o];
            const auto found = image_to_export.find(obs.image_id);
            if (found == image_to_export.end()) continue;
            vertex.views.emplace_back(found->second, 0.0f);
            if (track.has_color || !options.sample_colors) continue;
            const Image& image = scene.images[obs.image_id];
            if (obs.feature_id >= image.features.keypoints.size()) continue;
            const auto* rgb = cached_rgb(rgb_cache, scene, obs.image_id);
            if (!rgb) continue;
            const auto& kp = image.features.keypoints[obs.feature_id];
            double r = 0, g = 0, b = 0;
            if (!sample_rgb(*rgb, kp.x, kp.y, r, g, b)) continue;
            sum_r += r;
            sum_g += g;
            sum_b += b;
            ++color_samples;
        }
        if (vertex.views.size() < 2) continue;
        std::sort(vertex.views.begin(), vertex.views.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        if (track.has_color) {
            vertex.r = track.color_r;
            vertex.g = track.color_g;
            vertex.b = track.color_b;
            wrote_any_color = true;
        } else if (color_samples > 0) {
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

    writer.write_pod(std::uint64_t{0});
    if (options.sample_colors && wrote_any_color) {
        writer.write_pod(static_cast<std::uint64_t>(vertices.size()));
        for (const auto& vertex : vertices) {
            const std::uint8_t bgr[3] = {vertex.b, vertex.g, vertex.r};
            writer.write_array(bgr, 3);
        }
    } else {
        writer.write_pod(std::uint64_t{0});
    }

    writer.write_pod(std::uint64_t{0});
    writer.write_pod(std::uint64_t{0});
    writer.write_pod(std::uint64_t{0});
    const double identity4[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    writer.write_array(identity4, 16);
    writer.write_array(identity_r, 9);
    writer.write_array(zero_c, 3);
    writer.write_array(zero_c, 3);

    if (!writer.good())
        throw std::runtime_error("Failed while writing MVS file: " + path.string());
    core::Logger::instance().info(
        "OpenMVS export: path=", path, " images=", registered.size(),
        " vertices=", vertices.size(), " colors=",
        options.sample_colors && wrote_any_color);
}

}  // namespace aetherscan::sfm
