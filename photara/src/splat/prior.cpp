#include "splat/prior.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace photara::splat {
namespace {
template<class T> void write(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
template<class T> void read(std::istream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("Truncated training prior");
}
void write_camera(std::ostream& stream, const Camera& camera) {
    write(stream, camera.width); write(stream, camera.height);
    write(stream, camera.fx); write(stream, camera.fy); write(stream, camera.cx); write(stream, camera.cy);
    for (float value : camera.world_to_camera) write(stream, value);
}
Camera read_camera(std::istream& stream) {
    Camera camera;
    read(stream, camera.width); read(stream, camera.height);
    read(stream, camera.fx); read(stream, camera.fy); read(stream, camera.cx); read(stream, camera.cy);
    for (float& value : camera.world_to_camera) read(stream, value);
    return camera;
}
bool near(float a, float b) { return std::isfinite(a) && std::isfinite(b) && std::abs(a-b) <= 1e-4F * std::max(1.F, std::abs(b)); }
void check_camera(const Camera& stored, const Camera& requested) {
    if (stored.model != CameraModel::pinhole || requested.model != CameraModel::pinhole ||
        stored.width != requested.width || stored.height != requested.height ||
        !near(stored.fx, requested.fx) || !near(stored.fy, requested.fy) ||
        !near(stored.cx, requested.cx) || !near(stored.cy, requested.cy))
        throw std::invalid_argument("Training prior camera/resolution changed; regenerate the prior cache");
    for (std::size_t i = 0; i < 16; ++i)
        if (!near(stored.world_to_camera[i], requested.world_to_camera[i]))
            throw std::invalid_argument("Training prior camera pose changed; regenerate the prior cache");
}
std::string source_name(const std::filesystem::path& path) {
    const auto name = std::filesystem::absolute(path).lexically_normal().generic_u8string();
    return std::string(name.begin(), name.end());
}
}
void TrainingPrior::validate() const {
    const auto count = static_cast<std::uint64_t>(camera.width) * camera.height;
    if (camera.model != CameraModel::pinhole || !count || count > 100'000'000 ||
        (depth.empty() && normals.empty()) ||
        (!depth.empty() && depth.size() != count) ||
        (!normals.empty() && normals.size() != 3 * count) ||
        !std::isfinite(camera.fx) || camera.fx <= 0 || !std::isfinite(camera.fy) || camera.fy <= 0 ||
        !std::isfinite(camera.cx) || !std::isfinite(camera.cy))
        throw std::invalid_argument("Invalid training prior camera or map dimensions");
    for (float value : camera.world_to_camera)
        if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite training prior camera pose");
    for (std::size_t i = 0; i < count; ++i) {
        if (!depth.empty() && (!std::isfinite(depth[i]) || depth[i] < 0))
            throw std::invalid_argument("Invalid training prior depth");
        if (normals.empty()) continue;
        const float x = normals[i], y = normals[count+i], z = normals[2*count+i];
        const float length = std::sqrt(x*x+y*y+z*z);
        if (!std::isfinite(length) || (length > 0 && std::abs(length-1.F) > 1e-3F))
            throw std::invalid_argument("Training prior normals must be zero or unit vectors");
    }
}
std::filesystem::path training_prior_path(const std::filesystem::path& directory, std::uint32_t view_id) {
    return directory / ("view_" + std::to_string(view_id) + ".prior");
}
void save_training_prior(const std::filesystem::path& path,
    const std::filesystem::path& source_image, const TrainingPrior& prior) {
    prior.validate();
    const auto name = source_name(source_image);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write("PHPRIOR2", 8);
    const std::uint32_t maps = (prior.depth.empty() ? 0U : 1U) | (prior.normals.empty() ? 0U : 2U);
    write(output, maps);
    write_camera(output, prior.camera);
    write(output, static_cast<std::uint32_t>(name.size()));
    output.write(name.data(), name.size());
    write(output, static_cast<std::uint64_t>(std::filesystem::file_size(source_image)));
    write(output, static_cast<std::int64_t>(std::filesystem::last_write_time(source_image).time_since_epoch().count()));
    if (!prior.depth.empty()) output.write(reinterpret_cast<const char*>(prior.depth.data()), prior.depth.size()*sizeof(float));
    if (!prior.normals.empty()) output.write(reinterpret_cast<const char*>(prior.normals.data()), prior.normals.size()*sizeof(float));
    if (!output) throw std::runtime_error("Cannot write training prior: " + path.string());
}
TrainingPrior load_training_prior(const std::filesystem::path& path,
    const std::filesystem::path& source_image, const Camera& full_camera,
    bool require_depth, bool require_normals) {
    if (!require_depth && !require_normals)
        throw std::invalid_argument("No training prior maps requested");
    std::ifstream input(path, std::ios::binary);
    char magic[8]; input.read(magic, 8);
    if (!input || (std::string(magic, 8) != "PHPRIOR1" && std::string(magic, 8) != "PHPRIOR2"))
        throw std::runtime_error("Missing or invalid training prior: " + path.string());
    std::uint32_t maps = 3;
    if (std::string(magic, 8) == "PHPRIOR2") read(input, maps);
    if (!maps || (maps & ~3U)) throw std::runtime_error("Invalid training prior map flags");
    if ((require_depth && !(maps & 1U)) || (require_normals && !(maps & 2U)))
        throw std::runtime_error("Requested training prior map is missing; regenerate the prior cache");
    TrainingPrior prior;
    prior.camera = read_camera(input);
    check_camera(prior.camera, full_camera);
    std::uint32_t length; read(input, length);
    if (length > 32768) throw std::runtime_error("Invalid training prior source name");
    std::string name(length, '\0'); input.read(name.data(), length);
    std::uint64_t size; std::int64_t stamp;
    read(input, size); read(input, stamp);
    if (name != source_name(source_image) || size != std::filesystem::file_size(source_image) ||
        stamp != std::filesystem::last_write_time(source_image).time_since_epoch().count())
        throw std::runtime_error("Training prior image changed; regenerate the prior cache");
    const auto count = static_cast<std::uint64_t>(prior.camera.width) * prior.camera.height;
    const auto channels = ((maps & 1U) ? 1U : 0U) + ((maps & 2U) ? 3U : 0U);
    const auto offset = input.tellg();
    const auto bytes = std::filesystem::file_size(path);
    if (!count || count > 100'000'000 || offset < 0 ||
        bytes < static_cast<std::uint64_t>(offset) ||
        bytes - static_cast<std::uint64_t>(offset) != count*channels*sizeof(float))
        throw std::runtime_error("Invalid training prior data size");
    if (maps & 1U) {
        if (require_depth) {
            prior.depth.resize(static_cast<std::size_t>(count));
            input.read(reinterpret_cast<char*>(prior.depth.data()), prior.depth.size()*sizeof(float));
        } else input.seekg(static_cast<std::streamoff>(count*sizeof(float)), std::ios::cur);
    }
    if (maps & 2U) {
        if (require_normals) {
            prior.normals.resize(static_cast<std::size_t>(3*count));
            input.read(reinterpret_cast<char*>(prior.normals.data()), prior.normals.size()*sizeof(float));
        } else input.seekg(static_cast<std::streamoff>(3*count*sizeof(float)), std::ios::cur);
    }
    if (!input) throw std::runtime_error("Truncated training prior maps");
    prior.validate();
    return prior;
}
TrainingPrior resize_training_prior(const TrainingPrior& prior, const Camera& camera) {
    prior.validate();
    auto expected = prior.camera;
    const float sx = static_cast<float>(camera.width) / expected.width;
    const float sy = static_cast<float>(camera.height) / expected.height;
    expected.width = camera.width; expected.height = camera.height;
    expected.fx *= sx; expected.fy *= sy;
    expected.cx = (expected.cx+.5F)*sx-.5F; expected.cy = (expected.cy+.5F)*sy-.5F;
    check_camera(expected, camera);
    TrainingPrior result; result.camera = camera;
    const auto count = static_cast<std::size_t>(camera.width)*camera.height;
    const auto original_count = static_cast<std::size_t>(prior.camera.width)*prior.camera.height;
    if (!prior.depth.empty()) result.depth.resize(count);
    if (!prior.normals.empty()) result.normals.resize(3*count);
    // Nearest sampling preserves invalid regions and avoids inventing a surface
    // between foreground/background at discontinuities. No depth unit scaling.
    for (std::uint32_t y = 0; y < camera.height; ++y)
        for (std::uint32_t x = 0; x < camera.width; ++x) {
            const auto px = std::min(prior.camera.width-1, static_cast<std::uint32_t>((x+.5)*prior.camera.width/camera.width));
            const auto py = std::min(prior.camera.height-1, static_cast<std::uint32_t>((y+.5)*prior.camera.height/camera.height));
            const auto src = static_cast<std::size_t>(py)*prior.camera.width+px;
            const auto dst = static_cast<std::size_t>(y)*camera.width+x;
            if (!prior.depth.empty()) result.depth[dst] = prior.depth[src];
            if (!prior.normals.empty())
                for (int c = 0; c < 3; ++c) result.normals[c*count+dst] = prior.normals[c*original_count+src];
        }
    return result;
}
} // namespace photara::splat
