#include "sfm/export_nerfstudio.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <ostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace aetherscan::sfm {
namespace {

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string text = path.generic_u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::string unify_slash(std::string path) {
    for (char& ch : path)
        if (ch == '\\') ch = '/';
    return path;
}

std::string relative_or_name(
    const std::filesystem::path& path, const std::filesystem::path& base) {
    std::error_code error;
    if (!base.empty()) {
        const auto relative = std::filesystem::relative(path, base, error);
        if (!error) {
            const std::string text = unify_slash(path_utf8(relative));
            if (!text.empty() && text.find("..") == std::string::npos) return text;
        }
    }
    return unify_slash(path_utf8(path.filename()));
}

std::string json_string(const std::string& value) {
    std::string out;
    out.push_back('"');
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(hex[ch >> 4]);
                    out.push_back(hex[ch & 0x0F]);
                } else {
                    out.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

const char* nerfstudio_camera_model(const PinholeCamera& camera) {
    return camera.model == CameraModel::opencv_fisheye ? "OPENCV_FISHEYE"
                                                       : "OPENCV";
}

bool same_intrinsics(const PinholeCamera& a, const PinholeCamera& b) {
    auto close = [](const double x, const double y) {
        return std::abs(x - y) <= 1e-9 * (1.0 + std::abs(x));
    };
    return a.model == b.model && a.width == b.width && a.height == b.height &&
           close(a.fx, b.fx) && close(a.fy, b.fy) && close(a.cx, b.cx) &&
           close(a.cy, b.cy) && close(a.k1, b.k1) && close(a.k2, b.k2) &&
           close(a.p1, b.p1) && close(a.p2, b.p2);
}

void write_intrinsics(std::ostream& json, const PinholeCamera& camera) {
    json << "\"fl_x\":" << camera.fx << ",\"fl_y\":" << camera.fy
         << ",\"cx\":" << camera.cx << ",\"cy\":" << camera.cy
         << ",\"w\":" << camera.width << ",\"h\":" << camera.height
         << ",\"k1\":" << camera.k1 << ",\"k2\":" << camera.k2
         << ",\"p1\":" << camera.p1 << ",\"p2\":" << camera.p2
         << ",\"camera_model\":" << json_string(nerfstudio_camera_model(camera));
}

void write_c2w_gl(std::ostream& json, const Pose3D& pose) {
    // OpenCV camera-to-world, then flip camera Y/Z to OpenGL.
    Mat4 c2w = Mat4::Identity();
    c2w.block<3, 3>(0, 0) = pose.R.transpose();
    c2w.block<3, 1>(0, 3) = pose.C;
    c2w.col(1) *= -1.0;
    c2w.col(2) *= -1.0;
    json << '[';
    for (int row = 0; row < 4; ++row) {
        if (row != 0) json << ',';
        json << '[';
        for (int col = 0; col < 4; ++col) {
            if (col != 0) json << ',';
            json << c2w(row, col);
        }
        json << ']';
    }
    json << ']';
}

}  // namespace

void save_nerfstudio_transforms(
    const Scene& scene,
    const std::filesystem::path& json_path,
    const std::filesystem::path& image_path_base,
    const std::filesystem::path& ply_path) {
    if (json_path.empty())
        throw std::runtime_error("Nerfstudio export requires a transforms.json path");
    if (!json_path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(json_path.parent_path(), error);
    }

    std::vector<Index> registered;
    registered.reserve(scene.images.size());
    for (Index i = 0; i < scene.images.size(); ++i)
        if (scene.images[i].registered) registered.push_back(i);
    if (registered.empty())
        throw std::runtime_error("Nerfstudio export needs a registered view");

    bool shared = true;
    const PinholeCamera* first = nullptr;
    for (const Index id : registered) {
        const Image& image = scene.images[id];
        if (image.camera_id >= scene.cameras.size())
            throw std::runtime_error("Nerfstudio export image has an invalid camera");
        const PinholeCamera& camera = scene.cameras[image.camera_id];
        if (first == nullptr) first = &camera;
        else if (!same_intrinsics(*first, camera)) shared = false;
    }

    std::ofstream json(json_path);
    if (!json)
        throw std::runtime_error("Failed to create " + path_utf8(json_path));
    json << std::setprecision(17) << '{';
    if (shared && first != nullptr) {
        write_intrinsics(json, *first);
        json << ',';
    }
    if (!ply_path.empty()) {
        json << "\"ply_file_path\":"
             << json_string(relative_or_name(ply_path, json_path.parent_path()))
             << ',';
    }
    json << "\"frames\":[";
    const std::filesystem::path json_dir = json_path.parent_path().empty()
        ? std::filesystem::current_path()
        : json_path.parent_path();
    const std::filesystem::path name_base =
        image_path_base.empty() ? json_dir : image_path_base;
    for (std::size_t i = 0; i < registered.size(); ++i) {
        const Image& image = scene.images[registered[i]];
        const PinholeCamera& camera = scene.cameras[image.camera_id];
        if (i != 0) json << ',';
        json << "{\"file_path\":"
             << json_string(relative_or_name(image.path, name_base));
        if (!shared) {
            json << ',';
            write_intrinsics(json, camera);
        }
        json << ",\"transform_matrix\":";
        write_c2w_gl(json, image.pose);
        json << '}';
    }
    json << "]}";
    if (!json)
        throw std::runtime_error("Failed while writing " + path_utf8(json_path));
}

}  // namespace aetherscan::sfm
