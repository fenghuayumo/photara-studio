#include "splat/colmap.hpp"

#include "core/camera_projection.hpp"
#include "io/image.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aetherscan::splat {
namespace {

struct ColmapCamera {
    std::int32_t model{};
    std::uint64_t width{};
    std::uint64_t height{};
    std::vector<double> parameters;
};

struct ColmapImage {
    std::uint32_t id{};
    Eigen::Quaterniond rotation{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d translation{Eigen::Vector3d::Zero()};
    std::uint32_t camera_id{};
    std::filesystem::path name;
};

struct ColmapPoint {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    std::array<std::uint8_t, 3> color{};
    double error{};
    std::vector<std::uint32_t> image_ids;
};

int camera_parameter_count(const int model) {
    switch (model) {
    case 0: return 3;   // SIMPLE_PINHOLE
    case 1: return 4;   // PINHOLE
    case 2: return 4;   // SIMPLE_RADIAL
    case 3: return 5;   // RADIAL
    case 4: return 8;   // OPENCV
    case 5: return 8;   // OPENCV_FISHEYE
    case 6: return 12;  // FULL_OPENCV
    case 7: return 5;   // FOV
    case 8: return 4;   // SIMPLE_RADIAL_FISHEYE
    case 9: return 5;   // RADIAL_FISHEYE
    case 10: return 12; // THIN_PRISM_FISHEYE
    case 17: return 2;  // EQUIRECTANGULAR (LichtFeld / COLMAP fork)
    default: return -1;
    }
}

template <typename T>
T read_binary(std::istream& stream, const char* field) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream)
        throw std::runtime_error(std::string("Truncated COLMAP binary field: ") + field);
    return value;
}

std::string read_c_string(std::istream& stream) {
    std::string value;
    char character{};
    while (stream.get(character)) {
        if (character == '\0') return value;
        value.push_back(character);
    }
    throw std::runtime_error("Truncated COLMAP image name");
}

std::filesystem::path resolve_model_directory(
    const std::filesystem::path& requested) {
    const auto has_model = [](const std::filesystem::path& path) {
        return (std::filesystem::is_regular_file(path / "cameras.bin") &&
                std::filesystem::is_regular_file(path / "images.bin") &&
                std::filesystem::is_regular_file(path / "points3D.bin")) ||
               (std::filesystem::is_regular_file(path / "cameras.txt") &&
                std::filesystem::is_regular_file(path / "images.txt") &&
                std::filesystem::is_regular_file(path / "points3D.txt"));
    };
    if (has_model(requested)) return requested;
    if (has_model(requested / "sparse" / "0")) return requested / "sparse" / "0";
    if (has_model(requested / "sparse")) return requested / "sparse";
    if (has_model(requested / "0")) return requested / "0";
    throw std::invalid_argument(
        "COLMAP model must contain cameras/images/points3D .bin or .txt files: " +
        requested.string());
}

std::unordered_map<std::uint32_t, ColmapCamera> read_cameras_binary(
    const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open COLMAP cameras.bin");
    const std::uint64_t count = read_binary<std::uint64_t>(stream, "camera count");
    std::unordered_map<std::uint32_t, ColmapCamera> cameras;
    cameras.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        const std::uint32_t id = read_binary<std::uint32_t>(stream, "camera id");
        ColmapCamera camera;
        camera.model = read_binary<std::int32_t>(stream, "camera model");
        camera.width = read_binary<std::uint64_t>(stream, "camera width");
        camera.height = read_binary<std::uint64_t>(stream, "camera height");
        const int parameter_count = camera_parameter_count(camera.model);
        if (parameter_count < 0)
            throw std::runtime_error("Unsupported COLMAP camera model id " +
                                     std::to_string(camera.model));
        camera.parameters.resize(static_cast<std::size_t>(parameter_count));
        for (double& parameter : camera.parameters)
            parameter = read_binary<double>(stream, "camera parameter");
        cameras.emplace(id, std::move(camera));
    }
    return cameras;
}

std::vector<ColmapImage> read_images_binary(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open COLMAP images.bin");
    const std::uint64_t count = read_binary<std::uint64_t>(stream, "image count");
    std::vector<ColmapImage> images;
    images.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        ColmapImage image;
        image.id = read_binary<std::uint32_t>(stream, "image id");
        const double qw = read_binary<double>(stream, "qw");
        const double qx = read_binary<double>(stream, "qx");
        const double qy = read_binary<double>(stream, "qy");
        const double qz = read_binary<double>(stream, "qz");
        image.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
        for (int axis = 0; axis < 3; ++axis)
            image.translation(axis) = read_binary<double>(stream, "translation");
        image.camera_id = read_binary<std::uint32_t>(stream, "camera id");
        image.name = std::filesystem::u8path(read_c_string(stream));
        const std::uint64_t observations =
            read_binary<std::uint64_t>(stream, "point2D count");
        for (std::uint64_t point = 0; point < observations; ++point) {
            (void)read_binary<double>(stream, "point2D x");
            (void)read_binary<double>(stream, "point2D y");
            (void)read_binary<std::uint64_t>(stream, "point3D id");
        }
        images.push_back(std::move(image));
    }
    return images;
}

std::vector<ColmapPoint> read_points_binary(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open COLMAP points3D.bin");
    const std::uint64_t count = read_binary<std::uint64_t>(stream, "point count");
    std::vector<ColmapPoint> points;
    points.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        (void)read_binary<std::uint64_t>(stream, "point id");
        ColmapPoint point;
        for (int axis = 0; axis < 3; ++axis)
            point.position(axis) = read_binary<double>(stream, "point position");
        for (std::uint8_t& channel : point.color)
            channel = read_binary<std::uint8_t>(stream, "point color");
        point.error = read_binary<double>(stream, "point error");
        const std::uint64_t track = read_binary<std::uint64_t>(stream, "track length");
        point.image_ids.reserve(static_cast<std::size_t>(track));
        for (std::uint64_t observation = 0; observation < track; ++observation) {
            point.image_ids.push_back(
                read_binary<std::uint32_t>(stream, "track image id"));
            (void)read_binary<std::uint32_t>(stream, "track point2D index");
        }
        points.push_back(std::move(point));
    }
    return points;
}

bool next_data_line(std::istream& stream, std::string& line) {
    while (std::getline(stream, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first != std::string::npos && line[first] != '#') return true;
    }
    return false;
}

int camera_model_id(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    static const std::unordered_map<std::string, int> ids{
        {"SIMPLE_PINHOLE", 0}, {"PINHOLE", 1}, {"SIMPLE_RADIAL", 2},
        {"RADIAL", 3}, {"OPENCV", 4}, {"OPENCV_FISHEYE", 5},
        {"FULL_OPENCV", 6}, {"FOV", 7}, {"SIMPLE_RADIAL_FISHEYE", 8},
        {"RADIAL_FISHEYE", 9}, {"THIN_PRISM_FISHEYE", 10},
        {"EQUIRECTANGULAR", 17}, {"SPHERICAL", 17}};
    const auto iterator = ids.find(name);
    return iterator == ids.end() ? -1 : iterator->second;
}

std::unordered_map<std::uint32_t, ColmapCamera> read_cameras_text(
    const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open COLMAP cameras.txt");
    std::unordered_map<std::uint32_t, ColmapCamera> cameras;
    std::string line;
    while (next_data_line(stream, line)) {
        std::istringstream row(line);
        std::uint32_t id{};
        std::string model;
        ColmapCamera camera;
        row >> id >> model >> camera.width >> camera.height;
        camera.model = camera_model_id(model);
        if (!row || camera.model < 0)
            throw std::runtime_error("Invalid COLMAP camera row: " + line);
        double value{};
        while (row >> value) camera.parameters.push_back(value);
        const int parameter_count = camera_parameter_count(camera.model);
        if (parameter_count < 0 ||
            camera.parameters.size() != static_cast<std::size_t>(parameter_count))
            throw std::runtime_error("Wrong COLMAP camera parameter count: " + line);
        cameras.emplace(id, std::move(camera));
    }
    return cameras;
}

std::vector<ColmapImage> read_images_text(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open COLMAP images.txt");
    std::vector<ColmapImage> images;
    std::string line;
    while (next_data_line(stream, line)) {
        std::istringstream row(line);
        ColmapImage image;
        double qw{}, qx{}, qy{}, qz{};
        row >> image.id >> qw >> qx >> qy >> qz >>
            image.translation.x() >> image.translation.y() >> image.translation.z() >>
            image.camera_id;
        if (!row) throw std::runtime_error("Invalid COLMAP image row: " + line);
        std::string name;
        std::getline(row, name);
        const auto first = name.find_first_not_of(" \t");
        if (first == std::string::npos)
            throw std::runtime_error("COLMAP image row has no filename: " + line);
        name.erase(0, first);
        image.rotation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
        image.name = std::filesystem::u8path(name);
        images.push_back(std::move(image));
        // POINTS2D is always the following physical line and may be empty.
        std::getline(stream, line);
    }
    return images;
}

std::vector<ColmapPoint> read_points_text(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open COLMAP points3D.txt");
    std::vector<ColmapPoint> points;
    std::string line;
    while (next_data_line(stream, line)) {
        std::istringstream row(line);
        std::uint64_t id{};
        unsigned red{}, green{}, blue{};
        ColmapPoint point;
        row >> id >> point.position.x() >> point.position.y() >> point.position.z() >>
            red >> green >> blue >> point.error;
        if (!row) throw std::runtime_error("Invalid COLMAP point row: " + line);
        point.color = {
            static_cast<std::uint8_t>(std::min(red, 255U)),
            static_cast<std::uint8_t>(std::min(green, 255U)),
            static_cast<std::uint8_t>(std::min(blue, 255U))};
        std::uint32_t image_id{}, point2d{};
        while (row >> image_id >> point2d) point.image_ids.push_back(image_id);
        points.push_back(std::move(point));
    }
    return points;
}

void set_intrinsics(const ColmapCamera& camera, mvs::MvsView& view) {
    if (camera.width == 0 || camera.height == 0 ||
        camera.width > (std::numeric_limits<std::uint32_t>::max)() ||
        camera.height > (std::numeric_limits<std::uint32_t>::max)())
        throw std::runtime_error("Invalid COLMAP camera dimensions");
    view.width = view.src_width = static_cast<std::uint32_t>(camera.width);
    view.height = view.src_height = static_cast<std::uint32_t>(camera.height);
    const auto& p = camera.parameters;
    if (camera.model == 0 || camera.model == 2 || camera.model == 3) {
        view.fx = view.fy = static_cast<float>(p[0]);
        view.cx = static_cast<float>(p[1]);
        view.cy = static_cast<float>(p[2]);
        if (camera.model >= 2) view.k1 = static_cast<float>(p[3]);
        if (camera.model == 3) view.k2 = static_cast<float>(p[4]);
    } else if (camera.model == 1 || camera.model == 4) {
        view.fx = static_cast<float>(p[0]);
        view.fy = static_cast<float>(p[1]);
        view.cx = static_cast<float>(p[2]);
        view.cy = static_cast<float>(p[3]);
        if (camera.model == 4) {
            view.k1 = static_cast<float>(p[4]);
            view.k2 = static_cast<float>(p[5]);
            view.p1 = static_cast<float>(p[6]);
            view.p2 = static_cast<float>(p[7]);
        }
    } else if (camera.model == 5 || camera.model == 8 || camera.model == 9) {
        view.source_model = CameraModel::opencv_fisheye;
        if (camera.model == 5) {
            view.fx = static_cast<float>(p[0]);
            view.fy = static_cast<float>(p[1]);
            view.cx = static_cast<float>(p[2]);
            view.cy = static_cast<float>(p[3]);
            view.k1 = static_cast<float>(p[4]);
            view.k2 = static_cast<float>(p[5]);
            view.p1 = static_cast<float>(p[6]);
            view.p2 = static_cast<float>(p[7]);
        } else {
            view.fx = view.fy = static_cast<float>(p[0]);
            view.cx = static_cast<float>(p[1]);
            view.cy = static_cast<float>(p[2]);
            view.k1 = static_cast<float>(p[3]);
            if (camera.model == 9) view.k2 = static_cast<float>(p[4]);
        }
    } else if (camera.model == 17) {
        view.source_model = CameraModel::equirectangular;
        const float width = static_cast<float>(view.width);
        const float height = static_cast<float>(view.height);
        view.fx = view.fy = width / (2.F * 3.14159265358979323846F);
        view.cx = 0.5F * width;
        view.cy = 0.5F * height;
    } else {
        throw std::runtime_error(
            "COLMAP camera model " + std::to_string(camera.model) +
            " is not supported for splat training");
    }
    view.src_fx = view.fx;
    view.src_fy = view.fy;
    view.src_cx = view.cx;
    view.src_cy = view.cy;
}

void match_selected_image_resolution(
    const io::ImageSize& selected, mvs::MvsView& view) {
    if (selected.width == view.src_width &&
        selected.height == view.src_height)
        return;
    const float scale_x =
        static_cast<float>(selected.width) / view.src_width;
    const float scale_y =
        static_cast<float>(selected.height) / view.src_height;
    const auto scale_principal = [](const float value, const float scale) {
        return (value + 0.5F) * scale - 0.5F;
    };
    view.fx *= scale_x;
    view.fy *= scale_y;
    view.cx = scale_principal(view.cx, scale_x);
    view.cy = scale_principal(view.cy, scale_y);
    view.width = selected.width;
    view.height = selected.height;
    view.src_fx *= scale_x;
    view.src_fy *= scale_y;
    view.src_cx = scale_principal(view.src_cx, scale_x);
    view.src_cy = scale_principal(view.src_cy, scale_y);
    view.src_width = selected.width;
    view.src_height = selected.height;
}

}  // namespace

ColmapLoadResult load_colmap_scene(
    const std::filesystem::path& requested_model_directory,
    const std::filesystem::path& image_directory) {
    const std::filesystem::path model_directory =
        resolve_model_directory(requested_model_directory);
    const bool binary = std::filesystem::is_regular_file(
        model_directory / "cameras.bin");
    const auto cameras = binary
        ? read_cameras_binary(model_directory / "cameras.bin")
        : read_cameras_text(model_directory / "cameras.txt");
    auto images = binary
        ? read_images_binary(model_directory / "images.bin")
        : read_images_text(model_directory / "images.txt");
    const auto points = binary
        ? read_points_binary(model_directory / "points3D.bin")
        : read_points_text(model_directory / "points3D.txt");
    if (cameras.empty() || images.empty())
        throw std::runtime_error("COLMAP reconstruction has no cameras or images");

    std::sort(images.begin(), images.end(), [](const auto& left, const auto& right) {
        return left.name.generic_u8string() < right.name.generic_u8string();
    });
    ColmapLoadResult result;
    result.model_directory = model_directory;
    result.binary = binary;
    result.scene.views.reserve(images.size());
    std::unordered_map<std::uint32_t, mvs::Index> image_to_view;
    image_to_view.reserve(images.size());
    for (const ColmapImage& image : images) {
        const auto camera = cameras.find(image.camera_id);
        if (camera == cameras.end())
            throw std::runtime_error("COLMAP image references a missing camera");
        mvs::MvsView view;
        view.id = static_cast<mvs::Index>(result.scene.views.size());
        view.sfm_image_id = image.id;
        view.path = image_directory / image.name;
        if (!std::filesystem::is_regular_file(view.path))
            throw std::runtime_error("COLMAP image does not exist: " + view.path.string());
        set_intrinsics(camera->second, view);
        match_selected_image_resolution(io::load_image_size(view.path), view);
        view.pose.set_from_rt(image.rotation.toRotationMatrix(), image.translation);
        image_to_view.emplace(image.id, view.id);
        result.scene.views.push_back(std::move(view));
    }

    result.scene.sparse_points.reserve(points.size());
    result.scene.dense_cloud.points.reserve(points.size());
    for (const ColmapPoint& source : points) {
        if (!source.position.allFinite()) continue;
        mvs::SparsePoint sparse;
        sparse.position = source.position.cast<float>();
        for (const std::uint32_t image_id : source.image_ids) {
            const auto view = image_to_view.find(image_id);
            if (view != image_to_view.end()) sparse.view_ids.push_back(view->second);
        }
        std::sort(sparse.view_ids.begin(), sparse.view_ids.end());
        sparse.view_ids.erase(
            std::unique(sparse.view_ids.begin(), sparse.view_ids.end()),
            sparse.view_ids.end());
        sparse.color = mvs::Vec3f(
            source.color[0] / 255.F,
            source.color[1] / 255.F,
            source.color[2] / 255.F);

        mvs::DensePoint point;
        point.position = sparse.position;
        point.normal = mvs::Vec3f::UnitZ();
        point.color = mvs::Vec3f(
            source.color[0] / 255.F,
            source.color[1] / 255.F,
            source.color[2] / 255.F);
        point.weight = static_cast<float>(sparse.view_ids.size()) /
                       static_cast<float>(1.0 + std::max(source.error, 0.0));
        point.views = sparse.view_ids;
        result.scene.sparse_points.push_back(std::move(sparse));
        result.scene.dense_cloud.points.push_back(std::move(point));
    }
    return result;
}

}  // namespace aetherscan::splat
