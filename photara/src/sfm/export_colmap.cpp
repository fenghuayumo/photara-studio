#include "sfm/export_colmap.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace photara::sfm {
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

std::string image_name_for_export(
    const std::filesystem::path& image_path, const std::filesystem::path& base) {
    std::error_code error;
    if (!base.empty()) {
        const auto relative = std::filesystem::relative(image_path, base, error);
        if (!error) {
            const std::string relative_text = unify_slash(path_utf8(relative));
            if (!relative_text.empty() &&
                relative_text.find("..") == std::string::npos)
                return relative_text;
        }
    }
    return unify_slash(path_utf8(image_path.filename()));
}

struct ColmapCamera {
    std::uint32_t id{};
    std::string model;
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<double> parameters;
};

ColmapCamera colmap_camera_from(const PinholeCamera& camera, const std::uint32_t id) {
    ColmapCamera out;
    out.id = id;
    out.width = camera.width;
    out.height = camera.height;
    if (camera.model == CameraModel::equirectangular) {
        // COLMAP-fork spherical model (id 17). The chart is fixed by the image
        // size, and the parameter pair carries the resolution so that readers
        // which do look at it (spirula, LichtFeld-style converters) can rebuild
        // fx = width / 2 pi without guessing.
        out.model = "EQUIRECTANGULAR";
        out.parameters = {static_cast<double>(camera.width),
                          static_cast<double>(camera.height)};
    } else if (camera.model == CameraModel::opencv_fisheye) {
        out.model = "OPENCV_FISHEYE";
        out.parameters = {
            camera.fx, camera.fy, camera.cx, camera.cy, camera.k1, camera.k2,
            camera.p1, camera.p2};
    } else if (camera.k1 != 0.0 || camera.k2 != 0.0 || camera.p1 != 0.0 ||
               camera.p2 != 0.0) {
        out.model = "OPENCV";
        out.parameters = {
            camera.fx, camera.fy, camera.cx, camera.cy, camera.k1, camera.k2,
            camera.p1, camera.p2};
    } else {
        out.model = "PINHOLE";
        out.parameters = {camera.fx, camera.fy, camera.cx, camera.cy};
    }
    return out;
}

}  // namespace

void save_colmap_text(
    const Scene& scene,
    const std::filesystem::path& directory,
    const std::filesystem::path& image_path_base,
    const bool write_points) {
    if (directory.empty())
        throw std::runtime_error("COLMAP export requires an output folder");
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error)
        throw std::runtime_error(
            "Failed to create COLMAP folder: " + path_utf8(directory));

    std::vector<std::uint32_t> camera_id(scene.cameras.size(), 0);
    std::vector<ColmapCamera> cameras;
    cameras.reserve(scene.cameras.size());
    std::vector<std::uint32_t> image_id(scene.images.size(), 0);
    std::uint32_t next_image = 1;
    unsigned registered = 0;
    for (std::size_t i = 0; i < scene.images.size(); ++i) {
        const Image& image = scene.images[i];
        if (!image.registered) continue;
        if (image.camera_id >= scene.cameras.size())
            throw std::runtime_error("COLMAP export image has an invalid camera");
        if (camera_id[image.camera_id] == 0) {
            camera_id[image.camera_id] =
                static_cast<std::uint32_t>(cameras.size() + 1U);
            cameras.push_back(colmap_camera_from(
                scene.cameras[image.camera_id], camera_id[image.camera_id]));
        }
        image_id[i] = next_image++;
        ++registered;
    }
    if (registered == 0)
        throw std::runtime_error("COLMAP export needs a registered view");

    struct Point2D {
        double x{};
        double y{};
        std::int64_t point3d{-1};
    };
    std::vector<std::vector<Point2D>> points2d(scene.images.size());
    struct Point3D {
        std::uint64_t id{};
        Vec3 position{Vec3::Zero()};
        std::uint8_t r{200};
        std::uint8_t g{200};
        std::uint8_t b{200};
        std::vector<std::pair<std::uint32_t, std::uint32_t>> track;
    };
    std::vector<Point3D> points3d;
    if (write_points) {
        std::uint64_t next_point = 1;
        for (const Track& track : scene.tracks) {
            if (!track.is_triangulated()) continue;
            Point3D point;
            point.id = next_point;
            point.position = track.position;
            if (track.has_color) {
                point.r = track.color_r;
                point.g = track.color_g;
                point.b = track.color_b;
            }
            const std::size_t inliers = std::min<std::size_t>(
                track.num_inliers, track.observations.size());
            for (std::size_t i = 0; i < inliers; ++i) {
                const Observation& observation = track.observations[i];
                if (observation.image_id >= scene.images.size()) continue;
                const std::uint32_t cid = image_id[observation.image_id];
                if (cid == 0) continue;
                const Image& image = scene.images[observation.image_id];
                if (observation.feature_id >= image.features.keypoints.size())
                    continue;
                const auto& keypoint =
                    image.features.keypoints[observation.feature_id];
                auto& list = points2d[observation.image_id];
                const auto index = static_cast<std::uint32_t>(list.size());
                list.push_back({keypoint.x, keypoint.y, static_cast<std::int64_t>(point.id)});
                point.track.emplace_back(cid, index);
            }
            if (point.track.size() < 2) continue;
            points3d.push_back(std::move(point));
            ++next_point;
        }
    }

    {
        std::ofstream cameras_txt(directory / "cameras.txt");
        if (!cameras_txt)
            throw std::runtime_error("Failed to create COLMAP cameras.txt");
        cameras_txt << "# Camera list with one line of data per camera:\n"
                    << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                    << "# Number of cameras: " << cameras.size() << '\n';
        cameras_txt << std::setprecision(17);
        for (const auto& camera : cameras) {
            cameras_txt << camera.id << ' ' << camera.model << ' ' << camera.width
                        << ' ' << camera.height;
            for (const double parameter : camera.parameters)
                cameras_txt << ' ' << parameter;
            cameras_txt << '\n';
        }
    }

    {
        std::ofstream images_txt(directory / "images.txt");
        if (!images_txt)
            throw std::runtime_error("Failed to create COLMAP images.txt");
        images_txt << "# Image list with two lines of data per image:\n"
                   << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                   << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                   << "# Number of images: " << registered << '\n';
        images_txt << std::setprecision(17);
        for (std::size_t i = 0; i < scene.images.size(); ++i) {
            if (image_id[i] == 0) continue;
            const Image& image = scene.images[i];
            Quat rotation = image.pose.quaternion();
            if (rotation.w() < 0.0) rotation.coeffs() *= -1.0;
            const Vec3 translation = image.pose.translation();
            images_txt << image_id[i] << ' ' << rotation.w() << ' ' << rotation.x()
                       << ' ' << rotation.y() << ' ' << rotation.z() << ' '
                       << translation.x() << ' ' << translation.y() << ' '
                       << translation.z() << ' ' << camera_id[image.camera_id]
                       << ' '
                       << image_name_for_export(image.path, image_path_base)
                       << '\n';
            const auto& list = points2d[i];
            for (std::size_t p = 0; p < list.size(); ++p) {
                if (p != 0) images_txt << ' ';
                images_txt << list[p].x << ' ' << list[p].y << ' '
                           << list[p].point3d;
            }
            images_txt << '\n';
        }
    }

    {
        std::ofstream points_txt(directory / "points3D.txt");
        if (!points_txt)
            throw std::runtime_error("Failed to create COLMAP points3D.txt");
        points_txt << "# 3D point list with one line of data per point:\n"
                   << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as "
                      "(IMAGE_ID, POINT2D_IDX)\n"
                   << "# Number of points: " << points3d.size() << '\n';
        points_txt << std::setprecision(17);
        for (const auto& point : points3d) {
            points_txt << point.id << ' ' << point.position.x() << ' '
                       << point.position.y() << ' ' << point.position.z() << ' '
                       << static_cast<int>(point.r) << ' '
                       << static_cast<int>(point.g) << ' '
                       << static_cast<int>(point.b) << " 0";
            for (const auto& obs : point.track)
                points_txt << ' ' << obs.first << ' ' << obs.second;
            points_txt << '\n';
        }
    }
}

}  // namespace photara::sfm
