#include "dataset_internal.hpp"
#include "io/image.hpp"
#include <nlohmann/json.hpp>
#include <Eigen/Geometry>
#include <cmath>
#include <fstream>
#include <utility>

namespace photara::splat::dataset_detail {
namespace {
using Json = nlohmann::json;
std::filesystem::path transforms_path(const std::filesystem::path& source) {
    return std::filesystem::is_directory(source) ? source / "transforms.json" : source;
}
double number(const Json& frame, const Json& root, const char* key, double fallback) {
    const auto& object = frame.contains(key) ? frame : root;
    const double value = object.contains(key) ? object.at(key).get<double>() : fallback;
    if (!std::isfinite(value)) throw std::runtime_error(std::string("Invalid Nerfstudio value: ") + key);
    return value;
}
class NerfstudioReader final : public DatasetReader {
public:
    DatasetFormat format() const noexcept override { return DatasetFormat::nerfstudio; }
    bool probe(const DatasetLoadRequest& request) const override {
        const auto path = transforms_path(request.source);
        return lower_ascii(path.extension().string()) == ".json" &&
               std::filesystem::is_regular_file(path);
    }
    DatasetLoadResult load(const DatasetLoadRequest& request) const override {
        const auto path = transforms_path(request.source);
        std::ifstream stream(path);
        if (!stream) throw std::runtime_error("Cannot open Nerfstudio transforms: " + path.string());
        const Json root = Json::parse(stream);
        if (!root.contains("frames") || !root.at("frames").is_array() || root.at("frames").empty())
            throw std::runtime_error("Nerfstudio dataset needs a nonempty frames array");
        DatasetLoadResult result;
        result.format = format();
        result.resolved_source = path;
        for (const auto& frame : root.at("frames")) {
            const std::string model = frame.value("camera_model", root.value("camera_model", std::string("OPENCV")));
            if (model != "OPENCV" && model != "PINHOLE" && model != "SIMPLE_PINHOLE")
                throw std::runtime_error("Unsupported Nerfstudio camera_model: " + model);
            for (const char* key : {"k3", "k4"})
                if (number(frame, root, key, 0) != 0)
                    throw std::runtime_error(std::string("Unsupported Nerfstudio distortion: ") + key);
            auto name = std::filesystem::u8path(frame.at("file_path").get<std::string>());
            std::filesystem::path image_path;
            if (name.has_extension()) image_path = resolve_image(name, request, path.parent_path());
            else {
                for (const char* extension : {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".exr"}) {
                    auto candidate = name;
                    candidate += extension;
                    try { image_path = resolve_image(candidate, request, path.parent_path()); break; }
                    catch (const std::runtime_error&) {}
                }
                if (image_path.empty()) throw std::runtime_error("Nerfstudio image not found: " + name.string());
            }
            const auto size = io::load_image_size(image_path);
            const double width = number(frame, root, "w", size.width);
            const double height = number(frame, root, "h", size.height);
            if (width <= 0 || height <= 0 || !size.width || !size.height)
                throw std::runtime_error("Invalid Nerfstudio image dimensions");
            const auto focal = [&](const char* key, const char* angle_key, double pixels, double fallback) {
                if (frame.contains(key) || root.contains(key)) return number(frame, root, key, fallback);
                const double angle = number(frame, root, angle_key, 0);
                if (angle == 0) return fallback;
                if (angle <= 0 || angle >= 3.141592653589793)
                    throw std::runtime_error("Invalid Nerfstudio camera field of view");
                return 0.5 * pixels / std::tan(0.5 * angle);
            };
            const double fx = focal("fl_x", "camera_angle_x", width, 0);
            const double fy = focal("fl_y", "camera_angle_y", height, fx);
            if (fx <= 0 || fy <= 0 || !std::isfinite(fx) || !std::isfinite(fy))
                throw std::runtime_error("Nerfstudio needs positive focal lengths or camera_angle_x");
            mvs::MvsView view;
            view.id = static_cast<mvs::Index>(result.scene.views.size());
            view.sfm_image_id = view.id;
            view.path = image_path;
            view.width = view.src_width = size.width;
            view.height = view.src_height = size.height;
            const double sx = size.width / width, sy = size.height / height;
            view.fx = view.src_fx = static_cast<float>(fx * sx);
            view.fy = view.src_fy = static_cast<float>(fy * sy);
            view.cx = view.src_cx = static_cast<float>((number(frame, root, "cx", width / 2) + 0.5) * sx - 0.5);
            view.cy = view.src_cy = static_cast<float>((number(frame, root, "cy", height / 2) + 0.5) * sy - 0.5);
            view.k1 = static_cast<float>(number(frame, root, "k1", 0));
            view.k2 = static_cast<float>(number(frame, root, "k2", 0));
            view.p1 = static_cast<float>(number(frame, root, "p1", 0));
            view.p2 = static_cast<float>(number(frame, root, "p2", 0));
            const auto& matrix = frame.at("transform_matrix");
            if (!matrix.is_array() || matrix.size() != 4)
                throw std::runtime_error("Nerfstudio transform_matrix must be 4x4");
            Eigen::Matrix4d pose;
            for (int row = 0; row < 4; ++row) {
                if (!matrix[row].is_array() || matrix[row].size() != 4)
                    throw std::runtime_error("Nerfstudio transform_matrix must be 4x4");
                for (int col = 0; col < 4; ++col) pose(row, col) = matrix[row][col].get<double>();
            }
            const Eigen::Matrix3d rotation = pose.topLeftCorner<3,3>();
            if (!pose.allFinite() || (pose.row(3) - Eigen::RowVector4d(0,0,0,1)).norm() > 1e-6 ||
                (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-4 ||
                std::abs(rotation.determinant() - 1) > 1e-4)
                throw std::runtime_error("Nerfstudio camera pose must be a finite rigid transform");
            // Blender/OpenGL c2w -> OpenCV w2c. World axes and scale are preserved.
            view.pose.R = Eigen::Vector3d(1,-1,-1).asDiagonal() * rotation.transpose();
            view.pose.C = pose.topRightCorner<3,1>();
            result.scene.views.push_back(std::move(view));
        }
        if (request.initial_point_cloud.empty() && root.contains("ply_file_path")) {
            result.initial_point_cloud = resolve_image(
                std::filesystem::u8path(root.at("ply_file_path").get<std::string>()), request, path.parent_path());
        }
        return result;
    }
};
}
std::unique_ptr<DatasetReader> make_nerfstudio_reader() { return std::make_unique<NerfstudioReader>(); }
}
