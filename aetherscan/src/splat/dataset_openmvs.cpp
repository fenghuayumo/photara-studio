#include "dataset_internal.hpp"

#include "io/image.hpp"

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace aetherscan::splat::dataset_detail {
namespace {

constexpr std::uint32_t k_openmvs_version = 7;
constexpr std::uint32_t k_no_id =
    (std::numeric_limits<std::uint32_t>::max)();
constexpr std::uint64_t k_max_serialized_elements = 1'000'000'000ULL;

class BinaryReader {
public:
    explicit BinaryReader(const std::filesystem::path& path)
        : stream_(path, std::ios::binary), path_(path) {
        if (!stream_)
            throw std::runtime_error(
                "Cannot open OpenMVS interface: " + path.string());
    }

    template <typename T>
    T pod(const char* field) {
        T value{};
        bytes(&value, sizeof(T), field);
        return value;
    }

    void bytes(void* data, const std::size_t size, const char* field) {
        if (size != 0)
            stream_.read(
                static_cast<char*>(data),
                static_cast<std::streamsize>(size));
        if (!stream_)
            throw std::runtime_error(
                "Truncated OpenMVS field " + std::string(field) +
                " in " + path_.string());
    }

    std::string string(const char* field) {
        const std::uint64_t size = count(field);
        std::string value(static_cast<std::size_t>(size), '\0');
        bytes(value.data(), value.size(), field);
        return value;
    }

    std::uint64_t count(const char* field) {
        const std::uint64_t value = pod<std::uint64_t>(field);
        if (value > k_max_serialized_elements)
            throw std::runtime_error(
                "Unreasonable OpenMVS element count for " +
                std::string(field));
        return value;
    }

private:
    std::ifstream stream_;
    std::filesystem::path path_;
};

struct OpenMvsCamera {
    std::uint32_t width{};
    std::uint32_t height{};
    std::array<double, 9> K{};
    std::array<double, 9> R{};
    std::array<double, 3> C{};
};

struct OpenMvsPose {
    std::array<double, 9> R{};
    std::array<double, 3> C{};
};

struct OpenMvsPlatform {
    std::vector<OpenMvsCamera> cameras;
    std::vector<OpenMvsPose> poses;
};

struct OpenMvsImage {
    std::string name;
    std::uint32_t platform_id{k_no_id};
    std::uint32_t camera_id{k_no_id};
    std::uint32_t pose_id{k_no_id};
    std::uint32_t id{k_no_id};
};

struct OpenMvsVertex {
    std::array<float, 3> position{};
    std::vector<std::pair<std::uint32_t, float>> views;
};

template <typename T, std::size_t Size>
std::array<T, Size> read_array(BinaryReader& reader, const char* field) {
    std::array<T, Size> values{};
    reader.bytes(values.data(), sizeof(T) * Size, field);
    return values;
}

void skip_openmvs_view_scores(
    BinaryReader& reader, const std::uint32_t version) {
    if (version <= 6) return;
    const std::uint64_t count = reader.count("image view scores");
    for (std::uint64_t index = 0; index < count; ++index) {
        static_cast<void>(reader.pod<std::uint32_t>("view score image"));
        static_cast<void>(reader.pod<std::uint32_t>("view score points"));
        for (int value = 0; value < 4; ++value)
            static_cast<void>(reader.pod<float>("view score value"));
    }
}

void skip_openmvs_lines(BinaryReader& reader) {
    const std::uint64_t count = reader.count("lines");
    for (std::uint64_t index = 0; index < count; ++index) {
        static_cast<void>(read_array<float, 6>(reader, "line endpoints"));
        const std::uint64_t views = reader.count("line views");
        for (std::uint64_t view = 0; view < views; ++view) {
            static_cast<void>(reader.pod<std::uint32_t>("line image"));
            static_cast<void>(reader.pod<float>("line confidence"));
        }
    }
}

void skip_float3_vector(BinaryReader& reader, const char* field) {
    const std::uint64_t count = reader.count(field);
    for (std::uint64_t index = 0; index < count; ++index)
        static_cast<void>(read_array<float, 3>(reader, field));
}

void skip_color_vector(BinaryReader& reader, const char* field) {
    const std::uint64_t count = reader.count(field);
    std::array<std::uint8_t, 3> color{};
    for (std::uint64_t index = 0; index < count; ++index)
        reader.bytes(color.data(), color.size(), field);
}

class OpenMvsReader final : public DatasetReader {
public:
    [[nodiscard]] DatasetFormat format() const noexcept override {
        return DatasetFormat::openmvs;
    }

    [[nodiscard]] bool probe(
        const DatasetLoadRequest& request) const override {
        if (!std::filesystem::is_regular_file(request.source) ||
            lower_ascii(request.source.extension().string()) != ".mvs")
            return false;
        std::ifstream stream(request.source, std::ios::binary);
        std::array<char, 4> id{};
        stream.read(id.data(), id.size());
        return stream &&
               id == std::array<char, 4>{'M', 'V', 'S', 'I'};
    }

    [[nodiscard]] DatasetLoadResult load(
        const DatasetLoadRequest& request) const override {
        BinaryReader reader(request.source);
        const auto id = read_array<char, 4>(reader, "header");
        if (id != std::array<char, 4>{'M', 'V', 'S', 'I'})
            throw std::runtime_error(
                "OpenMVS legacy archives without MVSI headers are unsupported");
        const std::uint32_t version =
            reader.pod<std::uint32_t>("version");
        if (version > k_openmvs_version)
            throw std::runtime_error(
                "OpenMVS interface version is newer than supported version 7");
        static_cast<void>(reader.pod<std::uint32_t>("reserved"));

        std::vector<OpenMvsPlatform> platforms;
        const std::uint64_t platform_count = reader.count("platforms");
        platforms.resize(static_cast<std::size_t>(platform_count));
        for (auto& platform : platforms) {
            static_cast<void>(reader.string("platform name"));
            const std::uint64_t camera_count = reader.count("cameras");
            platform.cameras.resize(static_cast<std::size_t>(camera_count));
            for (auto& camera : platform.cameras) {
                static_cast<void>(reader.string("camera name"));
                if (version > 3)
                    static_cast<void>(reader.string("camera band"));
                if (version > 0) {
                    camera.width =
                        reader.pod<std::uint32_t>("camera width");
                    camera.height =
                        reader.pod<std::uint32_t>("camera height");
                }
                camera.K = read_array<double, 9>(reader, "camera K");
                camera.R = read_array<double, 9>(reader, "camera R");
                camera.C = read_array<double, 3>(reader, "camera C");
            }
            const std::uint64_t pose_count = reader.count("poses");
            platform.poses.resize(static_cast<std::size_t>(pose_count));
            for (auto& pose : platform.poses) {
                pose.R = read_array<double, 9>(reader, "pose R");
                pose.C = read_array<double, 3>(reader, "pose C");
            }
        }

        std::vector<OpenMvsImage> images;
        const std::uint64_t image_count = reader.count("images");
        images.resize(static_cast<std::size_t>(image_count));
        for (auto& image : images) {
            image.name = reader.string("image name");
            if (version > 4)
                static_cast<void>(reader.string("image mask"));
            image.platform_id =
                reader.pod<std::uint32_t>("image platform");
            image.camera_id = reader.pod<std::uint32_t>("image camera");
            image.pose_id = reader.pod<std::uint32_t>("image pose");
            if (version > 2)
                image.id = reader.pod<std::uint32_t>("image id");
            if (version > 6) {
                static_cast<void>(reader.pod<float>("image min depth"));
                static_cast<void>(reader.pod<float>("image average depth"));
                static_cast<void>(reader.pod<float>("image max depth"));
                skip_openmvs_view_scores(reader, version);
            }
        }

        std::vector<OpenMvsVertex> vertices;
        const std::uint64_t vertex_count = reader.count("vertices");
        vertices.resize(static_cast<std::size_t>(vertex_count));
        for (auto& vertex : vertices) {
            vertex.position =
                read_array<float, 3>(reader, "vertex position");
            const std::uint64_t view_count = reader.count("vertex views");
            vertex.views.reserve(static_cast<std::size_t>(view_count));
            for (std::uint64_t index = 0; index < view_count; ++index) {
                // Function-argument evaluation order is not guaranteed. Read
                // archive fields explicitly in their serialized order.
                const std::uint32_t image =
                    reader.pod<std::uint32_t>("vertex image");
                const float confidence =
                    reader.pod<float>("vertex confidence");
                vertex.views.emplace_back(image, confidence);
            }
        }
        std::vector<std::array<float, 3>> normals;
        const std::uint64_t normal_count = reader.count("vertex normals");
        normals.resize(static_cast<std::size_t>(normal_count));
        for (auto& normal : normals)
            normal = read_array<float, 3>(reader, "vertex normal");
        std::vector<std::array<std::uint8_t, 3>> colors;
        const std::uint64_t color_count = reader.count("vertex colors");
        colors.resize(static_cast<std::size_t>(color_count));
        for (auto& color : colors)
            reader.bytes(color.data(), color.size(), "vertex color");
        if (version > 0) {
            skip_openmvs_lines(reader);
            skip_float3_vector(reader, "line normals");
            skip_color_vector(reader, "line colors");
            if (version > 1) {
                static_cast<void>(
                    read_array<double, 16>(reader, "transform"));
                if (version > 5) {
                    static_cast<void>(
                        read_array<double, 9>(reader, "OBB rotation"));
                    static_cast<void>(
                        read_array<double, 3>(reader, "OBB minimum"));
                    static_cast<void>(
                        read_array<double, 3>(reader, "OBB maximum"));
                }
            }
        }

        DatasetLoadResult result;
        result.format = format();
        result.resolved_source = request.source;
        result.initial_points_dense =
            !vertices.empty() && normals.size() == vertices.size();
        const auto root = request.source.parent_path();
        std::vector<mvs::Index> image_to_view(
            images.size(), mvs::k_invalid);
        for (std::size_t image_index = 0;
             image_index < images.size(); ++image_index) {
            const auto& image = images[image_index];
            if (image.pose_id == k_no_id) continue;
            if (image.platform_id >= platforms.size() ||
                image.camera_id >=
                    platforms[image.platform_id].cameras.size() ||
                image.pose_id >=
                    platforms[image.platform_id].poses.size())
                throw std::runtime_error(
                    "OpenMVS image references an invalid platform camera pose");
            const auto& platform = platforms[image.platform_id];
            const auto& camera = platform.cameras[image.camera_id];
            const auto& pose = platform.poses[image.pose_id];
            const auto path = resolve_image(image.name, request, root);
            const io::RgbImage decoded = io::load_rgb(path);
            const double source_scale = camera.width != 0 && camera.height != 0
                ? std::max(camera.width, camera.height)
                : 1.0;
            const double target_scale = std::max(
                decoded.width, decoded.height);
            const double scale = target_scale / source_scale;
            const bool normalized =
                camera.K[2] < 3.0 && camera.K[5] < 3.0;
            const auto scale_principal = [scale, normalized](const double value) {
                return normalized
                    ? value * scale
                    : (value + 0.5) * scale - 0.5;
            };
            mvs::MvsView view;
            view.id = static_cast<mvs::Index>(result.scene.views.size());
            view.sfm_image_id =
                image.id == k_no_id ? static_cast<mvs::Index>(image_index)
                                    : image.id;
            view.path = path;
            view.width = view.src_width = decoded.width;
            view.height = view.src_height = decoded.height;
            view.fx = view.src_fx =
                static_cast<float>(camera.K[0] * scale);
            view.fy = view.src_fy =
                static_cast<float>(camera.K[4] * scale);
            view.cx = view.src_cx =
                static_cast<float>(scale_principal(camera.K[2]));
            view.cy = view.src_cy =
                static_cast<float>(scale_principal(camera.K[5]));

            const Eigen::Map<
                const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>
                camera_rotation(camera.R.data());
            const Eigen::Map<
                const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>
                platform_rotation(pose.R.data());
            const Eigen::Map<const Eigen::Vector3d>
                camera_center(camera.C.data());
            const Eigen::Map<const Eigen::Vector3d>
                platform_center(pose.C.data());
            view.pose.R = camera_rotation * platform_rotation;
            view.pose.C =
                platform_rotation.transpose() * camera_center +
                platform_center;
            image_to_view[image_index] = view.id;
            result.scene.views.push_back(std::move(view));
        }

        result.scene.sparse_points.reserve(vertices.size());
        result.scene.dense_cloud.points.reserve(vertices.size());
        for (std::size_t index = 0; index < vertices.size(); ++index) {
            const auto& vertex = vertices[index];
            const mvs::Vec3f position(
                vertex.position[0], vertex.position[1], vertex.position[2]);
            if (!position.allFinite()) continue;
            mvs::SparsePoint sparse;
            sparse.position = position;
            mvs::DensePoint point;
            point.position = position;
            point.normal = index < normals.size()
                ? mvs::Vec3f(
                      normals[index][0], normals[index][1], normals[index][2])
                : mvs::Vec3f::UnitZ();
            point.color = index < colors.size()
                ? mvs::Vec3f(
                      colors[index][2] / 255.F,
                      colors[index][1] / 255.F,
                      colors[index][0] / 255.F)
                : mvs::Vec3f::Constant(0.5F);
            float total_confidence = 0.F;
            for (const auto& [source_image, confidence] : vertex.views) {
                if (source_image >= image_to_view.size()) continue;
                const mvs::Index view = image_to_view[source_image];
                if (view == mvs::k_invalid) continue;
                sparse.view_ids.push_back(view);
                point.views.push_back(view);
                point.view_weights.push_back(confidence);
                total_confidence += confidence;
            }
            point.weight = total_confidence > 0.F
                ? total_confidence
                : static_cast<float>(point.views.size());
            result.scene.sparse_points.push_back(std::move(sparse));
            result.scene.dense_cloud.points.push_back(std::move(point));
        }
        if (result.scene.views.empty())
            throw std::runtime_error(
                "OpenMVS dataset has no registered images");
        return result;
    }
};

}  // namespace

std::unique_ptr<DatasetReader> make_openmvs_reader() {
    return std::make_unique<OpenMvsReader>();
}

}  // namespace aetherscan::splat::dataset_detail
