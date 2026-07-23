#include "splat/dataset.hpp"

#include "io/image.hpp"
#include "mvs/export.hpp"
#include "splat/colmap.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace aetherscan::splat {
namespace {

constexpr std::uint32_t k_openmvs_version = 7;
constexpr std::uint32_t k_no_id =
    (std::numeric_limits<std::uint32_t>::max)();
constexpr std::uint64_t k_max_serialized_elements = 1'000'000'000ULL;

std::string lower_ascii(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

std::filesystem::path source_directory(
    const std::filesystem::path& source) {
    return std::filesystem::is_directory(source)
        ? source
        : source.parent_path();
}

bool has_colmap_model(const std::filesystem::path& path) {
    const auto has_files = [](const std::filesystem::path& directory) {
        return (std::filesystem::is_regular_file(directory / "cameras.bin") &&
                std::filesystem::is_regular_file(directory / "images.bin") &&
                std::filesystem::is_regular_file(directory / "points3D.bin")) ||
               (std::filesystem::is_regular_file(directory / "cameras.txt") &&
                std::filesystem::is_regular_file(directory / "images.txt") &&
                std::filesystem::is_regular_file(directory / "points3D.txt"));
    };
    return has_files(path) || has_files(path / "sparse" / "0") ||
           has_files(path / "sparse") || has_files(path / "0");
}

std::filesystem::path resolve_image(
    const std::filesystem::path& name,
    const DatasetLoadRequest& request,
    const std::filesystem::path& source_root) {
    std::vector<std::filesystem::path> candidates;
    if (name.is_absolute()) candidates.push_back(name);
    if (!request.image_directory.empty()) {
        candidates.push_back(request.image_directory / name);
        candidates.push_back(request.image_directory / name.filename());
    }
    candidates.push_back(source_root / name);
    candidates.push_back(source_root / "images" / name);
    candidates.push_back(source_root / "Images" / name);
    for (const auto& candidate : candidates)
        if (std::filesystem::is_regular_file(candidate)) return candidate;

    const auto search_root = !request.image_directory.empty()
        ? request.image_directory
        : source_root;
    std::error_code error;
    if (std::filesystem::is_directory(search_root, error)) {
        for (std::filesystem::recursive_directory_iterator iterator(
                 search_root,
                 std::filesystem::directory_options::skip_permission_denied,
                 error),
             end;
             iterator != end && !error; iterator.increment(error)) {
            if (iterator->is_regular_file(error) &&
                iterator->path().filename() == name.filename())
                return iterator->path();
        }
    }
    throw std::runtime_error(
        "Dataset image does not exist: " + name.string());
}

void use_sparse_points_as_initial_cloud(mvs::MvsScene& scene) {
    if (!scene.dense_cloud.points.empty()) return;
    scene.dense_cloud.points.reserve(scene.sparse_points.size());
    for (const mvs::SparsePoint& sparse : scene.sparse_points) {
        if (!sparse.position.allFinite()) continue;
        mvs::DensePoint point;
        point.position = sparse.position;
        point.normal = mvs::Vec3f::UnitZ();
        point.color = mvs::Vec3f::Constant(0.5F);
        point.weight = static_cast<float>(sparse.view_ids.size());
        point.views = sparse.view_ids;
        scene.dense_cloud.points.push_back(std::move(point));
    }
}

float estimate_camera_scene_scale(const std::vector<mvs::MvsView>& views) {
    if (views.size() < 2) return 1.F;
    double total_nearest = 0.0;
    for (std::size_t left = 0; left < views.size(); ++left) {
        double nearest = (std::numeric_limits<double>::infinity)();
        for (std::size_t right = 0; right < views.size(); ++right) {
            if (left == right) continue;
            nearest = std::min(
                nearest,
                (views[left].pose.C - views[right].pose.C).norm());
        }
        if (std::isfinite(nearest)) total_nearest += nearest;
    }
    return std::max(
        static_cast<float>(3.0 * total_nearest / views.size()), 1.F);
}

void generate_camera_frustum_points(
    mvs::MvsScene& scene, const std::size_t count, const std::uint32_t seed) {
    if (count == 0 || scene.views.empty())
        throw std::runtime_error(
            "Camera-only splat datasets require random initial points");
    const float scene_scale = estimate_camera_scene_scale(scene.views);
    const float near_depth = 0.05F * scene_scale;
    const float log_near = std::log(near_depth);
    const float log_far = std::log(scene_scale);
    std::mt19937 random(seed);
    std::uniform_int_distribution<std::size_t> camera_distribution(
        0, scene.views.size() - 1);
    std::uniform_real_distribution<float> unit(0.F, 1.F);
    std::uniform_real_distribution<float> color(0.F, 1.F);

    scene.dense_cloud.points.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t view_index = camera_distribution(random);
        const auto& view = scene.views[view_index];
        const float half_fov_x = std::atan(
            0.5F * static_cast<float>(view.width) /
            std::max(view.fx, 1e-6F));
        const float half_fov_y = std::atan(
            0.5F * static_cast<float>(view.height) /
            std::max(view.fy, 1e-6F));
        const float angle_x = (2.F * unit(random) - 1.F) * half_fov_x;
        const float angle_y = (2.F * unit(random) - 1.F) * half_fov_y;
        const float depth = std::exp(
            log_near + unit(random) * (log_far - log_near));
        const Eigen::Vector3d camera_point(
            std::tan(angle_x) * depth,
            std::tan(angle_y) * depth,
            depth);
        mvs::DensePoint point;
        point.position =
            view.pose.transform_camera_to_world(camera_point).cast<float>();
        point.normal =
            (view.pose.R.transpose() * Eigen::Vector3d::UnitZ()).cast<float>();
        point.color = mvs::Vec3f(color(random), color(random), color(random));
        point.weight = 1.F;
        point.views.push_back(static_cast<mvs::Index>(view_index));
        scene.dense_cloud.points.push_back(std::move(point));
    }
}

std::unordered_map<std::string, std::size_t> parse_csv_header(
    const std::string& line) {
    std::unordered_map<std::string, std::size_t> result;
    std::istringstream stream(line);
    std::string field;
    std::size_t index = 0;
    while (std::getline(stream, field, ',')) {
        const auto first = field.find_first_not_of(" \t\r");
        const auto last = field.find_last_not_of(" \t\r");
        if (first != std::string::npos) {
            field = field.substr(first, last - first + 1);
            if (!field.empty() && field.front() == '#') field.erase(0, 1);
            result.emplace(lower_ascii(field), index);
        }
        ++index;
    }
    return result;
}

bool reality_capture_header(
    const std::unordered_map<std::string, std::size_t>& header) {
    for (const char* required :
         {"name", "x", "y", "alt", "heading", "pitch", "roll", "f"})
        if (!header.contains(required)) return false;
    return true;
}

std::filesystem::path find_reality_capture_csv(
    const std::filesystem::path& source) {
    std::vector<std::filesystem::path> candidates;
    if (std::filesystem::is_regular_file(source) &&
        lower_ascii(source.extension().string()) == ".csv")
        candidates.push_back(source);
    else if (std::filesystem::is_directory(source)) {
        for (const auto& entry : std::filesystem::directory_iterator(source))
            if (entry.is_regular_file() &&
                lower_ascii(entry.path().extension().string()) == ".csv")
                candidates.push_back(entry.path());
        std::sort(candidates.begin(), candidates.end());
    }
    for (const auto& candidate : candidates) {
        std::ifstream stream(candidate);
        std::string line;
        while (std::getline(stream, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            if (reality_capture_header(parse_csv_header(line))) return candidate;
            break;
        }
    }
    return {};
}

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

std::string csv_text(
    const std::vector<std::string>& fields,
    const std::unordered_map<std::string, std::size_t>& header,
    const char* name) {
    const auto found = header.find(name);
    if (found == header.end() || found->second >= fields.size()) return {};
    std::string result = fields[found->second];
    const auto first = result.find_first_not_of(" \t\r");
    const auto last = result.find_last_not_of(" \t\r");
    return first == std::string::npos
        ? std::string{}
        : result.substr(first, last - first + 1);
}

double csv_number(
    const std::vector<std::string>& fields,
    const std::unordered_map<std::string, std::size_t>& header,
    const char* name) {
    const std::string text = csv_text(fields, header, name);
    if (text.empty()) return 0.0;
    std::size_t parsed = 0;
    const double value = std::stod(text, &parsed);
    if (parsed != text.size())
        throw std::runtime_error(
            "Invalid RealityCapture numeric value for " +
            std::string(name) + ": " + text);
    return value;
}

class RealityCaptureReader final : public DatasetReader {
public:
    [[nodiscard]] DatasetFormat format() const noexcept override {
        return DatasetFormat::reality_capture;
    }

    [[nodiscard]] bool probe(
        const DatasetLoadRequest& request) const override {
        return !find_reality_capture_csv(request.source).empty();
    }

    [[nodiscard]] DatasetLoadResult load(
        const DatasetLoadRequest& request) const override {
        const auto csv_path = find_reality_capture_csv(request.source);
        if (csv_path.empty())
            throw std::invalid_argument(
                "RealityCapture dataset needs an Internal/External camera CSV");
        std::ifstream stream(csv_path);
        if (!stream)
            throw std::runtime_error(
                "Cannot open RealityCapture CSV: " + csv_path.string());
        std::string line;
        while (std::getline(stream, line))
            if (line.find_first_not_of(" \t\r") != std::string::npos) break;
        const auto header = parse_csv_header(line);
        if (!reality_capture_header(header))
            throw std::runtime_error("Invalid RealityCapture CSV header");

        DatasetLoadResult result;
        result.format = format();
        result.resolved_source = csv_path;
        const auto root = csv_path.parent_path();
        bool warned_k3 = false;
        bool warned_k4 = false;
        while (std::getline(stream, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            const auto fields = split_csv_row(line);
            const std::string name = csv_text(fields, header, "name");
            if (name.empty()) continue;
            const auto image_path = resolve_image(name, request, root);
            const io::RgbImage image = io::load_rgb(image_path);
            if (image.width == 0 || image.height == 0)
                throw std::runtime_error(
                    "RealityCapture image has empty dimensions: " +
                    image_path.string());
            const double scale = std::max(image.width, image.height);
            mvs::MvsView view;
            view.id = static_cast<mvs::Index>(result.scene.views.size());
            view.sfm_image_id = view.id;
            view.path = image_path;
            view.width = view.src_width = image.width;
            view.height = view.src_height = image.height;
            view.fx = view.fy = static_cast<float>(
                csv_number(fields, header, "f") * scale / 36.0);
            view.cx = static_cast<float>(
                csv_number(fields, header, "px") * scale +
                0.5 * image.width);
            view.cy = static_cast<float>(
                csv_number(fields, header, "py") * scale +
                0.5 * image.height);
            view.src_fx = view.fx;
            view.src_fy = view.fy;
            view.src_cx = view.cx;
            view.src_cy = view.cy;
            view.k1 = static_cast<float>(csv_number(fields, header, "k1"));
            view.k2 = static_cast<float>(csv_number(fields, header, "k2"));
            view.p1 = static_cast<float>(csv_number(fields, header, "t1"));
            view.p2 = static_cast<float>(csv_number(fields, header, "t2"));
            if (!warned_k3 && csv_number(fields, header, "k3") != 0.0) {
                result.warnings.emplace_back(
                    "RealityCapture k3 is not representable by the current "
                    "Brown2 raster input and was ignored");
                warned_k3 = true;
            }
            if (!warned_k4 && csv_number(fields, header, "k4") != 0.0) {
                result.warnings.emplace_back(
                    "RealityCapture k4 is not representable by the current "
                    "Brown2 raster input and was ignored");
                warned_k4 = true;
            }

            const double degrees = std::numbers::pi / 180.0;
            const Eigen::Matrix3d open_gl_camera_to_world =
                Eigen::AngleAxisd(
                    -csv_number(fields, header, "heading") * degrees,
                    Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                Eigen::AngleAxisd(
                    csv_number(fields, header, "pitch") * degrees,
                    Eigen::Vector3d::UnitX()).toRotationMatrix() *
                Eigen::AngleAxisd(
                    csv_number(fields, header, "roll") * degrees,
                    Eigen::Vector3d::UnitY()).toRotationMatrix();
            Eigen::Matrix3d camera_to_world = open_gl_camera_to_world;
            camera_to_world.col(1) *= -1.0;
            camera_to_world.col(2) *= -1.0;
            view.pose.R = camera_to_world.transpose();
            view.pose.C = Eigen::Vector3d(
                csv_number(fields, header, "x"),
                csv_number(fields, header, "y"),
                csv_number(fields, header, "alt"));
            if (!view.pose.R.allFinite() || !view.pose.C.allFinite() ||
                !std::isfinite(view.fx) || view.fx <= 0.F)
                throw std::runtime_error(
                    "RealityCapture camera contains invalid values: " + name);
            result.scene.views.push_back(std::move(view));
        }
        if (result.scene.views.empty())
            throw std::runtime_error(
                "RealityCapture dataset has no usable camera rows");
        return result;
    }
};

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
            for (std::uint64_t index = 0; index < view_count; ++index)
                vertex.views.emplace_back(
                    reader.pod<std::uint32_t>("vertex image"),
                    reader.pod<float>("vertex confidence"));
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

class ColmapReader final : public DatasetReader {
public:
    [[nodiscard]] DatasetFormat format() const noexcept override {
        return DatasetFormat::colmap;
    }

    [[nodiscard]] bool probe(
        const DatasetLoadRequest& request) const override {
        return has_colmap_model(request.source);
    }

    [[nodiscard]] DatasetLoadResult load(
        const DatasetLoadRequest& request) const override {
        std::filesystem::path images = request.image_directory;
        if (images.empty()) {
            auto root = source_directory(request.source);
            for (unsigned level = 0; level < 4 && !root.empty(); ++level) {
                if (std::filesystem::is_directory(root / "images")) {
                    images = root / "images";
                    break;
                }
                root = root.parent_path();
            }
        }
        auto colmap = load_colmap_scene(
            request.source, images);
        DatasetLoadResult result;
        result.scene = std::move(colmap.scene);
        result.format = format();
        result.resolved_source = std::move(colmap.model_directory);
        return result;
    }
};

void finalize_initial_cloud(
    DatasetLoadResult& result, const DatasetLoadRequest& request) {
    if (!request.initial_point_cloud.empty()) {
        result.scene.dense_cloud =
            mvs::load_dense_ply(request.initial_point_cloud);
        result.initial_point_cloud = request.initial_point_cloud;
        result.initial_points_dense = true;
    } else {
        use_sparse_points_as_initial_cloud(result.scene);
    }
    if (result.scene.dense_cloud.points.empty()) {
        generate_camera_frustum_points(
            result.scene, request.random_initial_point_count, request.seed);
        result.generated_initial_points = true;
        result.initial_points_dense = false;
        result.warnings.emplace_back(
            "Dataset contains camera poses but no point cloud; generated " +
            std::to_string(request.random_initial_point_count) +
            " deterministic camera-frustum initialization points");
    }
}

}  // namespace

DatasetFormat parse_dataset_format(const std::string_view value) {
    const std::string normalized = lower_ascii(std::string(value));
    if (normalized.empty() || normalized == "auto")
        return DatasetFormat::auto_detect;
    if (normalized == "colmap") return DatasetFormat::colmap;
    if (normalized == "realitycapture" ||
        normalized == "reality_capture" ||
        normalized == "reality-capture" ||
        normalized == "rc")
        return DatasetFormat::reality_capture;
    if (normalized == "openmvs" || normalized == "mvs")
        return DatasetFormat::openmvs;
    throw std::invalid_argument(
        "Unknown splat dataset format '" + std::string(value) +
        "' (expected auto, colmap, realitycapture, or openmvs)");
}

std::string_view dataset_format_name(const DatasetFormat format) noexcept {
    switch (format) {
        case DatasetFormat::auto_detect: return "auto";
        case DatasetFormat::colmap: return "colmap";
        case DatasetFormat::reality_capture: return "realitycapture";
        case DatasetFormat::openmvs: return "openmvs";
    }
    return "unknown";
}

void DatasetLoader::register_reader(std::unique_ptr<DatasetReader> reader) {
    if (!reader)
        throw std::invalid_argument("Cannot register a null dataset reader");
    const DatasetFormat incoming = reader->format();
    if (incoming == DatasetFormat::auto_detect)
        throw std::invalid_argument(
            "Dataset readers must expose a concrete format");
    if (std::any_of(
            readers_.begin(), readers_.end(),
            [incoming](const auto& existing) {
                return existing->format() == incoming;
            }))
        throw std::invalid_argument(
            "A reader for dataset format " +
            std::string(dataset_format_name(incoming)) +
            " is already registered");
    readers_.push_back(std::move(reader));
}

DatasetLoadResult DatasetLoader::load(
    const DatasetLoadRequest& request) const {
    if (request.source.empty())
        throw std::invalid_argument(
            "External splat dataset source path is empty");
    for (const auto& reader : readers_) {
        if (request.format != DatasetFormat::auto_detect &&
            reader->format() != request.format)
            continue;
        if (request.format == DatasetFormat::auto_detect &&
            !reader->probe(request))
            continue;
        DatasetLoadResult result = reader->load(request);
        finalize_initial_cloud(result, request);
        return result;
    }
    if (request.format != DatasetFormat::auto_detect)
        throw std::invalid_argument(
            "No registered reader can load explicit format " +
            std::string(dataset_format_name(request.format)));
    throw std::invalid_argument(
        "Could not detect a COLMAP, RealityCapture, or OpenMVS dataset at " +
        request.source.string());
}

DatasetLoader make_default_dataset_loader() {
    DatasetLoader loader;
    loader.register_reader(std::make_unique<ColmapReader>());
    loader.register_reader(std::make_unique<RealityCaptureReader>());
    loader.register_reader(std::make_unique<OpenMvsReader>());
    return loader;
}

DatasetLoadResult load_splat_dataset(const DatasetLoadRequest& request) {
    return make_default_dataset_loader().load(request);
}

}  // namespace aetherscan::splat
