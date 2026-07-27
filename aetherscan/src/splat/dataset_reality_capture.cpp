#include "dataset_internal.hpp"

#include "io/image.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numbers>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace aetherscan::splat::dataset_detail {
namespace {

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

}  // namespace

std::unique_ptr<DatasetReader> make_reality_capture_reader() {
    return std::make_unique<RealityCaptureReader>();
}

}  // namespace aetherscan::splat::dataset_detail
