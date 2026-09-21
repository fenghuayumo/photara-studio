#include "formats_glb.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace photara::splat::detail {
namespace {

using Bytes = std::vector<std::uint8_t>;
using boost::property_tree::ptree;

constexpr std::uint32_t k_glb_magic = 0x46546c67U;
constexpr std::uint32_t k_json_chunk = 0x4e4f534aU;
constexpr std::uint32_t k_bin_chunk = 0x004e4942U;
constexpr std::array<float, 24> k_rdf_to_rub_sh_signs{
    -1.F, -1.F, 1.F, -1.F, 1.F, 1.F, -1.F, 1.F,
    -1.F, 1.F, -1.F, -1.F, 1.F, -1.F, 1.F, -1.F,
    1.F, -1.F, 1.F, 1.F, -1.F, 1.F, -1.F, 1.F};

[[nodiscard]] std::uint32_t read_u32(
    const Bytes& bytes, const std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4U)
        throw std::runtime_error("Truncated GLB header");
    return static_cast<std::uint32_t>(bytes[offset]) |
        (static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U) |
        (static_cast<std::uint32_t>(bytes[offset + 2U]) << 16U) |
        (static_cast<std::uint32_t>(bytes[offset + 3U]) << 24U);
}

void append_u32(Bytes& bytes, const std::uint32_t value) {
    for (unsigned byte = 0; byte < 4; ++byte)
        bytes.push_back(static_cast<std::uint8_t>(value >> (8U * byte)));
}

[[nodiscard]] Bytes read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("Failed to open GLB file: " + path.string());
    const auto end = input.tellg();
    if (end < 0)
        throw std::runtime_error("Failed to size GLB file: " + path.string());
    Bytes bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty())
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    if (!input)
        throw std::runtime_error("Failed to read GLB file: " + path.string());
    return bytes;
}

[[nodiscard]] const ptree& array_item(
    const ptree& array, const std::size_t index, const char* label) {
    std::size_t current = 0;
    for (const auto& item : array) {
        if (current++ == index) return item.second;
    }
    throw std::runtime_error(
        std::string("GLB ") + label + " index is out of range");
}

[[nodiscard]] std::vector<float> number_array(const ptree& node) {
    std::vector<float> result;
    for (const auto& item : node) result.push_back(item.second.get_value<float>());
    return result;
}

[[nodiscard]] bool identity_node(const ptree& node) {
    constexpr float tolerance = 1e-6F;
    if (const auto matrix = node.get_child_optional("matrix")) {
        const auto values = number_array(*matrix);
        if (values.size() != 16U) return false;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const float expected = i % 5U == 0U ? 1.F : 0.F;
            if (std::abs(values[i] - expected) > tolerance) return false;
        }
    }
    const auto check = [tolerance](
                           const auto& value,
                           const std::vector<float>& expected) {
        if (!value) return true;
        const auto actual = number_array(*value);
        if (actual.size() != expected.size()) return false;
        for (std::size_t i = 0; i < actual.size(); ++i)
            if (std::abs(actual[i] - expected[i]) > tolerance) return false;
        return true;
    };
    return check(node.get_child_optional("translation"), {0.F, 0.F, 0.F}) &&
        check(node.get_child_optional("rotation"), {0.F, 0.F, 0.F, 1.F}) &&
        check(node.get_child_optional("scale"), {1.F, 1.F, 1.F});
}

struct AccessorData {
    std::vector<float> values;
    std::size_t count{};
    std::size_t components{};
};

[[nodiscard]] std::size_t component_size(const unsigned type) {
    switch (type) {
        case 5120:
        case 5121: return 1U;
        case 5122:
        case 5123: return 2U;
        case 5125:
        case 5126: return 4U;
        default: throw std::runtime_error("Unsupported GLB accessor component type");
    }
}

[[nodiscard]] std::size_t type_components(const std::string& type) {
    if (type == "SCALAR") return 1U;
    if (type == "VEC2") return 2U;
    if (type == "VEC3") return 3U;
    if (type == "VEC4") return 4U;
    throw std::runtime_error("Unsupported GLB Gaussian accessor type: " + type);
}

template <typename T>
[[nodiscard]] T load_scalar(const std::uint8_t* bytes) {
    T value{};
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

[[nodiscard]] float read_component(
    const std::uint8_t* bytes, const unsigned type, const bool normalized) {
    switch (type) {
        case 5120: {
            const auto value = load_scalar<std::int8_t>(bytes);
            return normalized
                ? std::max(-1.F, static_cast<float>(value) / 127.F)
                : static_cast<float>(value);
        }
        case 5121: {
            const auto value = load_scalar<std::uint8_t>(bytes);
            return normalized ? static_cast<float>(value) / 255.F
                              : static_cast<float>(value);
        }
        case 5122: {
            const auto value = load_scalar<std::int16_t>(bytes);
            return normalized
                ? std::max(-1.F, static_cast<float>(value) / 32767.F)
                : static_cast<float>(value);
        }
        case 5123: {
            const auto value = load_scalar<std::uint16_t>(bytes);
            return normalized ? static_cast<float>(value) / 65535.F
                              : static_cast<float>(value);
        }
        case 5125: {
            const auto value = load_scalar<std::uint32_t>(bytes);
            return normalized
                ? static_cast<float>(static_cast<double>(value) / 4294967295.0)
                : static_cast<float>(value);
        }
        case 5126: return load_scalar<float>(bytes);
        default: break;
    }
    throw std::runtime_error("Unsupported GLB accessor component type");
}

[[nodiscard]] AccessorData read_accessor(
    const ptree& root, const Bytes& binary, const std::size_t index,
    const std::string& expected_type, const char* label) {
    const auto& accessor = array_item(root.get_child("accessors"), index, "accessor");
    if (accessor.get_child_optional("sparse"))
        throw std::runtime_error(
            std::string("Sparse GLB accessors are unsupported for ") + label);
    const auto view_index = accessor.get_optional<std::size_t>("bufferView");
    if (!view_index)
        throw std::runtime_error(std::string("GLB accessor has no bufferView: ") + label);
    const auto& view = array_item(
        root.get_child("bufferViews"), *view_index, "bufferView");
    if (view.get<std::size_t>("buffer", 0U) != 0U)
        throw std::runtime_error("GLB Gaussian accessor references an external buffer");
    const std::string type = accessor.get<std::string>("type");
    if (type != expected_type)
        throw std::runtime_error(
            std::string("GLB ") + label + " must use accessor type " + expected_type);
    const auto components = type_components(type);
    const auto component_type = accessor.get<unsigned>("componentType");
    const auto scalar_size = component_size(component_type);
    const auto element_size = components * scalar_size;
    const auto stride = view.get<std::size_t>("byteStride", element_size);
    const auto count = accessor.get<std::size_t>("count");
    const auto view_offset = view.get<std::size_t>("byteOffset", 0U);
    const auto view_length = view.get<std::size_t>("byteLength");
    const auto accessor_offset = accessor.get<std::size_t>("byteOffset", 0U);
    const bool normalized = accessor.get<bool>("normalized", false);
    if (stride < element_size || view_offset > binary.size() ||
        view_length > binary.size() - view_offset ||
        accessor_offset > view_length ||
        (count != 0U &&
         ((count - 1U) > (view_length - accessor_offset) / stride ||
          element_size > view_length - accessor_offset - (count - 1U) * stride)))
        throw std::runtime_error(std::string("GLB accessor is truncated: ") + label);
    const auto offset = view_offset + accessor_offset;
    AccessorData result;
    result.count = count;
    result.components = components;
    result.values.resize(count * components);
    for (std::size_t row = 0; row < count; ++row)
        for (std::size_t component = 0; component < components; ++component)
            result.values[row * components + component] = read_component(
                binary.data() + offset + row * stride + component * scalar_size,
                component_type, normalized);
    return result;
}

[[nodiscard]] std::optional<std::size_t> attribute_index(
    const ptree& attributes, const std::string& semantic) {
    for (const auto& item : attributes)
        if (item.first == semantic)
            return item.second.get_value<std::size_t>();
    return std::nullopt;
}

[[nodiscard]] std::string sh_semantic(
    const unsigned degree, const unsigned coefficient) {
    return "KHR_gaussian_splatting:SH_DEGREE_" + std::to_string(degree) +
        "_COEF_" + std::to_string(coefficient);
}

void append_float(Bytes& bytes, const float value) {
    std::array<std::uint8_t, sizeof(float)> encoded{};
    std::memcpy(encoded.data(), &value, sizeof(value));
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
}

struct OutputAccessor {
    std::string semantic;
    std::string type;
    std::size_t components{};
    std::size_t offset{};
    std::size_t length{};
    std::optional<std::array<float, 3>> minimum;
    std::optional<std::array<float, 3>> maximum;
};

void append_output_accessor(
    Bytes& binary, std::vector<OutputAccessor>& accessors,
    std::string semantic, std::string type, const std::size_t components,
    const std::vector<float>& values,
    std::optional<std::array<float, 3>> minimum = std::nullopt,
    std::optional<std::array<float, 3>> maximum = std::nullopt) {
    while (binary.size() % 4U != 0U) binary.push_back(0U);
    OutputAccessor accessor;
    accessor.semantic = std::move(semantic);
    accessor.type = std::move(type);
    accessor.components = components;
    accessor.offset = binary.size();
    accessor.length = values.size() * sizeof(float);
    accessor.minimum = minimum;
    accessor.maximum = maximum;
    for (const float value : values) append_float(binary, value);
    accessors.push_back(std::move(accessor));
}

[[nodiscard]] std::string glb_json(
    const std::size_t count, const Bytes& binary,
    const std::vector<OutputAccessor>& accessors) {
    std::ostringstream json;
    json << std::setprecision(9)
         << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"Photara\"},"
         << "\"extensionsUsed\":[\"KHR_gaussian_splatting\"],"
         << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
         << "\"nodes\":[{\"mesh\":0}],\"meshes\":[{\"primitives\":[{"
         << "\"mode\":0,\"attributes\":{";
    for (std::size_t i = 0; i < accessors.size(); ++i) {
        if (i != 0U) json << ',';
        json << '\"' << accessors[i].semantic << "\":" << i;
    }
    json << "},\"extensions\":{\"KHR_gaussian_splatting\":{"
         << "\"kernel\":\"ellipse\",\"colorSpace\":\"srgb_rec709_display\"}}"
         << "}]}],\"buffers\":[{\"byteLength\":" << binary.size() << "}],"
         << "\"bufferViews\":[";
    for (std::size_t i = 0; i < accessors.size(); ++i) {
        if (i != 0U) json << ',';
        json << "{\"buffer\":0,\"byteOffset\":" << accessors[i].offset
             << ",\"byteLength\":" << accessors[i].length
             << ",\"target\":34962}";
    }
    json << "],\"accessors\":[";
    for (std::size_t i = 0; i < accessors.size(); ++i) {
        if (i != 0U) json << ',';
        const auto& accessor = accessors[i];
        json << "{\"bufferView\":" << i << ",\"componentType\":5126,"
             << "\"count\":" << count << ",\"type\":\"" << accessor.type << '\"';
        if (accessor.minimum && accessor.maximum) {
            json << ",\"min\":[" << (*accessor.minimum)[0] << ','
                 << (*accessor.minimum)[1] << ',' << (*accessor.minimum)[2]
                 << "],\"max\":[" << (*accessor.maximum)[0] << ','
                 << (*accessor.maximum)[1] << ',' << (*accessor.maximum)[2] << ']';
        }
        json << '}';
    }
    json << "]}";
    return json.str();
}

}  // namespace

Bytes save_glb(const GaussianModel& model) {
    const std::size_t count = model.size();
    if (count == 0U || model.sh_degree > 3U || !model.sh.is_valid())
        throw std::runtime_error("Cannot write an empty or invalid GLB Gaussian model");
    const std::size_t bases = static_cast<std::size_t>(model.sh_degree + 1U) *
        (model.sh_degree + 1U);
    const auto means = model.means.to_vector();
    const auto log_scales = model.log_scales.to_vector();
    const auto rotations = model.quaternions.to_vector();
    const auto logits = model.opacity_logits.to_vector();
    const auto sh = model.sh.to_vector();
    if (means.size() != count * 3U || log_scales.size() != count * 3U ||
        rotations.size() != count * 4U || logits.size() != count ||
        sh.size() != count * bases * 3U)
        throw std::runtime_error("Gaussian model has inconsistent GLB attribute sizes");

    std::vector<float> positions(count * 3U);
    std::vector<float> scales(count * 3U);
    std::vector<float> gltf_rotations(count * 4U);
    std::vector<float> opacities(count);
    std::vector<float> converted_sh(sh);
    std::array<float, 3> minimum{
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    std::array<float, 3> maximum{
        std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    for (std::size_t index = 0; index < count; ++index) {
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            const float position = axis == 0U ? means[3U * index]
                : -means[3U * index + axis];
            const float scale = std::exp(log_scales[3U * index + axis]);
            if (!std::isfinite(position) || !std::isfinite(scale) || scale <= 0.F)
                throw std::runtime_error("GLB Gaussian position or scale is invalid");
            positions[3U * index + axis] = position;
            scales[3U * index + axis] = scale;
            minimum[axis] = std::min(minimum[axis], position);
            maximum[axis] = std::max(maximum[axis], position);
        }
        float w = rotations[4U * index];
        float x = rotations[4U * index + 1U];
        float y = -rotations[4U * index + 2U];
        float z = -rotations[4U * index + 3U];
        const float norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (!std::isfinite(norm) || norm <= 1e-12F)
            throw std::runtime_error("GLB Gaussian quaternion is invalid");
        gltf_rotations[4U * index] = x / norm;
        gltf_rotations[4U * index + 1U] = y / norm;
        gltf_rotations[4U * index + 2U] = z / norm;
        gltf_rotations[4U * index + 3U] = w / norm;
        const float logit = logits[index];
        if (!std::isfinite(logit))
            throw std::runtime_error("GLB Gaussian opacity is invalid");
        opacities[index] = logit >= 0.F
            ? 1.F / (1.F + std::exp(-logit))
            : std::exp(logit) / (1.F + std::exp(logit));
        for (std::size_t coefficient = 1U; coefficient < bases; ++coefficient)
            for (std::size_t channel = 0; channel < 3U; ++channel) {
                const auto target = (index * bases + coefficient) * 3U + channel;
                converted_sh[target] *= k_rdf_to_rub_sh_signs[coefficient - 1U];
            }
    }
    if (!std::all_of(converted_sh.begin(), converted_sh.end(),
                     [](const float value) { return std::isfinite(value); }))
        throw std::runtime_error("GLB Gaussian SH coefficients are invalid");

    Bytes binary;
    std::vector<OutputAccessor> accessors;
    append_output_accessor(
        binary, accessors, "POSITION", "VEC3", 3U, positions, minimum, maximum);
    append_output_accessor(
        binary, accessors, "KHR_gaussian_splatting:ROTATION", "VEC4", 4U,
        gltf_rotations);
    append_output_accessor(
        binary, accessors, "KHR_gaussian_splatting:SCALE", "VEC3", 3U, scales);
    append_output_accessor(
        binary, accessors, "KHR_gaussian_splatting:OPACITY", "SCALAR", 1U,
        opacities);
    for (unsigned degree = 0; degree <= model.sh_degree; ++degree)
        for (unsigned local = 0; local < 2U * degree + 1U; ++local) {
            const std::size_t coefficient =
                static_cast<std::size_t>(degree) * degree + local;
            std::vector<float> values(count * 3U);
            for (std::size_t index = 0; index < count; ++index)
                std::copy_n(
                    converted_sh.begin() + static_cast<std::ptrdiff_t>(
                        (index * bases + coefficient) * 3U),
                    3U, values.begin() + static_cast<std::ptrdiff_t>(index * 3U));
            append_output_accessor(
                binary, accessors, sh_semantic(degree, local), "VEC3", 3U, values);
        }
    if (binary.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("GLB Gaussian binary payload exceeds 4 GiB");
    std::string json = glb_json(count, binary, accessors);
    while (json.size() % 4U != 0U) json.push_back(' ');
    while (binary.size() % 4U != 0U) binary.push_back(0U);
    const std::uint64_t total = 12ULL + 8ULL + json.size() + 8ULL + binary.size();
    if (total > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("GLB Gaussian file exceeds 4 GiB");
    Bytes output;
    output.reserve(static_cast<std::size_t>(total));
    append_u32(output, k_glb_magic);
    append_u32(output, 2U);
    append_u32(output, static_cast<std::uint32_t>(total));
    append_u32(output, static_cast<std::uint32_t>(json.size()));
    append_u32(output, k_json_chunk);
    output.insert(output.end(), json.begin(), json.end());
    append_u32(output, static_cast<std::uint32_t>(binary.size()));
    append_u32(output, k_bin_chunk);
    output.insert(output.end(), binary.begin(), binary.end());
    return output;
}

DecodedGlbGaussians load_glb(const std::filesystem::path& path) {
    const Bytes file = read_file(path);
    if (file.size() < 20U || read_u32(file, 0U) != k_glb_magic ||
        read_u32(file, 4U) != 2U || read_u32(file, 8U) != file.size())
        throw std::runtime_error("Invalid GLB 2.0 header: " + path.string());
    std::size_t cursor = 12U;
    std::string json_text;
    Bytes binary;
    while (cursor < file.size()) {
        if (file.size() - cursor < 8U)
            throw std::runtime_error("Truncated GLB chunk header");
        const std::size_t length = read_u32(file, cursor);
        const auto type = read_u32(file, cursor + 4U);
        cursor += 8U;
        if (length > file.size() - cursor)
            throw std::runtime_error("Truncated GLB chunk payload");
        if (type == k_json_chunk && json_text.empty())
            json_text.assign(
                reinterpret_cast<const char*>(file.data() + cursor), length);
        else if (type == k_bin_chunk && binary.empty())
            binary.assign(
                file.begin() + static_cast<std::ptrdiff_t>(cursor),
                file.begin() + static_cast<std::ptrdiff_t>(cursor + length));
        cursor += length;
    }
    if (json_text.empty() || binary.empty())
        throw std::runtime_error("GLB Gaussian file requires JSON and BIN chunks");
    ptree root;
    std::istringstream json(json_text);
    boost::property_tree::read_json(json, root);
    if (root.get<std::string>("asset.version", "") != "2.0")
        throw std::runtime_error("GLB asset is not glTF 2.0");
    const auto& buffers = root.get_child("buffers");
    const auto& buffer = array_item(buffers, 0U, "buffer");
    if (buffer.get_optional<std::string>("uri"))
        throw std::runtime_error("External GLB buffers are unsupported");
    if (buffer.get<std::size_t>("byteLength") > binary.size())
        throw std::runtime_error("GLB BIN chunk is shorter than buffer.byteLength");

    const ptree* primitive = nullptr;
    const ptree* extension = nullptr;
    std::size_t selected_mesh = 0U;
    std::size_t mesh_index = 0U;
    for (const auto& mesh_item : root.get_child("meshes")) {
        for (const auto& primitive_item : mesh_item.second.get_child("primitives")) {
            const auto candidate = primitive_item.second.get_child_optional(
                "extensions.KHR_gaussian_splatting");
            if (!candidate) continue;
            if (primitive != nullptr)
                throw std::runtime_error(
                    "GLB files with multiple Gaussian primitives are unsupported");
            primitive = &primitive_item.second;
            extension = &candidate.get();
            selected_mesh = mesh_index;
        }
        ++mesh_index;
    }
    if (primitive == nullptr || extension == nullptr)
        throw std::runtime_error("GLB has no KHR_gaussian_splatting primitive");
    if (primitive->get<unsigned>("mode", 4U) != 0U)
        throw std::runtime_error("GLB Gaussian primitive must use POINTS mode");
    if (extension->get<std::string>("kernel", "") != "ellipse")
        throw std::runtime_error("Unsupported GLB Gaussian kernel");
    const auto color_space = extension->get<std::string>("colorSpace", "");
    if (color_space != "srgb_rec709_display" &&
        color_space != "lin_rec709_display")
        throw std::runtime_error("Unsupported GLB Gaussian color space");

    std::size_t mesh_instances = 0U;
    if (const auto nodes = root.get_child_optional("nodes")) {
        std::vector<const ptree*> node_list;
        node_list.reserve(nodes->size());
        for (const auto& node_item : *nodes) node_list.push_back(&node_item.second);
        const auto no_parent = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> parents(node_list.size(), no_parent);
        for (std::size_t parent = 0; parent < node_list.size(); ++parent) {
            if (const auto children = node_list[parent]->get_child_optional("children"))
                for (const auto& child : *children) {
                    const auto index = child.second.get_value<std::size_t>();
                    if (index >= node_list.size() || parents[index] != no_parent)
                        throw std::runtime_error("GLB node hierarchy is invalid");
                    parents[index] = parent;
                }
        }
        for (std::size_t index = 0; index < node_list.size(); ++index) {
            if (node_list[index]->get<std::size_t>(
                    "mesh", no_parent) != selected_mesh)
                continue;
            ++mesh_instances;
            std::size_t current = index;
            std::size_t depth = 0U;
            while (current != no_parent) {
                if (++depth > node_list.size())
                    throw std::runtime_error("GLB node hierarchy contains a cycle");
                if (!identity_node(*node_list[current]))
                    throw std::runtime_error(
                        "GLB Gaussian node transforms are not supported; bake them first");
                current = parents[current];
            }
        }
    }
    if (mesh_instances > 1U)
        throw std::runtime_error("Instanced GLB Gaussian meshes are unsupported");

    const auto& attributes = primitive->get_child("attributes");
    const auto required = [&](const std::string& semantic) {
        const auto index = attribute_index(attributes, semantic);
        if (!index)
            throw std::runtime_error("Missing GLB Gaussian attribute: " + semantic);
        return *index;
    };
    const auto positions = read_accessor(
        root, binary, required("POSITION"), "VEC3", "POSITION");
    const auto rotations = read_accessor(
        root, binary, required("KHR_gaussian_splatting:ROTATION"), "VEC4",
        "ROTATION");
    const auto scales = read_accessor(
        root, binary, required("KHR_gaussian_splatting:SCALE"), "VEC3", "SCALE");
    const auto opacities = read_accessor(
        root, binary, required("KHR_gaussian_splatting:OPACITY"), "SCALAR",
        "OPACITY");
    const auto sh0 = read_accessor(
        root, binary, required(sh_semantic(0U, 0U)), "VEC3", "SH degree 0");
    const std::size_t count = positions.count;
    if (count == 0U || rotations.count != count || scales.count != count ||
        opacities.count != count || sh0.count != count)
        throw std::runtime_error("GLB Gaussian accessor counts are inconsistent");

    unsigned degree = 0U;
    std::array<std::vector<AccessorData>, 4> sh_degrees;
    sh_degrees[0].push_back(sh0);
    bool missing_degree = false;
    for (unsigned candidate = 1U; candidate <= 3U; ++candidate) {
        std::vector<AccessorData> values;
        bool any = false;
        bool all = true;
        for (unsigned local = 0U; local < 2U * candidate + 1U; ++local) {
            const auto found = attribute_index(attributes, sh_semantic(candidate, local));
            any = any || found.has_value();
            all = all && found.has_value();
            if (found) values.push_back(read_accessor(
                root, binary, *found, "VEC3", "SH coefficient"));
        }
        if (any && (!all || missing_degree))
            throw std::runtime_error(
                "GLB Gaussian SH degrees must be complete and contiguous");
        if (!all) {
            missing_degree = true;
            continue;
        }
        for (const auto& value : values)
            if (value.count != count)
                throw std::runtime_error("GLB Gaussian SH accessor counts differ");
        degree = candidate;
        sh_degrees[candidate] = std::move(values);
    }

    const std::size_t bases = static_cast<std::size_t>(degree + 1U) *
        (degree + 1U);
    DecodedGlbGaussians result;
    result.count = count;
    result.degree = degree;
    result.means.resize(count * 3U);
    result.log_scales.resize(count * 3U);
    result.rotations.resize(count * 4U);
    result.opacity_logits.resize(count);
    result.sh.resize(count * bases * 3U);
    for (std::size_t index = 0; index < count; ++index) {
        for (std::size_t axis = 0; axis < 3U; ++axis) {
            const float position = positions.values[index * 3U + axis];
            const float scale = scales.values[index * 3U + axis];
            if (!std::isfinite(position) || !std::isfinite(scale) || scale <= 0.F)
                throw std::runtime_error("GLB Gaussian position or scale is invalid");
            result.means[index * 3U + axis] = axis == 0U ? position : -position;
            result.log_scales[index * 3U + axis] = std::log(scale);
        }
        const float x = rotations.values[index * 4U];
        const float y = rotations.values[index * 4U + 1U];
        const float z = rotations.values[index * 4U + 2U];
        const float w = rotations.values[index * 4U + 3U];
        const float norm = std::sqrt(w * w + x * x + y * y + z * z);
        if (!std::isfinite(norm) || std::abs(norm - 1.F) > 2e-3F)
            throw std::runtime_error("GLB Gaussian quaternion is not normalized");
        result.rotations[index * 4U] = w / norm;
        result.rotations[index * 4U + 1U] = x / norm;
        result.rotations[index * 4U + 2U] = -y / norm;
        result.rotations[index * 4U + 3U] = -z / norm;
        const float opacity = opacities.values[index];
        if (!std::isfinite(opacity) || opacity < 0.F || opacity > 1.F)
            throw std::runtime_error("GLB Gaussian opacity is outside [0,1]");
        const float alpha = std::clamp(opacity, 1e-5F, 1.F - 1e-5F);
        result.opacity_logits[index] = std::log(alpha / (1.F - alpha));
        for (unsigned sh_degree = 0U; sh_degree <= degree; ++sh_degree)
            for (unsigned local = 0U; local < 2U * sh_degree + 1U; ++local) {
                const std::size_t coefficient =
                    static_cast<std::size_t>(sh_degree) * sh_degree + local;
                const auto& source = sh_degrees[sh_degree][local].values;
                const float sign = coefficient == 0U
                    ? 1.F : k_rdf_to_rub_sh_signs[coefficient - 1U];
                for (std::size_t channel = 0; channel < 3U; ++channel) {
                    const float value = source[index * 3U + channel] * sign;
                    if (!std::isfinite(value))
                        throw std::runtime_error("GLB Gaussian SH value is invalid");
                    result.sh[(index * bases + coefficient) * 3U + channel] = value;
                }
            }
    }
    return result;
}

}  // namespace photara::splat::detail
