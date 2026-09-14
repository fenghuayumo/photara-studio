#include "splat/formats.hpp"

#include "formats_glb.hpp"
#include "splat/trainer.hpp"

#include <webp/decode.h>
#include <webp/encode.h>
#include <zlib.h>
#include <zstd.h>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aetherscan::splat {
namespace {

using Bytes = std::vector<std::uint8_t>;
using boost::property_tree::ptree;

constexpr float k_sh_c0 = 0.28209479177387814F;
constexpr float k_spz_color_scale = 0.15F;
constexpr float k_spz_sqrt_half = 0.7071067811865475244F;
constexpr std::array<float, 24> k_rdf_to_rub_sh_signs{
    -1.F, -1.F, 1.F, -1.F, 1.F, 1.F, -1.F, 1.F,
    -1.F, 1.F, -1.F, -1.F, 1.F, -1.F, 1.F, -1.F,
    1.F, -1.F, 1.F, 1.F, -1.F, 1.F, -1.F, 1.F};

[[nodiscard]] std::string lower(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    return value;
}

[[nodiscard]] std::uint32_t read_u32(const Bytes& bytes, const std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4)
        throw std::runtime_error("Truncated splat format header");
    return static_cast<std::uint32_t>(bytes[offset]) |
        (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
}

[[nodiscard]] std::uint64_t read_u64(const Bytes& bytes, const std::size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 8)
        throw std::runtime_error("Truncated splat format header");
    std::uint64_t value = 0;
    for (unsigned byte = 0; byte < 8; ++byte)
        value |= static_cast<std::uint64_t>(bytes[offset + byte]) << (8U * byte);
    return value;
}

void append_u16(Bytes& bytes, const std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value & 0xffU));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void append_u32(Bytes& bytes, const std::uint32_t value) {
    for (unsigned byte = 0; byte < 4; ++byte)
        bytes.push_back(static_cast<std::uint8_t>(value >> (8U * byte)));
}

void append_u64(Bytes& bytes, const std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte)
        bytes.push_back(static_cast<std::uint8_t>(value >> (8U * byte)));
}

[[nodiscard]] Bytes read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("Failed to open splat file: " + path.string());
    const auto end = input.tellg();
    if (end < 0)
        throw std::runtime_error("Failed to size splat file: " + path.string());
    Bytes bytes(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!bytes.empty())
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    if (!input)
        throw std::runtime_error("Failed to read splat file: " + path.string());
    return bytes;
}

void write_file(const std::filesystem::path& path, const Bytes& bytes) {
    if (!path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error)
            throw std::runtime_error(
                "Failed to create splat output directory: " +
                path.parent_path().string());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output)
        throw std::runtime_error("Failed to create splat file: " + path.string());
    if (!bytes.empty())
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("Failed to write splat file: " + path.string());
}

[[nodiscard]] std::size_t checked_size(
    const std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error(std::string("Splat ") + label + " is too large");
    return static_cast<std::size_t>(value);
}

[[nodiscard]] std::size_t checked_product(
    const std::size_t a, const std::size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::runtime_error(std::string("Splat ") + label + " is too large");
    return a * b;
}

struct WebpImage {
    int width{};
    int height{};
    Bytes rgba;
};

[[nodiscard]] WebpImage decode_webp(
    const Bytes& bytes, const std::string& asset_name) {
    int width = 0;
    int height = 0;
    if (bytes.empty() || !WebPGetInfo(bytes.data(), bytes.size(), &width, &height) ||
        width <= 0 || height <= 0)
        throw std::runtime_error("Invalid SOG WebP asset: " + asset_name);
    const auto pixel_count = checked_product(
        static_cast<std::size_t>(width), static_cast<std::size_t>(height),
        "WebP dimensions");
    const auto byte_count = checked_product(pixel_count, std::size_t{4},
                                             "WebP pixels");
    std::unique_ptr<std::uint8_t, decltype(&WebPFree)> decoded(
        WebPDecodeRGBA(bytes.data(), bytes.size(), &width, &height), &WebPFree);
    if (!decoded)
        throw std::runtime_error("Failed to decode SOG WebP asset: " + asset_name);
    WebpImage image;
    image.width = width;
    image.height = height;
    image.rgba.assign(decoded.get(), decoded.get() + byte_count);
    return image;
}

[[nodiscard]] Bytes encode_webp(
    const std::uint8_t* pixels, const int width, const int height,
    const int channels, const std::string& asset_name) {
    if (pixels == nullptr || width <= 0 || height <= 0 ||
        (channels != 3 && channels != 4))
        throw std::invalid_argument("Invalid SOG WebP input: " + asset_name);
    std::uint8_t* encoded = nullptr;
    const std::size_t encoded_size = channels == 4
        ? WebPEncodeLosslessRGBA(
              pixels, width, height, width * channels, &encoded)
        : WebPEncodeLosslessRGB(
              pixels, width, height, width * channels, &encoded);
    if (encoded_size == 0 || encoded == nullptr)
        throw std::runtime_error("Failed to encode SOG WebP asset: " + asset_name);
    Bytes result(encoded, encoded + encoded_size);
    WebPFree(encoded);
    return result;
}

struct ZipEntry {
    std::string name;
    Bytes data;
};

void append_zip_entry(Bytes& output, const ZipEntry& entry) {
    if (entry.name.size() > std::numeric_limits<std::uint16_t>::max() ||
        entry.data.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("SOG ZIP entry is too large: " + entry.name);
    const auto crc = static_cast<std::uint32_t>(crc32(
        0, entry.data.data(), static_cast<uInt>(entry.data.size())));
    append_u32(output, 0x04034b50U);
    append_u16(output, 20);
    append_u16(output, 0);
    append_u16(output, 0);  // Stored: WebP assets are already compressed.
    append_u16(output, 0);
    append_u16(output, 0);
    append_u32(output, crc);
    append_u32(output, static_cast<std::uint32_t>(entry.data.size()));
    append_u32(output, static_cast<std::uint32_t>(entry.data.size()));
    append_u16(output, static_cast<std::uint16_t>(entry.name.size()));
    append_u16(output, 0);
    output.insert(output.end(), entry.name.begin(), entry.name.end());
    output.insert(output.end(), entry.data.begin(), entry.data.end());
}

[[nodiscard]] Bytes make_sog_zip(const std::vector<ZipEntry>& entries) {
    struct Central {
        std::string name;
        std::uint32_t crc{};
        std::uint32_t size{};
        std::uint32_t offset{};
    };
    Bytes output;
    std::vector<Central> central;
    central.reserve(entries.size());
    for (const auto& entry : entries) {
        const auto offset = output.size();
        append_zip_entry(output, entry);
        central.push_back({
            entry.name, static_cast<std::uint32_t>(crc32(
                             0, entry.data.data(),
                             static_cast<uInt>(entry.data.size()))),
            static_cast<std::uint32_t>(entry.data.size()),
            static_cast<std::uint32_t>(offset)});
    }
    const auto central_offset = output.size();
    for (const auto& entry : central) {
        append_u32(output, 0x02014b50U);
        append_u16(output, 20);
        append_u16(output, 20);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u32(output, entry.crc);
        append_u32(output, entry.size);
        append_u32(output, entry.size);
        append_u16(output, static_cast<std::uint16_t>(entry.name.size()));
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u16(output, 0);
        append_u32(output, 0);
        append_u32(output, entry.offset);
        output.insert(output.end(), entry.name.begin(), entry.name.end());
    }
    const auto central_size = output.size() - central_offset;
    if (central_offset > std::numeric_limits<std::uint32_t>::max() ||
        central_size > std::numeric_limits<std::uint32_t>::max() ||
        central.size() > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("SOG ZIP archive is too large");
    append_u32(output, 0x06054b50U);
    append_u16(output, 0);
    append_u16(output, 0);
    append_u16(output, static_cast<std::uint16_t>(central.size()));
    append_u16(output, static_cast<std::uint16_t>(central.size()));
    append_u32(output, static_cast<std::uint32_t>(central_size));
    append_u32(output, static_cast<std::uint32_t>(central_offset));
    append_u16(output, 0);
    return output;
}

[[nodiscard]] Bytes inflate_zip_entry(
    const std::uint8_t* compressed, const std::size_t compressed_size,
    const std::size_t uncompressed_size) {
    Bytes result(uncompressed_size);
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(compressed);
    stream.avail_in = static_cast<uInt>(compressed_size);
    stream.next_out = result.data();
    stream.avail_out = static_cast<uInt>(result.size());
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
        throw std::runtime_error("Failed to initialize SOG ZIP decompressor");
    const int status = inflate(&stream, Z_FINISH);
    const bool ok = status == Z_STREAM_END && stream.total_out == uncompressed_size;
    inflateEnd(&stream);
    if (!ok) throw std::runtime_error("Invalid compressed SOG ZIP entry");
    return result;
}

[[nodiscard]] std::map<std::string, Bytes> read_sog_zip(const Bytes& bytes) {
    if (bytes.size() < 22)
        throw std::runtime_error("SOG ZIP archive is truncated");
    const std::size_t search_begin = bytes.size() > 0xffffU + 22U
        ? bytes.size() - (0xffffU + 22U)
        : 0;
    std::size_t eocd = std::numeric_limits<std::size_t>::max();
    for (std::size_t offset = bytes.size() - 22U;; --offset) {
        if (read_u32(bytes, offset) == 0x06054b50U) {
            eocd = offset;
            break;
        }
        if (offset == search_begin) break;
    }
    if (eocd == std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("SOG is not a ZIP archive");
    const auto entries = read_u32(bytes, eocd + 10) & 0xffffU;
    const auto central_size = checked_size(read_u32(bytes, eocd + 12),
                                           "ZIP central directory");
    const auto central_offset = checked_size(read_u32(bytes, eocd + 16),
                                             "ZIP central directory");
    if (central_offset > bytes.size() || central_size > bytes.size() - central_offset)
        throw std::runtime_error("SOG ZIP central directory is truncated");
    std::map<std::string, Bytes> result;
    std::size_t cursor = central_offset;
    for (std::uint32_t index = 0; index < entries; ++index) {
        if (cursor > bytes.size() || bytes.size() - cursor < 46 ||
            read_u32(bytes, cursor) != 0x02014b50U)
            throw std::runtime_error("Invalid SOG ZIP central directory");
        const auto flags = read_u32(bytes, cursor + 8) & 0xffffU;
        const auto method = read_u32(bytes, cursor + 10) & 0xffffU;
        const auto compressed_size = checked_size(read_u32(bytes, cursor + 20),
                                                  "ZIP entry");
        const auto uncompressed_size = checked_size(read_u32(bytes, cursor + 24),
                                                    "ZIP entry");
        const auto name_size = static_cast<std::size_t>(read_u32(bytes, cursor + 28) & 0xffffU);
        const auto extra_size = static_cast<std::size_t>(read_u32(bytes, cursor + 30) & 0xffffU);
        const auto comment_size = static_cast<std::size_t>(read_u32(bytes, cursor + 32) & 0xffffU);
        const auto local_offset = checked_size(read_u32(bytes, cursor + 42),
                                               "ZIP entry offset");
        if ((flags & 1U) != 0 || name_size == 0 ||
            46U + name_size + extra_size + comment_size > bytes.size() - cursor)
            throw std::runtime_error("Unsupported or invalid SOG ZIP entry");
        const std::string name(
            reinterpret_cast<const char*>(bytes.data() + cursor + 46), name_size);
        if (local_offset > bytes.size() || bytes.size() - local_offset < 30 ||
            read_u32(bytes, local_offset) != 0x04034b50U)
            throw std::runtime_error("Invalid SOG ZIP local header");
        const auto local_name_size = static_cast<std::size_t>(
            read_u32(bytes, local_offset + 26) & 0xffffU);
        const auto local_extra_size = static_cast<std::size_t>(
            read_u32(bytes, local_offset + 28) & 0xffffU);
        const auto data_offset = local_offset + 30U + local_name_size + local_extra_size;
        if (data_offset > bytes.size() || compressed_size > bytes.size() - data_offset)
            throw std::runtime_error("SOG ZIP entry payload is truncated");
        Bytes payload;
        if (method == 0)
            payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(data_offset),
                           bytes.begin() + static_cast<std::ptrdiff_t>(data_offset + compressed_size));
        else if (method == 8)
            payload = inflate_zip_entry(
                bytes.data() + data_offset, compressed_size, uncompressed_size);
        else
            throw std::runtime_error("Unsupported SOG ZIP compression method");
        if (payload.size() != uncompressed_size ||
            crc32(0, payload.data(), static_cast<uInt>(payload.size())) !=
                read_u32(bytes, cursor + 16))
            throw std::runtime_error("SOG ZIP entry checksum mismatch");
        result.emplace(name, std::move(payload));
        cursor += 46U + name_size + extra_size + comment_size;
    }
    return result;
}

struct SogAssets {
    std::string meta;
    std::map<std::string, Bytes> files;
};

[[nodiscard]] SogAssets read_sog_assets(const std::filesystem::path& path) {
    SogAssets result;
    if (std::filesystem::is_directory(path)) {
        const auto meta_path = path / "meta.json";
        const Bytes meta_bytes = read_file(meta_path);
        result.meta.assign(meta_bytes.begin(), meta_bytes.end());
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (!entry.is_regular_file() || entry.path().filename() == "meta.json")
                continue;
            result.files.emplace(entry.path().filename().generic_string(),
                                 read_file(entry.path()));
        }
        return result;
    }
    const auto files = read_sog_zip(read_file(path));
    const auto meta = files.find("meta.json");
    if (meta == files.end()) throw std::runtime_error("SOG archive has no meta.json");
    result.meta.assign(meta->second.begin(), meta->second.end());
    result.files = files;
    return result;
}

[[nodiscard]] const Bytes& sog_file(
    const SogAssets& assets, const std::string& name) {
    const auto found = assets.files.find(name);
    if (found == assets.files.end())
        throw std::runtime_error("SOG asset is missing: " + name);
    return found->second;
}

[[nodiscard]] std::vector<std::string> string_array(const ptree& node) {
    std::vector<std::string> result;
    for (const auto& child : node) result.push_back(child.second.get_value<std::string>());
    return result;
}

[[nodiscard]] std::vector<float> float_array(
    const ptree& node, const std::size_t expected, const char* label) {
    std::vector<float> result;
    result.reserve(expected);
    for (const auto& child : node)
        result.push_back(child.second.get_value<float>());
    if (result.size() != expected)
        throw std::runtime_error(std::string("SOG ") + label + " must contain " +
                                 std::to_string(expected) + " values");
    return result;
}

[[nodiscard]] std::vector<float> sog_codebook(
    const ptree& node, const char* label) {
    return float_array(node.get_child("codebook"), 256, label);
}

[[nodiscard]] std::array<std::size_t, 2> sog_dimensions(
    const WebpImage& lower_image, const WebpImage& upper_image,
    const WebpImage& scales, const WebpImage& quats, const WebpImage& sh0,
    const std::size_t count) {
    const std::array<const WebpImage*, 5> images{
        &lower_image, &upper_image, &scales, &quats, &sh0};
    for (const WebpImage* image : images)
        if (image->width != lower_image.width || image->height != lower_image.height)
            throw std::runtime_error("SOG per-Gaussian images have different dimensions");
    const auto pixels = checked_product(
        static_cast<std::size_t>(lower_image.width),
        static_cast<std::size_t>(lower_image.height), "SOG image dimensions");
    if (count == 0 || count > pixels)
        throw std::runtime_error("SOG count exceeds its image capacity");
    return {static_cast<std::size_t>(lower_image.width),
            static_cast<std::size_t>(lower_image.height)};
}

[[nodiscard]] float unlog_position(const float value) {
    return std::copysign(std::expm1(std::abs(value)), value);
}

[[nodiscard]] std::uint8_t nearest_codebook(
    const std::vector<float>& codebook, const float value) {
    const auto found = std::min_element(
        codebook.begin(), codebook.end(),
        [value](const float a, const float b) {
            return std::abs(a - value) < std::abs(b - value);
        });
    return static_cast<std::uint8_t>(std::distance(codebook.begin(), found));
}

[[nodiscard]] std::vector<float> make_codebook(
    const std::vector<float>& values, const char* label) {
    if (values.empty()) throw std::runtime_error(std::string("Empty SOG ") + label);
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    if (!std::isfinite(*minimum) || !std::isfinite(*maximum))
        throw std::runtime_error(std::string("Non-finite SOG ") + label);
    std::vector<float> codebook(256);
    for (std::size_t i = 0; i < codebook.size(); ++i) {
        const float t = static_cast<float>(i) / 255.F;
        codebook[i] = *minimum + (*maximum - *minimum) * t;
    }
    return codebook;
}

[[nodiscard]] std::string json_float(const float value) {
    std::ostringstream output;
    output << std::setprecision(9) << value;
    return output.str();
}

void append_json_float_array(
    std::ostringstream& output, const std::vector<float>& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) output << ',';
        output << json_float(values[i]);
    }
    output << ']';
}

[[nodiscard]] GaussianModel make_model(
    const std::size_t count, const unsigned degree,
    const std::vector<float>& means, const std::vector<float>& scales,
    const std::vector<float>& rotations, const std::vector<float>& opacities,
    const std::vector<float>& sh) {
    if (count == 0 || degree > 3)
        throw std::runtime_error("Splat model has an unsupported point count or SH degree");
    const auto bases = static_cast<std::size_t>(degree + 1U) * (degree + 1U);
    if (means.size() != count * 3U || scales.size() != count * 3U ||
        rotations.size() != count * 4U || opacities.size() != count ||
        sh.size() != count * bases * 3U)
        throw std::runtime_error("Decoded splat model has inconsistent attribute sizes");
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(),
                           [](const float value) { return std::isfinite(value); });
    };
    if (!finite(means) || !finite(scales) || !finite(rotations) ||
        !finite(opacities) || !finite(sh))
        throw std::runtime_error("Decoded splat model contains non-finite values");

    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        means, {count, 3U}, tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        scales, {count, 3U}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        rotations, {count, 4U}, tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        opacities, {count, 1U}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(
        sh, {count, bases, 3U}, tinytensor::Device::CUDA);
    model.sh_degree = degree;
    // SOG/SPZ do not carry AetherScan's optional normal field. Seed it from the
    // thinnest Gaussian axis so a subsequently extracted surface remains usable.
    std::vector<float> normal_features(count * 4U, 0.F);
    for (std::size_t index = 0; index < count; ++index) {
        const auto scale = scales.begin() + static_cast<std::ptrdiff_t>(3U * index);
        const int axis = static_cast<int>(std::distance(
            scale, std::min_element(scale, scale + 3)));
        float w = rotations[4U * index];
        float x = rotations[4U * index + 1U];
        float y = rotations[4U * index + 2U];
        float z = rotations[4U * index + 3U];
        const float inverse = 1.F / std::max(
            std::sqrt(w * w + x * x + y * y + z * z), 1e-12F);
        w *= inverse;
        x *= inverse;
        y *= inverse;
        z *= inverse;
        const std::array<std::array<float, 3>, 3> columns{{
            {{1.F - 2.F * (y * y + z * z), 2.F * (x * y + w * z),
              2.F * (x * z - w * y)}},
            {{2.F * (x * y - w * z), 1.F - 2.F * (x * x + z * z),
              2.F * (y * z + w * x)}},
            {{2.F * (x * z + w * y), 2.F * (y * z - w * x),
              1.F - 2.F * (x * x + y * y)}}}};
        for (int component = 0; component < 3; ++component)
            normal_features[4U * index + static_cast<std::size_t>(component)] =
                columns[static_cast<std::size_t>(axis)]
                       [static_cast<std::size_t>(component)];
        normal_features[4U * index + 3U] = 1.F;
    }
    model.normal_features = tinytensor::Tensor::from_vector(
        normal_features, {count, 4U}, tinytensor::Device::CUDA);
    return model;
}

void validate_model_shape(
    const GaussianModel& model, std::size_t& count, std::size_t& bases) {
    count = model.size();
    if (count == 0 || !model.sh.is_valid() || model.sh.shape().rank() != 3)
        throw std::runtime_error("Cannot write an empty or invalid Gaussian model");
    bases = model.sh.shape()[1];
    const auto expected_bases = static_cast<std::size_t>(model.sh_degree + 1U) *
        (model.sh_degree + 1U);
    if (model.sh_degree > 3 || bases != expected_bases ||
        model.means.shape().rank() != 2 || model.means.shape()[1] != 3 ||
        model.log_scales.shape().rank() != 2 || model.log_scales.shape()[1] != 3 ||
        model.quaternions.shape().rank() != 2 || model.quaternions.shape()[1] != 4 ||
        model.opacity_logits.shape().rank() != 2 || model.opacity_logits.shape()[1] != 1)
        throw std::runtime_error("Gaussian model has unsupported tensor shapes");
}

[[nodiscard]] GaussianModel load_sog(const std::filesystem::path& path) {
    const SogAssets assets = read_sog_assets(path);
    std::istringstream json(assets.meta);
    ptree meta;
    boost::property_tree::read_json(json, meta);
    if (meta.get<int>("version", 0) != 2)
        throw std::runtime_error("Unsupported SOG metadata version: " + path.string());
    const auto count = static_cast<std::size_t>(meta.get<std::uint64_t>("count", 0));
    const auto& means_meta = meta.get_child("means");
    const auto& means_files = string_array(means_meta.get_child("files"));
    if (means_files.size() != 2)
        throw std::runtime_error("SOG means.files must contain two assets");
    const auto& scales_meta = meta.get_child("scales");
    const auto& scales_files = string_array(scales_meta.get_child("files"));
    const auto& quats_files = string_array(meta.get_child("quats.files"));
    const auto& sh0_meta = meta.get_child("sh0");
    const auto& sh0_files = string_array(sh0_meta.get_child("files"));
    if (scales_files.size() != 1 || quats_files.size() != 1 || sh0_files.size() != 1)
        throw std::runtime_error("SOG metadata has invalid per-Gaussian file lists");

    const WebpImage means_l = decode_webp(sog_file(assets, means_files[0]), means_files[0]);
    const WebpImage means_u = decode_webp(sog_file(assets, means_files[1]), means_files[1]);
    const WebpImage scales = decode_webp(sog_file(assets, scales_files[0]), scales_files[0]);
    const WebpImage quats = decode_webp(sog_file(assets, quats_files[0]), quats_files[0]);
    const WebpImage sh0 = decode_webp(sog_file(assets, sh0_files[0]), sh0_files[0]);
    const auto dimensions = sog_dimensions(means_l, means_u, scales, quats, sh0, count);
    const auto means_mins = float_array(means_meta.get_child("mins"), 3, "means.mins");
    const auto means_maxs = float_array(means_meta.get_child("maxs"), 3, "means.maxs");
    const auto scale_codebook = sog_codebook(scales_meta, "scales.codebook");
    const auto sh0_codebook = sog_codebook(sh0_meta, "sh0.codebook");

    unsigned degree = 0;
    std::vector<float> shn_codebook;
    WebpImage shn_centroids;
    WebpImage shn_labels;
    std::size_t palette_count = 0;
    std::size_t ac_count = 0;
    if (const auto shn = meta.get_child_optional("shN")) {
        palette_count = static_cast<std::size_t>(shn->get<std::uint64_t>("count", 0));
        degree = shn->get<unsigned>("bands", 0);
        if (degree == 0 || degree > 3 || palette_count == 0)
            throw std::runtime_error("SOG shN metadata has an unsupported degree");
        ac_count = static_cast<std::size_t>(degree + 1U) * (degree + 1U) - 1U;
        shn_codebook = sog_codebook(*shn, "shN.codebook");
        const auto shn_files = string_array(shn->get_child("files"));
        if (shn_files.size() != 2)
            throw std::runtime_error("SOG shN.files must contain two assets");
        shn_centroids = decode_webp(
            sog_file(assets, shn_files[0]), shn_files[0]);
        shn_labels = decode_webp(sog_file(assets, shn_files[1]), shn_files[1]);
        if (shn_labels.width != static_cast<int>(dimensions[0]) ||
            shn_labels.height != static_cast<int>(dimensions[1]) ||
            shn_centroids.width < static_cast<int>(64U * ac_count) ||
            shn_centroids.height < static_cast<int>((palette_count + 63U) / 64U))
            throw std::runtime_error("SOG shN image dimensions are invalid");
    }

    std::vector<float> means(count * 3U);
    std::vector<float> log_scales(count * 3U);
    std::vector<float> rotations(count * 4U);
    std::vector<float> opacities(count);
    const auto logit = [](const float alpha) {
        const float value = std::clamp(alpha, 1e-5F, 1.F - 1e-5F);
        return std::log(value / (1.F - value));
    };
    for (std::size_t index = 0; index < count; ++index) {
        const auto pixel = index * 4U;
        for (int axis = 0; axis < 3; ++axis) {
            const std::uint16_t quantized = static_cast<std::uint16_t>(
                means_u.rgba[pixel + static_cast<std::size_t>(axis)] << 8U |
                means_l.rgba[pixel + static_cast<std::size_t>(axis)]);
            const float normalized = static_cast<float>(quantized) / 65535.F;
            const float encoded = means_mins[static_cast<std::size_t>(axis)] +
                (means_maxs[static_cast<std::size_t>(axis)] -
                 means_mins[static_cast<std::size_t>(axis)]) * normalized;
            const float rub_position = unlog_position(encoded);
            means[3U * index + static_cast<std::size_t>(axis)] =
                axis == 0 ? rub_position : -rub_position;
            log_scales[3U * index + static_cast<std::size_t>(axis)] =
                scale_codebook[scales.rgba[pixel + static_cast<std::size_t>(axis)]];
        }
        // SOG stores quaternions in scalar-first (w,x,y,z) order.
        const float q0 = (static_cast<float>(quats.rgba[pixel + 0U]) / 255.F - 0.5F) *
            2.F / std::sqrt(2.F);
        const float q1 = (static_cast<float>(quats.rgba[pixel + 1U]) / 255.F - 0.5F) *
            2.F / std::sqrt(2.F);
        const float q2 = (static_cast<float>(quats.rgba[pixel + 2U]) / 255.F - 0.5F) *
            2.F / std::sqrt(2.F);
        const auto mode = quats.rgba[pixel + 3U];
        if (mode < 252U || mode > 255U)
            throw std::runtime_error("SOG contains an invalid quaternion mode");
        const float omitted = std::sqrt(std::max(0.F, 1.F - q0 * q0 - q1 * q1 - q2 * q2));
        std::array<float, 4> stored{};
        std::array<float, 3> kept{q0, q1, q2};
        std::size_t kept_index = 0;
        for (std::size_t component = 0; component < 4; ++component) {
            if (component == static_cast<std::size_t>(mode - 252U))
                stored[component] = omitted;
            else
                stored[component] = kept[kept_index++];
        }
        // SOG's coordinate system is RUB; AetherScan uses RDF.
        rotations[4U * index + 0U] = stored[0];
        rotations[4U * index + 1U] = stored[1];
        rotations[4U * index + 2U] = -stored[2];
        rotations[4U * index + 3U] = -stored[3];
        opacities[index] = logit(
            static_cast<float>(sh0.rgba[pixel + 3U]) / 255.F);
    }

    const auto bases = static_cast<std::size_t>(degree + 1U) * (degree + 1U);
    std::vector<float> sh(count * bases * 3U, 0.F);
    for (std::size_t index = 0; index < count; ++index) {
        const auto pixel = index * 4U;
        for (int channel = 0; channel < 3; ++channel)
            sh[(index * bases) * 3U + static_cast<std::size_t>(channel)] =
                sh0_codebook[sh0.rgba[pixel + static_cast<std::size_t>(channel)]];
        for (std::size_t coefficient = 0; coefficient < ac_count; ++coefficient) {
            const std::size_t palette_pixel =
                (static_cast<std::size_t>(shn_labels.rgba[pixel]) |
                 (static_cast<std::size_t>(shn_labels.rgba[pixel + 1U]) << 8U));
            if (palette_pixel >= palette_count)
                throw std::runtime_error("SOG shN label exceeds its palette");
            const auto x = (palette_pixel % 64U) * ac_count + coefficient;
            const auto y = palette_pixel / 64U;
            const auto centroid_pixel =
                (y * static_cast<std::size_t>(shn_centroids.width) + x) * 4U;
            for (int channel = 0; channel < 3; ++channel) {
                const auto target = (index * bases + coefficient + 1U) * 3U +
                    static_cast<std::size_t>(channel);
                sh[target] = shn_codebook[
                    shn_centroids.rgba[centroid_pixel + static_cast<std::size_t>(channel)]] *
                    k_rdf_to_rub_sh_signs[coefficient];
            }
        }
    }
    return make_model(count, degree, means, log_scales, rotations, opacities, sh);
}

[[nodiscard]] std::uint8_t spz_quantize_sh(
    const float value, const int bucket_size) {
    int quantized = static_cast<int>(std::lround(value * 128.F + 128.F));
    quantized = ((quantized + bucket_size / 2) / bucket_size) * bucket_size;
    return static_cast<std::uint8_t>(std::clamp(quantized, 0, 255));
}

[[nodiscard]] float spz_unquantize_sh(const std::uint8_t value) {
    return (static_cast<float>(value) - 128.F) / 128.F;
}

[[nodiscard]] std::uint16_t float_to_half(const float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    const std::uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU)
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa != 0));
    const int adjusted = static_cast<int>(exponent) - 127 + 15;
    if (adjusted >= 31) return static_cast<std::uint16_t>(sign | 0x7c00U);
    if (adjusted <= 0) {
        if (adjusted < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t shifted = (mantissa | 0x800000U) >> (1 - adjusted);
        return static_cast<std::uint16_t>(sign | ((shifted + 0x1000U) >> 13U));
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(adjusted) << 10U) |
        ((mantissa + 0x1000U) >> 13U));
}

[[nodiscard]] float half_to_float(const std::uint16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16U;
    const std::uint32_t exponent = (value >> 10U) & 0x1fU;
    std::uint32_t mantissa = value & 0x3ffU;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x3ffU;
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - shift) << 23U) |
                (mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result = 0.F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void pack_spz_quaternion(
    std::uint8_t output[4], const float* current_rotation) {
    // SPZ's GaussianCloud uses xyzw; convert AetherScan's wxyz and RDF axes.
    std::array<float, 4> q{
        current_rotation[1], -current_rotation[2],
        -current_rotation[3], current_rotation[0]};
    float norm = 0.F;
    for (const float value : q) norm += value * value;
    norm = 1.F / std::max(std::sqrt(norm), 1e-12F);
    for (float& value : q) value *= norm;
    std::size_t largest = 0;
    for (std::size_t i = 1; i < 4; ++i)
        if (std::abs(q[i]) > std::abs(q[largest])) largest = i;
    if (q[largest] < 0.F)
        for (float& value : q) value = -value;
    std::uint32_t packed = static_cast<std::uint32_t>(largest);
    for (std::size_t i = 0; i < 4; ++i) {
        if (i == largest) continue;
        const std::uint32_t negative = q[i] < 0.F ? 1U : 0U;
        const auto magnitude = static_cast<std::uint32_t>(std::lround(
            511.F * std::abs(q[i]) / k_spz_sqrt_half));
        packed = (packed << 10U) | (negative << 9U) |
            std::min<std::uint32_t>(magnitude, 511U);
    }
    for (unsigned byte = 0; byte < 4; ++byte)
        output[byte] = static_cast<std::uint8_t>(packed >> (8U * byte));
}

[[nodiscard]] std::array<float, 4> unpack_spz_quaternion(const std::uint8_t input[4]) {
    std::uint32_t packed = static_cast<std::uint32_t>(input[0]) |
        (static_cast<std::uint32_t>(input[1]) << 8U) |
        (static_cast<std::uint32_t>(input[2]) << 16U) |
        (static_cast<std::uint32_t>(input[3]) << 24U);
    const std::size_t largest = packed >> 30U;
    std::array<float, 4> q{};
    float sum = 0.F;
    for (int i = 3; i >= 0; --i) {
        if (static_cast<std::size_t>(i) == largest) continue;
        const auto magnitude = packed & 511U;
        const bool negative = ((packed >> 9U) & 1U) != 0;
        q[static_cast<std::size_t>(i)] = k_spz_sqrt_half *
            static_cast<float>(magnitude) / 511.F;
        if (negative) q[static_cast<std::size_t>(i)] = -q[static_cast<std::size_t>(i)];
        sum += q[static_cast<std::size_t>(i)] * q[static_cast<std::size_t>(i)];
        packed >>= 10U;
    }
    q[largest] = std::sqrt(std::max(0.F, 1.F - sum));
    // Convert SPZ xyzw/RUB back to AetherScan wxyz/RDF.
    return {q[3], q[0], -q[1], -q[2]};
}

void append_bytes(Bytes& target, const std::uint8_t* source, const std::size_t count) {
    target.insert(target.end(), source, source + count);
}

[[nodiscard]] Bytes zstd_compress(const Bytes& source) {
    Bytes compressed(ZSTD_compressBound(source.size()));
    const auto size = ZSTD_compress(
        compressed.data(), compressed.size(), source.data(), source.size(), 12);
    if (ZSTD_isError(size))
        throw std::runtime_error(
            std::string("SPZ Zstd compression failed: ") + ZSTD_getErrorName(size));
    compressed.resize(size);
    return compressed;
}

[[nodiscard]] Bytes zstd_decompress(
    const std::uint8_t* source, const std::size_t source_size,
    const std::size_t destination_size) {
    Bytes destination(destination_size);
    const auto size = ZSTD_decompress(
        destination.data(), destination.size(), source, source_size);
    if (ZSTD_isError(size) || size != destination_size)
        throw std::runtime_error("Invalid SPZ Zstd attribute stream");
    return destination;
}

[[nodiscard]] Bytes gzip_decompress(const Bytes& compressed) {
    z_stream stream{};
    stream.next_in = const_cast<Bytef*>(compressed.data());
    stream.avail_in = static_cast<uInt>(compressed.size());
    if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK)
        throw std::runtime_error("Failed to initialize SPZ gzip decompressor");
    Bytes output;
    std::array<std::uint8_t, 64U * 1024U> buffer{};
    int status = Z_OK;
    while (status == Z_OK) {
        stream.next_out = buffer.data();
        stream.avail_out = static_cast<uInt>(buffer.size());
        status = inflate(&stream, Z_NO_FLUSH);
        const auto produced = buffer.size() - stream.avail_out;
        output.insert(output.end(), buffer.begin(),
                      buffer.begin() + static_cast<std::ptrdiff_t>(produced));
    }
    inflateEnd(&stream);
    if (status != Z_STREAM_END)
        throw std::runtime_error("Invalid SPZ gzip stream");
    return output;
}

[[nodiscard]] Bytes gzip_compress(const Bytes& source) {
    z_stream stream{};
    if (deflateInit2(
            &stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 16 + MAX_WBITS,
            9, Z_DEFAULT_STRATEGY) != Z_OK)
        throw std::runtime_error("Failed to initialize SPZ gzip compressor");
    Bytes output;
    std::array<std::uint8_t, 64U * 1024U> buffer{};
    stream.next_in = const_cast<Bytef*>(source.data());
    stream.avail_in = static_cast<uInt>(source.size());
    int status = Z_OK;
    while (status != Z_STREAM_END) {
        stream.next_out = buffer.data();
        stream.avail_out = static_cast<uInt>(buffer.size());
        status = deflate(&stream, Z_FINISH);
        const auto produced = buffer.size() - stream.avail_out;
        output.insert(output.end(), buffer.begin(),
                      buffer.begin() + static_cast<std::ptrdiff_t>(produced));
        if (status != Z_OK && status != Z_STREAM_END) break;
    }
    deflateEnd(&stream);
    if (status != Z_STREAM_END)
        throw std::runtime_error("SPZ gzip compression failed");
    return output;
}

[[nodiscard]] Bytes save_spz_v4(const GaussianModel& model) {
    std::size_t count = 0;
    std::size_t bases = 0;
    validate_model_shape(model, count, bases);
    if (count > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("SPZ point count exceeds uint32 range");
    const auto means = model.means.to_vector();
    const auto scales = model.log_scales.to_vector();
    const auto rotations = model.quaternions.to_vector();
    const auto opacities = model.opacity_logits.to_vector();
    const auto sh = model.sh.to_vector();
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(),
                           [](const float value) { return std::isfinite(value); });
    };
    if (!finite(means) || !finite(scales) || !finite(rotations) ||
        !finite(opacities) || !finite(sh))
        throw std::runtime_error("Refusing to write non-finite SPZ values");

    Bytes positions;
    positions.reserve(count * 9U);
    for (std::size_t index = 0; index < count; ++index) {
        const std::array<float, 3> position{
            means[3U * index], -means[3U * index + 1U],
            -means[3U * index + 2U]};
        for (const float value : position) {
            const auto fixed = static_cast<std::int64_t>(std::llround(value * 4096.F));
            if (fixed < -0x800000LL || fixed > 0x7fffffLL)
                throw std::runtime_error("SPZ position exceeds 24-bit fixed-point range");
            const auto encoded = static_cast<std::uint32_t>(
                static_cast<std::int32_t>(fixed));
            positions.push_back(static_cast<std::uint8_t>(encoded & 0xffU));
            positions.push_back(static_cast<std::uint8_t>((encoded >> 8U) & 0xffU));
            positions.push_back(static_cast<std::uint8_t>((encoded >> 16U) & 0xffU));
        }
    }
    Bytes alphas;
    Bytes colors;
    Bytes packed_scales;
    alphas.reserve(count);
    colors.reserve(count * 3U);
    packed_scales.reserve(count * 3U);
    for (std::size_t index = 0; index < count; ++index) {
        const float alpha = 1.F / (1.F + std::exp(-opacities[index]));
        alphas.push_back(static_cast<std::uint8_t>(std::clamp(
            std::lround(alpha * 255.F), 0L, 255L)));
        for (int channel = 0; channel < 3; ++channel)
            colors.push_back(static_cast<std::uint8_t>(std::clamp(
                std::lround(sh[(index * bases) * 3U + static_cast<std::size_t>(channel)] *
                                k_spz_color_scale * 255.F + 127.5F),
                0L, 255L)));
        for (int axis = 0; axis < 3; ++axis)
            packed_scales.push_back(static_cast<std::uint8_t>(std::clamp(
                std::lround((scales[3U * index + static_cast<std::size_t>(axis)] +
                             10.F) * 16.F),
                0L, 255L)));
    }
    Bytes packed_rotations(count * 4U);
    for (std::size_t index = 0; index < count; ++index)
        pack_spz_quaternion(
            packed_rotations.data() + 4U * index,
            rotations.data() + 4U * index);
    Bytes packed_sh;
    const std::size_t ac_count = bases - 1U;
    packed_sh.reserve(count * ac_count * 3U);
    for (std::size_t index = 0; index < count; ++index)
        for (std::size_t coefficient = 0; coefficient < ac_count; ++coefficient)
            for (int channel = 0; channel < 3; ++channel) {
                const auto source = (index * bases + coefficient + 1U) * 3U +
                    static_cast<std::size_t>(channel);
                const float converted = sh[source] * k_rdf_to_rub_sh_signs[coefficient];
                const int bits = coefficient < 3U ? 5 : 4;
                packed_sh.push_back(spz_quantize_sh(converted, 1 << (8 - bits)));
            }

    const std::array<const Bytes*, 6> streams{
        &positions, &alphas, &colors, &packed_scales, &packed_rotations, &packed_sh};
    std::vector<Bytes> compressed;
    compressed.reserve(streams.size());
    for (const Bytes* stream : streams) compressed.push_back(zstd_compress(*stream));
    Bytes output;
    output.reserve(32U + streams.size() * 16U);
    // NgspFileHeader, version 4, no extensions, six streams.
    append_u32(output, 0x5053474eU);
    append_u32(output, 4U);
    append_u32(output, static_cast<std::uint32_t>(count));
    output.push_back(static_cast<std::uint8_t>(model.sh_degree));
    output.push_back(12U);
    output.push_back(0U);
    output.push_back(static_cast<std::uint8_t>(streams.size()));
    append_u32(output, 32U);
    output.insert(output.end(), 12U, 0U);
    for (std::size_t index = 0; index < compressed.size(); ++index) {
        append_u64(output, compressed[index].size());
        append_u64(output, streams[index]->size());
    }
    for (const Bytes& stream : compressed)
        output.insert(output.end(), stream.begin(), stream.end());
    return output;
}

[[nodiscard]] GaussianModel decode_spz_packed(
    const std::uint32_t version, const std::uint32_t count,
    const unsigned degree, const unsigned fractional_bits,
    const Bytes& positions, const Bytes& alphas, const Bytes& colors,
    const Bytes& scales, const Bytes& rotations, const Bytes& sh) {
    if (count == 0 || degree > 3 || fractional_bits > 23)
        throw std::runtime_error("Unsupported SPZ point count, degree, or precision");
    const auto bases = static_cast<std::size_t>(degree + 1U) * (degree + 1U);
    const auto ac_count = bases - 1U;
    const bool uses_float16 = version == 1;
    const auto expected_positions = count * 3U * (uses_float16 ? 2U : 3U);
    const auto expected_rotations = count * (version >= 3 ? 4U : 3U);
    if (positions.size() != expected_positions || alphas.size() != count ||
        colors.size() != count * 3U || scales.size() != count * 3U ||
        rotations.size() != expected_rotations ||
        sh.size() != count * ac_count * 3U)
        throw std::runtime_error("SPZ attribute stream sizes do not match its header");
    std::vector<float> means(count * 3U);
    std::vector<float> log_scales(count * 3U);
    std::vector<float> model_rotations(count * 4U);
    std::vector<float> opacities(count);
    std::vector<float> model_sh(count * bases * 3U, 0.F);
    std::size_t position_cursor = 0;
    for (std::size_t index = 0; index < count; ++index) {
        for (int axis = 0; axis < 3; ++axis) {
            float value = 0.F;
            if (uses_float16) {
                value = half_to_float(static_cast<std::uint16_t>(
                    positions[position_cursor] |
                    (static_cast<std::uint16_t>(positions[position_cursor + 1U]) << 8U)));
                position_cursor += 2U;
            } else {
                std::int32_t fixed = positions[position_cursor] |
                    (static_cast<std::int32_t>(positions[position_cursor + 1U]) << 8U) |
                    (static_cast<std::int32_t>(positions[position_cursor + 2U]) << 16U);
                if ((fixed & 0x800000) != 0) fixed |= ~0xffffff;
                value = static_cast<float>(fixed) /
                    static_cast<float>(std::uint32_t{1} << fractional_bits);
                position_cursor += 3U;
            }
            const std::size_t target = 3U * index + static_cast<std::size_t>(axis);
            means[target] = axis == 0 ? value : -value;
        }
        opacities[index] = std::log(
            std::clamp(static_cast<float>(alphas[index]) / 255.F, 1e-5F, 1.F - 1e-5F) /
            (1.F - std::clamp(static_cast<float>(alphas[index]) / 255.F,
                              1e-5F, 1.F - 1e-5F)));
        for (int axis = 0; axis < 3; ++axis)
            log_scales[3U * index + static_cast<std::size_t>(axis)] =
                static_cast<float>(scales[3U * index + static_cast<std::size_t>(axis)]) /
                    16.F - 10.F;
        const auto quaternion = version >= 3
            ? unpack_spz_quaternion(rotations.data() + 4U * index)
            : std::array<float, 4>{};
        if (version < 3) {
            std::array<float, 3> xyz{};
            for (int component = 0; component < 3; ++component)
                xyz[static_cast<std::size_t>(component)] =
                    static_cast<float>(rotations[3U * index + static_cast<std::size_t>(component)]) /
                        127.5F - 1.F;
            const float w = std::sqrt(std::max(
                0.F, 1.F - xyz[0] * xyz[0] - xyz[1] * xyz[1] - xyz[2] * xyz[2]));
            // Legacy SPZ stores xyz in its xyzw/RUB convention.
            model_rotations[4U * index] = w;
            model_rotations[4U * index + 1U] = xyz[0];
            model_rotations[4U * index + 2U] = -xyz[1];
            model_rotations[4U * index + 3U] = -xyz[2];
        } else {
            std::copy(quaternion.begin(), quaternion.end(),
                      model_rotations.begin() + static_cast<std::ptrdiff_t>(4U * index));
        }
        model_sh[(index * bases) * 3U + 0U] =
            (static_cast<float>(colors[3U * index]) / 255.F - 0.5F) /
            k_spz_color_scale;
        model_sh[(index * bases) * 3U + 1U] =
            (static_cast<float>(colors[3U * index + 1U]) / 255.F - 0.5F) /
            k_spz_color_scale;
        model_sh[(index * bases) * 3U + 2U] =
            (static_cast<float>(colors[3U * index + 2U]) / 255.F - 0.5F) /
            k_spz_color_scale;
        for (std::size_t coefficient = 0; coefficient < ac_count; ++coefficient)
            for (int channel = 0; channel < 3; ++channel) {
                const auto source = (index * ac_count + coefficient) * 3U +
                    static_cast<std::size_t>(channel);
                const auto target = (index * bases + coefficient + 1U) * 3U +
                    static_cast<std::size_t>(channel);
                model_sh[target] = spz_unquantize_sh(sh[source]) *
                    k_rdf_to_rub_sh_signs[coefficient];
            }
    }
    return make_model(count, degree, means, log_scales, model_rotations, opacities, model_sh);
}

[[nodiscard]] GaussianModel load_spz(const std::filesystem::path& path) {
    const Bytes file = read_file(path);
    if (file.size() < 2) throw std::runtime_error("SPZ file is truncated");
    const std::uint32_t magic = file.size() >= 4 ? read_u32(file, 0) : 0;
    if (magic == 0x5053474eU) {
        if (file.size() < 32) throw std::runtime_error("SPZ v4 header is truncated");
        const auto version = read_u32(file, 4);
        if (version < 4 || version > 4)
            throw std::runtime_error("Unsupported SPZ NGSP version");
        const auto count = read_u32(file, 8);
        const auto degree = static_cast<unsigned>(file[12]);
        const auto fractional_bits = static_cast<unsigned>(file[13]);
        const auto flags = static_cast<unsigned>(file[14]);
        const auto stream_count = static_cast<unsigned>(file[15]);
        if (degree > 3)
            throw std::runtime_error("SPZ SH degree exceeds AetherScan support");
        if ((flags & 0x2U) != 0)
            throw std::runtime_error(
                "SPZ vendor extensions are not supported by this loader");
        const auto toc_offset = checked_size(read_u32(file, 16), "SPZ TOC");
        if (stream_count != 6 || toc_offset < 32 ||
            toc_offset > file.size() ||
            static_cast<std::size_t>(stream_count) >
                (file.size() - toc_offset) / 16U)
            throw std::runtime_error("Invalid SPZ v4 stream table");
        const auto bases = static_cast<std::size_t>(degree + 1U) * (degree + 1U);
        const std::array<std::size_t, 6> expected{
            static_cast<std::size_t>(count) * 9U,
            static_cast<std::size_t>(count),
            static_cast<std::size_t>(count) * 3U,
            static_cast<std::size_t>(count) * 3U,
            static_cast<std::size_t>(count) * 4U,
            static_cast<std::size_t>(count) * (bases - 1U) * 3U};
        std::array<Bytes, 6> streams;
        std::size_t compressed_offset = toc_offset + 6U * 16U;
        for (std::size_t index = 0; index < streams.size(); ++index) {
            const auto compressed_size = checked_size(
                read_u64(file, toc_offset + index * 16U), "SPZ stream");
            const auto uncompressed_size = checked_size(
                read_u64(file, toc_offset + index * 16U + 8U), "SPZ stream");
            if (uncompressed_size != expected[index] ||
                compressed_offset > file.size() ||
                compressed_size > file.size() - compressed_offset)
                throw std::runtime_error("Invalid SPZ v4 stream size");
            streams[index] = zstd_decompress(
                file.data() + compressed_offset, compressed_size, uncompressed_size);
            compressed_offset += compressed_size;
        }
        if (compressed_offset != file.size())
            throw std::runtime_error("SPZ v4 has trailing bytes");
        return decode_spz_packed(
            version, count, degree, fractional_bits,
            streams[0], streams[1], streams[2], streams[3], streams[4], streams[5]);
    }
    if (file[0] != 0x1fU || file[1] != 0x8bU)
        throw std::runtime_error("Unrecognized SPZ file");
    const Bytes unpacked = gzip_decompress(file);
    if (unpacked.size() < 16 || read_u32(unpacked, 0) != 0x5053474eU)
        throw std::runtime_error("SPZ legacy header is invalid");
    const auto version = read_u32(unpacked, 4);
    if (version < 1 || version > 3)
        throw std::runtime_error("Unsupported legacy SPZ version");
    const auto count = read_u32(unpacked, 8);
    const auto degree = static_cast<unsigned>(unpacked[12]);
    const auto fractional_bits = static_cast<unsigned>(unpacked[13]);
    const auto flags = static_cast<unsigned>(unpacked[14]);
    if (degree > 3)
        throw std::runtime_error("SPZ SH degree exceeds AetherScan support");
    if ((flags & 0x2U) != 0)
        throw std::runtime_error(
            "SPZ vendor extensions are not supported by this loader");
    const auto bases = static_cast<std::size_t>(degree + 1U) * (degree + 1U);
    const auto positions_size = static_cast<std::size_t>(count) * 3U *
        (version == 1 ? 2U : 3U);
    const auto rotations_size = static_cast<std::size_t>(count) *
        (version >= 3 ? 4U : 3U);
    const std::array<std::size_t, 6> sizes{
        positions_size, count, static_cast<std::size_t>(count) * 3U,
        static_cast<std::size_t>(count) * 3U, rotations_size,
        static_cast<std::size_t>(count) * (bases - 1U) * 3U};
    std::array<Bytes, 6> streams;
    std::size_t offset = 16;
    for (std::size_t index = 0; index < streams.size(); ++index) {
        if (offset > unpacked.size() || sizes[index] > unpacked.size() - offset)
            throw std::runtime_error("SPZ legacy stream is truncated");
        streams[index].assign(
            unpacked.begin() + static_cast<std::ptrdiff_t>(offset),
            unpacked.begin() + static_cast<std::ptrdiff_t>(offset + sizes[index]));
        offset += sizes[index];
    }
    return decode_spz_packed(
        version, count, degree, fractional_bits,
        streams[0], streams[1], streams[2], streams[3], streams[4], streams[5]);
}

[[nodiscard]] Bytes save_sog(const GaussianModel& model) {
    std::size_t count = 0;
    std::size_t bases = 0;
    validate_model_shape(model, count, bases);
    const auto means = model.means.to_vector();
    const auto scales = model.log_scales.to_vector();
    const auto rotations = model.quaternions.to_vector();
    const auto opacities = model.opacity_logits.to_vector();
    const auto sh = model.sh.to_vector();
    const auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(),
                           [](const float value) { return std::isfinite(value); });
    };
    if (!finite(means) || !finite(scales) || !finite(rotations) ||
        !finite(opacities) || !finite(sh))
        throw std::runtime_error("Refusing to write non-finite SOG values");

    const std::size_t width = std::max<std::size_t>(1, static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<double>(count)))));
    const std::size_t height = (count + width - 1U) / width;
    const auto pixels = checked_product(width, height, "SOG image dimensions");
    if (width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("SOG image dimensions exceed WebP limits");
    const auto to_log = [](const float value) {
        return std::copysign(std::log1p(std::abs(value)), value);
    };
    std::vector<float> encoded_positions(count * 3U);
    std::vector<float> position_mins(3, std::numeric_limits<float>::max());
    std::vector<float> position_maxs(3, std::numeric_limits<float>::lowest());
    for (std::size_t index = 0; index < count; ++index) {
        const std::array<float, 3> position{
            means[3U * index], -means[3U * index + 1U],
            -means[3U * index + 2U]};
        for (int axis = 0; axis < 3; ++axis) {
            const float value = to_log(position[static_cast<std::size_t>(axis)]);
            encoded_positions[3U * index + static_cast<std::size_t>(axis)] = value;
            position_mins[static_cast<std::size_t>(axis)] = std::min(
                position_mins[static_cast<std::size_t>(axis)], value);
            position_maxs[static_cast<std::size_t>(axis)] = std::max(
                position_maxs[static_cast<std::size_t>(axis)], value);
        }
    }
    const auto scale_codebook = make_codebook(scales, "scale codebook");
    std::vector<float> dc_values(count * 3U);
    for (std::size_t index = 0; index < count; ++index)
        for (int channel = 0; channel < 3; ++channel)
            dc_values[3U * index + static_cast<std::size_t>(channel)] =
                sh[(index * bases) * 3U + static_cast<std::size_t>(channel)];
    const auto sh0_codebook = make_codebook(dc_values, "sh0 codebook");

    // WebP's RGBA encoder premultiplies RGB by alpha. Position, scale, and
    // label images use alpha only as padding, so keep those pixels opaque.
    Bytes means_l(pixels * 4U, 255), means_u(pixels * 4U, 255),
        scale_image(pixels * 4U, 255), quat_image(pixels * 4U, 0),
        sh0_image(pixels * 4U, 0);
    for (std::size_t index = 0; index < count; ++index) {
        const auto pixel = index * 4U;
        for (int axis = 0; axis < 3; ++axis) {
            const auto a = static_cast<std::size_t>(axis);
            const float range = position_maxs[a] - position_mins[a];
            const auto quantized = static_cast<std::uint16_t>(std::clamp(
                std::lround(range > 0.F
                    ? (encoded_positions[3U * index + a] - position_mins[a]) /
                          range * 65535.F
                    : 0.F),
                0L, 65535L));
            means_l[pixel + a] = static_cast<std::uint8_t>(quantized & 0xffU);
            means_u[pixel + a] = static_cast<std::uint8_t>(quantized >> 8U);
            scale_image[pixel + a] = nearest_codebook(
                scale_codebook, scales[3U * index + a]);
        }
        std::array<float, 4> q{
            rotations[4U * index], rotations[4U * index + 1U],
            -rotations[4U * index + 2U], -rotations[4U * index + 3U]};
        float norm = 0.F;
        for (const float value : q) norm += value * value;
        norm = 1.F / std::max(std::sqrt(norm), 1e-12F);
        for (float& value : q) value *= norm;
        std::size_t largest = 0;
        for (std::size_t component = 1; component < 4; ++component)
            if (std::abs(q[component]) > std::abs(q[largest])) largest = component;
        if (q[largest] < 0.F)
            for (float& value : q) value = -value;
        std::size_t kept = 0;
        for (std::size_t component = 0; component < 4; ++component) {
            if (component == largest) continue;
            quat_image[pixel + kept++] = static_cast<std::uint8_t>(std::clamp(
                std::lround((q[component] / 2.F / std::sqrt(0.5F) + 0.5F) * 255.F),
                0L, 255L));
        }
        quat_image[pixel + 3U] = static_cast<std::uint8_t>(252U + largest);
        for (int channel = 0; channel < 3; ++channel)
            sh0_image[pixel + static_cast<std::size_t>(channel)] = nearest_codebook(
                sh0_codebook, dc_values[3U * index + static_cast<std::size_t>(channel)]);
        const float alpha = 1.F / (1.F + std::exp(-opacities[index]));
        sh0_image[pixel + 3U] = static_cast<std::uint8_t>(std::clamp(
            std::lround(alpha * 255.F), 0L, 255L));
    }

    const std::size_t ac_count = bases - 1U;
    std::vector<float> encoded_ac;
    std::vector<float> shn_codebook;
    Bytes shn_centroid_image;
    Bytes shn_label_image;
    std::size_t palette_count = 0;
    if (ac_count != 0) {
        encoded_ac.resize(count * ac_count * 3U);
        for (std::size_t index = 0; index < count; ++index)
            for (std::size_t coefficient = 0; coefficient < ac_count; ++coefficient)
                for (int channel = 0; channel < 3; ++channel) {
                    const auto source = (index * bases + coefficient + 1U) * 3U +
                        static_cast<std::size_t>(channel);
                    encoded_ac[(index * ac_count + coefficient) * 3U +
                               static_cast<std::size_t>(channel)] =
                        sh[source] * k_rdf_to_rub_sh_signs[coefficient];
                }
        shn_codebook = make_codebook(encoded_ac, "shN codebook");
        palette_count = std::min<std::size_t>(count, 65536U);
        const std::size_t palette_height = (palette_count + 63U) / 64U;
        shn_centroid_image.assign(
            checked_product(64U * ac_count, palette_height, "SOG shN palette") * 3U, 0);
        shn_label_image.assign(pixels * 4U, 0);
        for (std::size_t index = 0; index < count; ++index) {
            const auto label = std::min<std::size_t>(
                palette_count - 1U, index * palette_count / count);
            shn_label_image[index * 4U] = static_cast<std::uint8_t>(label & 0xffU);
            shn_label_image[index * 4U + 1U] = static_cast<std::uint8_t>(label >> 8U);
            shn_label_image[index * 4U + 3U] = 255U;
        }
        for (std::size_t label = 0; label < palette_count; ++label) {
            const auto source_index = std::min<std::size_t>(
                count - 1U, label * count / palette_count);
            for (std::size_t coefficient = 0; coefficient < ac_count; ++coefficient) {
                const auto target =
                    ((label / 64U) * (64U * ac_count) +
                     (label % 64U) * ac_count + coefficient) * 3U;
                for (int channel = 0; channel < 3; ++channel)
                    shn_centroid_image[target + static_cast<std::size_t>(channel)] =
                        nearest_codebook(shn_codebook, encoded_ac[
                            (source_index * ac_count + coefficient) * 3U +
                            static_cast<std::size_t>(channel)]);
            }
        }
    }

    const std::string means_l_name = "means_l.webp";
    const std::string means_u_name = "means_u.webp";
    const std::string scales_name = "scales.webp";
    const std::string quats_name = "quats.webp";
    const std::string sh0_name = "sh0.webp";
    const auto means_l_webp = encode_webp(
        means_l.data(), static_cast<int>(width), static_cast<int>(height), 4, means_l_name);
    const auto means_u_webp = encode_webp(
        means_u.data(), static_cast<int>(width), static_cast<int>(height), 4, means_u_name);
    const auto scales_webp = encode_webp(
        scale_image.data(), static_cast<int>(width), static_cast<int>(height), 4, scales_name);
    const auto quats_webp = encode_webp(
        quat_image.data(), static_cast<int>(width), static_cast<int>(height), 4, quats_name);
    const auto sh0_webp = encode_webp(
        sh0_image.data(), static_cast<int>(width), static_cast<int>(height), 4, sh0_name);
    std::ostringstream meta;
    meta << "{\"version\":2,\"asset\":{\"generator\":\"AetherScan\"},"
         << "\"count\":" << count << ",\"antialias\":false,\"means\":{\"mins\":["
         << json_float(position_mins[0]) << ',' << json_float(position_mins[1]) << ','
         << json_float(position_mins[2]) << "],\"maxs\":["
         << json_float(position_maxs[0]) << ',' << json_float(position_maxs[1]) << ','
         << json_float(position_maxs[2]) << "],\"files\":[\"" << means_l_name
         << "\",\"" << means_u_name << "\"]},\"scales\":{\"codebook\":";
    append_json_float_array(meta, scale_codebook);
    meta << ",\"files\":[\"" << scales_name << "\"]},\"quats\":{\"files\":[\""
         << quats_name << "\"]},\"sh0\":{\"codebook\":";
    append_json_float_array(meta, sh0_codebook);
    meta << ",\"files\":[\"" << sh0_name << "\"]}";
    if (ac_count != 0) {
        const std::string centroid_name = "shN_centroids.webp";
        const std::string label_name = "shN_labels.webp";
        meta << ",\"shN\":{\"count\":" << palette_count
             << ",\"bands\":" << model.sh_degree << ",\"codebook\":";
        append_json_float_array(meta, shn_codebook);
        meta << ",\"files\":[\"" << centroid_name << "\",\"" << label_name
             << "\"]}";
        const auto centroid_webp = encode_webp(
            shn_centroid_image.data(), static_cast<int>(64U * ac_count),
            static_cast<int>((palette_count + 63U) / 64U), 3, centroid_name);
        const auto label_webp = encode_webp(
            shn_label_image.data(), static_cast<int>(width), static_cast<int>(height),
            4, label_name);
        meta << '}';
        const std::string meta_text = meta.str();
        const Bytes meta_bytes(meta_text.begin(), meta_text.end());
        return make_sog_zip({
            {"meta.json", meta_bytes},
            {means_l_name, means_l_webp}, {means_u_name, means_u_webp},
            {scales_name, scales_webp}, {quats_name, quats_webp},
            {sh0_name, sh0_webp}, {centroid_name, centroid_webp},
            {label_name, label_webp}});
    }
    meta << '}';
    const std::string meta_text = meta.str();
    const Bytes meta_bytes(meta_text.begin(), meta_text.end());
    return make_sog_zip({
        {"meta.json", meta_bytes}, {means_l_name, means_l_webp},
        {means_u_name, means_u_webp}, {scales_name, scales_webp},
        {quats_name, quats_webp}, {sh0_name, sh0_webp}});
}

}  // namespace

GaussianFormat parse_gaussian_format(const std::string_view value) {
    const std::string normalized = lower(std::string(value));
    if (normalized.empty() || normalized == "auto") return GaussianFormat::auto_detect;
    if (normalized == "ply") return GaussianFormat::ply;
    if (normalized == "sog") return GaussianFormat::sog;
    if (normalized == "spz") return GaussianFormat::spz;
    if (normalized == "glb") return GaussianFormat::glb;
    throw std::invalid_argument(
        "Unknown splat output format '" + std::string(value) +
        "' (expected auto, ply, sog, spz, or glb)");
}

GaussianFormat gaussian_format_from_path(const std::filesystem::path& path) noexcept {
    const auto extension = lower(path.extension().string());
    if (extension == ".sog") return GaussianFormat::sog;
    if (extension == ".spz") return GaussianFormat::spz;
    if (extension == ".glb") return GaussianFormat::glb;
    return GaussianFormat::ply;
}

const char* gaussian_format_name(const GaussianFormat format) noexcept {
    switch (format) {
        case GaussianFormat::auto_detect: return "auto";
        case GaussianFormat::ply: return "ply";
        case GaussianFormat::sog: return "sog";
        case GaussianFormat::spz: return "spz";
        case GaussianFormat::glb: return "glb";
    }
    return "auto";
}

const char* gaussian_format_extension(const GaussianFormat format) noexcept {
    switch (format) {
        case GaussianFormat::auto_detect: return "";
        case GaussianFormat::ply: return "ply";
        case GaussianFormat::sog: return "sog";
        case GaussianFormat::spz: return "spz";
        case GaussianFormat::glb: return "glb";
    }
    return "";
}

void restrict_sh_degree(GaussianModel& model, const unsigned degree) {
    if (degree > 3U)
        throw std::runtime_error("SH degree must be between 0 and 3");
    if (!model.sh.is_valid() || model.sh.shape().rank() != 3)
        throw std::runtime_error("Gaussian model has no spherical harmonics");
    if (degree >= model.sh_degree) return;
    const auto bases = static_cast<std::size_t>(degree + 1U) * (degree + 1U);
    model.sh = model.sh.slice(1, 0, bases);
    model.sh_degree = degree;
    if (!model.sh.is_valid() || model.sh.shape()[1] != bases)
        throw std::runtime_error("Failed to truncate spherical harmonics");
}

void save_gaussians(
    const GaussianModel& model, const std::filesystem::path& path,
    GaussianFormat format) {
    if (format == GaussianFormat::auto_detect)
        format = gaussian_format_from_path(path);
    switch (format) {
        case GaussianFormat::ply:
            save_gaussians_ply(model, path);
            return;
        case GaussianFormat::sog:
            write_file(path, save_sog(model));
            return;
        case GaussianFormat::spz:
            write_file(path, save_spz_v4(model));
            return;
        case GaussianFormat::glb:
            write_file(path, detail::save_glb(model));
            return;
        case GaussianFormat::auto_detect:
            break;
    }
    throw std::runtime_error("Invalid Gaussian output format");
}

GaussianModel load_gaussians(
    const std::filesystem::path& path, GaussianFormat format) {
    if (format == GaussianFormat::auto_detect) {
        format = std::filesystem::is_directory(path)
            ? GaussianFormat::sog
            : gaussian_format_from_path(path);
    }
    switch (format) {
        case GaussianFormat::ply: return load_gaussians_ply(path);
        case GaussianFormat::sog: return load_sog(path);
        case GaussianFormat::spz: return load_spz(path);
        case GaussianFormat::glb: {
            auto decoded = detail::load_glb(path);
            return make_model(
                decoded.count, decoded.degree, decoded.means,
                decoded.log_scales, decoded.rotations,
                decoded.opacity_logits, decoded.sh);
        }
        case GaussianFormat::auto_detect: break;
    }
    throw std::runtime_error("Invalid Gaussian input format");
}

}  // namespace aetherscan::splat
