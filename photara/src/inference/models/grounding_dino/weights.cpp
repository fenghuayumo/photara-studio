#include "weights.hpp"

#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace photara::inference::detail {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error("Grounding DINO weights: " + message);
}

struct TensorEntry {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::filesystem::path& path) {
#if defined(_WIN32)
        file_ = std::unique_ptr<FILE, FileCloser>(
            _wfopen(path.c_str(), L"rb"));
#else
        file_ = std::unique_ptr<FILE, FileCloser>(
            std::fopen(path.c_str(), "rb"));
#endif
        require(file_ != nullptr, "cannot open " + path.string());
        std::array<unsigned char, 8> header_size{};
        require(std::fread(header_size.data(), 1, header_size.size(), file_.get()) ==
                    header_size.size(),
            "truncated header length");
        std::uint64_t length = 0;
        for (unsigned int i = 0; i < header_size.size(); ++i)
            length |= static_cast<std::uint64_t>(header_size[i]) << (8U * i);
        require(length > 2 && length < (128U << 20U), "invalid header length");
        header_.resize(static_cast<std::size_t>(length));
        require(std::fread(header_.data(), 1, header_.size(), file_.get()) ==
                    header_.size(),
            "truncated JSON header");
        data_offset_ = header_.size() + 8U;
        parse_header();
    }

    [[nodiscard]] bool has(const std::string& name) const {
        return entries_.find(name) != entries_.end();
    }
    [[nodiscard]] std::vector<std::string> names() const {
        std::vector<std::string> result;
        result.reserve(entries_.size());
        for (const auto& [name, entry] : entries_)
            if (entry.dtype == "F32") result.push_back(name);
        return result;
    }
    [[nodiscard]] const TensorEntry& entry(const std::string& name) const {
        const auto found = entries_.find(name);
        require(found != entries_.end(), "missing tensor " + name);
        return found->second;
    }
    [[nodiscard]] std::vector<float> read(const std::string& name) const {
        const auto& tensor = entry(name);
        require(tensor.dtype == "F32", "tensor " + name + " is not F32");
        std::size_t count = 1;
        for (const auto dimension : tensor.shape) {
            require(dimension >= 0, "negative shape in " + name);
            require(dimension == 0 || count <= std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension), "tensor shape overflows in " + name);
            count *= static_cast<std::size_t>(dimension);
        }
        require(count <= std::numeric_limits<std::size_t>::max() / sizeof(float) &&
            tensor.end - tensor.begin == count * sizeof(float), "invalid tensor byte size in " + name);
        std::vector<float> values(count);
        read_at(data_offset_ + tensor.begin,
            values.data(), values.size() * sizeof(float));
        return values;
    }

private:
    struct FileCloser {
        void operator()(FILE* file) const {
            if (file) std::fclose(file);
        }
    };

    static void skip_space(const std::string& text, std::size_t& offset) {
        while (offset < text.size() &&
               std::isspace(static_cast<unsigned char>(text[offset])) != 0)
            ++offset;
    }

    static std::string parse_string(
        const std::string& text, std::size_t& offset) {
        skip_space(text, offset);
        require(offset < text.size() && text[offset] == '"',
            "expected JSON string");
        ++offset;
        std::string result;
        while (offset < text.size() && text[offset] != '"') {
            if (text[offset] != '\\') {
                result.push_back(text[offset++]);
                continue;
            }
            require(offset + 1 < text.size(), "truncated JSON escape");
            ++offset;
            const char escape = text[offset++];
            switch (escape) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    require(offset + 4 <= text.size(), "truncated Unicode escape");
                    unsigned int code = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char digit = text[offset++];
                        code <<= 4U;
                        if (digit >= '0' && digit <= '9') code |= static_cast<unsigned int>(digit - '0');
                        else if (digit >= 'a' && digit <= 'f') code |= static_cast<unsigned int>(digit - 'a' + 10);
                        else if (digit >= 'A' && digit <= 'F') code |= static_cast<unsigned int>(digit - 'A' + 10);
                        else throw std::runtime_error("Invalid JSON Unicode escape");
                    }
                    if (code < 0x80U) result.push_back(static_cast<char>(code));
                    else if (code < 0x800U) {
                        result.push_back(static_cast<char>(0xC0U | (code >> 6U)));
                        result.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
                    } else {
                        result.push_back(static_cast<char>(0xE0U | (code >> 12U)));
                        result.push_back(static_cast<char>(0x80U | ((code >> 6U) & 0x3FU)));
                        result.push_back(static_cast<char>(0x80U | (code & 0x3FU)));
                    }
                    break;
                }
                default: throw std::runtime_error("Unknown JSON escape");
            }
        }
        require(offset < text.size(), "unterminated JSON string");
        ++offset;
        return result;
    }

    static std::vector<std::int64_t> parse_integer_array(
        const std::string& text, std::size_t& offset) {
        skip_space(text, offset);
        require(offset < text.size() && text[offset] == '[',
            "expected JSON array");
        ++offset;
        std::vector<std::int64_t> values;
        skip_space(text, offset);
        if (offset < text.size() && text[offset] == ']') {
            ++offset;
            return values;
        }
        while (true) {
            skip_space(text, offset);
            std::size_t end = offset;
            while (end < text.size() &&
                   (std::isdigit(static_cast<unsigned char>(text[end])) != 0 ||
                    text[end] == '-' || text[end] == '+'))
                ++end;
            require(end > offset, "expected integer in JSON array");
            values.push_back(std::stoll(text.substr(offset, end - offset)));
            offset = end;
            skip_space(text, offset);
            require(offset < text.size(), "unterminated JSON array");
            if (text[offset] == ',') {
                ++offset;
                continue;
            }
            require(text[offset] == ']', "expected , or ] in JSON array");
            ++offset;
            break;
        }
        return values;
    }

    static std::size_t value_extent(
        const std::string& text, std::size_t& offset) {
        skip_space(text, offset);
        const std::size_t begin = offset;
        require(offset < text.size(), "missing JSON value");
        if (text[offset] == '"') {
            (void)parse_string(text, offset);
            return begin;
        }
        if (text[offset] == '[' || text[offset] == '{') {
            const char opening = text[offset++];
            const char closing = opening == '[' ? ']' : '}';
            int depth = 1;
            bool quoted = false;
            bool escaped = false;
            while (offset < text.size() && depth != 0) {
                const char value = text[offset++];
                if (quoted) {
                    if (escaped) escaped = false;
                    else if (value == '\\') escaped = true;
                    else if (value == '"') quoted = false;
                    continue;
                }
                if (value == '"') quoted = true;
                else if (value == opening) ++depth;
                else if (value == closing) --depth;
            }
            require(depth == 0, "unterminated JSON container");
            return begin;
        }
        while (offset < text.size() && text[offset] != ',' && text[offset] != '}' &&
               text[offset] != ']')
            ++offset;
        require(offset > begin, "empty JSON value");
        return begin;
    }

    void parse_header() {
        std::size_t offset = 0;
        skip_space(header_, offset);
        require(offset < header_.size() && header_[offset] == '{',
            "header is not a JSON object");
        ++offset;
        while (true) {
            skip_space(header_, offset);
            if (offset < header_.size() && header_[offset] == '}') {
                ++offset;
                break;
            }
            const std::string name = parse_string(header_, offset);
            skip_space(header_, offset);
            require(offset < header_.size() && header_[offset] == ':',
                "missing colon after tensor name");
            ++offset;
            const std::size_t value = value_extent(header_, offset);
            if (name != "__metadata__") parse_tensor(name, value);
            skip_space(header_, offset);
            require(offset < header_.size(), "unterminated header object");
            if (header_[offset] == ',') {
                ++offset;
                continue;
            }
            require(header_[offset] == '}', "expected , or } in header");
            ++offset;
            break;
        }
        require(!entries_.empty(), "header contains no tensors");
    }

    void parse_tensor(const std::string& name, const std::size_t begin) {
        TensorEntry entry;
        std::size_t offset = begin;
        skip_space(header_, offset);
        require(offset < header_.size() && header_[offset] == '{',
            "tensor entry is not an object");
        ++offset;
        while (true) {
            skip_space(header_, offset);
            if (offset < header_.size() && header_[offset] == '}') {
                ++offset;
                break;
            }
            const std::string key = parse_string(header_, offset);
            skip_space(header_, offset);
            require(offset < header_.size() && header_[offset] == ':',
                "missing colon in tensor entry");
            ++offset;
            if (key == "dtype") entry.dtype = parse_string(header_, offset);
            else if (key == "shape") entry.shape = parse_integer_array(header_, offset);
            else if (key == "data_offsets") {
                const auto offsets = parse_integer_array(header_, offset);
                require(offsets.size() == 2, "invalid data_offsets");
                entry.begin = static_cast<std::uint64_t>(offsets[0]);
                entry.end = static_cast<std::uint64_t>(offsets[1]);
                require(entry.begin <= entry.end, "invalid tensor byte range");
            }
            else
                (void)value_extent(header_, offset);
            skip_space(header_, offset);
            require(offset < header_.size(), "unterminated tensor entry");
            if (header_[offset] == ',') {
                ++offset;
                continue;
            }
            require(header_[offset] == '}', "expected , or } in tensor entry");
            ++offset;
            break;
        }
        require(entry.dtype == "F32" || entry.dtype == "I64" ||
                    entry.dtype == "F16",
            "unsupported dtype " + entry.dtype + " for " + name);
        require(entry.shape.size() <= 4, "unsupported rank for " + name);
        entries_.emplace(name, std::move(entry));
    }

    void read_at(const std::uint64_t offset, void* data,
        const std::size_t bytes) const {
        require(bytes <= static_cast<std::size_t>(
                             std::numeric_limits<long>::max()),
            "tensor is too large for a single read");
#if defined(_WIN32)
        require(_fseeki64(file_.get(), static_cast<long long>(offset), SEEK_SET) == 0,
            "cannot seek tensor data");
#else
        require(fseeko(file_.get(), static_cast<off_t>(offset), SEEK_SET) == 0,
            "cannot seek tensor data");
#endif
        require(std::fread(data, 1, bytes, file_.get()) == bytes,
            "cannot read tensor data");
    }

    struct StringLess {
        using is_transparent = void;
        bool operator()(std::string_view left, std::string_view right) const {
            return left < right;
        }
    };

    std::unique_ptr<FILE, FileCloser> file_;
    std::string header_;
    std::uint64_t data_offset_ = 0;
    std::map<std::string, TensorEntry, StringLess> entries_;
};

GroundingDinoConfig make_config(const SafetensorsFile& file) {
    GroundingDinoConfig config;
    const std::string prefix = "model.backbone.conv_encoder.model.";
    const auto& table = file.entry(prefix +
        "encoder.layers.0.blocks.0.attention.self."
        "relative_position_bias_table");
    const auto side = static_cast<float>(table.shape[0]);
    const auto window = static_cast<std::uint32_t>(
        (std::lround(std::sqrt(side)) + 1) / 2);
    config.swin_window = window;
    config.swin_embedding = static_cast<std::uint32_t>(
        file.entry(prefix + "embeddings.patch_embeddings.projection.bias")
            .shape[0]);
    for (unsigned int stage = 0; stage < 4; ++stage) {
        std::uint32_t depth = 0;
        while (file.has(prefix + "encoder.layers." + std::to_string(stage) +
                        ".blocks." + std::to_string(depth) +
                        ".layernorm_before.weight"))
            ++depth;
        require(depth > 0, "backbone stage has no blocks");
        config.swin_depths[stage] = depth;
        config.swin_heads[stage] = static_cast<std::uint32_t>(file.entry(
            prefix + "encoder.layers." + std::to_string(stage) +
            ".blocks.0.attention.self.relative_position_bias_table").shape[1]);
    }
    while (file.has("model.encoder.layers." + std::to_string(config.encoder_layers) +
                    ".fusion_layer.text_param"))
        ++config.encoder_layers;
    while (file.has("model.decoder.layers." + std::to_string(config.decoder_layers) +
                    ".fc1.weight"))
        ++config.decoder_layers;
    while (file.has("model.text_backbone.encoder.layer." +
                    std::to_string(config.text_layers) +
                    ".output.dense.weight"))
        ++config.text_layers;
    config.model_dimension = static_cast<std::uint32_t>(
        file.entry("model.level_embed").shape[1]);
    config.levels = static_cast<std::uint32_t>(
        file.entry("model.level_embed").shape[0]);
    config.queries = static_cast<std::uint32_t>(file.entry(
        "model.query_position_embeddings.weight").shape[0]);
    config.text_dimension = static_cast<std::uint32_t>(file.entry(
        "model.text_projection.weight").shape[1]);
    const std::uint64_t locations = config.levels * config.sampling_points;
    config.encoder_heads = static_cast<std::uint32_t>(file.entry(
        "model.encoder.layers.0.deformable_layer.self_attn.attention_weights.bias")
        .shape[0] / locations);
    config.decoder_heads = static_cast<std::uint32_t>(file.entry(
        "model.decoder.layers.0.encoder_attn.attention_weights.bias")
        .shape[0] / locations);
    require(config.model_dimension == 256 && config.levels == 4 &&
                config.encoder_layers > 0 && config.decoder_layers > 0 &&
                config.text_layers > 0 &&
                config.text_dimension % 64 == 0 &&
                config.encoder_heads > 0 && config.decoder_heads > 0,
        "unsupported architecture");
    return config;
}

}  // namespace

struct GroundingDinoWeights::Impl {
    GroundingDinoModelInfo info;
    GroundingDinoConfig config;
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    ~Impl() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
};

GroundingDinoWeights::GroundingDinoWeights(
    const std::filesystem::path& path,
    ggml_backend_t backend,
    const std::size_t maximum_tensor_bytes)
    : impl_(std::make_unique<Impl>()) {
    require(backend != nullptr, "null GGML backend");
    SafetensorsFile file(path);
    impl_->config = make_config(file);
    impl_->info.model.id = path.stem().string();
    impl_->info.model.weights = path;

    std::vector<std::string> names = file.names();
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    require(!names.empty(), "no usable tensors were found");

    const std::size_t context_bytes = 128U << 20U;
    const ggml_init_params parameters{
        static_cast<std::size_t>(context_bytes), nullptr, true};
    impl_->context = ggml_init(parameters);
    require(impl_->context != nullptr, "cannot create GGML metadata context");
    std::size_t tensor_bytes = 0;
    for (const std::string& name : names) {
        const auto& source = file.entry(name);
        if (source.dtype != "F32") continue;
        std::int64_t dimensions[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (std::size_t i = 0; i < source.shape.size(); ++i)
            dimensions[i] = source.shape[source.shape.size() - 1U - i];
        const bool matrix = source.shape.size() >= 2 &&
                            name.ends_with(".weight");
        const ggml_type type = matrix ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor* tensor = nullptr;
        switch (source.shape.size()) {
            case 1:
                tensor = ggml_new_tensor_1d(impl_->context, type, dimensions[0]);
                break;
            case 2:
                tensor = ggml_new_tensor_2d(
                    impl_->context, type, dimensions[0], dimensions[1]);
                break;
            case 3:
                tensor = ggml_new_tensor_3d(impl_->context, type, dimensions[0],
                    dimensions[1], dimensions[2]);
                break;
            default:
                tensor = ggml_new_tensor_4d(impl_->context, type, dimensions[0],
                    dimensions[1], dimensions[2], dimensions[3]);
                break;
        }
        require(tensor != nullptr, "cannot create tensor " + name);
        impl_->tensors.emplace(name, tensor);
        tensor_bytes += ggml_nbytes(tensor);
    }
    require(maximum_tensor_bytes == 0 || tensor_bytes <= maximum_tensor_bytes,
        "checkpoint exceeds requested tensor-byte limit");
    impl_->info.tensor_count = names.size();
    impl_->info.tensor_bytes = tensor_bytes;
    impl_->buffer = ggml_backend_alloc_ctx_tensors(impl_->context, backend);
    require(impl_->buffer != nullptr, "cannot allocate backend weights");

    std::vector<float> floats;
    std::vector<ggml_fp16_t> halves;
    for (const std::string& name : names) {
        const auto& source = file.entry(name);
        if (source.dtype != "F32") continue;
        const auto found = impl_->tensors.find(name);
        auto* tensor = found == impl_->tensors.end() ? nullptr : found->second;
        require(tensor != nullptr, "lost tensor " + name);
        std::size_t count = 1;
        for (const auto dimension : source.shape)
            count *= static_cast<std::size_t>(dimension);
        floats = file.read(name);
        require(floats.size() == count, "short read for " + name);
        if (tensor->type == GGML_TYPE_F16) {
            halves.resize(count);
            for (std::size_t i = 0; i < count; ++i)
                halves[i] = ggml_fp32_to_fp16(floats[i]);
            ggml_backend_tensor_set(
                tensor, halves.data(), 0, halves.size() * sizeof(ggml_fp16_t));
        } else {
            ggml_backend_tensor_set(
                tensor, floats.data(), 0, floats.size() * sizeof(float));
        }
    }

    impl_->info.encoder_layers = impl_->config.encoder_layers;
    impl_->info.decoder_layers = impl_->config.decoder_layers;
    impl_->info.text_layers = impl_->config.text_layers;
    impl_->info.model_dimension = impl_->config.model_dimension;
    impl_->info.text_dimension = impl_->config.text_dimension;
    impl_->info.queries = impl_->config.queries;
}

GroundingDinoWeights::~GroundingDinoWeights() = default;

const GroundingDinoModelInfo& GroundingDinoWeights::info() const noexcept {
    return impl_->info;
}

const GroundingDinoConfig& GroundingDinoWeights::config() const noexcept {
    return impl_->config;
}

bool GroundingDinoWeights::has(const std::string& name) const noexcept {
    return impl_->tensors.find(name) != impl_->tensors.end();
}

ggml_tensor* GroundingDinoWeights::tensor(const std::string& name) const {
    const auto found = impl_->tensors.find(name);
    auto* tensor = found == impl_->tensors.end() ? nullptr : found->second;
    require(tensor != nullptr, "missing tensor " + name);
    return tensor;
}

}  // namespace photara::inference::detail
