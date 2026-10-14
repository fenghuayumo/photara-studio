#include "weights.hpp"
#include "ggml-alloc.h"
#include "gguf.h"
#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace photara::inference {
namespace {
template<class T, void (*Free)(T*)> struct Deleter {
    void operator()(T* value) const { if (value) Free(value); }
};
using Metadata = std::unique_ptr<gguf_context, Deleter<gguf_context, gguf_free>>;
using Context = std::unique_ptr<ggml_context, Deleter<ggml_context, ggml_free>>;
using Buffer = std::unique_ptr<ggml_backend_buffer, Deleter<ggml_backend_buffer, ggml_backend_buffer_free>>;
struct FileCloser { void operator()(FILE* file) const { if (file) std::fclose(file); } };
using File = std::unique_ptr<FILE, FileCloser>;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("MoGe weights: " + message);
}
std::string string_value(const gguf_context* metadata, const char* key) {
    const auto index = gguf_find_key(metadata, key);
    require(index >= 0 && gguf_get_kv_type(metadata, index) == GGUF_TYPE_STRING,
        std::string("missing or mistyped metadata ") + key);
    return gguf_get_val_str(metadata, index);
}
std::uint32_t uint_value(const gguf_context* metadata, const char* key) {
    const auto index = gguf_find_key(metadata, key);
    require(index >= 0 && gguf_get_kv_type(metadata, index) == GGUF_TYPE_UINT32,
        std::string("missing or mistyped metadata ") + key);
    return gguf_get_val_u32(metadata, index);
}
void shape(ggml_context* context, const std::string& name,
    std::initializer_list<std::int64_t> dimensions) {
    auto* tensor = ggml_get_tensor(context, name.c_str());
    require(tensor != nullptr, "missing tensor " + name);
    int axis = 0;
    for (const auto dimension : dimensions)
        require(tensor->ne[axis++] == dimension, "unexpected shape of " + name);
    for (; axis < GGML_MAX_DIMS; ++axis)
        require(tensor->ne[axis] == 1, "unexpected rank of " + name);
}
struct ParsedFile {
    File file;
    Metadata metadata;
    Context context;
    MogeModelInfo info;
};
ParsedFile parse(const std::filesystem::path& path) {
    ParsedFile parsed;
#if defined(_WIN32)
    parsed.file.reset(_wfopen(path.c_str(), L"rb"));
#else
    parsed.file.reset(std::fopen(path.c_str(), "rb"));
#endif
    require(parsed.file != nullptr, "cannot open checkpoint");
    ggml_context* raw_context = nullptr;
    parsed.metadata.reset(gguf_init_from_file_ptr(parsed.file.get(), {true, &raw_context}));
    parsed.context.reset(raw_context);
    require(parsed.metadata && parsed.context, "invalid GGUF file");
    const auto* metadata = parsed.metadata.get();
    auto& info = parsed.info;
    require(string_value(metadata, "general.architecture") == "moge", "wrong architecture");
    info.format_version = uint_value(metadata, "moge.format_version");
    require(info.format_version == 1, "unsupported format version");
    info.model.id = string_value(metadata, "moge.source.model_id");
    require(info.model.id == "Ruicheng/moge-2-vitb-normal", "unsupported model id");
    info.model.revision = string_value(metadata, "moge.source.revision");
    require(!info.model.revision.empty(), "empty checkpoint revision");
    info.model.weights = path;
    info.source_sha256 = string_value(metadata, "moge.source.sha256");
    require(info.source_sha256.size() == 64 && std::all_of(
        info.source_sha256.begin(), info.source_sha256.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }), "invalid source SHA-256");
    info.model_config = string_value(metadata, "moge.model_config");
    require(!info.model_config.empty(), "empty model config");
    info.weight_precision = string_value(metadata, "moge.weight_precision");
    require(info.weight_precision == "f32" || info.weight_precision == "f16", "unsupported precision");
    info.embedding_length = uint_value(metadata, "moge.encoder.embedding_length");
    info.block_count = uint_value(metadata, "moge.encoder.block_count");
    info.patch_size = uint_value(metadata, "moge.encoder.patch_size");
    require(info.embedding_length == 768 && info.block_count == 12 && info.patch_size == 14,
        "unsupported encoder configuration");
    const auto count = gguf_get_n_tensors(metadata);
    require(count > 0, "checkpoint contains no tensors");
    info.tensor_count = static_cast<std::size_t>(count);
    const auto file_size = std::filesystem::file_size(path);
    const auto data_offset = gguf_get_data_offset(metadata);
    require(data_offset <= file_size, "truncated GGUF data");
    for (std::int64_t i = 0; i < count; ++i) {
        const auto type = gguf_get_tensor_type(metadata, i);
        require(type == GGML_TYPE_F32 || type == GGML_TYPE_F16, "unsupported tensor type");
        require(info.weight_precision != "f32" || type == GGML_TYPE_F32,
            "F32 checkpoint contains a non-F32 tensor");
        const auto offset = gguf_get_tensor_offset(metadata, i);
        const auto bytes = gguf_get_tensor_size(metadata, i);
        require(offset <= file_size - data_offset && bytes <= file_size - data_offset - offset,
            std::string("truncated tensor ") + gguf_get_tensor_name(metadata, i));
        require(bytes <= std::numeric_limits<std::size_t>::max() - info.tensor_bytes,
            "tensor byte count overflows");
        info.tensor_bytes += bytes;
    }
    auto* context = parsed.context.get();
    shape(context, "encoder.backbone.patch_embed.proj.weight", {14, 14, 3, 768});
    shape(context, "encoder.backbone.cls_token", {768});
    shape(context, "encoder.backbone.pos_embed", {768, 1370});
    shape(context, "encoder.backbone.norm.weight", {768});
    for (int block = 0; block < 12; ++block) {
        const auto prefix = "encoder.backbone.blocks." + std::to_string(block);
        shape(context, prefix + ".attn.qkv.weight", {768, 2304});
        shape(context, prefix + ".mlp.fc1.weight", {768, 3072});
    }
    shape(context, "points_head.output_blocks.4.weight", {1, 1, 32, 3});
    shape(context, "normal_head.output_blocks.4.weight", {1, 1, 32, 3});
    shape(context, "mask_head.output_blocks.4.weight", {1, 1, 32, 1});
    shape(context, "scale_head.4.weight", {768, 1});
    return parsed;
}
}

MogeModelInfo inspect_moge_model(const std::filesystem::path& path) {
    return parse(path).info;
}

namespace detail {
struct MogeWeights::Impl {
    MogeModelInfo info;
    Context context;
    Buffer buffer; // Released before tensor metadata.
};
MogeWeights::MogeWeights(const std::filesystem::path& path, ggml_backend_t backend,
    std::size_t maximum_tensor_bytes) : impl_(std::make_unique<Impl>()) {
    require(backend != nullptr, "null execution backend");
    auto parsed = parse(path);
    require(maximum_tensor_bytes == 0 || parsed.info.tensor_bytes <= maximum_tensor_bytes,
        "checkpoint exceeds requested tensor-byte limit");
    impl_->info = std::move(parsed.info);
    impl_->context = std::move(parsed.context);
    impl_->buffer.reset(ggml_backend_alloc_ctx_tensors(impl_->context.get(), backend));
    require(impl_->buffer != nullptr, "cannot allocate backend weights");
    std::vector<std::byte> staging;
    for (std::int64_t i = 0; i < gguf_get_n_tensors(parsed.metadata.get()); ++i) {
        auto* tensor = ggml_get_tensor(impl_->context.get(),
            gguf_get_tensor_name(parsed.metadata.get(), i));
        const auto offset = gguf_get_data_offset(parsed.metadata.get()) +
            gguf_get_tensor_offset(parsed.metadata.get(), i);
        require(offset <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
            "file offset exceeds supported range");
#if defined(_WIN32)
        require(_fseeki64(parsed.file.get(), static_cast<std::int64_t>(offset), SEEK_SET) == 0,
            "cannot seek tensor data");
#else
        require(fseeko(parsed.file.get(), static_cast<off_t>(offset), SEEK_SET) == 0,
            "cannot seek tensor data");
#endif
        staging.resize(ggml_nbytes(tensor));
        require(std::fread(staging.data(), 1, staging.size(), parsed.file.get()) == staging.size(),
            "cannot read tensor data");
        ggml_backend_tensor_set(tensor, staging.data(), 0, staging.size());
    }
}
MogeWeights::~MogeWeights() = default;
const MogeModelInfo& MogeWeights::info() const noexcept { return impl_->info; }
ggml_tensor* MogeWeights::tensor(const std::string& name) const {
    auto* tensor = ggml_get_tensor(impl_->context.get(), name.c_str());
    require(tensor != nullptr, "missing tensor " + name);
    return tensor;
}
} // namespace detail
} // namespace photara::inference
