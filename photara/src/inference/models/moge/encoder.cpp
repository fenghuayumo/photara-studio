#include "encoder.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>

namespace photara::inference::detail {
namespace {
constexpr int kEmbedding = 768, kPatch = 14, kHeads = 12, kHeadSize = 64;
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("MoGe encoder: " + message);
}
ggml_tensor* weight(MogeWeights& weights, const std::string& name,
    std::initializer_list<std::int64_t> dimensions) {
    auto* tensor = weights.tensor(name);
    int axis = 0;
    for (const auto dimension : dimensions)
        require(tensor->ne[axis++] == dimension, "unexpected shape of " + name);
    for (; axis < GGML_MAX_DIMS; ++axis)
        require(tensor->ne[axis] == 1, "unexpected rank of " + name);
    return tensor;
}
ggml_tensor* matmul(ggml_context* context, ggml_tensor* a, ggml_tensor* b) {
    auto* result = ggml_mul_mat(context, a, b);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}
ggml_tensor* linear(ggml_context* context, MogeWeights& weights,
    ggml_tensor* input, const std::string& prefix, int output_width) {
    auto* matrix = weight(weights, prefix + ".weight", {input->ne[0], output_width});
    auto* bias = weight(weights, prefix + ".bias", {output_width});
    return ggml_add(context, matmul(context, matrix, input), bias);
}
ggml_tensor* norm(ggml_context* context, MogeWeights& weights,
    ggml_tensor* input, const std::string& prefix) {
    auto* scale = weight(weights, prefix + ".weight", {kEmbedding});
    auto* bias = weight(weights, prefix + ".bias", {kEmbedding});
    return ggml_add(context, ggml_mul(context, ggml_norm(context, input, 1e-6F), scale), bias);
}

// PyTorch bicubic kernel, A=-0.75. Native DINOv2 uses the historical
// scale-factor offset rather than simply resizing to the target dimensions.
float cubic(float distance) {
    const float x = std::abs(distance), a = -0.75F;
    if (x <= 1.F) return ((a + 2.F) * x - (a + 3.F)) * x * x + 1.F;
    if (x < 2.F) return ((a * x - 5.F * a) * x + 8.F * a) * x - 4.F * a;
    return 0.F;
}
std::vector<float> positions(MogeWeights& weights, int rows, int columns) {
    auto* tensor = weight(weights, "encoder.backbone.pos_embed", {kEmbedding, 1370});
    require(tensor->type == GGML_TYPE_F32, "position embedding must remain F32");
    std::vector<float> original(1370 * kEmbedding);
    ggml_backend_tensor_get(tensor, original.data(), 0, original.size() * sizeof(float));
    if (rows == 37 && columns == 37) return original;
    std::vector<float> result((static_cast<std::size_t>(rows) * columns + 1) * kEmbedding);
    std::copy_n(original.begin(), kEmbedding, result.begin());
    const double scale_y = 37.0 / (static_cast<double>(rows) + 0.1);
    const double scale_x = 37.0 / (static_cast<double>(columns) + 0.1);
    for (int y = 0; y < rows; ++y) {
        const float source_y = static_cast<float>((y + 0.5) * scale_y - 0.5);
        const int iy = static_cast<int>(std::floor(source_y));
        for (int x = 0; x < columns; ++x) {
            const float source_x = static_cast<float>((x + 0.5) * scale_x - 0.5);
            const int ix = static_cast<int>(std::floor(source_x));
            auto* output = result.data() + (1 + y * columns + x) * kEmbedding;
            for (int dy = -1; dy <= 2; ++dy) {
                const auto sy = std::clamp(iy + dy, 0, 36);
                const float wy = cubic(source_y - (iy + dy));
                for (int dx = -1; dx <= 2; ++dx) {
                    const auto sx = std::clamp(ix + dx, 0, 36);
                    const float coefficient = wy * cubic(source_x - (ix + dx));
                    const auto* source = original.data() + (1 + sy * 37 + sx) * kEmbedding;
                    for (int channel = 0; channel < kEmbedding; ++channel)
                        output[channel] += source[channel] * coefficient;
                }
            }
        }
    }
    return result;
}
std::vector<float> patchify(int rows, int columns, std::span<const float> image) {
    const int height = rows * kPatch, width = columns * kPatch;
    constexpr int patch_size = 3 * kPatch * kPatch;
    std::vector<float> patches(static_cast<std::size_t>(rows) * columns * patch_size);
    std::size_t index = 0;
    for (int py = 0; py < rows; ++py)
        for (int px = 0; px < columns; ++px)
            for (int channel = 0; channel < 3; ++channel)
                for (int y = 0; y < kPatch; ++y)
                    for (int x = 0; x < kPatch; ++x)
                        patches[index++] = image[(static_cast<std::size_t>(channel) * height +
                            py * kPatch + y) * width + px * kPatch + x];
    return patches;
}
}

EncoderResult encode_moge_vitb(MogeWeights& weights, GgmlRuntime& runtime,
    std::uint32_t rows, std::uint32_t columns, std::span<const float> normalized_rgb,
    bool capture_stages) {
    // Dense attention is quadratic. Bound this correctness path to the model's
    // intended range; a memory planner/flash-attention path can extend it later.
    const auto patch_count = static_cast<std::uint64_t>(rows) * columns;
    require(rows > 0 && columns > 0 && patch_count <= 4096,
        "token grid must contain 1..4096 patches");
    require(normalized_rgb.size() == patch_count * 3 * kPatch * kPatch,
        "normalized CHW image size mismatch");
    require(std::all_of(normalized_rgb.begin(), normalized_rgb.end(),
        [](float value) { return std::isfinite(value); }), "input contains nonfinite pixels");
    const int tokens = static_cast<int>(patch_count) + 1;
    auto position_data = positions(weights, static_cast<int>(rows), static_cast<int>(columns));
    auto patch_data = patchify(static_cast<int>(rows), static_cast<int>(columns), normalized_rgb);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({16 * 1024 * 1024, nullptr, true}), &ggml_free);
    if (!context) throw std::runtime_error("Cannot create MoGe encoder graph context");
    auto* ctx = context.get();
    auto* graph = ggml_new_graph_custom(ctx, 8192, false);
    auto* patches = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * kPatch * kPatch, patch_count);
    auto* position = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kEmbedding, tokens);
    ggml_set_input(patches);
    ggml_set_input(position);
    std::vector<std::pair<std::string, ggml_tensor*>> stages;
    const auto trace = [&](const std::string& name, ggml_tensor* tensor) {
        if (capture_stages) {
            ggml_set_output(tensor);
            stages.emplace_back(name, tensor);
            ggml_build_forward_expand(graph, tensor);
        }
    };
    auto* patch_matrix = ggml_reshape_2d(ctx,
        weight(weights, "encoder.backbone.patch_embed.proj.weight", {14, 14, 3, kEmbedding}),
        3 * kPatch * kPatch, kEmbedding);
    auto* embedding = ggml_add(ctx, matmul(ctx, patch_matrix, patches),
        weight(weights, "encoder.backbone.patch_embed.proj.bias", {kEmbedding}));
    trace("patch_embed", embedding);
    trace("position_encoding", position);
    auto* cls = ggml_reshape_2d(ctx,
        weight(weights, "encoder.backbone.cls_token", {kEmbedding}), kEmbedding, 1);
    auto* x = ggml_add(ctx, ggml_concat(ctx, cls, embedding, 1), position);
    trace("prepared_tokens", x);
    ggml_tensor* taps[2]{};
    for (int block = 0; block < 12; ++block) {
        const auto prefix = "encoder.backbone.blocks." + std::to_string(block);
        auto* n1 = norm(ctx, weights, x, prefix + ".norm1");
        auto* qkv = linear(ctx, weights, n1, prefix + ".attn.qkv", 3 * kEmbedding);
        const auto split_head = [&](int part) {
            auto* slice = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kEmbedding, tokens,
                qkv->nb[1], part * kEmbedding * sizeof(float)));
            auto* heads = ggml_reshape_3d(ctx, slice, kHeadSize, kHeads, tokens);
            return ggml_cont(ctx, ggml_permute(ctx, heads, 0, 2, 1, 3));
        };
        auto* q = split_head(0);
        auto* k = split_head(1);
        auto* v = split_head(2);
        auto* probabilities = ggml_soft_max_ext(ctx, matmul(ctx, k, q), nullptr, 0.125F, 0.F);
        auto* attention = matmul(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), probabilities);
        attention = ggml_reshape_2d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, attention, 0, 2, 1, 3)), kEmbedding, tokens);
        auto* projected = linear(ctx, weights, attention, prefix + ".attn.proj", kEmbedding);
        x = ggml_add(ctx, x, ggml_mul(ctx, projected,
            weight(weights, prefix + ".ls1.gamma", {kEmbedding})));
        auto* n2 = norm(ctx, weights, x, prefix + ".norm2");
        auto* hidden = ggml_gelu_erf(ctx, linear(ctx, weights, n2, prefix + ".mlp.fc1", 3072));
        auto* mlp = linear(ctx, weights, hidden, prefix + ".mlp.fc2", kEmbedding);
        x = ggml_add(ctx, x, ggml_mul(ctx, mlp,
            weight(weights, prefix + ".ls2.gamma", {kEmbedding})));
        trace("block." + std::to_string(block), x);
        if (block == 5) taps[0] = x;
        if (block == 11) taps[1] = x;
    }
    ggml_tensor* projections[2]{};
    ggml_tensor* normalized_last = nullptr;
    for (int tap = 0; tap < 2; ++tap) {
        auto* normalized = norm(ctx, weights, taps[tap], "encoder.backbone.norm");
        trace(tap == 0 ? "normalized.5" : "normalized.11", normalized);
        auto* patch_tokens = ggml_cont(ctx, ggml_view_2d(ctx, normalized,
            kEmbedding, patch_count, normalized->nb[1], kEmbedding * sizeof(float)));
        const auto prefix = "encoder.output_projections." + std::to_string(tap);
        auto* matrix = ggml_reshape_2d(ctx,
            weight(weights, prefix + ".weight", {1, 1, kEmbedding, kEmbedding}),
            kEmbedding, kEmbedding);
        projections[tap] = ggml_add(ctx, matmul(ctx, matrix, patch_tokens),
            weight(weights, prefix + ".bias", {kEmbedding}));
        trace("projection." + std::to_string(tap), projections[tap]);
        normalized_last = normalized;
    }
    auto* features = ggml_add(ctx, projections[0], projections[1]);
    auto* class_token = ggml_cont(ctx, ggml_view_1d(ctx, normalized_last, kEmbedding, 0));
    ggml_set_output(features);
    ggml_set_output(class_token);
    ggml_build_forward_expand(graph, features);
    ggml_build_forward_expand(graph, class_token);
    runtime.allocate(graph);
    ggml_backend_tensor_set(patches, patch_data.data(), 0, patch_data.size() * sizeof(float));
    ggml_backend_tensor_set(position, position_data.data(), 0, position_data.size() * sizeof(float));
    runtime.compute(graph);
    const auto read = [](ggml_tensor* tensor) {
        std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        return values;
    };
    EncoderResult result;
    result.rows = rows;
    result.columns = columns;
    result.features = read(features);
    result.class_token = read(class_token);
    for (const auto& [name, tensor] : stages) result.stages.push_back({name, read(tensor)});
    return result;
}
} // namespace photara::inference::detail
