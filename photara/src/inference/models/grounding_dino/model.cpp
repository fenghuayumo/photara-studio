#include "model.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace photara::inference::detail {
namespace {

constexpr double k_pi = 3.14159265358979323846;
constexpr std::size_t k_graph_nodes = 262144;

void require(const bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error("Grounding DINO graph: " + message);
}

struct ContextCloser {
    void operator()(ggml_context* context) const {
        if (context != nullptr) ggml_free(context);
    }
};
using ContextPtr = std::unique_ptr<ggml_context, ContextCloser>;

struct GraphInput {
    ggml_tensor* tensor = nullptr;
    std::vector<unsigned char> bytes;
};

struct GraphOutput {
    std::string name;
    ggml_tensor* tensor = nullptr;
};

class GraphBuilder {
public:
    GraphBuilder() : context_(ggml_init({256U << 20U, nullptr, true})) {
        require(context_ != nullptr, "cannot create the graph context");
        graph_ = ggml_new_graph_custom(context(), k_graph_nodes, false);
        require(graph_ != nullptr, "cannot create the graph");
    }

    [[nodiscard]] ggml_context* context() const noexcept {
        return context_.get();
    }

    [[nodiscard]] ggml_tensor* float_input(
        const std::initializer_list<std::int64_t> dimensions,
        const std::vector<float>& values) {
        auto* tensor = new_float(dimensions);
        ggml_set_input(tensor);
        GraphInput input;
        input.tensor = tensor;
        input.bytes.resize(values.size() * sizeof(float));
        require(input.bytes.size() == ggml_nbytes(tensor), "graph input byte count mismatch");
        std::memcpy(input.bytes.data(), values.data(), input.bytes.size());
        inputs_.push_back(std::move(input));
        return tensor;
    }

    [[nodiscard]] ggml_tensor* integer_input(
        const std::int64_t elements,
        const std::vector<std::int32_t>& values) {
        require(static_cast<std::size_t>(elements) == values.size(),
            "integer input size mismatch");
        auto* tensor = ggml_new_tensor_1d(context(), GGML_TYPE_I32, elements);
        require(tensor != nullptr, "cannot create integer input");
        ggml_set_input(tensor);
        GraphInput input;
        input.tensor = tensor;
        input.bytes.resize(values.size() * sizeof(std::int32_t));
        std::memcpy(input.bytes.data(), values.data(), input.bytes.size());
        inputs_.push_back(std::move(input));
        return tensor;
    }

    void output(const std::string& name, ggml_tensor* tensor) {
        require(tensor != nullptr, "cannot output a null tensor");
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph_, tensor);
        outputs_.push_back({std::move(name), tensor});
    }

    [[nodiscard]] std::map<std::string, std::vector<float>> run(
        GgmlRuntime& runtime) {
        runtime.allocate(graph_);
        for (const GraphInput& input : inputs_) {
            if (!input.tensor->buffer) continue;
            ggml_backend_tensor_set(
                input.tensor, input.bytes.data(), 0, input.bytes.size());
        }
        runtime.compute(graph_);
        std::map<std::string, std::vector<float>> result;
        for (const GraphOutput& output : outputs_) {
            require(output.tensor->type == GGML_TYPE_F32,
                "output " + output.name + " is not F32");
            std::vector<float> values(
                static_cast<std::size_t>(ggml_nelements(output.tensor)));
            ggml_backend_tensor_get(
                output.tensor, values.data(), 0, values.size() * sizeof(float));
            result.emplace(output.name, std::move(values));
        }
        return result;
    }

private:
    [[nodiscard]] ggml_tensor* new_float(
        const std::initializer_list<std::int64_t>& dimensions) const {
        require(dimensions.size() >= 1 && dimensions.size() <= 4,
            "unsupported input rank");
        std::int64_t shape[4] = {1, 1, 1, 1};
        std::size_t index = 0;
        for (const auto dimension : dimensions) shape[index++] = dimension;
        if (dimensions.size() == 1)
            return ggml_new_tensor_1d(context(), GGML_TYPE_F32, shape[0]);
        if (dimensions.size() == 2)
            return ggml_new_tensor_2d(
                context(), GGML_TYPE_F32, shape[0], shape[1]);
        if (dimensions.size() == 3)
            return ggml_new_tensor_3d(
                context(), GGML_TYPE_F32, shape[0], shape[1], shape[2]);
        return ggml_new_tensor_4d(context(), GGML_TYPE_F32, shape[0], shape[1],
            shape[2], shape[3]);
    }

    ContextPtr context_;
    ggml_cgraph* graph_ = nullptr;
    std::vector<GraphInput> inputs_;
    std::vector<GraphOutput> outputs_;
};

[[nodiscard]] ggml_tensor* matrix(
    GroundingDinoWeights& weights, const std::string& name,
    const std::int64_t input, const std::int64_t output) {
    auto* tensor = weights.tensor(name);
    require(tensor->ne[0] == input && tensor->ne[1] == output,
            "unexpected shape of " + name);
    return tensor;
}

[[nodiscard]] ggml_tensor* vector(
    GroundingDinoWeights& weights, const std::string& name,
    const std::int64_t elements) {
    auto* tensor = weights.tensor(name);
    require(tensor->ne[0] == elements && tensor->ne[1] == 1,
            "unexpected shape of " + name);
    return tensor;
}

[[nodiscard]] ggml_tensor* matmul(
    ggml_context* context, ggml_tensor* weight, ggml_tensor* input) {
    auto* result = ggml_mul_mat(context, weight, input);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

[[nodiscard]] ggml_tensor* linear(
    ggml_context* context,
    GroundingDinoWeights& weights,
    ggml_tensor* input,
    const std::string& prefix,
    ggml_tensor* residual = nullptr) {
    const auto output_width = weights.tensor(prefix + ".weight")->ne[1];
    auto* weight = matrix(
        weights, prefix + ".weight", input->ne[0], output_width);
    const auto n1 = input->ne[1], n2 = input->ne[2], n3 = input->ne[3];
    auto* flat_input = ggml_reshape_2d(context, ggml_cont(context, input), input->ne[0], n1*n2*n3);
    auto* result = matmul(context, weight, flat_input);
    result = ggml_reshape_4d(context, result, output_width, n1, n2, n3);
    if (weights.has(prefix + ".bias")) {
        auto* bias = vector(weights, prefix + ".bias", output_width);
        result = ggml_add(context, result, bias);
    }
    if (residual != nullptr) result = ggml_add(context, result, residual);
    return result;
}

[[nodiscard]] ggml_tensor* layer_norm(
    ggml_context* context,
    GroundingDinoWeights& weights,
    ggml_tensor* input,
    const std::string& prefix,
    const float epsilon = 1e-5F) {
    const auto width = input->ne[0];
    auto* normalized = ggml_norm(context, input, epsilon);
    normalized = ggml_mul(
        context, normalized, vector(weights, prefix + ".weight", width));
    return ggml_add(
        context, normalized, vector(weights, prefix + ".bias", width));
}

[[nodiscard]] ggml_tensor* plain_attention(
    ggml_context* context,
    ggml_tensor* query,
    ggml_tensor* key,
    ggml_tensor* value,
    ggml_tensor* mask,
    const int heads,
    const int batch_count = 1) {
    const auto head_dimension = query->ne[0] / heads;
    const auto query_count = query->ne[1] / batch_count;
    const auto key_count = key->ne[1] / batch_count;
    require(head_dimension * heads == query->ne[0],
            "attention channels are not divisible by heads");
    const auto split = [&](ggml_tensor* tensor, const int count) {
        auto* shaped = ggml_reshape_4d(
            context, tensor, head_dimension, heads, count, batch_count);
        return ggml_cont(
            context, ggml_permute(context, shaped, 0, 2, 1, 3));
    };
    auto* q = split(query, query_count);
    auto* k = split(key, key_count);
    auto* v = split(value, key_count);
    auto* scores = ggml_mul_mat(context, k, q);
    scores = ggml_soft_max_ext(
        context, scores, mask, 1.0F / std::sqrt(static_cast<float>(head_dimension)), 0.0F);
    auto* value_transposed = ggml_cont(context, ggml_transpose(context, v));
    auto* attended = ggml_mul_mat(context, value_transposed, scores);
    attended = ggml_cont(
        context, ggml_permute(context, attended, 0, 2, 1, 3));
    return ggml_reshape_2d(
        context, attended, query->ne[0], query->ne[1]);
}

[[nodiscard]] ggml_tensor* window_partition(
    ggml_context* context,
    ggml_tensor* input,
    const int width,
    const int height,
    const int window) {
    const auto channels = input->ne[0];
    const int pad_x = (window - width % window) % window;
    const int pad_y = (window - height % window) % window;
    auto* padded = input;
    if (pad_x != 0 || pad_y != 0)
        padded = ggml_pad(context, padded, 0, pad_x, pad_y, 0);
    if (!ggml_is_contiguous(padded))
        padded = ggml_cont(context, padded);
    const int windows_x = static_cast<int>(padded->ne[1]) / window;
    const int padded_height = static_cast<int>(padded->ne[2]);
    auto* grouped = ggml_reshape_4d(context, padded,
        channels * window, windows_x, window, padded_height / window);
    grouped = ggml_cont(context, ggml_permute(context, grouped, 0, 2, 1, 3));
    return ggml_reshape_4d(context, grouped, channels, window, window,
        windows_x * (padded_height / window));
}

[[nodiscard]] ggml_tensor* window_unpartition(
    ggml_context* context,
    ggml_tensor* input,
    const int width,
    const int height,
    const int window) {
    const int windows_x = (width + window - 1) / window;
    const int windows_y = (height + window - 1) / window;
    const int padded_width = windows_x * window;
    const int padded_height = windows_y * window;
    require(input->ne[3] == windows_x * windows_y, "unexpected window count");
    auto* image = ggml_is_contiguous(input)
        ? input
        : ggml_cont(context, input);
    image = ggml_reshape_4d(context, image, input->ne[0] * window,
        window, windows_x, windows_y);
    image = ggml_cont(context, ggml_permute(context, image, 0, 2, 1, 3));
    image = ggml_reshape_4d(context, image, input->ne[0], padded_width, padded_height, 1);
    if (padded_width == width && padded_height == height) return image;
    return ggml_cont(context, ggml_view_4d(
        context, image, image->ne[0], width, height, 1,
        image->nb[1], image->nb[2], image->nb[3], 0));
}

[[nodiscard]] std::vector<float> tensor_values(ggml_tensor* tensor) {
    std::vector<float> values(
        static_cast<std::size_t>(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(values.size());
        ggml_backend_tensor_get(tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(half.data(), values.data(), values.size());
    } else {
        require(tensor->type == GGML_TYPE_F32, "unsupported tensor download type");
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    return values;
}

[[nodiscard]] std::vector<float> expand_relative_bias(
    const std::vector<float>& table,
    const int window,
    const int heads) {
    const int tokens = window * window;
    const int side = 2 * window - 1;
    require(table.size() ==
                static_cast<std::size_t>(side) * side * heads,
            "unexpected relative position table");
    std::vector<float> result(static_cast<std::size_t>(heads) * tokens * tokens);
    for (int i = 0; i < tokens; ++i)
        for (int j = 0; j < tokens; ++j) {
            const int index =
                (i / window - j / window + window - 1) * side +
                (i % window - j % window + window - 1);
            for (int head = 0; head < heads; ++head)
                result[(static_cast<std::size_t>(head) * tokens + i) *
                           tokens +
                       j] = table[static_cast<std::size_t>(index) * heads +
                                  head];
        }
    return result;
}

[[nodiscard]] std::vector<std::int32_t> roll_indices(
    const int width,
    const int height,
    const int shift) {
    std::vector<std::int32_t> result(
        static_cast<std::size_t>(width) * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const int source_x = (x - shift + width) % width;
            const int source_y = (y - shift + height) % height;
            result[static_cast<std::size_t>(y) * width + x] =
                static_cast<std::int32_t>(
                    static_cast<std::size_t>(source_y) * width + source_x);
        }
    return result;
}

[[nodiscard]] std::vector<float> window_mask(
    const std::vector<float>& relative,
    const int width,
    const int height,
    const int window,
    const int shift,
    const int heads) {
    const int windows_x = (width + window - 1) / window;
    const int windows_y = (height + window - 1) / window;
    const int window_count = windows_x * windows_y;
    const int tokens = window * window;
    const int padded_width = windows_x * window;
    const int padded_height = windows_y * window;
    std::vector<float> mask(
        static_cast<std::size_t>(tokens) * tokens * heads * window_count);
    const auto label = [&](const int x, const int y) {
        const auto region = [&](int v, int extent) {
            return v < extent - window ? 0 : (v < extent - shift ? 1 : 2);
        };
        return region(y, padded_height) * 3 + region(x, padded_width);
    };
    for (int window_index = 0; window_index < window_count; ++window_index) {
        const int wx = window_index % windows_x;
        const int wy = window_index / windows_x;
        for (int query = 0; query < tokens; ++query) {
            const int qx = wx * window + query % window;
            const int qy = wy * window + query / window;
            for (int key = 0; key < tokens; ++key) {
                const int kx = wx * window + key % window;
                const int ky = wy * window + key / window;
                const bool allowed = shift == 0 ||
                                     label(qx, qy) == label(kx, ky);
                for (int head = 0; head < heads; ++head) {
                    const std::size_t flat =
                        ((static_cast<std::size_t>(window_index) * heads + head) * tokens + query) * tokens + key;
                    mask[flat] = allowed
                        ? relative[(static_cast<std::size_t>(head) * tokens +
                                    query) * tokens + key]
                        : -1.0e9F;
                }
            }
        }
    }
    return mask;
}

}  // namespace

GroundingDinoModel::GroundingDinoModel(
    GroundingDinoWeights& weights,
    GgmlRuntime& runtime)
    : weights_(weights), runtime_(runtime) {}

GroundingDinoModel::SwinOutput GroundingDinoModel::run_swin(
    const std::vector<float>& image,
    const std::uint32_t height,
    const std::uint32_t width) {
    GraphBuilder graph;
    auto* context = graph.context();
    const auto& config = weights_.config();
    auto* current = graph.float_input(
        {static_cast<std::int64_t>(width), static_cast<std::int64_t>(height), 3, 1},
        image);
    current = ggml_pad(context, current, (4 - width % 4) % 4, (4 - height % 4) % 4, 0, 0);
    current = ggml_conv_2d(
        context,
        weights_.tensor("model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.weight"),
        current, 4, 4, 0, 0, 1, 1);
    current = ggml_add(
        context, current,
        ggml_reshape_4d(
            context,
            weights_.tensor("model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection.bias"),
            1, 1, config.swin_embedding, 1));
    current = ggml_cont(
        context, ggml_permute(context, current, 1, 2, 0, 3));
    current = layer_norm(
        context, weights_, current,
        "model.backbone.conv_encoder.model.embeddings.norm");

    int stage_width = static_cast<int>(current->ne[1]);
    int stage_height = static_cast<int>(current->ne[2]);
    std::uint32_t channels = config.swin_embedding;
    SwinOutput output;
    for (unsigned int stage = 0; stage < 4; ++stage) {
        const int heads = static_cast<int>(config.swin_heads[stage]);
        const int window = static_cast<int>(config.swin_window);
        const bool may_shift = std::min(stage_width, stage_height) > window;
        const int padded_width =
            (stage_width + window - 1) / window * window;
        const int padded_height =
            (stage_height + window - 1) / window * window;
        const int window_count =
            (padded_width / window) * (padded_height / window);
        for (unsigned int block = 0;
             block < config.swin_depths[stage]; ++block) {
            const std::string prefix =
                "model.backbone.conv_encoder.model.encoder.layers." +
                std::to_string(stage) + ".blocks." + std::to_string(block);
            const int shift =
                (block % 2 == 1 && may_shift) ? window / 2 : 0;
            auto* normalized = layer_norm(
                context, weights_, current, prefix + ".layernorm_before");
            auto* padded = ggml_pad(
                context, normalized, 0,
                padded_width - stage_width, padded_height - stage_height, 0);
            auto* flat = ggml_reshape_2d(
                context, padded, channels,
                static_cast<std::int64_t>(padded_width) * padded_height);
            if (shift != 0) {
                auto* rolled = graph.integer_input(
                    static_cast<std::int64_t>(padded_width) * padded_height,
                    roll_indices(padded_width, padded_height, -shift));
                flat = ggml_get_rows(context, flat, rolled);
            }
            auto* spatial = ggml_reshape_4d(
                context, flat, channels, padded_width, padded_height, 1);
            auto* query = linear(
                context, weights_, spatial, prefix + ".attention.self.query");
            auto* key = linear(
                context, weights_, spatial, prefix + ".attention.self.key");
            auto* value = linear(
                context, weights_, spatial, prefix + ".attention.self.value");
            auto* partitioned_query = window_partition(
                context, query, padded_width, padded_height, window);
            auto* partitioned_key = window_partition(
                context, key, padded_width, padded_height, window);
            auto* partitioned_value = window_partition(
                context, value, padded_width, padded_height, window);
            const auto relative = expand_relative_bias(
                tensor_values(weights_.tensor(
                    prefix + ".attention.self.relative_position_bias_table")),
                window, heads);
            const auto mask = window_mask(
                relative, padded_width, padded_height, window, shift, heads);
            auto* attention_mask = graph.float_input(
                {static_cast<std::int64_t>(window * window),
                 static_cast<std::int64_t>(window * window),
                 heads, window_count},
                mask);
            auto* attended = plain_attention(
                context,
                ggml_reshape_2d(context, partitioned_query, channels,
                    static_cast<std::int64_t>(window * window) * window_count),
                ggml_reshape_2d(context, partitioned_key, channels,
                    static_cast<std::int64_t>(window * window) * window_count),
                ggml_reshape_2d(context, partitioned_value, channels,
                    static_cast<std::int64_t>(window * window) * window_count),
                attention_mask, heads, window_count);
            attended = ggml_reshape_4d(
                context, attended, channels, window, window, window_count);
            auto* projected = linear(
                context, weights_, attended,
                prefix + ".attention.output.dense");
            auto* unpartitioned = window_unpartition(
                context, projected, padded_width, padded_height, window);
            flat = ggml_reshape_2d(
                context, unpartitioned, channels,
                static_cast<std::int64_t>(padded_width) * padded_height);
            if (shift != 0) {
                auto* inverse = graph.integer_input(
                    static_cast<std::int64_t>(padded_width) * padded_height,
                    roll_indices(padded_width, padded_height, shift));
                flat = ggml_get_rows(context, flat, inverse);
            }
            unpartitioned = ggml_reshape_4d(
                context, flat, channels, padded_width, padded_height, 1);
            if (padded_width != stage_width || padded_height != stage_height)
                unpartitioned = ggml_cont(context, ggml_view_4d(
                    context, unpartitioned, channels, stage_width,
                    stage_height, 1, unpartitioned->nb[1],
                    unpartitioned->nb[2], unpartitioned->nb[3], 0));
            current = ggml_add(context, current, unpartitioned);

            normalized = layer_norm(
                context, weights_, current, prefix + ".layernorm_after");
            auto* hidden = linear(
                context, weights_, normalized, prefix + ".intermediate.dense");
            hidden = ggml_gelu_erf(context, hidden);
            current = linear(
                context, weights_, hidden, prefix + ".output.dense", current);
        }

        if (stage > 0 && stage < 4) {
            const auto index = static_cast<std::size_t>(stage - 1);
            output.heights[index] = stage_height;
            output.widths[index] = stage_width;
            auto* normed = layer_norm(
                context, weights_, current,
                "model.backbone.conv_encoder.model.hidden_states_norms.stage" +
                    std::to_string(stage + 1));
            graph.output("stage" + std::to_string(stage), normed);
        }
        if (stage == 3) break;

        const int next_width = (stage_width + 1) / 2;
        const int next_height = (stage_height + 1) / 2;
        auto* padded_merge = ggml_pad(context, current, 0, next_width * 2 - stage_width, next_height * 2 - stage_height, 0);
        auto* source = ggml_reshape_2d(
            context, padded_merge, channels, next_width * next_height * 4);
        ggml_tensor* merged = nullptr;
        for (const int dx : {0, 1}) {
            for (const int dy : {0, 1}) {
                std::vector<std::int32_t> indices;
                indices.reserve(static_cast<std::size_t>(next_width) * next_height);
                for (int y = 0; y < next_height; ++y)
                    for (int x = 0; x < next_width; ++x) {
                        int source_x = x * 2 + dx;
                        int source_y = y * 2 + dy;
                        indices.push_back(
                            static_cast<std::int32_t>(source_y * next_width * 2 + source_x));
                    }
                auto* selected = ggml_get_rows(
                    context, source,
                    graph.integer_input(
                        static_cast<std::int64_t>(indices.size()), indices));
                merged = merged == nullptr
                    ? selected
                    : ggml_concat(context, merged, selected, 0);
            }
        }
        merged = layer_norm(
            context, weights_, merged,
            "model.backbone.conv_encoder.model.encoder.layers." +
                std::to_string(stage) + ".downsample.norm");
        current = linear(
            context, weights_, merged,
            "model.backbone.conv_encoder.model.encoder.layers." +
                std::to_string(stage) + ".downsample.reduction");
        current = ggml_reshape_4d(
            context, current, channels * 2, next_width, next_height, 1);
        stage_width = next_width;
        stage_height = next_height;
        channels *= 2;
    }

    auto result = graph.run(runtime_);
    for (unsigned int index = 0; index < 3; ++index) {
        const std::string name = "stage" + std::to_string(index + 1);
        output.maps[index] = std::move(result.at(name));
    }
    return output;
}

std::vector<float> GroundingDinoModel::run_text(
    const std::vector<std::int32_t>& tokens,
    const std::vector<std::int32_t>& positions,
    std::vector<float>& phrase_mask,
    std::vector<float>& text_position) {
    GraphBuilder graph;
    auto* context = graph.context();
    const auto token_count = static_cast<std::int64_t>(tokens.size());
    const auto text_dimension =
        static_cast<std::int64_t>(weights_.config().text_dimension);
    const int heads = static_cast<int>(text_dimension / 64);
    auto* ids = graph.integer_input(token_count, tokens);
    auto* position_ids = graph.integer_input(token_count, positions);
    auto* words = ggml_get_rows(
        context,
        weights_.tensor("model.text_backbone.embeddings.word_embeddings.weight"),
        ids);
    auto* positions_tensor = ggml_get_rows(
        context,
        weights_.tensor("model.text_backbone.embeddings.position_embeddings.weight"),
        position_ids);
    auto* input = ggml_add(context, words, positions_tensor);
    input = ggml_add(
        context, input,
        ggml_view_1d(
            context,
            weights_.tensor("model.text_backbone.embeddings.token_type_embeddings.weight"),
            text_dimension, 0));
    input = layer_norm(
        context, weights_, input,
        "model.text_backbone.embeddings.LayerNorm", 1e-12F);
    auto* mask = graph.float_input(
        {token_count, token_count}, phrase_mask);
    for (unsigned int layer = 0;
         layer < weights_.config().text_layers; ++layer) {
        const std::string prefix =
            "model.text_backbone.encoder.layer." + std::to_string(layer);
        auto* query = linear(
            context, weights_, input, prefix + ".attention.self.query");
        auto* key = linear(
            context, weights_, input, prefix + ".attention.self.key");
        auto* value = linear(
            context, weights_, input, prefix + ".attention.self.value");
        auto* attended = plain_attention(
            context, query, key, value, mask, heads);
        attended = linear(
            context, weights_, attended,
            prefix + ".attention.output.dense", input);
        input = layer_norm(
            context, weights_, attended,
            prefix + ".attention.output.LayerNorm", 1e-12F);
        auto* hidden = linear(
            context, weights_, input, prefix + ".intermediate.dense");
        hidden = ggml_gelu_erf(context, hidden);
        input = linear(
            context, weights_, hidden, prefix + ".output.dense", input);
        input = layer_norm(
            context, weights_, input, prefix + ".output.LayerNorm", 1e-12F);
    }
    auto* projected = linear(
        context, weights_, input, "model.text_projection");
    (void)text_position;
    graph.output("text", projected);
    auto result = graph.run(runtime_);
    return result.at("text");
}

GroundingDinoHeadOutput GroundingDinoModel::run(
    const std::vector<float>& chw_image,
    const std::uint32_t height,
    const std::uint32_t width,
    const std::vector<std::int32_t>& tokens) {
    require(!tokens.empty() && tokens.size() <= 255,
            "invalid token sequence length");
    const auto swin = run_swin(chw_image, height, width);

    const std::size_t token_count = tokens.size();
    std::vector<float> phrase_mask(token_count * token_count, -1.0e9F);
    std::vector<std::int32_t> positions(token_count, 0);
    for (std::size_t i = 0; i < token_count; ++i)
        phrase_mask[i * token_count + i] = 0.0F;
    std::size_t previous_separator = 0;
    auto is_separator = [](const std::int32_t id) {
        return id == 101 || id == 102 || id == 1012 || id == 1029;
    };
    for (std::size_t column = 0; column < token_count; ++column) {
        if (!is_separator(tokens[column])) continue;
        if (column != 0 && column + 1U != token_count) {
            for (std::size_t row = previous_separator + 1U;
                 row <= column; ++row) {
                positions[row] = static_cast<std::int32_t>(
                    row - previous_separator - 1U);
                for (std::size_t key = previous_separator + 1U;
                     key <= column; ++key)
                    phrase_mask[row * token_count + key] = 0.0F;
            }
        }
        previous_separator = column;
    }
    const auto text_dimension =
        static_cast<std::size_t>(weights_.config().model_dimension);
    std::vector<float> text_position(token_count * text_dimension, 0.0F);
    for (std::size_t token = 0; token < token_count; ++token) {
        const double coordinate = positions[token];
        for (std::size_t i = 0; i < text_dimension; ++i) {
            const double frequency = std::pow(
                10000.0, 2.0 * static_cast<double>(i / 2U) /
                             static_cast<double>(text_dimension));
            const double angle = coordinate * 2.0 * k_pi / frequency;
            text_position[token * text_dimension + i] =
                static_cast<float>(i % 2U == 0U ? std::sin(angle)
                                                : std::cos(angle));
        }
    }
    const auto text = run_text(
        tokens, positions, phrase_mask, text_position);

    int level_heights[4] = {
        swin.heights[0], swin.heights[1], swin.heights[2], 0};
    int level_widths[4] = {
        swin.widths[0], swin.widths[1], swin.widths[2], 0};
    level_heights[3] = (level_heights[2] - 1) / 2 + 1;
    level_widths[3] = (level_widths[2] - 1) / 2 + 1;
    std::size_t level_tokens[4] = {};
    std::size_t total_tokens = 0;
    for (int level = 0; level < 4; ++level) {
        level_tokens[level] = static_cast<std::size_t>(
            level_heights[level]) * level_widths[level];
        total_tokens += level_tokens[level];
    }
    const auto dimension =
        static_cast<std::size_t>(weights_.config().model_dimension);

    // Positions, references, validity and reference-box logits are all fixed
    // by the four feature-map sizes.
    std::vector<float> geometry_position(total_tokens * dimension, 0.0F);
    std::vector<float> geometry_references(total_tokens * 2, 0.0F);
    std::vector<float> geometry_valid(dimension * total_tokens, 0.0F);
    std::vector<float> proposals(total_tokens * 4, 0.0F);
    std::vector<float> deform_width(total_tokens * 2, 0.0F);
    auto level_embedding = tensor_values(
        weights_.tensor("model.level_embed"));
    std::size_t row = 0;
    const std::size_t half = dimension / 2U;
    for (int level = 0; level < 4; ++level) {
        const int map_height = level_heights[level];
        const int map_width = level_widths[level];
        for (int y = 0; y < map_height; ++y) {
            const double y_angle =
                static_cast<double>(y + 1) /
                (static_cast<double>(map_height) + 1.0e-6) * 2.0 * k_pi;
            for (int x = 0; x < map_width; ++x, ++row) {
                const double x_angle =
                    static_cast<double>(x + 1) /
                    (static_cast<double>(map_width) + 1.0e-6) * 2.0 * k_pi;
                for (std::size_t i = 0; i < half; ++i) {
                    const double frequency = std::pow(
                        20.0, 2.0 * static_cast<double>(i / 2U) /
                                  static_cast<double>(half));
                    const float y_value = static_cast<float>(
                        i % 2U == 0U ? std::sin(y_angle / frequency)
                                     : std::cos(y_angle / frequency));
                    const float x_value = static_cast<float>(
                        i % 2U == 0U ? std::sin(x_angle / frequency)
                                     : std::cos(x_angle / frequency));
                    geometry_position[row * dimension + i] =
                        y_value + level_embedding[
                            static_cast<std::size_t>(level) * dimension + i];
                    geometry_position[row * dimension + half + i] =
                        x_value + level_embedding[
                            static_cast<std::size_t>(level) * dimension +
                            half + i];
                }
                const float centre_x =
                    (static_cast<float>(x) + 0.5F) /
                    static_cast<float>(map_width);
                const float centre_y =
                    (static_cast<float>(y) + 0.5F) /
                    static_cast<float>(map_height);
                const float size = 0.05F * static_cast<float>(1 << level);
                geometry_references[row * 2] = centre_x;
                geometry_references[row * 2 + 1] = centre_y;
                deform_width[row * 2] = size;
                deform_width[row * 2 + 1] = size;
                const float candidate[4] = {
                    centre_x, centre_y, size, size};
                bool valid = true;
                for (const float value : candidate)
                    valid = valid && value > 0.01F && value < 0.99F;
                std::fill_n(
                    geometry_valid.begin() +
                        static_cast<std::ptrdiff_t>(row * dimension),
                    dimension, valid ? 1.0F : 0.0F);
                for (int part = 0; part < 4; ++part) {
                    const float value = candidate[part];
                    proposals[row * 4 + static_cast<std::size_t>(part)] =
                        valid
                            ? std::log(value / (1.0F - value))
                            : std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    // Build the four visual sources and the complete image-text encoder.
    GraphBuilder graph;
    auto* context = graph.context();
    const auto input_map = [&](const std::size_t level) {
        const auto channels = static_cast<std::int64_t>([&] {
            if (level < 3)
                return weights_.config().swin_embedding << (level + 1U);
            return weights_.config().swin_embedding << 3U;
        }());
        auto* map = graph.float_input(
            {channels,
             static_cast<std::int64_t>(level_widths[level]),
             static_cast<std::int64_t>(level_heights[level])},
            swin.maps[level < 3 ? level : 2]);
        return ggml_cont(context, ggml_permute(context, map, 2, 0, 1, 3));
    };
    std::vector<ggml_tensor*> sources;
    for (unsigned int level = 0; level < 4; ++level) {
        auto* backbone_map = input_map(level == 3 ? 2 : level);
        const std::string prefix =
            "model.input_proj_vision." + std::to_string(level);
        auto* projected = level == 3
            ? ggml_conv_2d(
                  context, weights_.tensor(prefix + ".0.weight"),
                  backbone_map, 2, 2, 1, 1, 1, 1)
            : ggml_conv_2d(
                  context, weights_.tensor(prefix + ".0.weight"),
                  backbone_map, 1, 1, 0, 0, 1, 1);
        projected = ggml_add(
            context, projected, ggml_reshape_4d(context, weights_.tensor(prefix + ".0.bias"), 1, 1, dimension, 1));
        const auto map_tokens =
            static_cast<std::int64_t>(level_widths[level]) *
            level_heights[level];
        auto* grouped = ggml_reshape_4d(
            context, projected, 1, map_tokens,
            static_cast<std::int64_t>(dimension), 1);
        grouped = ggml_group_norm(context, grouped, 32, 1.0e-5F);
        grouped = ggml_mul(
            context, grouped, ggml_reshape_4d(context, weights_.tensor(prefix + ".1.weight"), 1, 1, dimension, 1));
        grouped = ggml_add(
            context, grouped, ggml_reshape_4d(context, weights_.tensor(prefix + ".1.bias"), 1, 1, dimension, 1));
        grouped = ggml_cont(context, ggml_permute(context, grouped, 2, 1, 0, 3));
        auto* source = ggml_reshape_2d(
            context, grouped, static_cast<std::int64_t>(dimension),
            map_tokens);
        sources.push_back(source);
    }
    auto* vision = sources[0];
    for (std::size_t level = 1; level < sources.size(); ++level)
        vision = ggml_concat(context, vision, sources[level], 1);
    require(vision->ne[1] == static_cast<std::int64_t>(total_tokens),
            "unexpected multi-level token count");

    auto* text_tensor = graph.float_input(
        {static_cast<std::int64_t>(dimension),
         static_cast<std::int64_t>(token_count)},
        text);
    auto* position_tensor = graph.float_input(
        {static_cast<std::int64_t>(dimension),
         static_cast<std::int64_t>(total_tokens)},
        geometry_position);
    auto* references_tensor = graph.float_input(
        {2, static_cast<std::int64_t>(total_tokens)},
        geometry_references);
    auto* deform_width_tensor = graph.float_input(
        {2, static_cast<std::int64_t>(total_tokens)}, deform_width);
    auto* text_position_tensor = graph.float_input(
        {static_cast<std::int64_t>(dimension),
         static_cast<std::int64_t>(token_count)},
        text_position);
    auto* phrase_mask_tensor = graph.float_input(
        {static_cast<std::int64_t>(token_count),
         static_cast<std::int64_t>(token_count)},
        phrase_mask);

    const auto deform_attention = [&](
        GraphBuilder& active_graph,
        ggml_tensor* query,
        ggml_tensor* values,
        ggml_tensor* offsets,
        ggml_tensor* attention,
        ggml_tensor* reference_points,
        ggml_tensor* reference_width,
        const int heads,
        const int points) {
        auto* context = active_graph.context();
        const auto head_dimension =
            static_cast<std::int64_t>(dimension / heads);
        auto* shaped_values = ggml_reshape_3d(
            context, values, head_dimension, heads,
            static_cast<std::int64_t>(total_tokens));
        auto* shaped_attention = ggml_reshape_4d(
            context, attention, 4 * points, heads,
            query->ne[1], 1);
        shaped_attention = ggml_soft_max(context, shaped_attention);
        auto* output = ggml_new_tensor_2d(
            context, GGML_TYPE_F32,
            static_cast<std::int64_t>(dimension), query->ne[1]);
        ggml_set_input(output);
        ggml_set_output(output);
        auto* centre = ggml_cont(context, ggml_view_2d(
            context, reference_points, 2, query->ne[1],
            reference_points->nb[1], 0));
        auto* size = ggml_cont(context, ggml_view_2d(
            context, reference_width, 2, query->ne[1],
            reference_width->nb[1], 0));
        for (int head = 0; head < heads; ++head) {
            ggml_tensor* head_output = nullptr;
            std::size_t level_offset = 0;
            for (int level = 0; level < 4; ++level) {
                const auto map_width =
                    static_cast<std::int64_t>(level_widths[level]);
                const auto map_height =
                    static_cast<std::int64_t>(level_heights[level]);
                const auto map_tokens = map_width * map_height;
                auto* level_values = ggml_cont(context, ggml_view_2d(
                    context, shaped_values, head_dimension, map_tokens,
                    shaped_values->nb[2],
                    static_cast<std::size_t>(head) * shaped_values->nb[1] +
                        level_offset * shaped_values->nb[2]));
                for (int point = 0; point < points; ++point) {
                    const std::size_t flat =
                        static_cast<std::size_t>(head) * 4 * points +
                        static_cast<std::size_t>(level) * points +
                        static_cast<std::size_t>(point);
                    auto* offset = ggml_cont(context, ggml_view_2d(
                        context, offsets, 2, query->ne[1], offsets->nb[1],
                        flat * 2U * sizeof(float)));
                    ggml_tensor* location = nullptr;
                    if (reference_points->ne[0] == 2) {
                        std::vector<float> inverse(query->ne[1] * 2);
                        for (std::size_t i = 0; i < inverse.size(); i += 2) {
                            inverse[i] = 1.F / map_width;
                            inverse[i + 1] = 1.F / map_height;
                        }
                        auto* normalizer = active_graph.float_input({2, query->ne[1]}, inverse);
                        location = ggml_add(context, centre, ggml_mul(context, offset, normalizer));
                    } else {
                        location = ggml_add(context, centre, ggml_mul(context,
                            ggml_scale(context, offset, 0.5F / points), size));
                    }
                    auto* x = ggml_cont(context, ggml_view_2d(
                        context, location, 1, query->ne[1],
                        location->nb[1], 0));
                    auto* y = ggml_cont(context, ggml_view_2d(
                        context, location, 1, query->ne[1],
                        location->nb[1], sizeof(float)));
                    auto* sample_x = ggml_scale_bias(
                        context, x, static_cast<float>(map_width), -0.5F);
                    auto* sample_y = ggml_scale_bias(
                        context, y, static_cast<float>(map_height), -0.5F);
                    auto* x0 = ggml_floor(context, sample_x);
                    auto* y0 = ggml_floor(context, sample_y);
                    auto* x1 = ggml_scale_bias(context, x0, 1.0F, 1.0F);
                    auto* y1 = ggml_scale_bias(context, y0, 1.0F, 1.0F);
                    auto* weight_x1 = ggml_sub(context, sample_x, x0);
                    auto* weight_y1 = ggml_sub(context, sample_y, y0);
                    auto* weight_x0 = ggml_scale_bias(
                        context, weight_x1, -1.0F, 1.0F);
                    auto* weight_y0 = ggml_scale_bias(
                        context, weight_y1, -1.0F, 1.0F);
                    const auto in_bounds = [&](
                        ggml_tensor* coordinate,
                        const std::int64_t limit) {
                        auto* nonnegative = ggml_step(
                            context,
                            ggml_scale_bias(
                                context, coordinate, 1.0F, 0.5F));
                        auto* below = ggml_step(
                            context,
                            ggml_scale_bias(
                                context, coordinate, -1.0F,
                                static_cast<float>(limit - 1) + 0.5F));
                        return ggml_mul(context, nonnegative, below);
                    };
                    auto* x0_valid = in_bounds(x0, map_width);
                    auto* x1_valid = in_bounds(x1, map_width);
                    auto* y0_valid = in_bounds(y0, map_height);
                    auto* y1_valid = in_bounds(y1, map_height);
                    auto* clamped_x0 = ggml_clamp(
                        context, ggml_cont(context, x0), 0.0F,
                        static_cast<float>(map_width - 1));
                    auto* clamped_x1 = ggml_clamp(
                        context, ggml_cont(context, x1), 0.0F,
                        static_cast<float>(map_width - 1));
                    auto* clamped_y0 = ggml_clamp(
                        context, ggml_cont(context, y0), 0.0F,
                        static_cast<float>(map_height - 1));
                    auto* clamped_y1 = ggml_clamp(
                        context, ggml_cont(context, y1), 0.0F,
                        static_cast<float>(map_height - 1));
                    struct Corner {
                        ggml_tensor* x;
                        ggml_tensor* y;
                        ggml_tensor* weight_x;
                        ggml_tensor* weight_y;
                        ggml_tensor* valid_x;
                        ggml_tensor* valid_y;
                    };
                    const Corner corners[4] = {
                        {clamped_x0, clamped_y0, weight_x0, weight_y0,
                         x0_valid, y0_valid},
                        {clamped_x1, clamped_y0, weight_x1, weight_y0,
                         x1_valid, y0_valid},
                        {clamped_x0, clamped_y1, weight_x0, weight_y1,
                         x0_valid, y1_valid},
                        {clamped_x1, clamped_y1, weight_x1, weight_y1,
                         x1_valid, y1_valid}};
                    ggml_tensor* sample = nullptr;
                    for (const Corner& corner : corners) {
                        auto* row_float = ggml_add(
                            context,
                            ggml_scale(
                                context, corner.y,
                                static_cast<float>(map_width)),
                            corner.x);
                        auto* row_integer = ggml_cast(
                            context,
                            ggml_reshape_1d(
                                context, row_float, query->ne[1]),
                            GGML_TYPE_I32);
                        auto* gathered = ggml_get_rows(
                            context, level_values, row_integer);
                        auto* weight = ggml_mul(
                            context,
                            ggml_mul(context, corner.weight_x, corner.weight_y),
                            ggml_mul(context, corner.valid_x, corner.valid_y));
                        auto* contribution = ggml_mul(context, gathered, weight);
                        sample = sample == nullptr
                            ? contribution
                            : ggml_add(context, sample, contribution);
                    }
                    auto* attention_weight = ggml_cont(context, ggml_view_2d(
                        context, shaped_attention, 1, query->ne[1],
                        shaped_attention->nb[2],
                        static_cast<std::size_t>(level * points + point) *
                            shaped_attention->nb[0] +
                            static_cast<std::size_t>(head) *
                                shaped_attention->nb[1]));
                    auto* weighted = ggml_mul(context, sample, attention_weight);
                    head_output = head_output == nullptr
                        ? weighted
                        : ggml_add(context, head_output, weighted);
                }
                level_offset += static_cast<std::size_t>(map_tokens);
            }
            output = ggml_set_2d(
                context, output, head_output, output->nb[1],
                static_cast<std::size_t>(head) *
                    static_cast<std::size_t>(head_dimension) * sizeof(float));
        }
        return ggml_reshape_2d(
            context, output, static_cast<std::int64_t>(dimension),
            query->ne[1]);
    };

    for (unsigned int layer = 0;
         layer < weights_.config().encoder_layers; ++layer) {
        const std::string prefix =
            "model.encoder.layers." + std::to_string(layer);
        const std::string fusion = prefix + ".fusion_layer.";
        auto* vision_norm = layer_norm(
            context, weights_, vision, fusion + "layer_norm_vision");
        auto* text_norm = layer_norm(
            context, weights_, text_tensor, fusion + "layer_norm_text");
        auto* vision_query = linear(
            context, weights_, vision_norm, fusion + "attn.vision_proj");
        auto* vision_values = linear(
            context, weights_, vision_norm,
            fusion + "attn.values_vision_proj");
        auto* text_key = linear(
            context, weights_, text_norm, fusion + "attn.text_proj");
        auto* text_values = linear(
            context, weights_, text_norm, fusion + "attn.values_text_proj");
        auto* vision_attended = plain_attention(
            context, vision_query, text_key, text_values, nullptr, 4);
        auto* text_query = linear(
            context, weights_, text_norm, fusion + "attn.text_proj");
        auto* text_attended = plain_attention(
            context, text_query, vision_query, vision_values, nullptr, 4);
        vision = ggml_add(context, vision_norm, ggml_mul(context,
            linear(context, weights_, vision_attended, fusion + "attn.out_vision_proj"),
            weights_.tensor(fusion + "vision_param")));
        text_tensor = ggml_add(context, text_norm, ggml_mul(context,
            linear(context, weights_, text_attended, fusion + "attn.out_text_proj"),
            weights_.tensor(fusion + "text_param")));

        const std::string enhancer = prefix + ".text_enhancer_layer.";
        auto* query_with_position = ggml_add(
            context, text_tensor, text_position_tensor);
        auto* self_query = linear(
            context, weights_, query_with_position,
            enhancer + "self_attn.query");
        auto* self_key = linear(
            context, weights_, query_with_position,
            enhancer + "self_attn.key");
        auto* self_value = linear(
            context, weights_, text_tensor,
            enhancer + "self_attn.value");
        auto* attended = plain_attention(
            context, self_query, self_key, self_value,
            phrase_mask_tensor,
            static_cast<int>(weights_.config().encoder_heads / 2U));
        text_tensor = linear(
            context, weights_, attended,
            enhancer + "self_attn.out_proj", text_tensor);
        text_tensor = layer_norm(
            context, weights_, text_tensor, enhancer + "layer_norm_before");
        auto* hidden = linear(
            context, weights_, text_tensor, enhancer + "fc1");
        hidden = ggml_relu(context, hidden);
        text_tensor = linear(
            context, weights_, hidden, enhancer + "fc2", text_tensor);
        text_tensor = layer_norm(
            context, weights_, text_tensor, enhancer + "layer_norm_after");

        const std::string deformable = prefix + ".deformable_layer.";
        auto* deform_query = ggml_add(
            context, vision, position_tensor);
        auto* values = linear(
            context, weights_, vision,
            deformable + "self_attn.value_proj");
        auto* offsets = linear(
            context, weights_, deform_query,
            deformable + "self_attn.sampling_offsets");
        auto* attention = linear(
            context, weights_, deform_query,
            deformable + "self_attn.attention_weights");
        auto* attended_vision = deform_attention(
            graph, deform_query, values, offsets, attention,
            references_tensor, deform_width_tensor,
            static_cast<int>(weights_.config().encoder_heads), 4);
        vision = linear(
            context, weights_, attended_vision,
            deformable + "self_attn.output_proj", vision);
        vision = layer_norm(
            context, weights_, vision,
            deformable + "self_attn_layer_norm");
        hidden = linear(context, weights_, vision, deformable + "fc1");
        hidden = ggml_relu(context, hidden);
        vision = linear(
            context, weights_, hidden, deformable + "fc2", vision);
        vision = layer_norm(
            context, weights_, vision, deformable + "final_layer_norm");
    }
    graph.output("vision", vision);
    graph.output("text", text_tensor);
    auto encoder = graph.run(runtime_);
    std::vector<float> encoded_vision = std::move(encoder.at("vision"));
    std::vector<float> encoded_text = std::move(encoder.at("text"));

    // Two-stage query selection.
    GraphBuilder selection_graph;
    context = selection_graph.context();
    auto* encoded_vision_tensor = selection_graph.float_input(
        {static_cast<std::int64_t>(dimension),
         static_cast<std::int64_t>(total_tokens)},
        encoded_vision);
    auto* encoded_text_tensor = selection_graph.float_input(
        {static_cast<std::int64_t>(dimension),
         static_cast<std::int64_t>(token_count)},
        encoded_text);
    auto* valid_selection = selection_graph.float_input(
        {static_cast<std::int64_t>(dimension),
         static_cast<std::int64_t>(total_tokens)},
        geometry_valid);
    auto* masked = ggml_mul(
        context, encoded_vision_tensor, valid_selection);
    auto* queries = linear(
        context, weights_, masked, "model.enc_output");
    queries = layer_norm(
        context, weights_, queries, "model.enc_output_norm");
    auto* classifications = ggml_mul_mat(
        context, encoded_text_tensor, queries);

    auto bbox = queries;
    for (int layer = 0; layer < 3; ++layer) {
        bbox = linear(
            context, weights_, bbox,
            "model.encoder_output_bbox_embed.layers." + std::to_string(layer));
        if (layer < 2) bbox = ggml_relu(context, bbox);
    }
    selection_graph.output("classification", classifications);
    selection_graph.output("bbox", bbox);
    auto selection = selection_graph.run(runtime_);
    const auto& classification = selection.at("classification");
    const auto& selection_boxes = selection.at("bbox");
    std::vector<float> best(total_tokens);
    for (std::size_t token_row = 0; token_row < total_tokens; ++token_row) {
        float value = -std::numeric_limits<float>::infinity();
        for (std::size_t token = 0; token < token_count; ++token)
            value = std::max(
                value,
                classification[token_row * token_count + token]);
        best[token_row] = value;
    }
    std::vector<std::size_t> order(total_tokens);
    std::iota(order.begin(), order.end(), 0U);
    const auto query_count = static_cast<std::size_t>(
        weights_.config().queries);
    const std::size_t selected = std::min(query_count, total_tokens);
    std::partial_sort(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(selected),
        order.end(), [&](const std::size_t left, const std::size_t right) {
            if (best[left] != best[right]) return best[left] > best[right];
            return left < right;
        });
    std::vector<float> reference_points(query_count * 4);
    for (std::size_t query = 0; query < query_count; ++query) {
        const std::size_t source =
            order[std::min(query, selected - 1U)];
        for (std::size_t part = 0; part < 4; ++part) {
            const float delta =
                selection_boxes[source * 4 + part];
            const float proposal = proposals[source * 4 + part];
            reference_points[query * 4 + part] =
                static_cast<float>(
                    1.0 / (1.0 + std::exp(-static_cast<double>(
                                                 delta + proposal))));
        }
    }

    // Decoder layers. The iterative box refinement depends on a host-side
    // sigmoid, so each layer is a compact GGML graph.
    std::vector<float> state = tensor_values(
        weights_.tensor("model.query_position_embeddings.weight"));
    const auto make_sine = [&](const std::vector<float>& boxes) {
        std::vector<float> result(query_count * dimension * 2U, 0.0F);
        const std::size_t part_dimension = dimension / 2;
        for (std::size_t query = 0; query < query_count; ++query) {
            const float values[4] = {
                boxes[query * 4 + 1], boxes[query * 4],
                boxes[query * 4 + 2], boxes[query * 4 + 3]};
            for (int part = 0; part < 4; ++part)
                for (std::size_t i = 0; i < part_dimension; ++i) {
                    const double frequency = std::pow(
                        10000.0,
                        2.0 * static_cast<double>(i / 2U) /
                            static_cast<double>(part_dimension));
                    const double angle =
                        static_cast<double>(values[part]) * 2.0 * k_pi /
                        frequency;
                    const std::size_t offset =
                        query * dimension * 2U +
                        static_cast<std::size_t>(part) * part_dimension + i;
                    result[offset] = static_cast<float>(
                        i % 2U == 0U ? std::sin(angle) : std::cos(angle));
                }
        }
        return result;
    };
    auto logit = [](float value) {
        value = std::clamp(value, 1.0e-5F, 1.0F - 1.0e-5F);
        return std::log(value / (1.0F - value));
    };

    for (unsigned int layer = 0;
         layer < weights_.config().decoder_layers; ++layer) {
        const std::string prefix =
            "model.decoder.layers." + std::to_string(layer);
        GraphBuilder decoder_graph;
        context = decoder_graph.context();
        auto* state_tensor = decoder_graph.float_input(
            {static_cast<std::int64_t>(dimension),
             static_cast<std::int64_t>(query_count)},
            state);
        auto* vision_tensor = decoder_graph.float_input(
            {static_cast<std::int64_t>(dimension),
             static_cast<std::int64_t>(total_tokens)},
            encoded_vision);
        auto* decoder_text = decoder_graph.float_input(
            {static_cast<std::int64_t>(dimension),
             static_cast<std::int64_t>(token_count)},
            encoded_text);
        auto* reference_tensor = decoder_graph.float_input(
            {4, static_cast<std::int64_t>(query_count)}, reference_points);
        auto* sine_tensor = decoder_graph.float_input(
            {static_cast<std::int64_t>(dimension * 2U),
             static_cast<std::int64_t>(query_count)},
            make_sine(reference_points));
        auto* query_position = sine_tensor;
        for (int part = 0; part < 2; ++part) {
            query_position = linear(
                context, weights_, query_position,
                "model.decoder.reference_points_head.layers." +
                    std::to_string(part));
            if (part == 0) query_position = ggml_relu(context, query_position);
        }
        auto* position_query = ggml_add(
            context, state_tensor, query_position);
        auto* self_query = linear(
            context, weights_, position_query, prefix + ".self_attn.query");
        auto* self_key = linear(
            context, weights_, position_query, prefix + ".self_attn.key");
        auto* self_value = linear(
            context, weights_, state_tensor, prefix + ".self_attn.value");
        auto* attended = plain_attention(
            context, self_query, self_key, self_value, nullptr,
            static_cast<int>(weights_.config().decoder_heads));
        state_tensor = linear(
            context, weights_, attended, prefix + ".self_attn.out_proj",
            state_tensor);
        state_tensor = layer_norm(
            context, weights_, state_tensor, prefix + ".self_attn_layer_norm");

        position_query = ggml_add(context, state_tensor, query_position);
        auto* text_query = linear(
            context, weights_, position_query,
            prefix + ".encoder_attn_text.query");
        auto* text_key = linear(
            context, weights_, decoder_text,
            prefix + ".encoder_attn_text.key");
        auto* text_value = linear(
            context, weights_, decoder_text,
            prefix + ".encoder_attn_text.value");
        attended = plain_attention(
            context, text_query, text_key, text_value, nullptr,
            static_cast<int>(weights_.config().decoder_heads));
        state_tensor = linear(
            context, weights_, attended,
            prefix + ".encoder_attn_text.out_proj", state_tensor);
        state_tensor = layer_norm(
            context, weights_, state_tensor,
            prefix + ".encoder_attn_text_layer_norm");

        position_query = ggml_add(context, state_tensor, query_position);
        auto* values = linear(
            context, weights_, vision_tensor,
            prefix + ".encoder_attn.value_proj");
        auto* offsets = linear(
            context, weights_, position_query,
            prefix + ".encoder_attn.sampling_offsets");
        auto* attention = linear(
            context, weights_, position_query,
            prefix + ".encoder_attn.attention_weights");
        auto* size = ggml_cont(context, ggml_view_2d(
            context, reference_tensor, 2,
            static_cast<std::int64_t>(query_count), reference_tensor->nb[1],
            2U * sizeof(float)));
        auto* deform_output = deform_attention(decoder_graph, position_query,
            values, offsets, attention, reference_tensor, size,
            static_cast<int>(weights_.config().decoder_heads), 4);
        attended = linear(
            context, weights_,
            ggml_reshape_2d(
                context, deform_output, static_cast<std::int64_t>(dimension),
                static_cast<std::int64_t>(query_count)),
            prefix + ".encoder_attn.output_proj", state_tensor);
        state_tensor = layer_norm(
            context, weights_, attended,
            prefix + ".encoder_attn_layer_norm");
        auto* hidden = linear(context, weights_, state_tensor, prefix + ".fc1");
        hidden = ggml_relu(context, hidden);
        state_tensor = linear(
            context, weights_, hidden, prefix + ".fc2", state_tensor);
        state_tensor = layer_norm(
            context, weights_, state_tensor, prefix + ".final_layer_norm");
        auto* delta = state_tensor;
        for (int part = 0; part < 3; ++part) {
            delta = linear(
                context, weights_, delta,
                "bbox_embed.0.layers." + std::to_string(part));
            if (part < 2) delta = ggml_relu(context, delta);
        }
        decoder_graph.output("state", state_tensor);
        decoder_graph.output("delta", delta);
        auto decoded = decoder_graph.run(runtime_);
        state = std::move(decoded.at("state"));
        const auto& deltas = decoded.at("delta");
        if (layer + 1U < weights_.config().decoder_layers) {
            for (std::size_t i = 0; i < reference_points.size(); ++i)
                reference_points[i] = static_cast<float>(
                    1.0 / (1.0 + std::exp(-static_cast<double>(
                                           deltas[i] +
                                           logit(reference_points[i])))));
        } else {
            GroundingDinoHeadOutput output;
            output.queries = query_count;
            output.tokens = token_count;
            GraphBuilder head_graph;
            auto* head_context = head_graph.context();
            auto* final_state = head_graph.float_input(
                {static_cast<std::int64_t>(dimension),
                 static_cast<std::int64_t>(query_count)},
                state);
            auto* final_text = head_graph.float_input(
                {static_cast<std::int64_t>(dimension),
                 static_cast<std::int64_t>(token_count)},
                encoded_text);
            auto* normed = layer_norm(
                head_context, weights_, final_state, "model.decoder.layer_norm");
            auto* logits = ggml_mul_mat(head_context, final_text, normed);
            auto* final_delta = normed;
            for (int part = 0; part < 3; ++part) {
                final_delta = linear(
                    head_context, weights_, final_delta,
                    "bbox_embed.0.layers." + std::to_string(part));
                if (part < 2)
                    final_delta = ggml_relu(head_context, final_delta);
            }
            head_graph.output("logits", logits);
            head_graph.output("boxes", final_delta);
            auto head = head_graph.run(runtime_);
            output.logits = std::move(head.at("logits"));
            output.boxes.resize(query_count * 4U);
            const auto& raw_boxes = head.at("boxes");
            for (std::size_t i = 0; i < output.boxes.size(); ++i)
                output.boxes[i] = static_cast<float>(
                    1.0 / (1.0 + std::exp(-static_cast<double>(
                                           raw_boxes[i] +
                                           logit(reference_points[i])))));
            return output;
        }
    }
    throw std::runtime_error("Grounding DINO graph: decoder produced no output");
}

}  // namespace photara::inference::detail
