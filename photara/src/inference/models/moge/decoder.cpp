#include "decoder.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>

namespace photara::inference::detail {
namespace {
constexpr std::array<int, 5> kChannels{768, 256, 128, 64, 32};
struct Input {
    ggml_tensor* tensor = nullptr;
    std::vector<float> floats;
    std::vector<std::int32_t> indices;
};
struct Map { ggml_tensor* tensor = nullptr; int width = 0, height = 0; };

class DecoderGraph {
public:
    DecoderGraph(ggml_context* context, MogeWeights& weights, bool capture)
        : ctx(context), weights(weights), capture(capture),
          graph(ggml_new_graph_custom(context, 16384, false)) {}
    ggml_tensor* float_input(int channels, int pixels, std::vector<float> values) {
        auto* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, pixels);
        ggml_set_input(tensor);
        inputs.push_back({tensor, std::move(values), {}});
        return tensor;
    }
    void trace(const std::string& name, ggml_tensor* tensor) {
        if (capture) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
            stages.emplace_back(name, tensor);
        }
    }
    ggml_tensor* weight(const std::string& name, std::initializer_list<std::int64_t> shape) {
        auto* tensor = weights.tensor(name);
        int axis = 0;
        for (const auto size : shape) {
            if (tensor->ne[axis++] != size)
                throw std::invalid_argument("MoGe decoder weight shape: " + name);
        }
        for (; axis < GGML_MAX_DIMS; ++axis)
            if (tensor->ne[axis] != 1) throw std::invalid_argument("MoGe decoder weight rank: " + name);
        return tensor;
    }
    ggml_tensor* matmul(ggml_tensor* matrix, ggml_tensor* data) {
        auto* result = ggml_mul_mat(ctx, matrix, data);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        return result;
    }
    ggml_tensor* linear(ggml_tensor* data, const std::string& prefix, int out) {
        return ggml_add(ctx, matmul(weight(prefix + ".weight", {data->ne[0], out}), data),
            weight(prefix + ".bias", {out}));
    }
    ggml_tensor* gather_indices(int width, int height, bool deconvolution) {
        const auto key = std::make_tuple(width, height, deconvolution);
        if (auto found = index_cache.find(key); found != index_cache.end()) return found->second;
        const int out_width = deconvolution ? width * 2 : width + 2;
        const int out_height = deconvolution ? height * 2 : height + 2;
        std::vector<std::int32_t> indices(out_width * out_height);
        for (int y = 0; y < out_height; ++y)
            for (int x = 0; x < out_width; ++x) {
                indices[y * out_width + x] = deconvolution
                    ? 4 * ((y / 2) * width + x / 2) + (y % 2) * 2 + x % 2
                    : std::clamp(y - 1, 0, height - 1) * width + std::clamp(x - 1, 0, width - 1);
            }
        auto* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, indices.size());
        ggml_set_input(tensor);
        inputs.push_back({tensor, {}, std::move(indices)});
        index_cache.emplace(key, tensor);
        return tensor;
    }
    ggml_tensor* spatial(Map map) {
        return ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, map.tensor)),
            map.width, map.height, map.tensor->ne[0]);
    }
    Map token_major(ggml_tensor* spatial) {
        auto* flat = ggml_reshape_2d(ctx, spatial, spatial->ne[0] * spatial->ne[1], spatial->ne[2]);
        return {ggml_cont(ctx, ggml_transpose(ctx, flat)),
            static_cast<int>(spatial->ne[0]), static_cast<int>(spatial->ne[1])};
    }
    Map conv(Map input, const std::string& prefix, int output_channels, int kernel) {
        const int channels = static_cast<int>(input.tensor->ne[0]);
        auto* matrix = weight(prefix + ".weight", {kernel, kernel, channels, output_channels});
        auto* bias = weight(prefix + ".bias", {output_channels});
        auto* data = input.tensor;
        if (kernel == 3) {
            auto* padded = ggml_get_rows(ctx, data, gather_indices(input.width, input.height, false));
            auto* image = spatial({padded, input.width + 2, input.height + 2});
            auto* patches = ggml_im2col(ctx, matrix, image, 1, 1, 0, 0, 1, 1, true, GGML_TYPE_F32);
            data = ggml_reshape_2d(ctx, patches, channels * kernel * kernel, input.width * input.height);
        }
        matrix = ggml_reshape_2d(ctx, matrix, channels * kernel * kernel, output_channels);
        return {ggml_add(ctx, matmul(matrix, data), bias), input.width, input.height};
    }
    Map residual(Map input, const std::string& prefix) {
        const int channels = static_cast<int>(input.tensor->ne[0]);
        auto first = conv({ggml_relu(ctx, input.tensor), input.width, input.height}, prefix + ".layers.2", channels, 3);
        auto second = conv({ggml_relu(ctx, first.tensor), first.width, first.height}, prefix + ".layers.5", channels, 3);
        return {ggml_add(ctx, input.tensor, second.tensor), input.width, input.height};
    }
    Map resample(Map input, const std::string& prefix, int output_channels, bool bilinear) {
        Map output;
        if (bilinear) {
            auto* resized = ggml_interpolate(ctx, spatial(input), input.width * 2, input.height * 2,
                input.tensor->ne[0], 1, GGML_SCALE_MODE_BILINEAR);
            output = token_major(resized);
        } else {
            // Kernel 2/stride 2 has no overlapping contributions. Project all
            // four phases in F32, then gather them into output pixel order.
            const int channels = static_cast<int>(input.tensor->ne[0]);
            auto* kernel = weight(prefix + ".0.weight", {2, 2, output_channels, channels});
            auto* matrix = ggml_cont(ctx, ggml_transpose(ctx,
                ggml_reshape_2d(ctx, kernel, 4 * output_channels, channels)));
            auto* phases = matmul(matrix, input.tensor);
            phases = ggml_reshape_3d(ctx, phases, 4, output_channels, input.width * input.height);
            phases = ggml_cont(ctx, ggml_permute(ctx, phases, 1, 0, 2, 3));
            phases = ggml_reshape_2d(ctx, phases, output_channels, 4 * input.width * input.height);
            auto* pixels = ggml_get_rows(ctx, phases, gather_indices(input.width, input.height, true));
            output = {ggml_add(ctx, pixels, weight(prefix + ".0.bias", {output_channels})),
                input.width * 2, input.height * 2};
        }
        trace(prefix + ".0", output.tensor);
        output = conv(output, prefix + ".1", output_channels, 3);
        trace(prefix + ".1", output.tensor);
        return output;
    }
    std::array<Map, 5> stack(const std::string& prefix, const std::array<Map, 5>& source, int final_channels) {
        std::array<Map, 5> output;
        Map current;
        for (int level = 0; level < 5; ++level) {
            const auto suffix = std::to_string(level);
            auto projected = conv(source[level], prefix + ".input_blocks." + suffix, kChannels[level], 1);
            trace(prefix + ".input." + suffix, projected.tensor);
            if (level == 0) current = projected;
            else current.tensor = ggml_add(ctx, current.tensor, projected.tensor);
            if (level >= 1 && level <= 3)
                current = residual(current, prefix + ".res_blocks." + suffix + ".0");
            trace(prefix + ".residual." + suffix, current.tensor);
            output[level] = level == 4 && final_channels != 0
                ? conv(current, prefix + ".output_blocks.4", final_channels, 1) : current;
            trace(prefix + ".output." + suffix, output[level].tensor);
            if (level < 4) current = resample(current, prefix + ".resamplers." + suffix, kChannels[level + 1], level == 3);
        }
        return output;
    }
    void upload() {
        for (const auto& input : inputs) {
            if (!input.floats.empty())
                ggml_backend_tensor_set(input.tensor, input.floats.data(), 0, input.floats.size() * sizeof(float));
            else ggml_backend_tensor_set(input.tensor, input.indices.data(), 0, input.indices.size() * sizeof(std::int32_t));
        }
    }
    ggml_context* ctx;
    MogeWeights& weights;
    bool capture;
    ggml_cgraph* graph;
    std::vector<Input> inputs;
    std::vector<std::pair<std::string, ggml_tensor*>> stages;
    std::map<std::tuple<int, int, bool>, ggml_tensor*> index_cache;
};
std::vector<float> uv(int width, int height, float aspect_ratio) {
    const double span_y = 1.0 / std::sqrt(1.0 + static_cast<double>(aspect_ratio) * aspect_ratio);
    const double span_x = aspect_ratio * span_y;
    std::vector<float> values(static_cast<std::size_t>(width) * height * 2);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            values[2 * (y * width + x)] = static_cast<float>((2.0 * x + 1.0 - width) / width * span_x);
            values[2 * (y * width + x) + 1] = static_cast<float>((2.0 * y + 1.0 - height) / height * span_y);
        }
    return values;
}
std::vector<float> read(ggml_tensor* tensor) {
    std::vector<float> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}
}

DecoderResult decode_moge_vitb(MogeWeights& weights, GgmlRuntime& runtime,
    const EncoderResult& encoder, float input_aspect_ratio, bool capture_stages) {
    const auto pixels = static_cast<std::uint64_t>(encoder.rows) * encoder.columns;
    if (!encoder.rows || !encoder.columns || pixels > 4096 || encoder.features.size() != pixels * 768 ||
        encoder.class_token.size() != 768 || !std::isfinite(input_aspect_ratio) || input_aspect_ratio <= 0)
        throw std::invalid_argument("Invalid MoGe decoder input");
    if (!std::all_of(encoder.features.begin(), encoder.features.end(), [](float x) { return std::isfinite(x); }) ||
        !std::all_of(encoder.class_token.begin(), encoder.class_token.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("Nonfinite MoGe decoder input");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({32 * 1024 * 1024, nullptr, true}), &ggml_free);
    if (!context) throw std::runtime_error("Cannot create MoGe decoder graph context");
    DecoderGraph builder(context.get(), weights, capture_stages);
    auto* ctx = context.get();
    auto* features = builder.float_input(768, static_cast<int>(pixels), encoder.features);
    auto* class_token = builder.float_input(768, 1, encoder.class_token);
    std::array<Map, 5> inputs;
    for (int level = 0; level < 5; ++level) {
        const int width = static_cast<int>(encoder.columns) << level;
        const int height = static_cast<int>(encoder.rows) << level;
        auto* coordinates = builder.float_input(2, width * height, uv(width, height, input_aspect_ratio));
        builder.trace("uv." + std::to_string(level), coordinates);
        inputs[level] = {level == 0 ? ggml_concat(ctx, features, coordinates, 0) : coordinates, width, height};
    }
    auto neck = builder.stack("neck", inputs, 0);
    auto points = builder.stack("points_head", neck, 3)[4];
    auto normals = builder.stack("normal_head", neck, 3)[4];
    auto mask = builder.stack("mask_head", neck, 1)[4];
    auto* scale = class_token;
    for (int layer = 0; layer < 3; ++layer) {
        scale = builder.linear(scale, "scale_head." + std::to_string(layer * 2), layer == 2 ? 1 : 768);
        if (layer < 2) scale = ggml_relu(ctx, scale);
        builder.trace("scale_head." + std::to_string(layer), scale);
    }
    for (auto* tensor : {points.tensor, normals.tensor, mask.tensor, scale}) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(builder.graph, tensor);
    }
    runtime.allocate(builder.graph);
    builder.upload();
    runtime.compute(builder.graph);
    DecoderResult result;
    result.width = points.width;
    result.height = points.height;
    result.points = read(points.tensor);
    result.normals = read(normals.tensor);
    result.mask_logits = read(mask.tensor);
    result.log_metric_scale = read(scale)[0];
    for (const auto& [name, tensor] : builder.stages) result.stages.push_back({name, read(tensor)});
    return result;
}
} // namespace photara::inference::detail
