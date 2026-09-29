#include "ppisp.hpp"
#include "bilateral_grid.hpp"
#include "optimizer.hpp"

#include "splat_drender/vulkan_api.h"
#include "vulkan/backend.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace photara::splat::detail {
namespace {

using DeviceView = splat_drender::vulkan::SplatBufferView;
using Command = splat_drender::vulkan::SplatColorCorrectionCommand;
using Buffers = std::array<DeviceView, 6>;
using Push = std::array<std::uint32_t, 20>;

struct CorrectionBackend {
    splat_drender::vulkan::Context context;
    splat_drender::vulkan::SplatRasterizer dispatcher;

    CorrectionBackend() : context([] {
        const auto handles = tinytensor::vulkan::device_handles();
        splat_drender::vulkan::ContextOptions options;
        options.external_device.instance = handles.instance;
        options.external_device.physical_device = handles.physical_device;
        options.external_device.device = handles.device;
        options.external_device.queue = handles.queue;
        options.external_device.queue_family = handles.queue_family;
        options.external_device.push_descriptors =
            tinytensor::vulkan::device_info().push_descriptors;
        options.external_device.buffer_float32_atomic_add =
            tinytensor::vulkan::device_info().buffer_atomic_f32;
        return options;
    }()), dispatcher(context) {}
};

CorrectionBackend& backend() {
    static thread_local CorrectionBackend instance;
    return instance;
}

DeviceView buffer(const tinytensor::Tensor& tensor) {
    if (!tensor.is_valid()) return {};
    const auto view = tinytensor::vulkan::buffer_view(tensor);
    return {view.buffer, view.offset, view.bytes};
}

std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }
std::uint32_t groups(std::size_t count) {
    return static_cast<std::uint32_t>((count + 63) / 64);
}

void launch(std::uint32_t mode, const Buffers& buffers, Push push,
            std::uint32_t group_count) {
    push[7] = mode;
    tinytensor::vulkan::submit_async();
    backend().dispatcher.color_correction_device(buffers, push, group_count);
}

Command command(std::uint32_t mode, const Buffers& buffers, Push push,
                std::uint32_t group_count) {
    push[7] = mode;
    return {buffers, push, group_count};
}

void launch_batch(std::span<const Command> commands) {
    tinytensor::vulkan::submit_async();
    backend().dispatcher.color_correction_batch_device(commands);
}

void image_shape(const tinytensor::Tensor& color) {
    if (color.device() != tinytensor::Device::Vulkan ||
        color.shape().rank() != 3 || color.shape()[0] != 3)
        throw std::invalid_argument("Vulkan color correction expects [3,H,W]");
}

Push image_push(const tinytensor::Tensor& color, std::size_t view) {
    Push push{};
    push[0] = static_cast<std::uint32_t>(color.shape()[2]);
    push[1] = static_cast<std::uint32_t>(color.shape()[1]);
    push[2] = static_cast<std::uint32_t>(view);
    return push;
}

}  // namespace

PpispState make_ppisp_state_vulkan(
    std::size_t views, const TrainingOptions& options) {
    PpispState state;
    if (views == 0) return state;
    state.type = options.ppisp_type;
    state.num_params = ppisp_parameter_count(options.ppisp_type);
    state.clamp_output = options.ppisp_clamp_output;
    std::vector<float> initial(views * state.num_params, 0.F);
    if (state.num_params == 36)
        for (std::size_t view = 0; view < views; ++view)
            for (int c = 0; c < 3; ++c) {
                initial[view * 36 + 24 + 4 * c] = 0.013658988289535046F;
                initial[view * 36 + 25 + 4 * c] = 0.013658988289535046F;
                initial[view * 36 + 26 + 4 * c] = 0.37816452980041504F;
            }
    state.parameters = tinytensor::Tensor::from_vector(
        initial, {views, static_cast<std::size_t>(state.num_params)},
        tinytensor::Device::Vulkan);
    state.gradient = tinytensor::Tensor::zeros_like(state.parameters);
    state.adam = make_adam_state(state.parameters);
    state.colour_pullback = tinytensor::Tensor::empty(
        {std::size_t{81}}, tinytensor::Device::Vulkan);
    return state;
}

void apply_ppisp_vulkan(
    const tinytensor::Tensor& color, PpispState& state,
    const Camera& camera, std::size_t view) {
    image_shape(color);
    if (!state.is_valid() || view >= state.parameters.shape()[0])
        throw std::invalid_argument("PPISP view index is out of range");
    if (!state.output.is_valid() || state.output.shape() != color.shape())
        state.output = tinytensor::Tensor::empty(
            color.shape(), tinytensor::Device::Vulkan);
    Push push = image_push(color, view);
    push[3] = static_cast<std::uint32_t>(state.num_params);
    push[9] = state.clamp_output;
    push[10] = bits(camera.cx);
    push[11] = bits(camera.cy);
    launch(0, {buffer(color), buffer(state.parameters), {},
               buffer(state.output), {}, {}}, push,
           groups(color.shape()[1] * color.shape()[2]));
}

void backward_ppisp_vulkan(
    PpispState& state, const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient, const Camera& camera,
    std::size_t view) {
    image_shape(color);
    if (view >= state.parameters.shape()[0] ||
        output_gradient.shape() != color.shape())
        throw std::invalid_argument("PPISP backward shape/view mismatch");
    const std::size_t pixels = color.shape()[1] * color.shape()[2];
    const std::size_t block_count = (pixels + 2047) / 2048;
    const tinytensor::TensorShape partial_shape{
        block_count, static_cast<std::size_t>(state.num_params)};
    if (!state.raw_sums.is_valid() || state.raw_sums.shape() != partial_shape)
        state.raw_sums = tinytensor::Tensor::empty(
            partial_shape, tinytensor::Device::Vulkan);
    if (!state.input_grad.is_valid() || state.input_grad.shape() != color.shape())
        state.input_grad = tinytensor::Tensor::empty(
            color.shape(), tinytensor::Device::Vulkan);
    state.gradient.zero_();
    Push push = image_push(color, view);
    push[3] = static_cast<std::uint32_t>(state.num_params);
    push[9] = state.clamp_output;
    push[10] = bits(camera.cx);
    push[11] = bits(camera.cy);
    const auto precompute = command(11,
        {buffer(state.parameters), {}, {}, buffer(state.colour_pullback), {}, {}},
        push, 2);
    const auto backward = command(state.num_params == 9 ? 12 : 13,
        {buffer(color), buffer(state.parameters), buffer(output_gradient),
         buffer(state.input_grad), buffer(state.raw_sums),
         buffer(state.colour_pullback)},
        push, static_cast<std::uint32_t>(block_count));
    push[4] = static_cast<std::uint32_t>(block_count);
    const auto reduce = command(2,
        {DeviceView{}, DeviceView{}, DeviceView{}, DeviceView{},
         buffer(state.raw_sums), buffer(state.gradient)},
        push, 1);
    launch_batch(std::array{precompute, backward, reduce});
}

void step_ppisp_vulkan(
    PpispState& state, const TrainingOptions& options,
    unsigned iteration) {
    if (!state.is_valid()) return;
    if (iteration == 0)
        throw std::invalid_argument("PPISP Adam step must be positive");
    Push push{};
    push[3] = static_cast<std::uint32_t>(state.num_params);
    push[4] = static_cast<std::uint32_t>(state.parameters.shape()[0]);
    push[10] = bits(options.ppisp_reg_exposure_mean);
    push[11] = bits(options.ppisp_reg_color_mean);
    push[12] = bits(options.ppisp_reg_vig_center);
    push[13] = bits(options.ppisp_reg_vig_non_pos);
    push[14] = bits(options.ppisp_reg_vig_channel_var);
    push[15] = bits(options.ppisp_reg_crf_channel_var);
    if (state.parameters.numel() <= 256) {
        push[0] = bits(options.ppisp_lr);
        push[1] = bits(options.beta1);
        push[2] = bits(options.beta2);
        push[5] = bits(1.F - std::pow(options.beta1, static_cast<float>(iteration)));
        push[6] = bits(1.F - std::pow(options.beta2, static_cast<float>(iteration)));
        push[8] = bits(options.adam_epsilon);
        push[9] = options.ppisp_identity_projection;
        push[16] = bits(options.ppisp_exposure_limit);
        push[17] = bits(options.ppisp_color_limit);
        launch(15, {buffer(state.parameters), buffer(state.gradient),
                    buffer(state.adam.first), buffer(state.parameters),
                    buffer(state.adam.second), buffer(state.adam.first)},
               push, 1);
        return;
    }
    launch(3, {buffer(state.parameters), {}, {}, buffer(state.gradient),
               {}, {}}, push,
           groups(state.parameters.numel()));
    adam_step(state.parameters, state.gradient, state.adam,
              options.ppisp_lr, iteration, options);
    push[12] = options.ppisp_identity_projection;
    push[13] = bits(options.ppisp_exposure_limit);
    push[14] = bits(options.ppisp_color_limit);
    launch(4, {buffer(state.parameters), {}, {}, buffer(state.parameters),
               {}, {}}, push, 1);
}

BilateralGridState make_bilateral_grid_state_vulkan(
    std::size_t views, const TrainingOptions& options) {
    BilateralGridState state;
    if (views == 0) return state;
    state.shared = options.bilateral_grid_shared;
    state.grid_width = static_cast<int>(std::max(options.bilateral_grid_width, 1U));
    state.grid_height = static_cast<int>(std::max(options.bilateral_grid_height, 1U));
    state.luma = static_cast<int>(std::max(options.bilateral_grid_luma, 1U));
    const std::size_t rows = state.shared ? 1 : views;
    const std::size_t cells = rows * state.luma * state.grid_height *
        state.grid_width;
    std::vector<float> initial(cells * 12, 0.F);
    for (std::size_t cell = 0; cell < cells; ++cell)
        for (int c = 0; c < 3; ++c)
            initial[cell * 12 + 5 * c] = 1.F;
    state.grids = tinytensor::Tensor::from_vector(initial,
        {rows, static_cast<std::size_t>(state.luma),
         static_cast<std::size_t>(state.grid_height),
         static_cast<std::size_t>(state.grid_width), std::size_t{12}},
        tinytensor::Device::Vulkan);
    state.gradient = tinytensor::Tensor::zeros_like(state.grids);
    state.adam = make_adam_state(state.grids);
    return state;
}

void apply_bilateral_grid_vulkan(
    const tinytensor::Tensor& color, BilateralGridState& state,
    std::size_t view, bool wrap_horizontal) {
    image_shape(color);
    if (!state.is_valid() || (!state.shared && view >= state.grids.shape()[0]))
        throw std::invalid_argument("bilateral grid view index is out of range");
    if (!state.output.is_valid() || state.output.shape() != color.shape())
        state.output = tinytensor::Tensor::empty(
            color.shape(), tinytensor::Device::Vulkan);
    Push push = image_push(color, state.shared ? 0 : view);
    push[3] = state.luma;
    push[4] = state.grid_width;
    push[5] = state.grid_height;
    push[8] = wrap_horizontal;
    launch(5, {buffer(color), buffer(state.grids), {},
               buffer(state.output), {}, {}}, push,
           groups(color.shape()[1] * color.shape()[2]));
}

void backward_bilateral_grid_vulkan(
    BilateralGridState& state, const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient, std::size_t view,
    bool wrap_horizontal) {
    image_shape(color);
    if ((!state.shared && view >= state.grids.shape()[0]) ||
        output_gradient.shape() != color.shape())
        throw std::invalid_argument("bilateral grid backward shape/view mismatch");
    if (!state.input_grad.is_valid() || state.input_grad.shape() != color.shape())
        state.input_grad = tinytensor::Tensor::empty(
            color.shape(), tinytensor::Device::Vulkan);
    Push push = image_push(color, state.shared ? 0 : view);
    push[3] = state.luma;
    push[4] = state.grid_width;
    push[5] = state.grid_height;
    push[6] = static_cast<std::uint32_t>(state.grids.shape()[0]);
    push[8] = wrap_horizontal;
    const std::size_t pixels = color.shape()[1] * color.shape()[2];
    if (pixels >= 128 * 128) {
        launch(9, {buffer(color), buffer(state.grids),
                   buffer(output_gradient), buffer(state.input_grad),
                   {}, {}}, push, groups(pixels));
        launch(10, {buffer(color), {}, buffer(output_gradient),
                    buffer(state.gradient), {}, {}}, push,
               static_cast<std::uint32_t>(state.grids.numel() / 12));
    } else {
        state.gradient.zero_();
        launch(6, {buffer(color), buffer(state.grids),
                   buffer(output_gradient), buffer(state.input_grad),
                   buffer(state.gradient), {}}, push, groups(pixels));
    }
}

void step_bilateral_grid_vulkan(
    BilateralGridState& state, const TrainingOptions& options,
    unsigned iteration, bool wrap_horizontal) {
    if (!state.is_valid()) return;
    Push push{};
    push[3] = state.luma;
    push[4] = state.grid_width;
    push[5] = state.grid_height;
    push[6] = static_cast<std::uint32_t>(state.grids.shape()[0]);
    push[8] = wrap_horizontal;
    push[10] = bits(options.bilateral_grid_tv_weight);
    if (options.bilateral_grid_tv_weight > 0.F)
        launch(7, {buffer(state.grids), {}, {}, buffer(state.gradient),
                   {}, {}}, push, groups(state.grids.numel()));
    adam_step(state.grids, state.gradient, state.adam,
              options.bilateral_grid_lr, iteration, options);
    push[12] = options.bilateral_grid_identity_projection;
    push[13] = bits(options.bilateral_grid_deviation_limit);
    if (options.bilateral_grid_identity_projection) {
        const tinytensor::TensorShape mean_shape{
            state.grids.shape()[0], std::size_t{12}};
        if (!state.projection_means.is_valid() ||
            state.projection_means.shape() != mean_shape)
            state.projection_means = tinytensor::Tensor::empty(
                mean_shape, tinytensor::Device::Vulkan);
        launch(14, {buffer(state.grids), {}, {},
                    buffer(state.projection_means), {}, {}}, push,
               static_cast<std::uint32_t>(state.grids.shape()[0] * 12));
    }
    launch(8, {buffer(state.grids), buffer(state.projection_means), {},
               buffer(state.grids),
               {}, {}}, push,
           groups(state.grids.numel()));
}

}  // namespace photara::splat::detail
