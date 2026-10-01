#pragma once

// Quantized SH Adam lives beside AdamState. zero_adam_rows and the CUDA
// optimizer are float-only, and the headers they include cannot change
// without rebuilding cuda_ops.cu. Densify consults the bound state.

#include "splat/types.hpp"

#include <cstddef>

namespace photara::splat::detail {

enum class ShAdamQuantFormat {
    fp16_m_logq8_v,
    fp16_u_logq8_v,
};

struct ShAdamQuant {
    // float16 [N, stride] signed first moment.
    tinytensor::Tensor first;
    // uint8 [N, stride] log_s code for the second moment.
    tinytensor::Tensor packed;
    // float [N, 4]: DC and non-DC each own (log_s_min, log_s_max).
    tinytensor::Tensor bounds;
    int stride = 0;
    ShAdamQuantFormat format{ShAdamQuantFormat::fp16_u_logq8_v};

    [[nodiscard]] bool active() const { return first.is_valid(); }
};

[[nodiscard]] ShAdamQuant make_sh_adam_quant(
    std::size_t rows, int stride, tinytensor::Device device,
    ShAdamQuantFormat format = ShAdamQuantFormat::fp16_u_logq8_v);

void sh_adam_quant_select_rows(
    ShAdamQuant& state, const tinytensor::Tensor& indices);
void sh_adam_quant_zero_rows(
    ShAdamQuant& state, const tinytensor::Tensor& indices);
void sh_adam_quant_append_zeros(ShAdamQuant& state, std::size_t count);

// Unfused / empty-render step. `active_stride` columns are updated;
// the rest of the row is re-encoded so the new bounds do not move them.
// Pass regularization_factor 0 when the gradient already contains it.
void sh_adam_quant_step(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    ShAdamQuant& state, int active_stride, float learning_rate,
    float rest_learning_rate, float beta1, float beta2, float correction1,
    float correction2, float adam_epsilon, float regularization_factor);

}  // namespace photara::splat::detail
