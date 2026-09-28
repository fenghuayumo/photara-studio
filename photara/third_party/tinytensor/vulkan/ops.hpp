#pragma once

#include "core/data_types.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tinytensor {

class Tensor;
class TensorShape;

namespace vulkan {

enum class ElementwiseOp : std::uint32_t {
    Fill = 0,
    Convert = 1,
    Not = 2,
    And = 3,
    Or = 4,
    Xor = 5,
    Eq = 6,
    Ne = 7,
    Lt = 8,
    Le = 9,
    Gt = 10,
    Ge = 11,
    Neg = 12,
    Abs = 13,
    Sign = 14,
    Reciprocal = 15,
    Exp = 16,
    Exp2 = 17,
    Log = 18,
    Log2 = 19,
    Log10 = 20,
    Log1p = 21,
    Sqrt = 22,
    Rsqrt = 23,
    Square = 24,
    Sin = 25,
    Cos = 26,
    Tan = 27,
    Asin = 28,
    Acos = 29,
    Atan = 30,
    Sinh = 31,
    Cosh = 32,
    Tanh = 33,
    Sigmoid = 34,
    Relu = 35,
    Gelu = 36,
    Swish = 37,
    Floor = 38,
    Ceil = 39,
    Round = 40,
    Trunc = 41,
    IsNan = 42,
    IsInf = 43,
    IsFinite = 44,
    Add = 45,
    Sub = 46,
    Mul = 47,
    Div = 48,
    Pow = 49,
    Mod = 50,
    Maximum = 51,
    Minimum = 52,
    Clamp = 53,
    Where = 54,
    MaskFill = 55,
};

enum class ReduceKind : std::uint32_t {
    Sum = 0,
    Mean = 1,
    Max = 2,
    Min = 3,
    Prod = 4,
    Any = 5,
    All = 6,
    Argmax = 9,
    Argmin = 10,
};

class TensorStorage {
public:
    static Tensor empty(const TensorShape& shape, DataType dtype, std::size_t dim0_capacity = 0);
    static Tensor alias(const Tensor& source, TensorShape shape, std::size_t logical_dim0);
};

bool is_vulkan_tensor(const Tensor& tensor);
void fill(Tensor& tensor, float value);
// Expands a packed [H, W] Int32 RGBA image (r | g << 8 | b << 16 | a << 24)
// into a planar [3, H, W] Float32 RGB tensor scaled to [0, 1]. `gray` receives
// the luma plane and `mask` the alpha plane when the pointers are non-null.
void unpack_training_pixels(const Tensor& packed, Tensor& rgb, Tensor* gray,
                            Tensor* mask);
void copy_same_layout(Tensor& dst, const Tensor& src);
void upload(Tensor& dst, const void* data, std::size_t bytes);
// Immediate non-blocking upload for a newly allocated tensor. Its first GPU
// consumer must be recorded after this call so queue submission order provides
// the dependency.
void upload_async(Tensor& dst, const void* data, std::size_t bytes);
void download(const Tensor& src, void* data, std::size_t bytes);
Tensor convert(const Tensor& src, DataType dtype);
Tensor make_contiguous(const Tensor& src);
Tensor broadcast_to(const Tensor& src, const TensorShape& target);
void cat_dim0_into(Tensor& dst, const std::vector<Tensor>& tensors, std::size_t first_rows);
void cat_along(Tensor& dst, const std::vector<Tensor>& tensors, int dim);
void index_select_into(const Tensor& src, const Tensor& indices, Tensor& out, int dim, int mode);
void index_fill(Tensor& dst, const Tensor& indices, float value, int dim);
void scatter(Tensor& dst, const Tensor& indices, const Tensor& src, int dim, bool accumulate,
             bool wrap_negative = true);
// indices has the same rank as src. Each output element copies src at the
// output coordinate with axis `dim` replaced by indices[element].
void gather(const Tensor& src, const Tensor& indices, Tensor& out, int dim, int mode);
// Fused ADC prune sweep (the Vulkan twin of the CUDA adc_plus_prune_kernel):
// one dispatch over all Gaussian rows writes the keep/hard masks and sigmoid
// opacities without any host round trip. Every tensor must be contiguous with
// dim0 == count; per-row strides are derived from numel / count.
void prune_masks(const Tensor& means, const Tensor& log_scales,
                 const Tensor& quaternions, const Tensor& opacity_logits,
                 const Tensor& sh, Tensor& keep, Tensor& hard,
                 Tensor& opacities, float minimum_opacity,
                 float maximum_bounds, float center_x, float center_y,
                 float center_z);
// Exact angular size of the projected 3D Gaussian in a panorama tangent plane.
void panorama_sizes(const Tensor& means, const Tensor& log_scales,
                    const Tensor& quaternions, Tensor& sizes,
                    float camera_x, float camera_y, float camera_z,
                    float angular_normalization, float scale_modifier);
// Projects each Gaussian's [N,3] log scales into a maximum longest/shortest
// axis ratio, preserving the row's log-scale midpoint. maximum_ratio must
// exceed 1; rows already inside the bound are left untouched.
void constrain_scale_ratio(Tensor& log_scales, float maximum_ratio);
std::size_t count_nonzero(const Tensor& src);
Tensor nonzero(const Tensor& src);
Tensor masked_select(const Tensor& src, const Tensor& mask);
void masked_fill(Tensor& dst, const Tensor& mask, float value);
Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y);
Tensor elementwise_unary(const Tensor& src, ElementwiseOp op);
Tensor elementwise_binary(const Tensor& a, const Tensor& b, ElementwiseOp op);
Tensor elementwise_scalar(const Tensor& a, float scalar, ElementwiseOp op, float scalar2 = 0.0F);
Tensor fused_pointwise(const Tensor& src, std::span<const std::uint32_t> kinds,
                       std::span<const float> scalars);
Tensor clamp(const Tensor& src, float lo, float hi);
Tensor reduce(const Tensor& src, const std::vector<int>& axes, bool keepdim, ReduceKind kind);
Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_b, std::size_t batch);
Tensor cumsum(const Tensor& src, int dim);
void random_uniform(Tensor& dst, float low, float high, std::uint32_t seed);
void random_normal(Tensor& dst, float mean, float stddev, std::uint32_t seed);
void random_bernoulli(Tensor& dst, float p, std::uint32_t seed);
void random_randint(Tensor& dst, int low, int high, std::uint32_t seed);
void arange(Tensor& dst, float start, float step);
void eye(Tensor& dst, std::uint32_t rows, std::uint32_t cols);
void diag(Tensor& dst, const Tensor& diagonal);
void max_pool2d(const Tensor& src, Tensor& dst, int kernel, int stride, int padding,
                int h_out, int w_out);
void adaptive_avg_pool2d(const Tensor& src, Tensor& dst, int h_out, int w_out);
Tensor multinomial(const Tensor& weights, int num_samples, bool replacement);
std::pair<Tensor, Tensor> sort_1d_f32(const Tensor& values, bool descending);

// Exact order statistics over a Float32 array, restricted to the rows `mask`
// selects (a Bool array of the same length, or an invalid tensor for "all
// rows"). Both are the device form of sorting a masked copy and taking the
// front of it, but they cost two histogram passes and one emit pass instead of
// a full bitonic sort's log2(n)^2 dispatches.
//
// `rank` is 1-based: select_nth_value writes the exact `rank`-th value in the
// requested direction and returns false when nothing qualifies.
bool select_nth_value(const Tensor& values, const Tensor& mask,
                      std::uint32_t rank, bool descending, float* value);

// Device form of "sort `values` descending, keep the front `count` indices".
// Returns Int32 row indices of the `count` selected rows; when the mask
// selects fewer than `count` rows the result is that shorter prefix, so the
// caller never needs a separate count. `emitted` receives the number of rows
// the emit pass actually wrote.
Tensor select_topk_indices(const Tensor& values, const Tensor& mask,
                           std::uint32_t count, bool descending,
                           std::uint32_t* emitted = nullptr);

} // namespace vulkan
} // namespace tinytensor
