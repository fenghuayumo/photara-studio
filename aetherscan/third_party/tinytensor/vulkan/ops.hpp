#pragma once

#include "core/data_types.hpp"

#include <cstddef>
#include <cstdint>
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
};

class TensorStorage {
public:
    static Tensor empty(const TensorShape& shape, DataType dtype, std::size_t dim0_capacity = 0);
    static Tensor alias(const Tensor& source, TensorShape shape, std::size_t logical_dim0);
};

bool is_vulkan_tensor(const Tensor& tensor);
void fill(Tensor& tensor, float value);
void copy_same_layout(Tensor& dst, const Tensor& src);
void upload(Tensor& dst, const void* data, std::size_t bytes);
void download(const Tensor& src, void* data, std::size_t bytes);
Tensor convert(const Tensor& src, DataType dtype);
Tensor make_contiguous(const Tensor& src);
void cat_dim0_into(Tensor& dst, const std::vector<Tensor>& tensors, std::size_t first_rows);
void index_select_into(const Tensor& src, const Tensor& indices, Tensor& out, int dim, int mode);
void index_fill(Tensor& dst, const Tensor& indices, float value);
std::size_t count_nonzero(const Tensor& src);
Tensor nonzero(const Tensor& src);
Tensor elementwise_unary(const Tensor& src, ElementwiseOp op);
Tensor elementwise_binary(const Tensor& a, const Tensor& b, ElementwiseOp op);
Tensor elementwise_scalar(const Tensor& a, float scalar, ElementwiseOp op);
Tensor multinomial(const Tensor& weights, int num_samples, bool replacement);

} // namespace vulkan
} // namespace tinytensor
