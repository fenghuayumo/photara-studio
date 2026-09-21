#include "vulkan/ops.hpp"

#include "internal/tensor_impl.hpp"
#include "vulkan/runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tinytensor::vulkan {
namespace {

using runtime::Buffer;
using runtime::BufferBinding;
using runtime::ceil_div;
using runtime::Context;
using runtime::kGroupSize;
using runtime::ShaderId;

runtime::Buffer& buffer_of(const Tensor& tensor) {
    auto owner = tensor.external_storage_owner();
    if (!owner) {
        throw std::runtime_error("Vulkan tensor is missing its buffer owner");
    }
    return *static_cast<runtime::Buffer*>(owner.get());
}

std::size_t byte_offset(const Tensor& tensor) {
    return tensor.storage_offset() * dtype_size(tensor.dtype());
}

std::uint32_t u32(std::size_t value) {
    return static_cast<std::uint32_t>(value);
}

std::uint32_t groups_for(std::size_t count) {
    return std::max(1U, ceil_div(u32(count), kGroupSize));
}

BufferBinding bind(const Tensor& tensor) {
    return buffer_of(tensor).binding();
}

BufferBinding dummy_binding() {
    return Context::get().dummy().binding();
}

void dispatch_elementwise(ElementwiseOp op, std::size_t count, const Tensor* a, const Tensor* b,
                          Tensor& out, float scalar) {
    struct Push {
        std::uint32_t op;
        std::uint32_t count;
        std::uint32_t dtype_a;
        std::uint32_t dtype_b;
        std::uint32_t dtype_out;
        std::uint32_t off_a;
        std::uint32_t off_b;
        std::uint32_t off_out;
        std::uint32_t elem_size_a;
        std::uint32_t elem_size_b;
        std::uint32_t elem_size_out;
        std::uint32_t has_b;
        std::uint32_t scalar_bits;
        std::uint32_t pad;
    } push{};
    push.op = static_cast<std::uint32_t>(op);
    push.count = u32(count);
    push.dtype_out = static_cast<std::uint32_t>(out.dtype());
    push.off_out = u32(byte_offset(out));
    push.elem_size_out = u32(dtype_size(out.dtype()));
    float bits = scalar;
    std::memcpy(&push.scalar_bits, &bits, sizeof(bits));
    if (a != nullptr) {
        push.dtype_a = static_cast<std::uint32_t>(a->dtype());
        push.off_a = u32(byte_offset(*a));
        push.elem_size_a = u32(dtype_size(a->dtype()));
    }
    if (b != nullptr) {
        push.dtype_b = static_cast<std::uint32_t>(b->dtype());
        push.off_b = u32(byte_offset(*b));
        push.elem_size_b = u32(dtype_size(b->dtype()));
        push.has_b = 1;
    }

    std::array<BufferBinding, 3> bindings{
        a != nullptr ? bind(*a) : dummy_binding(),
        b != nullptr ? bind(*b) : dummy_binding(),
        bind(out),
    };
    Context::get().dispatch(ShaderId::Elementwise, bindings, &push, sizeof(push), groups_for(count));
}

void exclusive_or_inclusive_scan(Buffer& input, Buffer& output, std::uint32_t count, bool is_float,
                                 bool exclusive) {
    if (count == 0) {
        return;
    }
    auto& ctx = Context::get();
    const std::uint32_t groups = groups_for(count);
    auto sums = ctx.alloc(groups * sizeof(std::uint32_t));
    struct ScanPush {
        std::uint32_t count;
        std::uint32_t is_float;
        std::uint32_t src_offset;
        std::uint32_t dst_offset;
        std::uint32_t sums_offset;
        std::uint32_t exclusive;
    } push{count, is_float ? 1U : 0U, 0, 0, 0, exclusive ? 1U : 0U};
    std::array<BufferBinding, 3> bindings{input.binding(), output.binding(), sums->binding()};
    ctx.dispatch(ShaderId::ScanBlock, bindings, &push, sizeof(push), groups);
    if (groups == 1) {
        return;
    }
    auto sums_scan = ctx.alloc(groups * sizeof(std::uint32_t));
    exclusive_or_inclusive_scan(*sums, *sums_scan, groups, is_float, true);
    struct AddPush {
        std::uint32_t count;
        std::uint32_t is_float;
        std::uint32_t data_offset;
        std::uint32_t sums_offset;
    } add{count, is_float ? 1U : 0U, 0, 0};
    std::array<BufferBinding, 2> add_bindings{output.binding(), sums_scan->binding()};
    ctx.dispatch(ShaderId::ScanAdd, add_bindings, &add, sizeof(add), groups);
}

Tensor prepared(const Tensor& tensor) {
    if (!tensor.is_contiguous()) {
        return make_contiguous(tensor);
    }
    return tensor;
}

} // namespace

Tensor TensorStorage::empty(const TensorShape& shape, DataType dtype, std::size_t dim0_capacity) {
    const std::size_t elem = dtype_size(dtype);
    const std::size_t logical = shape.elements();
    std::size_t row = 1;
    if (shape.rank() > 0) {
        for (std::size_t i = 1; i < shape.rank(); ++i) {
            row *= shape[i];
        }
    }
    const std::size_t capacity = dim0_capacity == 0 && shape.rank() > 0 ? shape[0] : dim0_capacity;
    const std::size_t bytes = std::max(logical * elem, capacity * row * elem);
    auto buffer = Context::get().alloc(bytes == 0 ? 16 : bytes);
    return Tensor::from_external_owner(
        buffer.get(), shape, Device::Vulkan, dtype, std::shared_ptr<void>(buffer), capacity, nullptr,
        "vulkan");
}

Tensor TensorStorage::alias(const Tensor& source, TensorShape shape, std::size_t logical_dim0) {
    Tensor result = source;
    result.shape_ = std::move(shape);
    result.strides_ = result.shape_.strides();
    result.storage_offset_ = 0;
    result.is_contiguous_ = true;
    result.is_view_ = false;
    result.state_ = std::make_shared<Tensor::TensorState>(*source.state_);
    result.state_->capacity = source.capacity();
    result.state_->logical_size = logical_dim0;
    result.id_ = Tensor::next_id_++;
    return result;
}

bool is_vulkan_tensor(const Tensor& tensor) {
    return tensor.device() == Device::Vulkan;
}

void fill(Tensor& tensor, float value) {
    if (!tensor.is_valid() || tensor.numel() == 0) {
        return;
    }
    if (value == 0.0F) {
        Context::get().fill_zero(buffer_of(tensor), byte_offset(tensor), tensor.bytes());
        return;
    }
    dispatch_elementwise(ElementwiseOp::Fill, tensor.numel(), nullptr, nullptr, tensor, value);
}

void copy_same_layout(Tensor& dst, const Tensor& src) {
    if (dst.numel() == 0) {
        return;
    }
    Context::get().copy(buffer_of(dst), byte_offset(dst), buffer_of(src), byte_offset(src),
                        src.bytes());
}

void upload(Tensor& dst, const void* data, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    Context::get().upload(buffer_of(dst), byte_offset(dst), data, bytes);
}

void download(const Tensor& src, void* data, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    Context::get().download(buffer_of(src), byte_offset(src), data, bytes);
}

Tensor convert(const Tensor& src, DataType dtype) {
    Tensor input = prepared(src);
    Tensor out = TensorStorage::empty(input.shape(), dtype, input.capacity());
    if (input.numel() == 0) {
        return out;
    }
    dispatch_elementwise(ElementwiseOp::Convert, input.numel(), &input, nullptr, out, 0.0F);
    return out;
}

Tensor make_contiguous(const Tensor& src) {
    if (src.is_contiguous()) {
        return src;
    }
    Tensor out = TensorStorage::empty(src.shape(), src.dtype());
    if (src.numel() == 0) {
        return out;
    }
    if (src.ndim() > 4) {
        auto host = src.to(Device::CPU);
        upload(out, host.data_ptr(), host.bytes());
        return out;
    }
    struct Push {
        std::uint32_t count;
        std::uint32_t rank;
        std::uint32_t elem_size;
        std::uint32_t src_offset_elems;
        std::uint32_t dst_offset_elems;
        std::uint32_t shape[4];
        std::uint32_t stride[4];
    } push{};
    push.count = u32(src.numel());
    push.rank = u32(src.ndim());
    push.elem_size = u32(dtype_size(src.dtype()));
    push.src_offset_elems = u32(src.storage_offset());
    for (std::size_t i = 0; i < src.ndim(); ++i) {
        push.shape[i] = u32(src.size(i));
        push.stride[i] = u32(src.stride(i));
    }
    std::array<BufferBinding, 2> bindings{bind(src), bind(out)};
    Context::get().dispatch(ShaderId::StridedCopy, bindings, &push, sizeof(push),
                            groups_for(src.numel()));
    return out;
}

void cat_dim0_into(Tensor& dst, const std::vector<Tensor>& tensors, std::size_t first_rows) {
    auto& ctx = Context::get();
    const std::size_t elem = dtype_size(dst.dtype());
    std::size_t row = 1;
    if (dst.ndim() > 0) {
        for (std::size_t i = 1; i < dst.ndim(); ++i) {
            row *= dst.size(i);
        }
    }
    std::size_t offset = byte_offset(dst) + first_rows * row * elem;
    const std::size_t start = first_rows == 0 ? 0 : 1;
    for (std::size_t i = start; i < tensors.size(); ++i) {
        Tensor src = prepared(tensors[i]);
        ctx.copy(buffer_of(dst), offset, buffer_of(src), byte_offset(src), src.bytes());
        offset += src.bytes();
    }
}

void index_select_into(const Tensor& src, const Tensor& indices, Tensor& out, int dim, int mode) {
    Tensor input = prepared(src);
    Tensor idx = prepared(indices);
    std::size_t outer = 1;
    std::size_t inner = 1;
    for (int i = 0; i < dim; ++i) {
        outer *= input.size(static_cast<std::size_t>(i));
    }
    for (std::size_t i = static_cast<std::size_t>(dim) + 1; i < input.ndim(); ++i) {
        inner *= input.size(i);
    }
    struct Push {
        std::uint32_t outer;
        std::uint32_t dim_size;
        std::uint32_t inner;
        std::uint32_t n_indices;
        std::uint32_t elem_size;
        std::uint32_t src_offset;
        std::uint32_t dst_offset;
        std::uint32_t idx_offset;
        std::uint32_t idx_is_i64;
        std::uint32_t mode;
    } push{};
    push.outer = u32(outer);
    push.dim_size = u32(input.size(static_cast<std::size_t>(dim)));
    push.inner = u32(std::max<std::size_t>(inner, 1));
    push.n_indices = u32(idx.numel());
    push.elem_size = u32(dtype_size(input.dtype()));
    push.src_offset = u32(byte_offset(input));
    push.dst_offset = u32(byte_offset(out));
    push.idx_offset = u32(byte_offset(idx));
    push.idx_is_i64 = idx.dtype() == DataType::Int64 ? 1U : 0U;
    push.mode = static_cast<std::uint32_t>(mode);
    const std::size_t total = outer * idx.numel() * std::max<std::size_t>(inner, 1);
    std::array<BufferBinding, 3> bindings{bind(input), bind(idx), bind(out)};
    Context::get().dispatch(ShaderId::IndexSelect, bindings, &push, sizeof(push), groups_for(total));
}

void index_fill(Tensor& dst, const Tensor& indices, float value) {
    Tensor idx = prepared(indices);
    struct Push {
        std::uint32_t n_indices;
        std::uint32_t dim_size;
        std::uint32_t elem_size;
        std::uint32_t dst_offset;
        std::uint32_t idx_offset;
        std::uint32_t idx_is_i64;
        std::uint32_t dtype;
        std::uint32_t value_bits;
    } push{};
    push.n_indices = u32(idx.numel());
    push.dim_size = u32(dst.size(0));
    push.elem_size = u32(dtype_size(dst.dtype()));
    push.dst_offset = u32(byte_offset(dst));
    push.idx_offset = u32(byte_offset(idx));
    push.idx_is_i64 = idx.dtype() == DataType::Int64 ? 1U : 0U;
    push.dtype = static_cast<std::uint32_t>(dst.dtype());
    std::memcpy(&push.value_bits, &value, sizeof(value));
    std::array<BufferBinding, 2> bindings{bind(dst), bind(idx)};
    Context::get().dispatch(ShaderId::IndexFill, bindings, &push, sizeof(push),
                            groups_for(idx.numel()));
}

std::size_t count_nonzero(const Tensor& src) {
    Tensor input = prepared(src);
    if (input.numel() == 0) {
        return 0;
    }
    auto& ctx = Context::get();
    const std::uint32_t count = u32(input.numel());
    auto flags = ctx.alloc(count * 4U);
    auto scan = ctx.alloc(count * 4U);
    struct MaskPush {
        std::uint32_t count;
        std::uint32_t dtype;
        std::uint32_t src_offset;
        std::uint32_t dst_offset;
    } mask{count, static_cast<std::uint32_t>(input.dtype()), u32(byte_offset(input)), 0};
    std::array<BufferBinding, 2> mask_bindings{bind(input), flags->binding()};
    ctx.dispatch(ShaderId::MaskFlags, mask_bindings, &mask, sizeof(mask), groups_for(count));
    exclusive_or_inclusive_scan(*flags, *scan, count, false, true);
    std::uint32_t last_scan = 0;
    std::uint32_t last_flag = 0;
    ctx.download(*scan, (count - 1U) * 4U, &last_scan, 4);
    ctx.download(*flags, (count - 1U) * 4U, &last_flag, 4);
    return static_cast<std::size_t>(last_scan + last_flag);
}

Tensor nonzero(const Tensor& src) {
    Tensor input = prepared(src);
    const std::size_t found = count_nonzero(input);
    const std::size_t ndim = std::max<std::size_t>(input.ndim(), 1);
    if (found == 0) {
        return TensorStorage::empty(TensorShape{std::size_t{0}, ndim}, DataType::Int64);
    }
    auto& ctx = Context::get();
    const std::uint32_t count = u32(input.numel());
    auto flags = ctx.alloc(count * 4U);
    auto scan = ctx.alloc(count * 4U);
    struct MaskPush {
        std::uint32_t count;
        std::uint32_t dtype;
        std::uint32_t src_offset;
        std::uint32_t dst_offset;
    } mask{count, static_cast<std::uint32_t>(input.dtype()), u32(byte_offset(input)), 0};
    std::array<BufferBinding, 2> mask_bindings{bind(input), flags->binding()};
    ctx.dispatch(ShaderId::MaskFlags, mask_bindings, &mask, sizeof(mask), groups_for(count));
    exclusive_or_inclusive_scan(*flags, *scan, count, false, true);

    Tensor out = TensorStorage::empty(TensorShape{found, ndim}, DataType::Int64);
    struct CompactPush {
        std::uint32_t count;
        std::uint32_t ndim;
        std::uint32_t flags_offset;
        std::uint32_t scan_offset;
        std::uint32_t out_offset;
        std::uint32_t shape0;
        std::uint32_t shape1;
        std::uint32_t shape2;
        std::uint32_t shape3;
    } compact{};
    compact.count = count;
    compact.ndim = u32(ndim);
    compact.out_offset = u32(byte_offset(out));
    if (input.ndim() > 0) {
        compact.shape0 = u32(input.size(0));
    }
    if (input.ndim() > 1) {
        compact.shape1 = u32(input.size(1));
    }
    if (input.ndim() > 2) {
        compact.shape2 = u32(input.size(2));
    }
    if (input.ndim() > 3) {
        compact.shape3 = u32(input.size(3));
    }
    std::array<BufferBinding, 3> compact_bindings{flags->binding(), scan->binding(), bind(out)};
    ctx.dispatch(ShaderId::Compact, compact_bindings, &compact, sizeof(compact), groups_for(count));
    return out;
}

Tensor elementwise_unary(const Tensor& src, ElementwiseOp op) {
    Tensor input = prepared(src);
    const DataType out_dtype =
        (op == ElementwiseOp::Not || op == ElementwiseOp::Eq || op == ElementwiseOp::Ne ||
         op == ElementwiseOp::Lt || op == ElementwiseOp::Le || op == ElementwiseOp::Gt ||
         op == ElementwiseOp::Ge)
            ? DataType::Bool
            : input.dtype();
    Tensor out = TensorStorage::empty(input.shape(), out_dtype);
    if (input.numel() == 0) {
        return out;
    }
    dispatch_elementwise(op, input.numel(), &input, nullptr, out, 0.0F);
    return out;
}

Tensor elementwise_binary(const Tensor& a, const Tensor& b, ElementwiseOp op) {
    Tensor left = prepared(a);
    Tensor right = prepared(b);
    if (left.shape() != right.shape()) {
        auto broadcast = left.broadcast_shape(right.shape());
        if (left.shape() != broadcast) {
            left = make_contiguous(left.broadcast_to(broadcast));
        }
        if (right.shape() != broadcast) {
            right = make_contiguous(right.broadcast_to(broadcast));
        }
    }
    const bool logical = op == ElementwiseOp::And || op == ElementwiseOp::Or ||
                         op == ElementwiseOp::Xor || op == ElementwiseOp::Eq ||
                         op == ElementwiseOp::Ne || op == ElementwiseOp::Lt ||
                         op == ElementwiseOp::Le || op == ElementwiseOp::Gt ||
                         op == ElementwiseOp::Ge;
    Tensor out = TensorStorage::empty(left.shape(), logical ? DataType::Bool : left.dtype());
    if (left.numel() == 0) {
        return out;
    }
    dispatch_elementwise(op, left.numel(), &left, &right, out, 0.0F);
    return out;
}

Tensor elementwise_scalar(const Tensor& a, float scalar, ElementwiseOp op) {
    Tensor input = prepared(a);
    const bool logical = op == ElementwiseOp::Not || op == ElementwiseOp::Eq ||
                         op == ElementwiseOp::Ne || op == ElementwiseOp::Lt ||
                         op == ElementwiseOp::Le || op == ElementwiseOp::Gt ||
                         op == ElementwiseOp::Ge;
    Tensor out = TensorStorage::empty(input.shape(), logical ? DataType::Bool : input.dtype());
    if (input.numel() == 0) {
        return out;
    }
    dispatch_elementwise(op, input.numel(), &input, nullptr, out, scalar);
    return out;
}

Tensor multinomial(const Tensor& weights, int num_samples, bool replacement) {
    Tensor input = prepared(weights);
    const int samples = replacement
                            ? num_samples
                            : std::min(num_samples, static_cast<int>(input.numel()));
    Tensor out = TensorStorage::empty(TensorShape{static_cast<std::size_t>(samples)}, DataType::Int64);
    if (samples <= 0 || input.numel() == 0) {
        return out;
    }
    auto& ctx = Context::get();
    const std::uint32_t n = u32(input.numel());
    auto cdf = ctx.alloc(n * 4U);
    exclusive_or_inclusive_scan(buffer_of(input), *cdf, n, true, false);
    const std::uint32_t seed =
        static_cast<std::uint32_t>(RandomGenerator::instance().get_next_cuda_seed());

    struct Push {
        std::uint32_t op;
        std::uint32_t n_weights;
        std::uint32_t n_samples;
        std::uint32_t seed;
        std::uint32_t weights_offset;
        std::uint32_t cdf_offset;
        std::uint32_t keys_offset;
        std::uint32_t out_offset;
    } push{};
    push.n_weights = n;
    push.seed = seed;
    push.weights_offset = u32(byte_offset(input));
    push.out_offset = u32(byte_offset(out));

    if (replacement) {
        push.op = 0;
        push.n_samples = static_cast<std::uint32_t>(samples);
        std::array<BufferBinding, 3> bindings{bind(input), cdf->binding(), bind(out)};
        ctx.dispatch(ShaderId::Multinomial, bindings, &push, sizeof(push), groups_for(samples));
        return out;
    }

    auto keys = ctx.alloc(n * 4U);
    push.op = 1;
    push.keys_offset = 0;
    std::array<BufferBinding, 3> key_bindings{bind(input), keys->binding(), bind(out)};
    ctx.dispatch(ShaderId::Multinomial, key_bindings, &push, sizeof(push), groups_for(n));
    for (int sample = 0; sample < samples; ++sample) {
        push.op = 2;
        push.n_samples = static_cast<std::uint32_t>(sample);
        std::array<BufferBinding, 3> argmax_bindings{bind(input), keys->binding(), bind(out)};
        ctx.dispatch(ShaderId::Multinomial, argmax_bindings, &push, sizeof(push), 1);
    }
    return out;
}

} // namespace tinytensor::vulkan
