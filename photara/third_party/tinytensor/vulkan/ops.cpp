#include "vulkan/ops.hpp"

#include "internal/tensor_impl.hpp"
#include "vulkan/backend.hpp"
#include "vulkan/runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
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
    // This is also the materialization boundary for a deferred Vulkan tensor.
    // storage_ptr() is valid for externally owned storage and, unlike ptr(),
    // deliberately does not expose a host-dereferenceable Vulkan address.
    (void)tensor.storage_ptr();
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

bool writes_bool(ElementwiseOp op) {
    return op == ElementwiseOp::Not || op == ElementwiseOp::And || op == ElementwiseOp::Or ||
           op == ElementwiseOp::Xor || op == ElementwiseOp::Eq || op == ElementwiseOp::Ne ||
           op == ElementwiseOp::Lt || op == ElementwiseOp::Le || op == ElementwiseOp::Gt ||
           op == ElementwiseOp::Ge || op == ElementwiseOp::IsNan || op == ElementwiseOp::IsInf ||
           op == ElementwiseOp::IsFinite;
}

void dispatch_elementwise(ElementwiseOp op, std::size_t count, const Tensor* a, const Tensor* b,
                          const Tensor* c, Tensor& out, float scalar, float scalar2) {
    struct Push {
        std::uint32_t op;
        std::uint32_t count;
        std::uint32_t dtype_a;
        std::uint32_t dtype_b;
        std::uint32_t dtype_c;
        std::uint32_t dtype_out;
        std::uint32_t off_a;
        std::uint32_t off_b;
        std::uint32_t off_c;
        std::uint32_t off_out;
        std::uint32_t elem_size_a;
        std::uint32_t elem_size_b;
        std::uint32_t elem_size_c;
        std::uint32_t elem_size_out;
        std::uint32_t has_b;
        std::uint32_t has_c;
        std::uint32_t scalar_bits;
        std::uint32_t scalar2_bits;
    } push{};
    static_assert(sizeof(push) == 72);
    push.op = static_cast<std::uint32_t>(op);
    push.count = u32(count);
    push.dtype_out = static_cast<std::uint32_t>(out.dtype());
    push.off_out = u32(byte_offset(out));
    push.elem_size_out = u32(dtype_size(out.dtype()));
    std::memcpy(&push.scalar_bits, &scalar, sizeof(scalar));
    std::memcpy(&push.scalar2_bits, &scalar2, sizeof(scalar2));
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
    if (c != nullptr) {
        push.dtype_c = static_cast<std::uint32_t>(c->dtype());
        push.off_c = u32(byte_offset(*c));
        push.elem_size_c = u32(dtype_size(c->dtype()));
        push.has_c = 1;
    }

    std::array<BufferBinding, 4> bindings{
        a != nullptr ? bind(*a) : dummy_binding(),
        b != nullptr ? bind(*b) : dummy_binding(),
        bind(out),
        c != nullptr ? bind(*c) : dummy_binding(),
    };
    Context::get().dispatch(ShaderId::Elementwise, bindings, &push, sizeof(push), groups_for(count));
}

void exclusive_or_inclusive_scan(Buffer& input, std::uint32_t src_offset, Buffer& output,
                                 std::uint32_t dst_offset, std::uint32_t count, bool is_float,
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
    } push{count, is_float ? 1U : 0U, src_offset, dst_offset, 0, exclusive ? 1U : 0U};
    std::array<BufferBinding, 3> bindings{input.binding(), output.binding(), sums->binding()};
    ctx.dispatch(ShaderId::ScanBlock, bindings, &push, sizeof(push), groups);
    if (groups == 1) {
        return;
    }
    auto sums_scan = ctx.alloc(groups * sizeof(std::uint32_t));
    exclusive_or_inclusive_scan(*sums, 0, *sums_scan, 0, groups, is_float, true);
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

BufferView buffer_view(const Tensor& tensor) {
    if (!is_vulkan_tensor(tensor))
        throw std::invalid_argument("buffer_view requires a Vulkan tensor");
    Buffer& buffer = buffer_of(tensor);
    return {buffer.handle(), static_cast<VkDeviceSize>(byte_offset(tensor)),
            static_cast<VkDeviceSize>(tensor.bytes())};
}

void adam_step(Tensor& parameter, const Tensor& gradient, Tensor& first, Tensor& second,
               const AdamStepOptions& options) {
    const auto valid = [&](const Tensor& tensor) {
        return is_vulkan_tensor(tensor) && tensor.dtype() == DataType::Float32 &&
               tensor.is_contiguous() && tensor.numel() == parameter.numel();
    };
    if (!parameter.is_valid() || !valid(parameter) || !valid(gradient) || !valid(first) ||
        !valid(second))
        throw std::invalid_argument(
            "Vulkan Adam requires equal-size contiguous Float32 Vulkan tensors");
    if (parameter.numel() == 0) return;
    if (!(options.correction1 > 0.0F) || !(options.correction2 > 0.0F) ||
        options.clamp_min > options.clamp_max)
        throw std::invalid_argument("Vulkan Adam received invalid correction or clamp bounds");
    if (options.active_row_stride != 0 &&
        (options.group_stride == 0 ||
         options.active_row_stride > options.group_stride))
        throw std::invalid_argument("Vulkan Adam received invalid active row stride");
    struct Push {
        std::uint32_t count, parameter_offset, gradient_offset, first_offset, second_offset;
        float learning_rate, secondary_learning_rate;
        std::uint32_t group_stride, active_row_stride;
        float beta1, beta2, correction1, correction2, epsilon, clamp_min, clamp_max;
        float grouped_rest_regularization;
    } push{u32(parameter.numel()), u32(byte_offset(parameter)), u32(byte_offset(gradient)),
           u32(byte_offset(first)), u32(byte_offset(second)), options.learning_rate,
           options.secondary_learning_rate, options.group_stride,
           options.active_row_stride, options.beta1, options.beta2,
           options.correction1, options.correction2, options.epsilon, options.clamp_min,
           options.clamp_max, options.grouped_rest_regularization};
    static_assert(sizeof(Push) == 68);
    std::array<BufferBinding, 4> bindings{
        bind(parameter), bind(gradient), bind(first), bind(second)};
    Context::get().dispatch(
        ShaderId::AdamF32, bindings, &push, sizeof(push), groups_for(parameter.numel()));
}

void fill(Tensor& tensor, float value) {
    if (!tensor.is_valid() || tensor.numel() == 0) {
        return;
    }
    if (!tensor.is_contiguous()) {
        struct Push {
            std::uint32_t count;
            std::uint32_t rank;
            std::uint32_t elem_size;
            std::uint32_t src_offset_elems;
            std::uint32_t dst_offset_elems;
            std::uint32_t mode;
            std::uint32_t fill_bits;
            std::uint32_t dtype;
            std::uint32_t shape[8];
            std::uint32_t stride[8];
        } push{};
        static_assert(sizeof(push) == 96);
        push.count = u32(tensor.numel());
        push.rank = u32(std::min<std::size_t>(tensor.ndim(), 8));
        push.elem_size = u32(dtype_size(tensor.dtype()));
        push.src_offset_elems = u32(tensor.storage_offset());
        push.mode = 2;
        push.dtype = static_cast<std::uint32_t>(tensor.dtype());
        std::memcpy(&push.fill_bits, &value, sizeof(value));
        for (std::size_t i = 0; i < tensor.ndim() && i < 8; ++i) {
            push.shape[i] = u32(tensor.size(i));
            push.stride[i] = u32(tensor.stride(i));
        }
        std::array<BufferBinding, 2> bindings{dummy_binding(), bind(tensor)};
        Context::get().dispatch(ShaderId::StridedCopy, bindings, &push, sizeof(push),
                                groups_for(tensor.numel()));
        return;
    }
    const bool aligned = byte_offset(tensor) % 4 == 0 && tensor.bytes() % 4 == 0;
    if (value == 0.0F && aligned) {
        Context::get().fill_zero(buffer_of(tensor), byte_offset(tensor), tensor.bytes());
        return;
    }
    dispatch_elementwise(ElementwiseOp::Fill, tensor.numel(), nullptr, nullptr, nullptr, tensor, value,
                         0.0F);
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

void unpack_training_pixels(
    const Tensor& packed, Tensor& rgb, Tensor* gray, Tensor* mask) {
    const std::size_t pixels = packed.numel();
    if (pixels == 0) {
        return;
    }
    struct Push {
        std::uint32_t pixels;
        std::uint32_t flags;
        std::uint32_t packed_offset;
        std::uint32_t rgb_offset;
        std::uint32_t gray_offset;
        std::uint32_t mask_offset;
    } push{};
    static_assert(sizeof(push) == 24);
    push.pixels = u32(pixels);
    // The optional planes stay bound to the dummy buffer when the caller does
    // not ask for them; the flag keeps the kernel from writing there.
    if (gray != nullptr) push.flags |= 1U;
    if (mask != nullptr) push.flags |= 2U;
    push.packed_offset = u32(byte_offset(packed));
    push.rgb_offset = u32(byte_offset(rgb));
    if (gray != nullptr) push.gray_offset = u32(byte_offset(*gray));
    if (mask != nullptr) push.mask_offset = u32(byte_offset(*mask));
    std::array<BufferBinding, 4> bindings{
        bind(packed),
        bind(rgb),
        gray != nullptr ? bind(*gray) : dummy_binding(),
        mask != nullptr ? bind(*mask) : dummy_binding(),
    };
    Context::get().dispatch(
        ShaderId::UnpackRgba, bindings, &push, sizeof(push), groups_for(pixels));
}

Tensor convert(const Tensor& src, DataType dtype) {
    Tensor input = prepared(src);
    Tensor out = TensorStorage::empty(input.shape(), dtype, input.capacity());
    if (input.numel() == 0) {
        return out;
    }
    dispatch_elementwise(ElementwiseOp::Convert, input.numel(), &input, nullptr, nullptr, out, 0.0F,
                         0.0F);
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
    if (src.ndim() > 8) {
        std::size_t max_index = src.storage_offset();
        for (std::size_t axis = 0; axis < src.ndim(); ++axis) {
            if (src.size(axis) == 0) {
                continue;
            }
            const std::size_t end = src.storage_offset() + (src.size(axis) - 1) * src.stride(axis);
            max_index = std::max(max_index, end);
        }
        const std::size_t elem = dtype_size(src.dtype());
        std::vector<unsigned char> raw((max_index + 1) * elem);
        Context::get().download(buffer_of(src), 0, raw.data(), raw.size());
        std::vector<unsigned char> packed(src.numel() * elem);
        std::vector<std::size_t> coord(src.ndim(), 0);
        for (std::size_t i = 0; i < src.numel(); ++i) {
            std::size_t src_index = src.storage_offset();
            for (std::size_t axis = 0; axis < src.ndim(); ++axis) {
                src_index += coord[axis] * src.stride(axis);
            }
            std::memcpy(packed.data() + i * elem, raw.data() + src_index * elem, elem);
            for (std::size_t axis = src.ndim(); axis-- > 0;) {
                coord[axis] += 1;
                if (coord[axis] < src.size(axis)) {
                    break;
                }
                coord[axis] = 0;
            }
        }
        upload(out, packed.data(), packed.size());
        return out;
    }
    struct Push {
        std::uint32_t count;
        std::uint32_t rank;
        std::uint32_t elem_size;
        std::uint32_t src_offset_elems;
        std::uint32_t dst_offset_elems;
        std::uint32_t mode;
        std::uint32_t fill_bits;
        std::uint32_t dtype;
        std::uint32_t shape[8];
        std::uint32_t stride[8];
    } push{};
    static_assert(sizeof(push) == 96);
    push.count = u32(src.numel());
    push.rank = u32(src.ndim());
    push.elem_size = u32(dtype_size(src.dtype()));
    push.src_offset_elems = u32(src.storage_offset());
    push.dtype = static_cast<std::uint32_t>(src.dtype());
    for (std::size_t i = 0; i < src.ndim(); ++i) {
        push.shape[i] = u32(src.size(i));
        push.stride[i] = u32(src.stride(i));
    }
    std::array<BufferBinding, 2> bindings{bind(src), bind(out)};
    Context::get().dispatch(ShaderId::StridedCopy, bindings, &push, sizeof(push),
                            groups_for(src.numel()));
    return out;
}

Tensor broadcast_to(const Tensor& src, const TensorShape& target) {
    if (target.rank() > 8) {
        throw std::invalid_argument("Vulkan broadcast supports rank <= 8");
    }
    if (src.shape() == target) {
        return src.clone();
    }
    Tensor input = prepared(src);
    Tensor out = TensorStorage::empty(target, input.dtype());
    if (out.numel() == 0) {
        return out;
    }
    struct Push {
        std::uint32_t count;
        std::uint32_t rank;
        std::uint32_t elem_size;
        std::uint32_t src_offset_elems;
        std::uint32_t dst_offset_elems;
        std::uint32_t mode;
        std::uint32_t fill_bits;
        std::uint32_t dtype;
        std::uint32_t shape[8];
        std::uint32_t stride[8];
    } push{};
    static_assert(sizeof(push) == 96);
    push.count = u32(out.numel());
    push.rank = u32(target.rank());
    push.elem_size = u32(dtype_size(input.dtype()));
    push.src_offset_elems = u32(input.storage_offset());
    push.dtype = static_cast<std::uint32_t>(input.dtype());
    const std::size_t leading = target.rank() - input.ndim();
    for (std::size_t axis = 0; axis < target.rank(); ++axis) {
        push.shape[axis] = u32(target[axis]);
        if (axis < leading) {
            push.stride[axis] = 0;
            continue;
        }
        const std::size_t src_axis = axis - leading;
        push.stride[axis] = input.size(src_axis) == 1 && target[axis] != 1
                                ? 0U
                                : u32(input.stride(src_axis));
    }
    std::array<BufferBinding, 2> bindings{bind(input), bind(out)};
    Context::get().dispatch(ShaderId::StridedCopy, bindings, &push, sizeof(push),
                            groups_for(out.numel()));
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

void cat_along(Tensor& dst, const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty() || dst.numel() == 0) return;
    auto& ctx = Context::get();
    std::vector<Tensor> inputs;
    inputs.reserve(tensors.size());
    std::size_t packed_bytes = 0;
    for (const Tensor& tensor : tensors) {
        inputs.push_back(prepared(tensor));
        packed_bytes += inputs.back().bytes();
    }
    auto packed = ctx.alloc(std::max<std::size_t>(packed_bytes, 16));
    std::vector<std::uint32_t> metadata(inputs.size() * 2);
    std::size_t packed_offset = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        ctx.copy(*packed, packed_offset, buffer_of(inputs[i]), byte_offset(inputs[i]),
                 inputs[i].bytes());
        metadata[i * 2] = u32(packed_offset);
        metadata[i * 2 + 1] = u32(inputs[i].size(static_cast<std::size_t>(dim)));
        packed_offset += inputs[i].bytes();
    }
    auto meta = ctx.alloc(std::max<std::size_t>(metadata.size() * sizeof(std::uint32_t), 16));
    ctx.upload(*meta, 0, metadata.data(), metadata.size() * sizeof(std::uint32_t));
    std::size_t outer = 1;
    std::size_t inner = 1;
    for (int axis = 0; axis < dim; ++axis) outer *= dst.size(static_cast<std::size_t>(axis));
    for (std::size_t axis = static_cast<std::size_t>(dim) + 1; axis < dst.ndim(); ++axis)
        inner *= dst.size(axis);
    struct Push {
        std::uint32_t count, outer, inner, n_tensors, elem_size, dst_offset;
    } push{u32(dst.numel()), u32(outer), u32(inner), u32(inputs.size()),
           u32(dtype_size(dst.dtype())), u32(byte_offset(dst))};
    std::array<BufferBinding, 3> bindings{packed->binding(), bind(dst), meta->binding()};
    ctx.dispatch(ShaderId::Cat, bindings, &push, sizeof(push), groups_for(dst.numel()));
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

void index_fill(Tensor& dst, const Tensor& indices, float value, int dim) {
    Tensor idx = prepared(indices);
    std::size_t outer = 1;
    std::size_t inner = 1;
    for (int axis = 0; axis < dim; ++axis) {
        outer *= dst.size(static_cast<std::size_t>(axis));
    }
    for (std::size_t axis = static_cast<std::size_t>(dim) + 1; axis < dst.ndim(); ++axis) {
        inner *= dst.size(axis);
    }
    struct Push {
        std::uint32_t n_indices;
        std::uint32_t outer;
        std::uint32_t dim_size;
        std::uint32_t inner;
        std::uint32_t elem_size;
        std::uint32_t dst_offset;
        std::uint32_t idx_offset;
        std::uint32_t idx_is_i64;
        std::uint32_t dtype;
        std::uint32_t value_bits;
    } push{};
    static_assert(sizeof(push) == 40);
    push.n_indices = u32(idx.numel());
    push.outer = u32(outer);
    push.dim_size = u32(dst.size(static_cast<std::size_t>(dim)));
    push.inner = u32(inner);
    push.elem_size = u32(dtype_size(dst.dtype()));
    push.dst_offset = u32(byte_offset(dst));
    push.idx_offset = u32(byte_offset(idx));
    push.idx_is_i64 = idx.dtype() == DataType::Int64 ? 1U : 0U;
    push.dtype = static_cast<std::uint32_t>(dst.dtype());
    std::memcpy(&push.value_bits, &value, sizeof(value));
    std::array<BufferBinding, 2> bindings{bind(dst), bind(idx)};
    Context::get().dispatch(ShaderId::IndexFill, bindings, &push, sizeof(push),
                            groups_for(outer * idx.numel() * inner));
}

void scatter(Tensor& dst, const Tensor& indices, const Tensor& src, int dim, bool accumulate,
             bool wrap_negative) {
    Tensor idx = prepared(indices);
    Tensor values = prepared(src);
    std::size_t outer = 1;
    std::size_t inner = 1;
    for (int axis = 0; axis < dim; ++axis) outer *= dst.size(static_cast<std::size_t>(axis));
    for (std::size_t axis = static_cast<std::size_t>(dim) + 1; axis < dst.ndim(); ++axis)
        inner *= dst.size(axis);
    struct Push {
        std::uint32_t op, outer, dim_size, inner, n_indices, elem_size;
        std::uint32_t src_offset, dst_offset, idx_offset, idx_is_i64, dtype, wrap;
    } push{};
    static_assert(sizeof(push) == 48);
    push.op = accumulate ? 1U : 0U;
    push.outer = u32(outer);
    push.dim_size = u32(dst.size(static_cast<std::size_t>(dim)));
    push.inner = u32(inner);
    push.n_indices = u32(idx.numel());
    push.elem_size = u32(dtype_size(dst.dtype()));
    push.src_offset = u32(byte_offset(values));
    push.dst_offset = u32(byte_offset(dst));
    push.idx_offset = u32(byte_offset(idx));
    push.idx_is_i64 = idx.dtype() == DataType::Int64 ? 1U : 0U;
    push.dtype = static_cast<std::uint32_t>(dst.dtype());
    push.wrap = wrap_negative ? 1U : 0U;
    std::array<BufferBinding, 3> bindings{bind(values), bind(idx), bind(dst)};
    Context::get().dispatch(ShaderId::Scatter, bindings, &push, sizeof(push),
                            groups_for(outer * idx.numel() * inner));
}

void gather(const Tensor& src, const Tensor& indices, Tensor& out, int dim, int mode) {
    if (src.ndim() > 8 || indices.ndim() > 8) {
        throw std::runtime_error("Vulkan gather supports rank <= 8");
    }
    if (out.numel() == 0) return;
    Tensor input = prepared(src);
    Tensor idx = prepared(indices);
    struct Push {
        std::uint32_t rank;
        std::uint32_t dim;
        std::uint32_t count;
        std::uint32_t mode;
        std::uint32_t elem_size;
        std::uint32_t src_offset;
        std::uint32_t dst_offset;
        std::uint32_t idx_offset;
        std::uint32_t idx_is_i64;
        std::uint32_t idx_rank;
        std::uint32_t src_shape[8];
        std::uint32_t idx_shape[8];
    } push{};
    static_assert(sizeof(push) == 104);
    push.rank = u32(input.ndim());
    push.dim = u32(static_cast<std::size_t>(dim));
    push.count = u32(out.numel());
    push.mode = static_cast<std::uint32_t>(mode);
    push.elem_size = u32(dtype_size(input.dtype()));
    push.src_offset = u32(byte_offset(input));
    push.dst_offset = u32(byte_offset(out));
    push.idx_offset = u32(byte_offset(idx));
    push.idx_is_i64 = idx.dtype() == DataType::Int64 ? 1U : 0U;
    push.idx_rank = u32(idx.ndim());
    for (std::size_t axis = 0; axis < input.ndim() && axis < 8; ++axis) {
        push.src_shape[axis] = u32(input.size(axis));
    }
    for (std::size_t axis = 0; axis < idx.ndim() && axis < 8; ++axis) {
        push.idx_shape[axis] = u32(idx.size(axis));
    }
    std::array<BufferBinding, 3> bindings{bind(input), bind(idx), bind(out)};
    Context::get().dispatch(ShaderId::Gather, bindings, &push, sizeof(push), groups_for(out.numel()));
}

void prune_masks(const Tensor& means, const Tensor& log_scales,
                 const Tensor& quaternions, const Tensor& opacity_logits,
                 const Tensor& sh, Tensor& keep, Tensor& hard,
                 Tensor& opacities, const float minimum_opacity,
                 const float maximum_bounds, const float center_x,
                 const float center_y, const float center_z) {
    const std::size_t count = static_cast<std::size_t>(means.shape()[0]);
    if (count == 0) return;
    const auto prepared_rows = [count](const Tensor& tensor) {
        if (tensor.device() != Device::Vulkan ||
            tensor.dtype() != DataType::Float32 || !tensor.is_contiguous() ||
            tensor.storage_offset() != 0 || tensor.shape()[0] != count)
            throw std::runtime_error(
                "Vulkan prune expects contiguous float32 rows sharing dim0");
        return tensor;
    };
    const auto row_means = prepared_rows(means);
    const auto row_scales = prepared_rows(log_scales);
    const auto row_rotations = prepared_rows(quaternions);
    const auto row_logits = prepared_rows(opacity_logits);
    const auto row_sh = prepared_rows(sh);
    for (const Tensor* mask : {&keep, &hard}) {
        if (mask->dtype() != DataType::Bool || !mask->is_contiguous() ||
            mask->numel() != count)
            throw std::runtime_error("Vulkan prune expects Bool [N] masks");
    }
    if (opacities.dtype() != DataType::Float32 || !opacities.is_contiguous() ||
        opacities.numel() != count)
        throw std::runtime_error("Vulkan prune expects a Float32 [N] output");

    struct Push {
        std::uint32_t count, mean_stride, scale_stride, rotation_stride;
        std::uint32_t opacity_stride, sh_stride;
        float minimum_opacity, maximum_bounds;
        float center_x, center_y, center_z;
    } push{};
    static_assert(sizeof(push) == 44);
    push.count = u32(count);
    push.mean_stride = u32(row_means.numel() / count);
    push.scale_stride = u32(row_scales.numel() / count);
    push.rotation_stride = u32(row_rotations.numel() / count);
    push.opacity_stride = u32(row_logits.numel() / count);
    push.sh_stride = u32(row_sh.numel() / count);
    push.minimum_opacity = minimum_opacity;
    push.maximum_bounds = maximum_bounds;
    push.center_x = center_x;
    push.center_y = center_y;
    push.center_z = center_z;
    std::array<BufferBinding, 8> bindings{
        bind(row_means), bind(row_scales), bind(row_rotations),
        bind(row_logits), bind(row_sh), bind(keep), bind(hard),
        bind(opacities)};
    Context::get().dispatch(ShaderId::Prune, bindings, &push, sizeof(push),
                            groups_for(count));
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
    exclusive_or_inclusive_scan(*flags, 0, *scan, 0, count, false, true);
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
    exclusive_or_inclusive_scan(*flags, 0, *scan, 0, count, false, true);

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
        std::uint32_t shape4;
        std::uint32_t shape5;
        std::uint32_t shape6;
        std::uint32_t shape7;
    } compact{};
    static_assert(sizeof(compact) == 52);
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
    if (input.ndim() > 4) compact.shape4 = u32(input.size(4));
    if (input.ndim() > 5) compact.shape5 = u32(input.size(5));
    if (input.ndim() > 6) compact.shape6 = u32(input.size(6));
    if (input.ndim() > 7) compact.shape7 = u32(input.size(7));
    std::array<BufferBinding, 3> compact_bindings{flags->binding(), scan->binding(), bind(out)};
    ctx.dispatch(ShaderId::Compact, compact_bindings, &compact, sizeof(compact), groups_for(count));
    return out;
}

Tensor masked_select(const Tensor& src, const Tensor& mask) {
    Tensor input = prepared(src);
    Tensor selector = prepared(mask);
    const std::size_t found = count_nonzero(selector);
    Tensor out = TensorStorage::empty(TensorShape{found}, input.dtype());
    if (found == 0) return out;
    auto& ctx = Context::get();
    const std::uint32_t count = u32(input.numel());
    auto flags = ctx.alloc(count * sizeof(std::uint32_t));
    auto scan = ctx.alloc(count * sizeof(std::uint32_t));
    struct MaskPush {
        std::uint32_t count, dtype, src_offset, dst_offset;
    } mask_push{count, static_cast<std::uint32_t>(selector.dtype()),
                u32(byte_offset(selector)), 0};
    std::array<BufferBinding, 2> mask_bindings{bind(selector), flags->binding()};
    ctx.dispatch(ShaderId::MaskFlags, mask_bindings, &mask_push, sizeof(mask_push),
                 groups_for(count));
    exclusive_or_inclusive_scan(*flags, 0, *scan, 0, count, false, true);
    struct SelectPush {
        std::uint32_t count, elem_size, dtype, src_offset;
        std::uint32_t flags_offset, scan_offset, dst_offset;
    } push{count, u32(dtype_size(input.dtype())), static_cast<std::uint32_t>(input.dtype()),
           u32(byte_offset(input)), 0, 0, u32(byte_offset(out))};
    std::array<BufferBinding, 4> bindings{
        bind(input), flags->binding(), scan->binding(), bind(out)};
    ctx.dispatch(ShaderId::SelectCompact, bindings, &push, sizeof(push), groups_for(count));
    return out;
}

void masked_fill(Tensor& dst, const Tensor& mask, float value) {
    Tensor selector = prepared(mask);
    dispatch_elementwise(ElementwiseOp::MaskFill, dst.numel(), &dst, &selector, nullptr, dst,
                         value, 0.0F);
}

Tensor where(const Tensor& cond, const Tensor& x, const Tensor& y) {
    Tensor condition = prepared(cond);
    Tensor lhs = prepared(x);
    Tensor rhs = prepared(y);
    Tensor out = TensorStorage::empty(lhs.shape(), lhs.dtype());
    if (out.numel() != 0)
        dispatch_elementwise(ElementwiseOp::Where, out.numel(), &condition, &lhs, &rhs, out,
                             0.0F, 0.0F);
    return out;
}

Tensor elementwise_unary(const Tensor& src, ElementwiseOp op) {
    Tensor input = prepared(src);
    const DataType out_dtype = writes_bool(op) ? DataType::Bool : input.dtype();
    Tensor out = TensorStorage::empty(input.shape(), out_dtype);
    if (input.numel() == 0) {
        return out;
    }
    dispatch_elementwise(op, input.numel(), &input, nullptr, nullptr, out, 0.0F, 0.0F);
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
    Tensor out = TensorStorage::empty(left.shape(), writes_bool(op) ? DataType::Bool : left.dtype());
    if (left.numel() == 0) {
        return out;
    }
    dispatch_elementwise(op, left.numel(), &left, &right, nullptr, out, 0.0F, 0.0F);
    return out;
}

Tensor elementwise_scalar(const Tensor& a, float scalar, ElementwiseOp op, float scalar2) {
    Tensor input = prepared(a);
    Tensor out = TensorStorage::empty(input.shape(), writes_bool(op) ? DataType::Bool : input.dtype());
    if (input.numel() == 0) {
        return out;
    }
    dispatch_elementwise(op, input.numel(), &input, nullptr, nullptr, out, scalar, scalar2);
    return out;
}

Tensor fused_pointwise(const Tensor& src, std::span<const std::uint32_t> kinds,
                       std::span<const float> scalars) {
    if (kinds.empty() || kinds.size() != scalars.size() || kinds.size() > 16)
        throw std::invalid_argument("Vulkan fused pointwise recipe must contain 1..16 ops");
    Tensor input = prepared(src);
    if (input.dtype() != DataType::Float32)
        throw std::invalid_argument("Vulkan fused pointwise requires Float32");
    Tensor out = TensorStorage::empty(input.shape(), DataType::Float32);
    struct RecipeOp {
        std::uint32_t kind;
        float scalar;
    };
    std::array<RecipeOp, 16> recipe{};
    for (std::size_t i = 0; i < kinds.size(); ++i) recipe[i] = {kinds[i], scalars[i]};
    auto& ctx = Context::get();
    auto recipe_buffer = ctx.alloc(kinds.size() * sizeof(RecipeOp));
    ctx.upload(*recipe_buffer, 0, recipe.data(), kinds.size() * sizeof(RecipeOp));
    struct Push {
        std::uint32_t count, src_offset, dst_offset, num_ops;
    } push{u32(input.numel()), u32(byte_offset(input)), u32(byte_offset(out)), u32(kinds.size())};
    std::array<BufferBinding, 3> bindings{bind(input), bind(out), recipe_buffer->binding()};
    ctx.dispatch(ShaderId::FusedPointwise, bindings, &push, sizeof(push), groups_for(input.numel()));
    return out;
}

Tensor clamp(const Tensor& src, float lo, float hi) {
    return elementwise_scalar(src, lo, ElementwiseOp::Clamp, hi);
}

Tensor reduce(const Tensor& src, const std::vector<int>& axes, bool keepdim, ReduceKind kind) {
    Tensor input = prepared(src);
    if (input.ndim() > 8) throw std::invalid_argument("Vulkan reduce supports rank <= 8");
    std::array<bool, 8> reduced{};
    if (axes.empty()) {
        for (std::size_t i = 0; i < input.ndim(); ++i) reduced[i] = true;
    } else {
        for (int axis : axes) {
            if (axis < 0) axis += static_cast<int>(input.ndim());
            if (axis < 0 || axis >= static_cast<int>(input.ndim()))
                throw std::invalid_argument("Vulkan reduce axis out of range");
            reduced[static_cast<std::size_t>(axis)] = true;
        }
    }
    std::vector<std::size_t> out_dims;
    std::size_t reduce_count = 1;
    for (std::size_t axis = 0; axis < input.ndim(); ++axis) {
        if (reduced[axis]) {
            reduce_count *= input.size(axis);
            if (keepdim) out_dims.push_back(1);
        } else {
            out_dims.push_back(input.size(axis));
        }
    }
    DataType out_dtype = input.dtype();
    if (kind == ReduceKind::Any || kind == ReduceKind::All) out_dtype = DataType::Bool;
    if (kind == ReduceKind::Argmax || kind == ReduceKind::Argmin ||
        (input.dtype() == DataType::Bool && kind != ReduceKind::Any && kind != ReduceKind::All))
        out_dtype = DataType::Int64;
    Tensor out = TensorStorage::empty(TensorShape(out_dims), out_dtype);
    if (out.numel() == 0) return out;
    if (input.numel() == 0) {
        float identity = 0.0F;
        if (kind == ReduceKind::Prod || kind == ReduceKind::All) identity = 1.0F;
        if (kind == ReduceKind::Max) identity = -std::numeric_limits<float>::infinity();
        if (kind == ReduceKind::Min) identity = std::numeric_limits<float>::infinity();
        fill(out, identity);
        return out;
    }

    // The generic shader assigns one thread to each output and walks the whole
    // reduced axis. A full reduction is then a single thread, and a long
    // contiguous row is a long serial loop. Workgroup trees and one group per
    // row cover those cases. Intermediate buffers stay alive until this call
    // returns; the context keeps recorded batches from recycling them.
    const bool all_axes = axes.empty() ||
        std::all_of(reduced.begin(), reduced.begin() + input.ndim(), [](bool value) {
            return value;
        });
    const bool scalar_float = all_axes && input.dtype() == DataType::Float32 &&
                              out_dtype == DataType::Float32 &&
                              (kind == ReduceKind::Sum || kind == ReduceKind::Mean ||
                               kind == ReduceKind::Max || kind == ReduceKind::Min);
    const bool scalar_mask = all_axes &&
                             (kind == ReduceKind::Any || kind == ReduceKind::All) &&
                             (input.dtype() == DataType::Bool || input.dtype() == DataType::Float32);
    if (scalar_float || scalar_mask) {
        constexpr std::uint32_t items_per_group = kGroupSize * 4U;
        struct FastPush {
            std::uint32_t count, op, src_offset, dst_offset, original_count, final_pass, src_elem;
        } push{};
        static_assert(sizeof(push) == 28);
        push.op = static_cast<std::uint32_t>(kind);
        push.original_count = u32(input.numel());
        std::uint32_t src_elem = scalar_mask ? u32(dtype_size(input.dtype())) : 4U;

        auto& ctx = Context::get();
        Buffer* current = &buffer_of(input);
        std::uint32_t current_offset = u32(byte_offset(input));
        std::uint32_t current_count = u32(input.numel());
        std::vector<std::shared_ptr<Buffer>> partials;
        while (true) {
            const std::uint32_t groups = std::max(1U, ceil_div(current_count, items_per_group));
            const bool final_pass = groups == 1U;
            std::shared_ptr<Buffer> partial;
            Buffer* destination = nullptr;
            std::uint32_t destination_offset = 0;
            if (final_pass) {
                destination = &buffer_of(out);
                destination_offset = u32(byte_offset(out));
            } else {
                partial = ctx.alloc(static_cast<std::size_t>(groups) * sizeof(float));
                destination = partial.get();
            }
            push.count = current_count;
            push.src_offset = current_offset;
            push.dst_offset = destination_offset;
            push.final_pass = final_pass ? 1U : 0U;
            push.src_elem = src_elem;
            std::array<BufferBinding, 2> bindings{current->binding(), destination->binding()};
            ctx.dispatch(ShaderId::ReduceAllF32, bindings, &push, sizeof(push), groups);
            if (final_pass) break;
            partials.push_back(std::move(partial));
            current = partials.back().get();
            current_offset = 0;
            current_count = groups;
            src_elem = 4U;
        }
        return out;
    }
    if (all_axes && input.dtype() == DataType::Float32 &&
        (kind == ReduceKind::Argmax || kind == ReduceKind::Argmin)) {
        constexpr std::uint32_t items_per_group = kGroupSize * 4U;
        struct ArgPush {
            std::uint32_t count, argmax, src_offset, dst_offset, src_is_pair, final_pass;
        } push{};
        static_assert(sizeof(push) == 24);
        push.argmax = kind == ReduceKind::Argmax ? 1U : 0U;
        auto& ctx = Context::get();
        Buffer* current = &buffer_of(input);
        std::uint32_t current_offset = u32(byte_offset(input));
        std::uint32_t current_count = u32(input.numel());
        std::uint32_t src_is_pair = 0;
        std::vector<std::shared_ptr<Buffer>> partials;
        while (true) {
            const std::uint32_t groups = std::max(1U, ceil_div(current_count, items_per_group));
            const bool final_pass = groups == 1U;
            std::shared_ptr<Buffer> partial;
            Buffer* destination = nullptr;
            std::uint32_t destination_offset = 0;
            if (final_pass) {
                destination = &buffer_of(out);
                destination_offset = u32(byte_offset(out));
            } else {
                partial = ctx.alloc(static_cast<std::size_t>(groups) * sizeof(float) * 2U);
                destination = partial.get();
            }
            push.count = current_count;
            push.src_offset = current_offset;
            push.dst_offset = destination_offset;
            push.src_is_pair = src_is_pair;
            push.final_pass = final_pass ? 1U : 0U;
            std::array<BufferBinding, 2> bindings{current->binding(), destination->binding()};
            ctx.dispatch(ShaderId::ReduceAllArg, bindings, &push, sizeof(push), groups);
            if (final_pass) break;
            partials.push_back(std::move(partial));
            current = partials.back().get();
            current_offset = 0;
            current_count = groups;
            src_is_pair = 1U;
        }
        return out;
    }
    int reduced_axis = -1;
    int reduced_axes = 0;
    for (std::size_t axis = 0; axis < input.ndim(); ++axis) {
        if (!reduced[axis]) continue;
        reduced_axis = static_cast<int>(axis);
        ++reduced_axes;
    }
    if (reduced_axes == 1 && input.is_contiguous() &&
        (input.dtype() == DataType::Float32 || input.dtype() == DataType::Bool)) {
        std::size_t outer = 1;
        std::size_t inner = 1;
        for (int axis = 0; axis < reduced_axis; ++axis) outer *= input.size(static_cast<std::size_t>(axis));
        for (std::size_t axis = static_cast<std::size_t>(reduced_axis) + 1; axis < input.ndim(); ++axis)
            inner *= input.size(axis);
        const std::size_t reduce_n = input.size(static_cast<std::size_t>(reduced_axis));
        const bool supported =
            kind == ReduceKind::Sum || kind == ReduceKind::Mean || kind == ReduceKind::Max ||
            kind == ReduceKind::Min || kind == ReduceKind::Prod || kind == ReduceKind::Any ||
            kind == ReduceKind::All || kind == ReduceKind::Argmax || kind == ReduceKind::Argmin;
        if (supported && outer * std::max<std::size_t>(inner, 1) > 1) {
            struct AxisPush {
                std::uint32_t mode, outer, reduce, inner, op;
                std::uint32_t src_offset, dst_offset, elem_in, elem_out, dtype_in, dtype_out;
            } push{};
            static_assert(sizeof(push) == 44);
            const bool contiguous_row = inner == 1 && reduce_n >= 64;
            push.mode = contiguous_row ? 1U : 0U;
            push.outer = u32(outer);
            push.reduce = u32(reduce_n);
            push.inner = u32(std::max<std::size_t>(inner, 1));
            push.op = static_cast<std::uint32_t>(kind);
            push.src_offset = u32(byte_offset(input));
            push.dst_offset = u32(byte_offset(out));
            push.elem_in = u32(dtype_size(input.dtype()));
            push.elem_out = u32(dtype_size(out.dtype()));
            push.dtype_in = static_cast<std::uint32_t>(input.dtype());
            push.dtype_out = static_cast<std::uint32_t>(out_dtype);
            std::array<BufferBinding, 2> bindings{bind(input), bind(out)};
            const std::uint32_t groups = contiguous_row
                                             ? std::max(1U, push.outer)
                                             : groups_for(outer * std::max<std::size_t>(inner, 1));
            Context::get().dispatch(ShaderId::ReduceAxis, bindings, &push, sizeof(push), groups);
            return out;
        }
    }

    std::array<std::uint32_t, 24> metadata{};
    for (std::size_t axis = 0; axis < input.ndim(); ++axis) {
        metadata[axis] = u32(input.size(axis));
        metadata[8 + axis] = reduced[axis] ? 1U : 0U;
        metadata[16 + axis] = reduced[axis] ? 1U : u32(input.size(axis));
    }
    auto meta = Context::get().alloc(sizeof(metadata));
    Context::get().upload(*meta, 0, metadata.data(), sizeof(metadata));
    struct Push {
        std::uint32_t out_count, rank, reduce_count, op, dtype_in;
        std::uint32_t dtype_out, src_offset, dst_offset, elem_in, elem_out;
    } push{u32(out.numel()), u32(input.ndim()), u32(reduce_count),
           static_cast<std::uint32_t>(kind), static_cast<std::uint32_t>(input.dtype()),
           static_cast<std::uint32_t>(out_dtype), u32(byte_offset(input)), u32(byte_offset(out)),
           u32(dtype_size(input.dtype())), u32(dtype_size(out_dtype))};
    std::array<BufferBinding, 3> bindings{bind(input), bind(out), meta->binding()};
    Context::get().dispatch(ShaderId::Reduce, bindings, &push, sizeof(push),
                            groups_for(out.numel()));
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_b, std::size_t batch) {
    Tensor left = prepared(a);
    Tensor right = prepared(b);
    if (left.dtype() != DataType::Float32 || right.dtype() != DataType::Float32)
        throw std::invalid_argument("Vulkan matmul currently requires Float32 tensors");
    const std::size_t m = left.size(left.ndim() - 2);
    const std::size_t k = left.size(left.ndim() - 1);
    const std::size_t n = transpose_b ? right.size(right.ndim() - 2)
                                      : right.size(right.ndim() - 1);
    const bool batched_output = left.ndim() == 3 || right.ndim() == 3;
    TensorShape out_shape = batched_output ? TensorShape{batch, m, n} : TensorShape{m, n};
    Tensor out = TensorStorage::empty(out_shape, DataType::Float32);
    struct Push {
        std::uint32_t m, n, k, batch, a_offset, b_offset, c_offset;
        std::uint32_t a_row_stride, a_col_stride, b_row_stride, b_col_stride;
        std::uint32_t batch_stride_a, batch_stride_b, batch_stride_c;
    } push{};
    static_assert(sizeof(push) == 56);
    push.m = u32(m); push.n = u32(n); push.k = u32(k); push.batch = u32(batch);
    push.a_offset = u32(byte_offset(left)); push.b_offset = u32(byte_offset(right));
    push.c_offset = u32(byte_offset(out));
    push.a_row_stride = u32(k); push.a_col_stride = 1;
    push.b_row_stride = transpose_b ? 1U : u32(n);
    push.b_col_stride = transpose_b ? u32(k) : 1U;
    push.batch_stride_a = left.ndim() == 3 ? u32(m * k) : 0U;
    push.batch_stride_b = right.ndim() == 3 ? u32(k * n) : 0U;
    push.batch_stride_c = u32(m * n);
    std::array<BufferBinding, 3> bindings{bind(left), bind(right), bind(out)};
    Context::get().dispatch(ShaderId::Matmul, bindings, &push, sizeof(push),
                            std::max(1U, ceil_div(u32(n), 16)),
                            std::max(1U, ceil_div(u32(m), 16)), u32(batch));
    return out;
}

Tensor cumsum(const Tensor& src, int dim) {
    Tensor input = prepared(src);
    Tensor out = TensorStorage::empty(input.shape(), input.dtype());
    std::size_t outer = 1, inner = 1;
    for (int axis = 0; axis < dim; ++axis) outer *= input.size(static_cast<std::size_t>(axis));
    for (std::size_t axis = static_cast<std::size_t>(dim) + 1; axis < input.ndim(); ++axis)
        inner *= input.size(axis);
    const std::size_t length = input.size(static_cast<std::size_t>(dim));
    // A single thread walking a million-element line dominates the step.
    // Contiguous Float32 lines of at least 4096 elements scan in 1024-wide
    // chunks, then one group turns the chunk totals into exclusive prefixes.
    constexpr std::uint32_t kChunk = 1024;
    const std::uint32_t chunks = length == 0 ? 0U : ceil_div(u32(length), kChunk);
    if (input.dtype() == DataType::Float32 && inner == 1 && length >= 4096 && chunks <= kChunk &&
        outer > 0) {
        auto& ctx = Context::get();
        auto sums = ctx.alloc(outer * static_cast<std::size_t>(chunks) * sizeof(float));
        struct ScanPush {
            std::uint32_t mode, rows, length, num_chunks, src_offset, dst_offset, sums_offset, chunk;
        } scan{};
        static_assert(sizeof(scan) == 32);
        scan.rows = u32(outer);
        scan.length = u32(length);
        scan.num_chunks = chunks;
        scan.chunk = kChunk;
        scan.sums_offset = 0;
        std::array<BufferBinding, 3> bindings{bind(input), bind(out), sums->binding()};
        scan.mode = 0;
        scan.src_offset = u32(byte_offset(input));
        scan.dst_offset = u32(byte_offset(out));
        ctx.dispatch(ShaderId::CumsumScan, bindings, &scan, sizeof(scan),
                     std::max(1U, scan.rows * chunks));
        if (chunks > 1U) {
            scan.mode = 1;
            scan.src_offset = 0;
            scan.dst_offset = 0;
            scan.length = chunks;
            std::array<BufferBinding, 3> sum_bindings{sums->binding(), sums->binding(), sums->binding()};
            ctx.dispatch(ShaderId::CumsumScan, sum_bindings, &scan, sizeof(scan), scan.rows);
            scan.mode = 2;
            scan.length = u32(length);
            scan.src_offset = u32(byte_offset(out));
            scan.dst_offset = u32(byte_offset(out));
            std::array<BufferBinding, 3> add_bindings{bind(out), bind(out), sums->binding()};
            ctx.dispatch(ShaderId::CumsumScan, add_bindings, &scan, sizeof(scan),
                         groups_for(outer * length));
        }
        return out;
    }
    struct Push {
        std::uint32_t outer, dim_size, inner, src_offset, dst_offset, dtype, elem_size;
    } push{u32(outer), u32(length), u32(inner),
           u32(byte_offset(input)), u32(byte_offset(out)),
           static_cast<std::uint32_t>(input.dtype()), u32(dtype_size(input.dtype()))};
    std::array<BufferBinding, 2> bindings{bind(input), bind(out)};
    Context::get().dispatch(ShaderId::Cumsum, bindings, &push, sizeof(push),
                            groups_for(outer * inner));
    return out;
}

namespace {
void dispatch_random(Tensor& dst, const Tensor* src, std::uint32_t op, float p0, float p1,
                     std::uint32_t seed, std::uint32_t rows = 0, std::uint32_t cols = 0) {
    struct Push {
        std::uint32_t op, count, seed, dst_offset, src_offset, dtype;
        std::uint32_t elem_size, rows, cols, p0_bits, p1_bits;
    } push{};
    static_assert(sizeof(push) == 44);
    push.op = op; push.count = u32(dst.numel()); push.seed = seed;
    push.dst_offset = u32(byte_offset(dst));
    push.src_offset = src != nullptr ? u32(byte_offset(*src)) : 0U;
    push.dtype = static_cast<std::uint32_t>(dst.dtype());
    push.elem_size = u32(dtype_size(dst.dtype())); push.rows = rows; push.cols = cols;
    std::memcpy(&push.p0_bits, &p0, sizeof(p0));
    std::memcpy(&push.p1_bits, &p1, sizeof(p1));
    std::array<BufferBinding, 2> bindings{bind(dst), src != nullptr ? bind(*src) : dummy_binding()};
    Context::get().dispatch(ShaderId::Random, bindings, &push, sizeof(push),
                            groups_for(dst.numel()));
}

void dispatch_pool(const Tensor& src, Tensor& dst, std::uint32_t op, int kernel, int stride,
                   int padding, int h_out, int w_out) {
    Tensor input = prepared(src);
    struct Push {
        std::uint32_t op, n, c, h_in, w_in, h_out, w_out, kernel, stride, padding;
        std::uint32_t src_offset, dst_offset;
    } push{op, u32(input.size(0)), u32(input.size(1)), u32(input.size(2)), u32(input.size(3)),
           u32(h_out), u32(w_out), u32(kernel), u32(stride), u32(padding),
           u32(byte_offset(input)), u32(byte_offset(dst))};
    std::array<BufferBinding, 2> bindings{bind(input), bind(dst)};
    Context::get().dispatch(ShaderId::Pool, bindings, &push, sizeof(push),
                            groups_for(dst.numel()));
}
} // namespace

void random_uniform(Tensor& dst, float low, float high, std::uint32_t seed) {
    dispatch_random(dst, nullptr, 0, low, high, seed);
}

void random_normal(Tensor& dst, float mean, float stddev, std::uint32_t seed) {
    dispatch_random(dst, nullptr, 1, mean, stddev, seed);
}

void random_bernoulli(Tensor& dst, float p, std::uint32_t seed) {
    dispatch_random(dst, nullptr, 2, p, 0.0F, seed);
}

void random_randint(Tensor& dst, int low, int high, std::uint32_t seed) {
    dispatch_random(dst, nullptr, 3, static_cast<float>(low), static_cast<float>(high), seed);
}

void arange(Tensor& dst, float start, float step) {
    dispatch_random(dst, nullptr, 4, start, step, 0);
}

void eye(Tensor& dst, std::uint32_t rows, std::uint32_t cols) {
    dispatch_random(dst, nullptr, 5, 0.0F, 0.0F, 0, rows, cols);
}

void diag(Tensor& dst, const Tensor& diagonal) {
    Tensor src = diagonal.dtype() == DataType::Float32 ? prepared(diagonal)
                                                       : convert(diagonal, DataType::Float32);
    dispatch_random(dst, &src, 6, 0.0F, 0.0F, 0, u32(src.numel()), u32(dst.size(1)));
}

void max_pool2d(const Tensor& src, Tensor& dst, int kernel, int stride, int padding,
                int h_out, int w_out) {
    dispatch_pool(src, dst, 0, kernel, stride, padding, h_out, w_out);
}

void adaptive_avg_pool2d(const Tensor& src, Tensor& dst, int h_out, int w_out) {
    dispatch_pool(src, dst, 1, 0, 0, 0, h_out, w_out);
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
    exclusive_or_inclusive_scan(buffer_of(input), u32(byte_offset(input)), *cdf, 0, n, true,
                                false);
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

std::pair<Tensor, Tensor> sort_1d_f32(const Tensor& values, bool descending) {
    const std::size_t count = values.numel();
    if (count == 0) {
        return {Tensor::empty({0}, Device::Vulkan, DataType::Float32),
                Tensor::empty({0}, Device::Vulkan, DataType::Int64)};
    }
    if (count == 1) {
        return {values.clone(), Tensor::zeros({1}, Device::Vulkan, DataType::Int64)};
    }
    std::size_t padded = 1;
    while (padded < count) padded <<= 1;
    if (padded > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Vulkan sort exceeds 32-bit indexing");
    }
    auto a_values = Tensor::empty({padded}, Device::Vulkan, DataType::Float32);
    auto a_indices = Tensor::empty({padded}, Device::Vulkan, DataType::Int64);
    auto b_values = Tensor::empty({padded}, Device::Vulkan, DataType::Float32);
    auto b_indices = Tensor::empty({padded}, Device::Vulkan, DataType::Int64);
    auto current_values = values;
    Tensor current_indices;
    bool write_a = true;
    bool first_pass = true;
    struct Push {
        std::uint32_t count;
        std::uint32_t padded;
        std::uint32_t merge_size;
        std::uint32_t compare_distance;
        std::uint32_t descending;
        std::uint32_t first_pass;
        std::uint32_t src_offset;
    } push{};
    static_assert(sizeof(push) == 28);
    push.count = u32(count);
    push.padded = u32(padded);
    push.descending = descending ? 1U : 0U;
    auto& ctx = Context::get();
    for (std::size_t merge = 2; merge <= padded; merge <<= 1) {
        push.merge_size = u32(merge);
        for (std::size_t distance = merge >> 1; distance != 0; distance >>= 1) {
            push.compare_distance = u32(distance);
            push.first_pass = first_pass ? 1U : 0U;
            push.src_offset = first_pass ? u32(byte_offset(values)) : 0U;
            auto& dst_values = write_a ? a_values : b_values;
            auto& dst_indices = write_a ? a_indices : b_indices;
            std::array<BufferBinding, 4> bindings{
                bind(current_values),
                first_pass ? dummy_binding() : bind(current_indices),
                bind(dst_values), bind(dst_indices)};
            ctx.dispatch(ShaderId::SortBitonic, bindings, &push, sizeof(push),
                         groups_for(padded / 2));
            current_values = dst_values;
            current_indices = dst_indices;
            write_a = !write_a;
            first_pass = false;
        }
    }
    return {current_values.slice(0, 0, count),
            current_indices.slice(0, 0, count)};
}

namespace {

// Two 16-bit digits cover the whole 32-bit key, so a selection costs two
// histogram passes and one emit pass no matter how large the row count is.
// The bins stay device-local; only the 256 KiB histogram crosses to the host.
constexpr std::uint32_t kSelectBinCount = 1U << 16;

struct SelectPush {
    std::uint32_t count;
    std::uint32_t src_offset;
    std::uint32_t mask_offset;
    std::uint32_t has_mask;
    std::uint32_t bins_offset;
    std::uint32_t dst_offset;
    std::uint32_t prefix_hi;
    std::uint32_t prefix_lo;
    std::uint32_t prefix_bits;
    std::uint32_t limit;
    std::uint32_t descending;
    std::uint32_t mode;
    std::uint32_t threshold_hi;
    std::uint32_t threshold_lo;
};
static_assert(sizeof(SelectPush) == 56);

// Host twin of the shader's key transform, used to turn the resolved key back
// into the value it came from without a third pass.
float value_of_select_key(std::uint32_t select_key, bool descending) {
    const std::uint32_t ordered = descending ? ~select_key : select_key;
    const std::uint32_t bits =
        (ordered & 0x80000000U) != 0U ? (ordered & 0x7FFFFFFFU) : ~ordered;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Resolves the exact key of the `rank`-th row (1-based) in the requested
// direction among the rows `mask` selects, and reports how many rows hold a
// strictly smaller key. The key is 64 bits wide in effect - the value's
// transform in the high half, the row index in the low half - which is what
// makes ties resolve by row index exactly as the sort did.
bool select_resolve_rank(const Tensor& values, Buffer& bins, const Tensor& mask,
                         const std::uint32_t rank, const bool descending,
                         std::uint32_t* key_hi, std::uint32_t* key_lo,
                         std::uint64_t* below) {
    const std::size_t rows = values.numel();
    if (rows == 0 || rank == 0) {
        return false;
    }
    const bool masked = mask.is_valid();
    if (masked && mask.numel() != rows) {
        throw std::invalid_argument("Vulkan select mask must match the values");
    }
    auto& ctx = Context::get();
    std::array<BufferBinding, 4> bindings{
        bind(values), masked ? bind(mask) : dummy_binding(), bins.binding(),
        dummy_binding()};
    SelectPush push{};
    push.count = u32(rows);
    push.src_offset = u32(byte_offset(values));
    push.mask_offset = masked ? u32(byte_offset(mask)) : 0U;
    push.has_mask = masked ? 1U : 0U;
    push.bins_offset = 0U;
    push.descending = descending ? 1U : 0U;
    push.mode = 0U;

    std::vector<std::uint32_t> histogram(kSelectBinCount);
    std::uint32_t prefix_hi = 0;
    std::uint32_t prefix_lo = 0;
    std::uint64_t below_total = 0;
    std::uint64_t remaining = rank;
    for (std::uint32_t pass = 0; pass < 4U; ++pass) {
        ctx.fill_zero(bins, 0, kSelectBinCount * sizeof(std::uint32_t));
        push.prefix_hi = prefix_hi;
        push.prefix_lo = prefix_lo;
        push.prefix_bits = pass * 16U;
        ctx.dispatch(ShaderId::Select, bindings, &push, sizeof(push), groups_for(rows));
        ctx.download(bins, 0, histogram.data(), kSelectBinCount * sizeof(std::uint32_t));

        // Ascending walk: the first bin whose running total reaches the
        // remaining rank holds the answer, and every row counted before it
        // sorts strictly below the answer.
        std::uint32_t chosen = 0;
        bool found = false;
        std::uint64_t running = 0;
        for (std::uint32_t bin = 0; bin < kSelectBinCount; ++bin) {
            const std::uint64_t next = running + histogram[bin];
            if (next >= remaining) {
                chosen = bin;
                found = true;
                break;
            }
            running = next;
        }
        if (!found) {
            // Fewer rows qualify than the caller asked for. Taking the
            // highest key instead of the rank-th gives the caller exactly
            // those rows, which is what a sort-and-truncate would have
            // returned; only an empty selection is a genuine failure.
            if (running == 0) {
                return false;
            }
            remaining = running;
            for (std::uint32_t bin = kSelectBinCount; bin-- > 0;) {
                if (histogram[bin] != 0) {
                    chosen = bin;
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
            running -= histogram[chosen];
        }
        below_total += running;
        remaining -= running;
        if (pass < 2U) {
            prefix_hi = (prefix_hi << 16U) | chosen;
        } else {
            prefix_lo = (prefix_lo << 16U) | chosen;
        }
    }
    *key_hi = prefix_hi;
    *key_lo = prefix_lo;
    *below = below_total;
    return true;
}

}  // namespace

bool select_nth_value(const Tensor& values, const Tensor& mask, const std::uint32_t rank,
                      const bool descending, float* value) {
    if (value == nullptr) {
        return false;
    }
    Tensor input = prepared(values);
    Tensor selector = mask.is_valid() ? prepared(mask) : Tensor{};
    auto bins = Context::get().alloc(kSelectBinCount * sizeof(std::uint32_t));
    std::uint32_t key_hi = 0;
    std::uint32_t key_lo = 0;
    std::uint64_t below = 0;
    if (!select_resolve_rank(input, *bins, selector, rank, descending, &key_hi, &key_lo,
                             &below)) {
        return false;
    }
    *value = value_of_select_key(key_hi, descending);
    return true;
}

Tensor select_topk_indices(const Tensor& values, const Tensor& mask, const std::uint32_t count,
                           const bool descending, std::uint32_t* emitted) {
    if (emitted != nullptr) {
        *emitted = 0;
    }
    if (count == 0) {
        return {};
    }
    Tensor input = prepared(values);
    Tensor selector = mask.is_valid() ? prepared(mask) : Tensor{};
    auto& ctx = Context::get();
    auto bins = ctx.alloc(kSelectBinCount * sizeof(std::uint32_t));
    std::uint32_t key_hi = 0;
    std::uint32_t key_lo = 0;
    std::uint64_t below = 0;
    if (!select_resolve_rank(input, *bins, selector, count, descending, &key_hi, &key_lo,
                             &below)) {
        return {};
    }
    // The histogram passes are done with the bins; the first one becomes the
    // emit ticket, which is also where the caller reads the written count.
    Tensor out = TensorStorage::empty(TensorShape{count}, DataType::Int32);
    const bool masked = selector.is_valid();
    ctx.fill_zero(*bins, 0, sizeof(std::uint32_t));
    SelectPush push{};
    push.count = u32(input.numel());
    push.src_offset = u32(byte_offset(input));
    push.mask_offset = masked ? u32(byte_offset(selector)) : 0U;
    push.has_mask = masked ? 1U : 0U;
    push.bins_offset = 0U;
    push.dst_offset = u32(byte_offset(out));
    push.limit = count;
    push.descending = descending ? 1U : 0U;
    push.mode = 1U;
    push.threshold_hi = key_hi;
    push.threshold_lo = key_lo;
    std::array<BufferBinding, 4> bindings{
        bind(input), masked ? bind(selector) : dummy_binding(), bins->binding(), bind(out)};
    ctx.dispatch(ShaderId::Select, bindings, &push, sizeof(push),
                 groups_for(input.numel()));
    std::uint32_t written = 0;
    ctx.download(*bins, 0, &written, sizeof(written));
    if (written == 0) {
        return {};
    }
    const std::size_t keep = std::min<std::size_t>(written, count);
    if (emitted != nullptr) {
        *emitted = u32(keep);
    }
    return out.slice(0, 0, keep);
}

} // namespace tinytensor::vulkan
