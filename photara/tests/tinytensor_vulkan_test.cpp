#include "internal/tensor_impl.hpp"
#include "vulkan/backend.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(const float actual, const float expected, const float eps, const char* message) {
    if (std::abs(actual - expected) > eps) {
        throw std::runtime_error(
            std::string(message) + " actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
    }
}

void require_vector_near(const std::vector<float>& actual, const std::vector<float>& expected,
                         const float eps, const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require_near(actual[i], expected[i], eps, message);
    }
}

void test_factory_and_roundtrip() {
    using namespace tinytensor;
    auto zeros = Tensor::zeros({4, 3}, Device::Vulkan, DataType::Float32);
    require(zeros.device() == Device::Vulkan, "zeros device");
    require(zeros.numel() == 12, "zeros numel");
    const auto zero_values = zeros.to_vector();
    require(zero_values.size() == 12, "zeros download size");
    for (float value : zero_values) {
        require_near(value, 0.0F, 1e-6F, "zeros value");
    }

    const std::vector<float> host{1.F, 2.F, 3.F, 4.F, 5.F, 6.F};
    auto tensor = Tensor::from_vector(host, {2, 3}, Device::Vulkan);
    require(tensor.to_vector() == host, "from_vector roundtrip");
    const auto handles = vulkan::device_handles();
    require(handles.instance != VK_NULL_HANDLE && handles.physical_device != VK_NULL_HANDLE &&
                handles.device != VK_NULL_HANDLE && handles.queue != VK_NULL_HANDLE &&
                handles.queue_family != VK_QUEUE_FAMILY_IGNORED,
            "Vulkan device interop handles");
    const auto view = vulkan::buffer_view(tensor);
    require(view.buffer != VK_NULL_HANDLE && view.offset == 0 && view.bytes == host.size() * 4,
            "Vulkan tensor buffer view");

    auto cloned = tensor.clone();
    require(cloned.to_vector() == host, "clone");

    auto cpu = tensor.to(Device::CPU);
    require(cpu.device() == Device::CPU, "to cpu device");
    require(cpu.to_vector() == host, "to cpu values");
    auto back = cpu.to(Device::Vulkan);
    require(back.to_vector() == host, "cpu->vulkan");
}

void test_cat_index_squeeze() {
    using namespace tinytensor;
    auto a = Tensor::from_vector(std::vector<float>{1.F, 2.F, 3.F}, {1, 3}, Device::Vulkan);
    auto b = Tensor::from_vector(std::vector<float>{4.F, 5.F, 6.F}, {1, 3}, Device::Vulkan);
    auto cat = Tensor::cat({a, b}, 0);
    require(cat.shape() == TensorShape({2, 3}), "cat shape");
    const auto cat_values = cat.to_vector();
    require(cat_values == std::vector<float>({1.F, 2.F, 3.F, 4.F, 5.F, 6.F}), "cat values");

    auto indices = Tensor::from_vector(std::vector<int>{1, 0}, {2}, Device::Vulkan);
    auto selected = cat.index_select(0, indices);
    require(selected.to_vector() == std::vector<float>({4.F, 5.F, 6.F, 1.F, 2.F, 3.F}),
            "index_select");

    auto row = Tensor::from_vector(std::vector<float>{7.F, 8.F, 9.F}, {1, 3}, Device::Vulkan);
    auto squeezed = row.squeeze(0);
    require(squeezed.ndim() == 1, "squeeze rank");
    require(squeezed.to_vector() == std::vector<float>({7.F, 8.F, 9.F}), "squeeze values");
}

void test_mask_and_convert() {
    using namespace tinytensor;
    auto mask = Tensor::from_vector(
        std::vector<bool>{true, false, true, false, true}, {5}, Device::Vulkan);
    auto inverted = !mask;
    require(inverted.to_vector_bool() == std::vector<bool>({false, true, false, true, false}),
            "logical_not");

    auto other = Tensor::from_vector(
        std::vector<bool>{true, true, false, false, true}, {5}, Device::Vulkan);
    auto both = mask.logical_and(other);
    require(both.to_vector_bool() == std::vector<bool>({true, false, false, false, true}),
            "logical_and");

    auto nz = mask.nonzero().squeeze(1).to(DataType::Int32);
    require(nz.to_vector_int() == std::vector<int>({0, 2, 4}), "nonzero");

    auto values = Tensor::from_vector(std::vector<float>{0.F, 1.F, 2.F, 0.F, 4.F}, {5}, Device::Vulkan);
    require(values.gt(0.F).to_vector_bool() ==
                std::vector<bool>({false, true, true, false, true}),
            "gt scalar");
}

void test_multinomial() {
    using namespace tinytensor;
    auto weights = Tensor::from_vector(std::vector<float>{0.F, 1.F, 0.F, 0.F}, {4}, Device::Vulkan);
    auto samples = Tensor::multinomial(weights, 8, true);
    require(samples.dtype() == DataType::Int64, "multinomial dtype");
    require(samples.numel() == 8, "multinomial count");
    for (const auto index : samples.to_vector_int64()) {
        require(index == 1, "multinomial should sample the only nonzero bin");
    }

    auto unique = Tensor::multinomial(
        Tensor::from_vector(std::vector<float>{1.F, 1.F, 1.F}, {3}, Device::Vulkan), 3, false);
    auto picked = unique.to_vector_int64();
    require(picked.size() == 3, "without-replacement size");
    std::vector<int> seen(3, 0);
    for (const auto index : picked) {
        require(index >= 0 && index < 3, "without-replacement range");
        seen[static_cast<int>(index)] += 1;
    }
    require(seen[0] == 1 && seen[1] == 1 && seen[2] == 1, "without-replacement unique");
}

void test_elementwise_and_where() {
    using namespace tinytensor;
    auto a = Tensor::from_vector(std::vector<float>{-1.F, 0.F, 1.F, 2.F}, {2, 2}, Device::Vulkan);
    auto b = Tensor::from_vector(std::vector<float>{2.F, 3.F}, {1, 2}, Device::Vulkan);
    require_vector_near((a + b).to_vector(), {1.F, 3.F, 3.F, 5.F}, 1e-5F, "broadcast add");
    require_vector_near(a.abs().to_vector(), {1.F, 0.F, 1.F, 2.F}, 1e-5F, "abs");
    require_vector_near(a.exp().log().to_vector(), a.to_vector(), 2e-5F, "exp/log");
    require_vector_near(a.clamp(0.F, 1.F).to_vector(), {0.F, 0.F, 1.F, 1.F}, 1e-5F,
                        "clamp");
    auto condition = a.gt(0.F);
    auto selected = Tensor::where(condition, a, Tensor::zeros(a.shape(), Device::Vulkan));
    require_vector_near(selected.to_vector(), {0.F, 0.F, 1.F, 2.F}, 1e-5F, "where");
}

void test_reduce_and_cumsum() {
    using namespace tinytensor;
    auto a = Tensor::from_vector(std::vector<float>{1.F, 2.F, 3.F, 4.F, 5.F, 6.F},
                                 {2, 3}, Device::Vulkan);
    require_vector_near(a.sum(1).to_vector(), {6.F, 15.F}, 1e-5F, "sum dim1");
    require_vector_near(a.mean(0).to_vector(), {2.5F, 3.5F, 4.5F}, 1e-5F, "mean dim0");
    require_vector_near(a.max(1).to_vector(), {3.F, 6.F}, 1e-5F, "max dim1");
    require_vector_near(a.var(1).to_vector(), {1.F, 1.F}, 1e-5F, "var dim1");
    require_vector_near(a.std(1).to_vector(), {1.F, 1.F}, 1e-5F, "std dim1");
    require_vector_near(a.cumsum(1).to_vector(), {1.F, 3.F, 6.F, 4.F, 9.F, 15.F}, 1e-5F,
                        "cumsum");
    constexpr std::array<int, 1> axis{1};
    require(a.argmax(axis).to_vector_int64() == std::vector<std::int64_t>({2, 2}), "argmax");

    auto empty = Tensor::empty({0, 3}, Device::Vulkan);
    require_vector_near(empty.sum(0).to_vector(), {0.F, 0.F, 0.F}, 1e-5F, "empty sum");
    auto scalar = Tensor::from_vector(std::vector<float>{7.F}, {}, Device::Vulkan);
    require_vector_near(scalar.sum().to_vector(), {7.F}, 1e-5F, "scalar sum");

    constexpr std::size_t large_count = 1U << 20;
    auto large = Tensor::from_vector(std::vector<float>(large_count, 1.F), {large_count},
                                     Device::Vulkan);
    require_near(large.sum().to_vector()[0], static_cast<float>(large_count), 1e-3F,
                 "hierarchical full sum");
    require_near(large.mean().to_vector()[0], 1.F, 1e-6F, "hierarchical full mean");
}

void test_fused_pointwise() {
    using namespace tinytensor;
    constexpr std::size_t count = 300000;
    Tensor value = Tensor::from_vector(std::vector<float>(count, 2.F), {count}, Device::Vulkan);
    float expected = 2.F;
    // More than one 16-op recipe chunk verifies the long-chain path as well as
    // the ordinary single-dispatch fusion path.
    for (int i = 0; i < 20; ++i) {
        value = value.add(1.F).mul(0.5F);
        expected = (expected + 1.F) * 0.5F;
    }
    const auto result = value.to_vector();
    require_near(result.front(), expected, 1e-5F, "fused pointwise first");
    require_near(result.back(), expected, 1e-5F, "fused pointwise last");
}

void test_fused_adam() {
    using namespace tinytensor;
    auto parameter = Tensor::from_vector(std::vector<float>{1.F, 1.F, 1.F, 1.F}, {4}, Device::Vulkan);
    auto gradient = Tensor::from_vector(std::vector<float>{0.5F, 0.5F, 0.5F, 0.5F}, {4}, Device::Vulkan);
    auto first = Tensor::zeros({4}, Device::Vulkan);
    auto second = Tensor::zeros({4}, Device::Vulkan);
    vulkan::AdamStepOptions options;
    options.learning_rate = 0.01F;
    options.secondary_learning_rate = 0.02F;
    options.group_stride = 4;
    options.correction1 = 0.1F;
    options.correction2 = 0.001F;
    vulkan::adam_step(parameter, gradient, first, second, options);
    require_vector_near(parameter.to_vector(), {0.99F, 0.99F, 0.99F, 0.98F}, 2e-5F,
                        "fused Adam parameter");
    require_vector_near(first.to_vector(), {0.05F, 0.05F, 0.05F, 0.05F}, 1e-6F,
                        "fused Adam first moment");
    require_vector_near(second.to_vector(), {0.00025F, 0.00025F, 0.00025F, 0.00025F}, 1e-7F,
                        "fused Adam second moment");

    auto bad_parameter = Tensor::from_vector(
        std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 2.F}, {2}, Device::Vulkan);
    auto bad_gradient = Tensor::from_vector(
        std::vector<float>{1.F, std::numeric_limits<float>::infinity()}, {2}, Device::Vulkan);
    auto bad_first = Tensor::from_vector(std::vector<float>{3.F, 3.F}, {2}, Device::Vulkan);
    auto bad_second = Tensor::from_vector(std::vector<float>{4.F, 4.F}, {2}, Device::Vulkan);
    options.group_stride = 0;
    options.clamp_min = -1.F;
    options.clamp_max = 1.F;
    vulkan::adam_step(bad_parameter, bad_gradient, bad_first, bad_second, options);
    require_vector_near(bad_parameter.to_vector(), {0.F, 1.F}, 1e-6F,
                        "fused Adam non-finite parameter guard");
    require_vector_near(bad_first.to_vector(), {0.F, 0.F}, 1e-6F,
                        "fused Adam non-finite first guard");
    require_vector_near(bad_second.to_vector(), {0.F, 0.F}, 1e-6F,
                        "fused Adam non-finite second guard");
}

void test_matmul_cat_and_indexing() {
    using namespace tinytensor;
    auto a = Tensor::from_vector(std::vector<float>{1.F, 2.F, 3.F, 4.F, 5.F, 6.F},
                                 {2, 3}, Device::Vulkan);
    auto b = Tensor::from_vector(std::vector<float>{7.F, 8.F, 9.F, 10.F, 11.F, 12.F},
                                 {3, 2}, Device::Vulkan);
    require_vector_near(a.mm(b).to_vector(), {58.F, 64.F, 139.F, 154.F}, 1e-4F, "mm");

    auto batch_a = a.reshape({1, 2, 3});
    auto batch_b = b.reshape({1, 3, 2});
    auto batch_product = batch_a.bmm(batch_b);
    require(batch_product.shape() == TensorShape({1, 2, 2}), "bmm batch-one shape");
    require_vector_near(batch_product.to_vector(), {58.F, 64.F, 139.F, 154.F}, 1e-4F,
                        "bmm batch-one values");

    auto left = Tensor::from_vector(std::vector<float>{1.F, 2.F, 3.F, 4.F}, {2, 2}, Device::Vulkan);
    auto right = Tensor::from_vector(std::vector<float>{5.F, 6.F}, {2, 1}, Device::Vulkan);
    auto joined = Tensor::cat({left, right}, 1);
    require_vector_near(joined.to_vector(), {1.F, 2.F, 5.F, 3.F, 4.F, 6.F}, 1e-5F, "cat dim1");

    auto indices = Tensor::from_vector(std::vector<int>{2, 0}, {2}, Device::Vulkan);
    auto filled = Tensor::zeros({2, 3}, Device::Vulkan);
    filled.index_fill_(1, indices, 7.F);
    require_vector_near(filled.to_vector(), {7.F, 0.F, 7.F, 7.F, 0.F, 7.F}, 1e-5F,
                        "index_fill dim1");
    auto source = Tensor::from_vector(std::vector<float>{1.F, 2.F, 3.F, 4.F}, {2, 2}, Device::Vulkan);
    filled.scatter_(1, indices, source);
    require_vector_near(filled.to_vector(), {2.F, 0.F, 1.F, 4.F, 0.F, 3.F}, 1e-5F, "scatter");
}

void test_mask_factory_and_pool() {
    using namespace tinytensor;
    auto values = Tensor::from_vector(std::vector<float>{1.F, 2.F, 3.F, 4.F}, {4}, Device::Vulkan);
    auto mask = Tensor::from_vector(std::vector<bool>{true, false, true, false}, {4}, Device::Vulkan);
    require_vector_near(values.masked_select(mask).to_vector(), {1.F, 3.F}, 1e-5F,
                        "masked_select");
    values.masked_fill_(mask, -2.F);
    require_vector_near(values.to_vector(), {-2.F, 2.F, -2.F, 4.F}, 1e-5F, "masked_fill");

    auto random = Tensor::uniform({128}, -2.F, 3.F, Device::Vulkan);
    for (float value : random.to_vector()) require(value >= -2.F && value < 3.F, "uniform range");
    require_vector_near(Tensor::eye(3, Device::Vulkan).to_vector(),
                        {1.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 1.F}, 1e-5F, "eye");
    auto diagonal = Tensor::from_vector(std::vector<float>{2.F, 3.F}, {2}, Device::Vulkan);
    require_vector_near(Tensor::diag(diagonal).to_vector(), {2.F, 0.F, 0.F, 3.F}, 1e-5F,
                        "diag");

    auto image = Tensor::from_vector(
        std::vector<float>{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F, 9.F},
        {1, 1, 3, 3}, Device::Vulkan);
    require_vector_near(image.max_pool2d(2, 1).to_vector(), {5.F, 6.F, 8.F, 9.F}, 1e-5F,
                        "max_pool2d");
    require_vector_near(image.adaptive_avg_pool2d(1, 1).to_vector(), {5.F}, 1e-5F,
                        "adaptive_avg_pool2d");
}

} // namespace

int main() {
    try {
        if (!tinytensor::vulkan::available()) {
            std::cout << "tinytensor vulkan backend unavailable; skipping\n";
            return 0;
        }
        const auto& info = tinytensor::vulkan::device_info();
        std::cout << "tinytensor vulkan device=" << info.name
                  << " subgroup=" << info.subgroup_size
                  << " atomic_f32=" << (info.buffer_atomic_f32 ? "yes" : "no")
                  << "\n";
        const auto run = [](const char* name, auto&& test) {
            std::cout << "  " << name << "\n";
            test();
        };
        run("factory_and_roundtrip", test_factory_and_roundtrip);
        run("cat_index_squeeze", test_cat_index_squeeze);
        run("mask_and_convert", test_mask_and_convert);
        run("multinomial", test_multinomial);
        run("elementwise_and_where", test_elementwise_and_where);
        run("reduce_and_cumsum", test_reduce_and_cumsum);
        run("fused_pointwise", test_fused_pointwise);
        run("fused_adam", test_fused_adam);
        run("matmul_cat_and_indexing", test_matmul_cat_and_indexing);
        run("mask_factory_and_pool", test_mask_factory_and_pool);
        std::cout << "tinytensor vulkan tests passed\n";
        tinytensor::vulkan::shutdown();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tinytensor vulkan test failed: " << error.what() << "\n";
        return 1;
    }
}
