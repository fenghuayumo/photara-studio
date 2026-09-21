#include "internal/tensor_impl.hpp"
#include "vulkan/backend.hpp"

#include <cmath>
#include <iostream>
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
        test_factory_and_roundtrip();
        test_cat_index_squeeze();
        test_mask_and_convert();
        test_multinomial();
        std::cout << "tinytensor vulkan tests passed\n";
        tinytensor::vulkan::shutdown();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tinytensor vulkan test failed: " << error.what() << "\n";
        return 1;
    }
}
