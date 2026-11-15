#include "../src/splat/ops.hpp"
#include "../src/splat/ppisp.hpp"
#include "../src/splat/sh_adam_quant.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace photara::splat;
using tinytensor::Device;
using tinytensor::Tensor;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Exception, class Function> void requires_exception(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("Expected a backend/input error");
}

void contracts() {
    const auto cpu = Tensor::zeros({2, 4}, Device::CPU);
    requires_exception<std::runtime_error>([&] { detail::normal_features_to_normals(cpu); });
    requires_exception<std::invalid_argument>([&] { detail::percentile_bounds(cpu, .01F); });
    detail::ShAdamQuant inactive;
    detail::sh_adam_quant_select_rows(inactive, {});
    detail::sh_adam_quant_zero_rows(inactive, {});
#if !defined(PHOTARA_HAS_CUDA_SPLAT)
    requires_exception<std::runtime_error>([] { detail::make_ppisp_state(1, {}); });
#endif
}

void device_dispatch(Device device) {
    auto means =
        Tensor::from_vector({-2.F, -1.F, 0.F, 1.F, 2.F, 3.F, 4.F, 5.F, 6.F}, {3, 3}, device);
    const auto bounds = detail::percentile_bounds(means, 1.F);
    const std::array<float, 6> expected{-2.F, -1.F, 0.F, 4.F, 5.F, 6.F};
    require(bounds == expected, "Percentile dispatch returned incorrect bounds");

    // Check the shared entry point against a finite-difference derivative,
    // exercising both the forward and backward backend implementations.
    const std::vector<float> values{.3F, -.2F, .7F, .6F};
    auto features = Tensor::from_vector(values, {1, 4}, device);
    auto normals = detail::normal_features_to_normals(features);
    require(normals.device() == device, "Normal output changed device");
    const auto gradient = Tensor::from_vector({.2F, .4F, -.3F}, {1, 3}, device);
    const auto analytic = detail::normal_features_backward(features, gradient).to_vector();
    for (unsigned i = 0; i < values.size(); ++i) {
        auto plus = values, minus = values;
        plus[i] += .001F;
        minus[i] -= .001F;
        auto a = detail::normal_features_to_normals(Tensor::from_vector(plus, {1, 4}, device))
                     .to_vector();
        auto b = detail::normal_features_to_normals(Tensor::from_vector(minus, {1, 4}, device))
                     .to_vector();
        const float numeric =
            ((a[0] - b[0]) * .2F + (a[1] - b[1]) * .4F - (a[2] - b[2]) * .3F) / .002F;
        require(
            std::abs(analytic[i] - numeric) < 2e-4F,
            "Normal gradient dispatch failed finite differences");
    }

    auto state = detail::make_sh_adam_quant(2, 3, device);
    auto parameter = Tensor::full({2, 3}, 1.F, device);
    detail::sh_adam_quant_step(
        parameter, Tensor::full({2, 3}, .25F, device), state, 3, .01F, .01F, .9F, .999F, .1F, .001F,
        1e-8F, 0.F);
    for (float value : parameter.to_vector())
        require(
            std::abs(value - .99F) < 1e-4F, "Quantized Adam dispatch returned incorrect update");
    auto indices = Tensor::from_vector({1.F}, {1}, device).to(tinytensor::DataType::Int32);
    detail::sh_adam_quant_select_rows(state, indices);
    require(state.first.shape()[0] == 1, "Quantized Adam row selection failed");
    indices = Tensor::from_vector({0.F}, {1}, device).to(tinytensor::DataType::Int32);
    detail::sh_adam_quant_zero_rows(state, indices);
    for (float value : state.bounds.to_vector())
        require(value == 0.F, "Adam row reset failed");
    detail::sh_adam_quant_append_zeros(state, 2);
    require(state.first.shape()[0] == 3, "Quantized Adam append failed");
    detail::sh_adam_quant_select_rows(state, {});
    require(state.first.shape()[0] == 0, "Quantized Adam empty selection failed");
}

int main(int argc, char** argv) {
    try {
        contracts();
        if (argc > 1) {
            const std::string backend = argv[1];
            require(backend == "--vulkan" || backend == "--cuda", "Unknown backend");
            device_dispatch(backend == "--vulkan" ? Device::Vulkan : Device::CUDA);
        }
        std::cout << "Shared backend dispatch contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
