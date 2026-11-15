#include "splat/rasterizer.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using tinytensor::Tensor;
int main() {
    try {
        auto device = tinytensor::Device::Vulkan;
        auto a = Tensor::from_vector({1.F, 2.F, 3.F}, {3}, device);
        auto values = a.add(a).to(tinytensor::Device::CPU);
        if (std::abs(values.ptr<float>()[2] - 6.F) > 1e-5F)
            throw std::runtime_error("Vulkan tensor addition failed");
        photara::splat::GaussianModel m;
        m.sh_degree = 0;
        m.means = Tensor::from_vector({0.F, 0.F, 3.F}, {1,3}, device);
        m.log_scales = Tensor::full({1,3}, -2.F, device);
        m.quaternions = Tensor::from_vector({1.F,0.F,0.F,0.F}, {1,4}, device);
        m.opacity_logits = Tensor::zeros({1,1}, device);
        m.sh = Tensor::zeros({1,1,3}, device);
        photara::splat::Camera c;
        c.width = c.height = 32;
        c.fx = c.fy = 32.F; c.cx = c.cy = 16.F;
        c.world_to_camera = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        photara::splat::RasterizeOptions o;
        o.active_sh_degree = 0; o.require_depth = false; o.record_backward_state = false;
        auto r = photara::splat::Rasterizer{}.forward(m,c,o);
        auto pixels = r.color.to(tinytensor::Device::CPU);
        if (pixels.numel() != 3*32*32 || !std::isfinite(pixels.ptr<float>()[16*32+16]))
            throw std::runtime_error("Vulkan rasterizer output is invalid");
        std::cout << "Vulkan tensor addition and Gaussian render passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
