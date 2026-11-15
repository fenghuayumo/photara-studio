#include "vulkan/runtime/runtime.hpp"
#include "splat_drender/vulkan_api.h"
#include <iostream>
int main() {
 try {
  auto& tensor = tinytensor::vulkan::runtime::Context::get();
  std::cout << "Tensor device: " << tensor.info().name << '\n';
  { splat_drender::vulkan::Context splat;
    std::cout << "Splat device initialized\n"; }
  tinytensor::vulkan::runtime::Context::shutdown();
 } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
