#pragma once
#include "splat/types.hpp"
#include <span>
#include <vector>
namespace photara::splat::detail {
std::vector<std::uint8_t> save_usdc(const GaussianModel &);
GaussianModel load_usdc(std::span<const std::uint8_t>, tinytensor::Device);
} // namespace photara::splat::detail
