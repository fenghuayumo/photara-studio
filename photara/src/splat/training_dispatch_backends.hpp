#pragma once
#include "training_dispatch.hpp"

namespace photara::splat::detail {
// Internal implementation contract; only the shared C++ dispatcher selects it.
decltype(compute_step_loss) compute_step_loss_vulkan;
} // namespace photara::splat::detail
