#pragma once
#include "features/features.hpp"

namespace photara::features {
std::unique_ptr<FeatureMatcher> make_native_cuda_matcher(SiftGpuMatcherOptions options);
}
