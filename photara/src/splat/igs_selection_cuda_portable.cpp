// Vulkan and backend-independent entry points. CUDA kernels stay in .cu files.
#include "cuda_ops.hpp"
#include "geometry_vulkan.hpp"
#include "adc_backends.hpp"
#include "rasterizer_vulkan.hpp"
#include "fused_ssim.hpp"
#include "optimizer_backends.hpp"
#include "sh_adam_quant.hpp"
#include "ppisp.hpp"
#include "bilateral_grid.hpp"
#include "pam_cuda.hpp"
#include "formats_sog_xt.hpp"
#include "igs_selection_cuda.hpp"
#include "core/vram_profiler.hpp"
#include "vulkan/ops.hpp"
#include "vulkan/backend.hpp"
#include "pam_vulkan.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace photara::splat::densification::gpu_detail {
tinytensor::Tensor igs_weighted_sample_cuda(
    const tinytensor::Tensor& weights,
    const std::size_t requested,
    const std::uint64_t seed,
    double* prepare_ms,
    double* sort_ms) {

    throw std::runtime_error("igs_weighted_sample_cuda: this operation requires the CUDA backend");

}

}
