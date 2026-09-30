#pragma once

#include "ba/optimizer.hpp"
#include "sfm/scene.hpp"

namespace photara::sfm {

enum class BundleBackend {
    cpu,
    cuda,
    vulkan,
};

// Process-wide backend override for A/B validation of the joint BA.
// Set once from the CLI before reconstruction starts.
// automatic prefers a GPU: CUDA when a device is present, otherwise Vulkan,
// then CPU. Explicit cuda/vulkan do not fall through to the other GPU.
enum class BundleBackendPreference {
    automatic,  // CUDA, else Vulkan, else CPU (default)
    cpu,        // never use a GPU solver
    cuda,       // prefer CUDA; warn when any solve falls back to CPU
    vulkan,     // prefer Vulkan; warn and use CPU when it cannot run
};

void set_bundle_backend_preference(BundleBackendPreference preference);
[[nodiscard]] BundleBackendPreference bundle_backend_preference();

struct BundleOptions {
    ba::OptimizerOptions optimizer{};
    bool optimize_points{true};
    // Write optimized shared intrinsics back into Scene::cameras.
    bool write_intrinsics{true};
    // If non-empty, only these image IDs have free poses; others are fixed
    // inside the BA problem.
    std::vector<Index> free_image_ids;
    // Boundary views kept in the local problem as constant poses. Their
    // observations anchor shared points to the existing reconstruction.
    std::vector<Index> fixed_image_ids;
    // If true and free_image_ids empty, optimize all registered images.
    bool optimize_all_registered{true};
    // Freeze intrinsic groups that lack enough views / parallax support.
    bool gate_intrinsics_by_observability{true};
    unsigned min_views_for_intrinsics{3};
    float min_median_parallax_deg{1.0F};
    // Use a GPU Schur/PCG backend when the problem is large enough and the
    // requested parameterization is supported. The process-wide preference
    // chooses CUDA or Vulkan; partial pose locks stay on the CPU.
    // Default on: automatic tries CUDA first, then Vulkan.
    bool prefer_cuda{true};
    // Shared CUDA/Vulkan threshold; the legacy name is kept for API stability.
    std::size_t cuda_min_observations{50'000};
};

struct BundleSummary {
    bool success{false};
    ba::OptimizerSummary optimizer;
    unsigned num_cameras{0};
    unsigned num_points{0};
    unsigned num_observations{0};
    BundleBackend backend{BundleBackend::cpu};
};

// Builds a BA problem from triangulated inlier tracks and runs photara::ba.
BundleSummary run_bundle_adjustment(Scene& scene, const BundleOptions& options = {});

}  // namespace photara::sfm
