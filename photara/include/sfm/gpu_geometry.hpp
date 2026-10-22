#pragma once

#include "sfm/geometry.hpp"
#include "sfm/triangulation.hpp"

namespace photara::sfm {

enum class GeometryBackend { automatic, cpu, cuda };
// Overrides PHOTARA_SFM_GEOMETRY_BACKEND=automatic|cpu|cuda. This controls
// batched post-estimation verification, linear-consensus triangulation and
// reprojection filtering; PoseLib RANSAC/minimal solvers, LO-RANSAC and
// complex-track splitting remain on CPU. automatic currently selects CPU:
// the full-scene CUDA stages measured slower than the 32-thread CPU path, so
// CUDA is an explicit opt-in used by benchmarks and parity tests.
void set_geometry_backend(GeometryBackend backend);
GeometryBackend geometry_backend();
bool geometry_cuda_available();

// Internal pipeline hooks. False / empty handled flags mean CPU fallback.
bool try_filter_matches_cuda(const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2, const PinholeCamera& camera1,
    const PinholeCamera& camera2, const Pose3D& pose,
    const RelativePoseOptions& options, std::vector<char>& mask,
    float& mean_ray_angle, std::vector<Vec2>* inlier_pixels, unsigned& count);

std::vector<unsigned char> try_triangulate_tracks_cuda(Scene& scene,
    const std::vector<Index>* selected, bool outliers_only,
    const TriangulationOptions& options);

struct GpuTrackFilterStats {
    double sum_px{}, sum_cos_angle{}, average_distance{};
    unsigned observations{};
    bool triangulated{}, handled{};
};
std::vector<GpuTrackFilterStats> try_filter_tracks_cuda(Scene& scene,
    const std::vector<Index>* selected, float threshold, float min_angle);

} // namespace photara::sfm
