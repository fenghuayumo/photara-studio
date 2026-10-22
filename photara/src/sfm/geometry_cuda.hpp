#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace photara::sfm::gpu {

// POD-only boundary: CUDA compilation does not depend on Eigen or Scene.
struct V3 { double x{}, y{}, z{}; };
struct View {
    double R[9]{}, C[3]{};
    double fx{}, fy{}, cx{}, cy{}, k1{}, k2{}, p1{}, p2{};
    unsigned model{};
};
struct RayPair {
    V3 first, second;
    unsigned job{};
};
struct PairJob {
    View view;
    V3 epipole1, epipole2;
    double cos_epipole{}, cos_reproj{}, min_cos_angle{};
    bool epipole{}, reprojection{}, angle{};
};
struct PairResult { double cosine{}, weight{}; unsigned accepted{}, uncertain{}; };
// Views are deduplicated per batch (one pose/camera record per image), so an
// observation only carries its bearing, pixel and view index.
struct Observation {
    V3 bearing;
    double u{}, v{}, minimum_cosine{};
    unsigned view{};
};
struct TrackJob {
    unsigned begin{}, count{};
    V3 position;
};
struct TrackResult {
    V3 position;
    double sum_error{}, sum_cosine{}, distance{};
    unsigned inliers{}, fallback{};
};
struct TriangulationConfig {
    double threshold{}, min_angle{};
    unsigned min_inliers{}, refine_iterations{};
    bool refine{};
};

bool cuda_available() noexcept;
void cuda_filter_pairs(const std::vector<RayPair>& rays,
    const std::vector<PairJob>& jobs, std::vector<PairResult>& results);
// Linear-consensus tracks only. Long LO-RANSAC tracks remain on the CPU: a
// per-track CUDA replay measured slower and is not part of the production path.
void cuda_triangulate(const std::vector<View>& views,
    const std::vector<Observation>& observations,
    const std::vector<TrackJob>& jobs, const TriangulationConfig& config,
    std::vector<TrackResult>& results,
    std::vector<unsigned char>& inliers);
void cuda_filter_tracks(const std::vector<View>& views,
    const std::vector<Observation>& observations,
    const std::vector<TrackJob>& jobs, double threshold, double min_angle,
    std::vector<TrackResult>& results, std::vector<unsigned char>& inliers);

} // namespace photara::sfm::gpu
