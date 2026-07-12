#include "ba/optimizer.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <utility>

namespace {

using namespace aetherscan::ba;

std::pair<double, double> project(
    const Pose& pose, const PinholeIntrinsics& intrinsics, const Point3& point) {
    const double dx = point.x - pose.cx;
    const double dy = point.y - pose.cy;
    const double dz = point.z - pose.cz;
    const double r00 = 1.0 - 2.0 * (pose.qy * pose.qy + pose.qz * pose.qz);
    const double r01 = 2.0 * (pose.qx * pose.qy - pose.qw * pose.qz);
    const double r02 = 2.0 * (pose.qx * pose.qz + pose.qw * pose.qy);
    const double r10 = 2.0 * (pose.qx * pose.qy + pose.qw * pose.qz);
    const double r11 = 1.0 - 2.0 * (pose.qx * pose.qx + pose.qz * pose.qz);
    const double r12 = 2.0 * (pose.qy * pose.qz - pose.qw * pose.qx);
    const double r20 = 2.0 * (pose.qx * pose.qz - pose.qw * pose.qy);
    const double r21 = 2.0 * (pose.qy * pose.qz + pose.qw * pose.qx);
    const double r22 = 1.0 - 2.0 * (pose.qx * pose.qx + pose.qy * pose.qy);
    const double px = r00 * dx + r01 * dy + r02 * dz;
    const double py = r10 * dx + r11 * dy + r12 * dz;
    const double pz = r20 * dx + r21 * dy + r22 * dz;
    return {intrinsics.fx * px / pz + intrinsics.cx,
            intrinsics.fy * py / pz + intrinsics.cy};
}

Problem make_problem() {
    Problem problem;
    problem.poses.resize(5);
    problem.intrinsics.resize(5, PinholeIntrinsics{800.0, 800.0, 640.0, 360.0});
    problem.points.resize(80);
    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera) {
        problem.poses[camera].cx = -1.0 + 0.5 * static_cast<double>(camera);
    }
    std::mt19937 random(42);
    std::uniform_real_distribution<double> xy(-1.5, 1.5);
    std::uniform_real_distribution<double> z(4.0, 8.0);
    for (auto& point : problem.points) point = Point3{xy(random), xy(random), z(random)};
    for (std::size_t point = 0; point < problem.points.size(); ++point) {
        for (std::size_t camera = 0; camera < problem.poses.size(); ++camera) {
            const auto [x, y] = project(
                problem.poses[camera], problem.intrinsics[camera], problem.points[point]);
            problem.observations.push_back(
                static_cast<Index>(camera), static_cast<Index>(point), x, y);
        }
    }
    std::normal_distribution<double> camera_noise(0.0, 0.025);
    std::normal_distribution<double> point_noise(0.0, 0.04);
    for (std::size_t camera = 1; camera < problem.poses.size(); ++camera) {
        problem.poses[camera].cx += camera_noise(random);
        problem.poses[camera].cy += camera_noise(random);
        problem.poses[camera].cz += camera_noise(random);
    }
    for (std::size_t point = 1; point < problem.points.size(); ++point) {
        problem.points[point].x += point_noise(random);
        problem.points[point].y += point_noise(random);
        problem.points[point].z += point_noise(random);
    }
    return problem;
}

}  // namespace

int main() {
    Problem problem = make_problem();
    Problem gpu_problem = problem;
    const double initial = evaluate_cost(problem);
    OptimizerOptions options;
    options.maximum_iterations = 15;
    options.maximum_pcg_iterations = 80;
    options.huber_delta = 100.0;
    const OptimizerSummary summary = optimize_cpu(problem, options);
    std::cout << summary.brief_report() << '\n';
    if (!summary.usable() || summary.successful_steps == 0 ||
        !(summary.final_cost < initial * 1e-3)) {
        std::cerr << "optimizer failed to reduce synthetic reprojection cost\n";
        return 1;
    }
    if (std::abs(problem.poses.front().cx + 1.0) > 1e-15) {
        std::cerr << "fixed gauge pose was modified\n";
        return 2;
    }
    Problem boundary_problem = make_problem();
    boundary_problem.pose_constant.resize(boundary_problem.poses.size(), 0);
    boundary_problem.pose_constant[2] = 1;
    const Pose fixed_boundary = boundary_problem.poses[2];
    const double boundary_initial = evaluate_cost(boundary_problem);
    const OptimizerSummary boundary_summary =
        optimize_cpu(boundary_problem, options);
    const Pose& boundary_after = boundary_problem.poses[2];
    if (!boundary_summary.usable() ||
        !(boundary_summary.final_cost < boundary_initial) ||
        std::abs(boundary_after.qw - fixed_boundary.qw) > 1e-15 ||
        std::abs(boundary_after.qx - fixed_boundary.qx) > 1e-15 ||
        std::abs(boundary_after.qy - fixed_boundary.qy) > 1e-15 ||
        std::abs(boundary_after.qz - fixed_boundary.qz) > 1e-15 ||
        std::abs(boundary_after.cx - fixed_boundary.cx) > 1e-15 ||
        std::abs(boundary_after.cy - fixed_boundary.cy) > 1e-15 ||
        std::abs(boundary_after.cz - fixed_boundary.cz) > 1e-15) {
        std::cerr << "constant boundary pose was modified or BA did not improve\n";
        return 3;
    }
#if defined(AETHERSCAN_HAS_CUDA)
    if (CudaOptimizer::is_available()) {
        const OptimizerSummary gpu_summary = optimize_cuda(gpu_problem, options);
        std::cout << "GPU " << gpu_summary.brief_report() << '\n';
        if (!gpu_summary.usable() || gpu_summary.successful_steps == 0 ||
            !(gpu_summary.final_cost < initial * 1e-3)) {
            std::cerr << "GPU optimizer failed to reduce synthetic reprojection cost\n";
            return 4;
        }
        if (std::abs(gpu_problem.poses.front().cx + 1.0) > 1e-15) {
            std::cerr << "GPU optimizer modified the fixed gauge pose\n";
            return 5;
        }
    }
#endif
    return 0;
}
