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
    problem.intrinsics.assign(
        1, PinholeIntrinsics{800.0, 800.0, 640.0, 360.0});
    problem.pose_intrinsic.assign(problem.poses.size(), 0);
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
                problem.poses[camera], problem.intrinsics.front(),
                problem.points[point]);
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

Problem make_grouped_intrinsics_problem() {
    Problem problem;
    problem.poses.resize(6);
    problem.intrinsics = {
        PinholeIntrinsics{630.0, 630.0, 640.0, 360.0},
        PinholeIntrinsics{1210.0, 1210.0, 640.0, 360.0}};
    problem.pose_intrinsic = {0, 0, 0, 1, 1, 1};
    problem.pose_constant.assign(problem.poses.size(), 1);
    const PinholeIntrinsics truth[] = {
        PinholeIntrinsics{700.0, 700.0, 640.0, 360.0},
        PinholeIntrinsics{1100.0, 1100.0, 640.0, 360.0}};
    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera)
        problem.poses[camera].cx =
            -1.25 + 0.5 * static_cast<double>(camera);

    std::mt19937 random(73);
    std::uniform_real_distribution<double> xy(-1.2, 1.2);
    std::uniform_real_distribution<double> z(4.0, 8.0);
    std::normal_distribution<double> point_noise(0.0, 0.03);
    problem.points.resize(100);
    for (std::size_t point = 0; point < problem.points.size(); ++point) {
        const Point3 exact{xy(random), xy(random), z(random)};
        problem.points[point] = {
            exact.x + point_noise(random),
            exact.y + point_noise(random),
            exact.z + point_noise(random)};
        for (std::size_t camera = 0; camera < problem.poses.size(); ++camera) {
            const auto [x, y] = project(
                problem.poses[camera],
                truth[problem.pose_intrinsic[camera]], exact);
            problem.observations.push_back(
                static_cast<Index>(camera), static_cast<Index>(point), x, y);
        }
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
    Problem fixed_point_problem = make_problem();
    const std::vector<Point3> fixed_points = fixed_point_problem.points;
    OptimizerOptions fixed_point_options = options;
    fixed_point_options.optimize_points = false;
    const double fixed_point_initial = evaluate_cost(
        fixed_point_problem, fixed_point_options.huber_delta,
        fixed_point_options.minimum_depth);
    const OptimizerSummary fixed_point_summary =
        optimize_cpu(fixed_point_problem, fixed_point_options);
    bool points_unchanged = true;
    for (std::size_t point = 0; point < fixed_points.size(); ++point) {
        const Point3& before = fixed_points[point];
        const Point3& after = fixed_point_problem.points[point];
        points_unchanged = points_unchanged &&
            before.x == after.x && before.y == after.y && before.z == after.z;
    }
    if (!fixed_point_summary.usable() ||
        !(fixed_point_summary.final_cost < fixed_point_initial) ||
        !points_unchanged) {
        std::cerr << "fixed-point BA modified landmarks or failed to improve cost: "
                  << fixed_point_summary.brief_report()
                  << ", unchanged=" << points_unchanged
                  << ", termination="
                  << static_cast<int>(fixed_point_summary.termination)
                  << ", iterations=" << fixed_point_summary.iterations.size();
        if (!fixed_point_summary.iterations.empty())
            std::cerr << ", last_step="
                      << fixed_point_summary.iterations.back().step_norm;
        std::cerr << '\n';
        return 3;
    }
    Problem fixed_translation_problem = make_problem();
    const std::vector<Pose> fixed_translation_poses =
        fixed_translation_problem.poses;
    OptimizerOptions fixed_translation_options = options;
    fixed_translation_options.optimize_translations = false;
    const double fixed_translation_initial =
        evaluate_cost(fixed_translation_problem);
    const OptimizerSummary fixed_translation_summary =
        optimize_cpu(fixed_translation_problem, fixed_translation_options);
    bool translations_unchanged = true;
    for (std::size_t pose = 0;
         pose < fixed_translation_problem.poses.size(); ++pose) {
        const Pose& before = fixed_translation_poses[pose];
        const Pose& after = fixed_translation_problem.poses[pose];
        translations_unchanged = translations_unchanged &&
            before.cx == after.cx && before.cy == after.cy &&
            before.cz == after.cz;
    }
    if (!fixed_translation_summary.usable() ||
        !(fixed_translation_summary.final_cost < fixed_translation_initial) ||
        !translations_unchanged) {
        std::cerr <<
            "fixed-translation BA modified centers or failed to improve cost\n";
        return 13;
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
        return 4;
    }
    Problem grouped_problem = make_grouped_intrinsics_problem();
    const double grouped_initial = evaluate_cost(grouped_problem);
    OptimizerOptions grouped_options = options;
    grouped_options.fix_first_pose = false;
    grouped_options.optimize_focal = true;
    grouped_options.focal_prior_weight = 0.0;  // isolate grouping behavior
    const OptimizerSummary grouped_summary =
        optimize_cpu(grouped_problem, grouped_options);
    if (!grouped_summary.usable() ||
        !(grouped_summary.final_cost < grouped_initial) ||
        !(std::abs(grouped_problem.intrinsics[0].fx - 700.0) < 70.0) ||
        !(std::abs(grouped_problem.intrinsics[1].fx - 1100.0) < 110.0) ||
        !(std::abs(grouped_problem.intrinsics[0].fx -
                   grouped_problem.intrinsics[1].fx) > 100.0)) {
        std::cerr << "grouped intrinsics were broadcast or failed to optimize\n";
        return 5;
    }

    // A non-square resize or changing digital stabilization can produce
    // different horizontal and vertical focal scales. The independent focal
    // model must recover both instead of forcing the error into camera poses.
    {
        Problem aspect_problem = make_problem();
        aspect_problem.pose_constant.assign(aspect_problem.poses.size(), 1);
        aspect_problem.observations = Observations{};
        const PinholeIntrinsics truth{760.0, 920.0, 640.0, 360.0};
        for (std::size_t point = 0; point < aspect_problem.points.size(); ++point) {
            for (std::size_t camera = 0; camera < aspect_problem.poses.size();
                 ++camera) {
                const auto [x, y] = project(
                    aspect_problem.poses[camera], truth,
                    aspect_problem.points[point]);
                aspect_problem.observations.push_back(
                    static_cast<Index>(camera), static_cast<Index>(point), x, y);
            }
        }
        aspect_problem.intrinsics.front() =
            PinholeIntrinsics{840.0, 840.0, 640.0, 360.0};
        aspect_problem.initial_intrinsics = aspect_problem.intrinsics;
        OptimizerOptions aspect_options = options;
        aspect_options.fix_first_pose = false;
        aspect_options.optimize_points = false;
        aspect_options.optimize_rotations = false;
        aspect_options.optimize_focal = true;
        aspect_options.optimize_aspect_ratio = true;
        aspect_options.focal_prior_weight = 0.0;
        const OptimizerSummary aspect_summary =
            optimize_cpu(aspect_problem, aspect_options);
        if (!aspect_summary.usable() ||
            std::abs(aspect_problem.intrinsics[0].fx - truth.fx) > 1e-3 ||
            std::abs(aspect_problem.intrinsics[0].fy - truth.fy) > 1e-3) {
            std::cerr << "independent fx/fy optimization failed: fx="
                      << aspect_problem.intrinsics[0].fx << " fy="
                      << aspect_problem.intrinsics[0].fy << '\n';
            return 12;
        }
    }

    // Strong focal prior should keep f near the declared initial value even
    // when observations prefer a modestly different truth focal.
    {
        Problem prior_problem = make_problem();
        const PinholeIntrinsics truth = prior_problem.intrinsics.front();
        prior_problem.intrinsics.front() =
            PinholeIntrinsics{900.0, 900.0, truth.cx, truth.cy};
        prior_problem.initial_intrinsics = prior_problem.intrinsics;
        prior_problem.pose_constant.assign(prior_problem.poses.size(), 1);
        // Rebuild observations at a nearby truth focal with fixed poses/points.
        prior_problem.observations = Observations{};
        for (std::size_t point = 0; point < prior_problem.points.size(); ++point) {
            for (std::size_t camera = 0; camera < prior_problem.poses.size();
                 ++camera) {
                const auto [x, y] = project(
                    prior_problem.poses[camera],
                    PinholeIntrinsics{880.0, 880.0, truth.cx, truth.cy},
                    prior_problem.points[point]);
                prior_problem.observations.push_back(
                    static_cast<Index>(camera), static_cast<Index>(point), x,
                    y);
            }
        }
        OptimizerOptions prior_options = options;
        prior_options.optimize_focal = true;
        prior_options.optimize_points = false;
        prior_options.optimize_rotations = false;
        prior_options.fix_first_pose = false;
        prior_options.focal_prior_weight = 1e6;
        prior_options.max_focal_ratio = 2.0;
        prior_options.min_focal_ratio = 0.5;
        const OptimizerSummary prior_summary =
            optimize_cpu(prior_problem, prior_options);
        const double f =
            0.5 *
            (prior_problem.intrinsics[0].fx + prior_problem.intrinsics[0].fy);
        if (!prior_summary.usable() || std::abs(f - 900.0) > 5.0) {
            std::cerr << "focal prior failed to restrain focal: f=" << f
                      << " summary=" << prior_summary.brief_report() << '\n';
            return 9;
        }
    }

    // Hard ratio bounds must clamp updates relative to initial_intrinsics.
    {
        Problem bound_problem = make_problem();
        const PinholeIntrinsics truth = bound_problem.intrinsics.front();
        bound_problem.intrinsics.front() =
            PinholeIntrinsics{800.0, 800.0, truth.cx, truth.cy};
        bound_problem.initial_intrinsics = bound_problem.intrinsics;
        bound_problem.observations = Observations{};
        for (std::size_t point = 0; point < bound_problem.points.size();
             ++point) {
            for (std::size_t camera = 0; camera < bound_problem.poses.size();
                 ++camera) {
                const auto [x, y] = project(
                    bound_problem.poses[camera],
                    PinholeIntrinsics{1200.0, 1200.0, truth.cx, truth.cy},
                    bound_problem.points[point]);
                bound_problem.observations.push_back(
                    static_cast<Index>(camera), static_cast<Index>(point), x,
                    y);
            }
        }
        OptimizerOptions bound_options = options;
        bound_options.optimize_focal = true;
        bound_options.optimize_points = false;
        bound_options.focal_prior_weight = 0.0;
        bound_options.max_focal_ratio = 1.05;
        bound_options.min_focal_ratio = 0.95;
        const OptimizerSummary bound_summary =
            optimize_cpu(bound_problem, bound_options);
        const double f =
            0.5 *
            (bound_problem.intrinsics[0].fx + bound_problem.intrinsics[0].fy);
        if (!bound_summary.usable() || f > 800.0 * 1.05 + 1e-6 ||
            f < 800.0 * 0.95 - 1e-6) {
            std::cerr << "focal bounds were violated: f=" << f << '\n';
            return 10;
        }
    }

    // Frozen intrinsic groups must stay exactly at their input values.
    {
        Problem frozen_problem = make_problem();
        frozen_problem.initial_intrinsics = frozen_problem.intrinsics;
        frozen_problem.intrinsic_constant.assign(1, 1);
        const double f_before = frozen_problem.intrinsics[0].fx;
        OptimizerOptions frozen_options = options;
        frozen_options.optimize_focal = true;
        frozen_options.focal_prior_weight = 0.0;
        const OptimizerSummary frozen_summary =
            optimize_cpu(frozen_problem, frozen_options);
        if (!frozen_summary.usable() ||
            frozen_problem.intrinsics[0].fx != f_before ||
            frozen_problem.intrinsics[0].fy != f_before) {
            std::cerr << "constant intrinsic group was modified\n";
            return 11;
        }
    }
#if defined(AETHERSCAN_HAS_CUDA)
    if (CudaOptimizer::is_available()) {
        const OptimizerSummary gpu_summary = optimize_cuda(gpu_problem, options);
        std::cout << "GPU " << gpu_summary.brief_report() << '\n';
        if (!gpu_summary.usable() || gpu_summary.successful_steps == 0 ||
            !(gpu_summary.final_cost < initial * 1e-3)) {
            std::cerr << "GPU optimizer failed to reduce synthetic reprojection cost\n";
            return 6;
        }
        if (std::abs(gpu_problem.poses.front().cx + 1.0) > 1e-15) {
            std::cerr << "GPU optimizer modified the fixed gauge pose\n";
            return 7;
        }
        Problem gpu_fixed_point_problem = make_problem();
        const std::vector<Point3> gpu_fixed_points =
            gpu_fixed_point_problem.points;
        OptimizerOptions gpu_fixed_options = options;
        gpu_fixed_options.optimize_points = false;
        const double gpu_fixed_initial = evaluate_cost(
            gpu_fixed_point_problem, gpu_fixed_options.huber_delta,
            gpu_fixed_options.minimum_depth);
        const OptimizerSummary gpu_fixed_summary =
            optimize_cuda(gpu_fixed_point_problem, gpu_fixed_options);
        bool gpu_points_unchanged = true;
        for (std::size_t point = 0; point < gpu_fixed_points.size(); ++point) {
            const Point3& before = gpu_fixed_points[point];
            const Point3& after = gpu_fixed_point_problem.points[point];
            gpu_points_unchanged = gpu_points_unchanged &&
                before.x == after.x && before.y == after.y && before.z == after.z;
        }
        if (!gpu_fixed_summary.usable() ||
            !(gpu_fixed_summary.final_cost < gpu_fixed_initial) ||
            !gpu_points_unchanged) {
            std::cerr << "GPU fixed-point BA modified landmarks or failed to improve cost\n";
            return 8;
        }
    }
#endif
    return 0;
}
