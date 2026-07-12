#include "ba/linearizer.hpp"
#include "ba/optimizer.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using aetherscan::ba::Index;
using aetherscan::ba::LinearizationOutput;
using aetherscan::ba::PinholeIntrinsics;
using aetherscan::ba::Point3;
using aetherscan::ba::Pose;
using aetherscan::ba::Problem;

struct BenchmarkConfig {
    std::size_t cameras{256};
    std::size_t points{100'000};
    std::size_t observations_per_point{6};
    std::size_t iterations{20};
    std::size_t solver_iterations{0};
    std::size_t gpu_solver_iterations{0};
};

std::size_t parse_size(const std::string_view text, const char* option) {
    std::size_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value == 0) {
        throw std::invalid_argument(std::string("Invalid value for ") + option);
    }
    return value;
}

BenchmarkConfig parse_arguments(const int argc, char** argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--help") {
            std::cout
                << "Usage: aetherscan_ba_benchmark [options]\n"
                << "  --cameras N                 Camera count (default 256)\n"
                << "  --points N                  Point count (default 100000)\n"
                << "  --observations-per-point N  Track length (default 6)\n"
                << "  --iterations N              CUDA timed iterations (default 20)\n"
                << "  --solver-iterations N       Run N CPU BA iterations (default disabled)\n"
                << "  --gpu-solver-iterations N   Run N GPU BA iterations (default disabled)\n";
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value after " + std::string(argument));
        }
        const std::string_view value = argv[++i];
        if (argument == "--cameras") {
            config.cameras = parse_size(value, "--cameras");
        } else if (argument == "--points") {
            config.points = parse_size(value, "--points");
        } else if (argument == "--observations-per-point") {
            config.observations_per_point = parse_size(value, "--observations-per-point");
        } else if (argument == "--iterations") {
            config.iterations = parse_size(value, "--iterations");
        } else if (argument == "--solver-iterations") {
            config.solver_iterations = parse_size(value, "--solver-iterations");
        } else if (argument == "--gpu-solver-iterations") {
            config.gpu_solver_iterations = parse_size(value, "--gpu-solver-iterations");
        } else {
            throw std::invalid_argument("Unknown option: " + std::string(argument));
        }
    }
    if (config.cameras > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
        config.points > static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::invalid_argument("Camera and point counts must fit in 32-bit indices");
    }
    return config;
}

std::pair<double, double> project(
    const Pose& pose,
    const PinholeIntrinsics& intrinsics,
    const Point3& point) {
    const double qw = pose.qw;
    const double qx = pose.qx;
    const double qy = pose.qy;
    const double qz = pose.qz;
    const double dx = point.x - pose.cx;
    const double dy = point.y - pose.cy;
    const double dz = point.z - pose.cz;
    const double px =
        (1.0 - 2.0 * (qy * qy + qz * qz)) * dx +
        2.0 * (qx * qy - qw * qz) * dy +
        2.0 * (qx * qz + qw * qy) * dz;
    const double py =
        2.0 * (qx * qy + qw * qz) * dx +
        (1.0 - 2.0 * (qx * qx + qz * qz)) * dy +
        2.0 * (qy * qz - qw * qx) * dz;
    const double pz =
        2.0 * (qx * qz - qw * qy) * dx +
        2.0 * (qy * qz + qw * qx) * dy +
        (1.0 - 2.0 * (qx * qx + qy * qy)) * dz;
    const double x = px / pz;
    const double y = py / pz;
    const double r2 = x * x + y * y;
    const double radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r2 * r2;
    const double xd =
        x * radial + 2.0 * intrinsics.p1 * x * y +
        intrinsics.p2 * (r2 + 2.0 * x * x);
    const double yd =
        y * radial + intrinsics.p1 * (r2 + 2.0 * y * y) +
        2.0 * intrinsics.p2 * x * y;
    return {
        intrinsics.fx * xd + intrinsics.cx,
        intrinsics.fy * yd + intrinsics.cy};
}

Problem make_problem(const BenchmarkConfig& config) {
    Problem problem;
    problem.poses.resize(config.cameras);
    problem.intrinsics.resize(1);
    problem.pose_intrinsic.assign(config.cameras, 0);
    problem.points.resize(config.points);
    problem.observations.reserve(config.points * config.observations_per_point);

    std::mt19937_64 random(0xA37E5CAFull);
    std::uniform_real_distribution<double> point_x(-3.0, 3.0);
    std::uniform_real_distribution<double> point_y(-2.0, 2.0);
    std::uniform_real_distribution<double> point_z(4.0, 12.0);
    std::normal_distribution<double> pixel_noise(0.0, 0.35);
    std::normal_distribution<double> point_noise(0.0, 0.01);
    std::normal_distribution<double> center_noise(0.0, 0.002);
    std::normal_distribution<double> angle_noise(0.0, 0.0005);

    for (std::size_t camera = 0; camera < config.cameras; ++camera) {
        const double interpolation =
            config.cameras > 1
                ? static_cast<double>(camera) / static_cast<double>(config.cameras - 1)
                : 0.5;
        problem.poses[camera].cx = -2.0 + 4.0 * interpolation;
        problem.intrinsics.front() =
            PinholeIntrinsics{1800.0, 1800.0, 1920.0, 1080.0, -0.02, 0.003, 0.0002, -0.0001};
    }
    for (auto& point : problem.points) {
        point = Point3{point_x(random), point_y(random), point_z(random)};
    }

    for (std::size_t point = 0; point < config.points; ++point) {
        const std::size_t start_camera =
            (point * 2654435761ULL) % config.cameras;
        for (std::size_t view = 0; view < config.observations_per_point; ++view) {
            const auto camera = static_cast<Index>(
                (start_camera + view * 17) % config.cameras);
            const auto [x, y] = project(
                problem.poses[camera],
                problem.intrinsics.front(),
                problem.points[point]);
            problem.observations.push_back(
                camera,
                static_cast<Index>(point),
                x + pixel_noise(random),
                y + pixel_noise(random));
        }
    }

    for (auto& pose : problem.poses) {
        pose.cx += center_noise(random);
        pose.cy += center_noise(random);
        pose.cz += center_noise(random);
        const double angle = angle_noise(random);
        pose.qw = std::cos(0.5 * angle);
        pose.qy = std::sin(0.5 * angle);
    }
    for (auto& point : problem.points) {
        point.x += point_noise(random);
        point.y += point_noise(random);
        point.z += point_noise(random);
    }
    return problem;
}

double maximum_difference(
    const LinearizationOutput& cpu,
    const LinearizationOutput& cuda) {
    if (cpu.observations.size() != cuda.observations.size()) {
        throw std::logic_error("CPU and CUDA result sizes differ");
    }
    double maximum = 0.0;
    for (std::size_t i = 0; i < cpu.observations.size(); ++i) {
        const auto& expected = cpu.observations[i];
        const auto& actual = cuda.observations[i];
        if (expected.valid != actual.valid) {
            std::cerr << "Validity mismatch at observation " << i
                      << ": CPU=" << expected.valid
                      << ", CUDA=" << actual.valid << '\n';
            return std::numeric_limits<double>::infinity();
        }
        for (std::size_t k = 0; k < 2; ++k) {
            maximum = std::max(maximum, std::abs(expected.residual[k] - actual.residual[k]));
        }
        for (std::size_t k = 0; k < 12; ++k) {
            maximum = std::max(
                maximum, std::abs(expected.pose_jacobian[k] - actual.pose_jacobian[k]));
        }
        for (std::size_t k = 0; k < 6; ++k) {
            maximum = std::max(
                maximum, std::abs(expected.point_jacobian[k] - actual.point_jacobian[k]));
        }
        maximum = std::max(
            maximum, std::abs(expected.robust_weight - actual.robust_weight));
    }
    return maximum;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const BenchmarkConfig config = parse_arguments(argc, argv);
        std::cout << "Generating " << config.cameras << " cameras, " << config.points
                  << " points, "
                  << config.points * config.observations_per_point
                  << " observations\n";
        Problem problem = make_problem(config);

        if (config.solver_iterations > 0) {
            Problem cpu_problem = problem;
            aetherscan::ba::OptimizerOptions optimizer_options;
            optimizer_options.maximum_iterations = config.solver_iterations;
            const auto summary = aetherscan::ba::optimize_cpu(cpu_problem, optimizer_options);
            std::cout << "CPU " << summary.brief_report() << '\n';
            std::size_t total_pcg = 0;
            for (const auto& iteration : summary.iterations) total_pcg += iteration.pcg_iterations;
            std::cout << "CPU PCG iterations: " << total_pcg << '\n';
        }
#if defined(AETHERSCAN_HAS_CUDA)
        if (config.gpu_solver_iterations > 0) {
            Problem gpu_problem = problem;
            aetherscan::ba::OptimizerOptions optimizer_options;
            optimizer_options.maximum_iterations = config.gpu_solver_iterations;
            aetherscan::ba::CudaOptimizer optimizer(optimizer_options);
            optimizer.upload(gpu_problem);
            const auto summary = optimizer.optimize();
            optimizer.download(gpu_problem);
            std::cout << "GPU " << summary.brief_report() << " (upload/download excluded)\n";
            std::size_t total_pcg = 0;
            for (const auto& iteration : summary.iterations) total_pcg += iteration.pcg_iterations;
            std::cout << "GPU PCG iterations: " << total_pcg << '\n';
        }
#endif

        LinearizationOutput cpu_output;
        const auto cpu_stats = aetherscan::ba::linearize_cpu(problem, cpu_output);
        std::cout << std::fixed << std::setprecision(3)
                  << "CPU:  " << cpu_stats.elapsed_ms << " ms, "
                  << cpu_stats.million_observations_per_second() << " Mobs/s\n";

#if !defined(AETHERSCAN_HAS_CUDA)
        std::cout << "CUDA backend was not built\n";
        return 0;
#else
        if (!aetherscan::ba::CudaLinearizer::is_available()) {
            std::cerr << "No CUDA device is available\n";
            return 2;
        }
        aetherscan::ba::CudaLinearizer cuda;
        cuda.upload(problem);
        cuda.evaluate();

        double total_milliseconds = 0.0;
        for (std::size_t iteration = 0; iteration < config.iterations; ++iteration) {
            total_milliseconds += cuda.evaluate().elapsed_ms;
        }
        LinearizationOutput cuda_output;
        cuda.download(cuda_output);
        const double average_milliseconds =
            total_milliseconds / static_cast<double>(config.iterations);
        const double throughput =
            static_cast<double>(problem.observations.size()) /
            (average_milliseconds * 1000.0);
        const double difference = maximum_difference(cpu_output, cuda_output);

        std::cout << "CUDA device: " << aetherscan::ba::CudaLinearizer::device_name() << '\n'
                  << "CUDA: " << average_milliseconds << " ms, "
                  << throughput << " Mobs/s (kernel only)\n"
                  << "Speedup: " << cpu_stats.elapsed_ms / average_milliseconds << "x\n"
                  << std::scientific
                  << "CPU/CUDA maximum absolute difference: " << difference << '\n';
        return difference <= 1e-8 ? 0 : 3;
#endif
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
