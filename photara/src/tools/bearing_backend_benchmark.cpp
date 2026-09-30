#include "ba/bearing_cuda.hpp"
#include "ba/bearing_vulkan.hpp"
#include "ba/optimizer.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using photara::ba::BearingProblem;
using photara::ba::BearingSolveSummary;

unsigned parse_unsigned(const char *text, std::string_view name) {
    unsigned value = 0;
    const std::string_view input(text);
    const auto [end, error] = std::from_chars(input.data(), input.data() + input.size(), value);
    if (error != std::errc{} || end != input.data() + input.size() || value == 0) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return value;
}

BearingProblem make_problem(unsigned camera_count, unsigned point_count, unsigned views_per_point) {
    if (camera_count < 4 || camera_count > 2000) {
        throw std::invalid_argument("camera count must be in [4, 2000]");
    }
    if (views_per_point < 2 || views_per_point > camera_count) {
        throw std::invalid_argument("views per point must be in [2, cameras]");
    }

    BearingProblem problem;
    problem.anchor = 0;
    problem.baseline_first = 0;
    problem.baseline_second = camera_count / 2;
    problem.huber = 0.03;
    problem.cameras.reserve(camera_count);
    problem.points.reserve(point_count);
    problem.observations.reserve(static_cast<std::size_t>(point_count) * views_per_point);

    constexpr double pi = 3.14159265358979323846;
    for (unsigned camera = 0; camera < camera_count; ++camera) {
        const double angle = 2.0 * pi * static_cast<double>(camera) / camera_count;
        problem.cameras.push_back(
            {6.0 * std::cos(angle), 6.0 * std::sin(angle), 0.35 * std::sin(3.0 * angle)});
    }
    const auto &first = problem.cameras[problem.baseline_first];
    const auto &second = problem.cameras[problem.baseline_second];
    double baseline_squared = 0;
    for (unsigned axis = 0; axis < 3; ++axis) {
        const double delta = first[axis] - second[axis];
        baseline_squared += delta * delta;
    }
    problem.baseline = std::sqrt(baseline_squared);

    std::mt19937 random(0x4a17u);
    std::uniform_real_distribution<double> xy(-2.0, 2.0);
    std::uniform_real_distribution<double> z(-1.0, 1.0);
    std::normal_distribution<double> perturbation(0.0, 0.025);
    for (unsigned point = 0; point < point_count; ++point) {
        problem.points.push_back({xy(random), xy(random), z(random)});
        const unsigned first_view = static_cast<unsigned>(
            (static_cast<std::uint64_t>(point) * camera_count) / std::max(point_count, 1u));
        for (unsigned view = 0; view < views_per_point; ++view) {
            const unsigned camera = (first_view + view) % camera_count;
            std::array<double, 3> direction{};
            double length_squared = 0;
            for (unsigned axis = 0; axis < 3; ++axis) {
                direction[axis] = problem.points.back()[axis] - problem.cameras[camera][axis];
                length_squared += direction[axis] * direction[axis];
            }
            const double inverse_length = 1.0 / std::sqrt(length_squared);
            for (double &value : direction)
                value *= inverse_length;
            problem.observations.push_back({camera, point, direction});
        }
    }

    for (unsigned camera = 1; camera < camera_count; ++camera) {
        for (double &value : problem.cameras[camera])
            value += perturbation(random);
    }
    for (auto &point : problem.points) {
        for (double &value : point)
            value += perturbation(random);
    }
    return problem;
}

double maximum_parameter_difference(const BearingProblem &left, const BearingProblem &right) {
    double difference = 0;
    for (std::size_t index = 0; index < left.cameras.size(); ++index) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            difference = std::max(difference,
                                  std::abs(left.cameras[index][axis] - right.cameras[index][axis]));
        }
    }
    for (std::size_t index = 0; index < left.points.size(); ++index) {
        for (unsigned axis = 0; axis < 3; ++axis) {
            difference = std::max(difference,
                                  std::abs(left.points[index][axis] - right.points[index][axis]));
        }
    }
    return difference;
}

struct Measurement {
    BearingProblem problem;
    BearingSolveSummary summary;
};

Measurement run_cuda(const BearingProblem &input, const std::vector<double> &weights,
                     unsigned iterations) {
    Measurement result{input, {}};
    photara::ba::CudaBearingOptimizer optimizer(result.problem);
    result.summary = optimizer.solve(weights, iterations, 0.0, 0.0);
    optimizer.download(result.problem);
    return result;
}

Measurement run_vulkan(const BearingProblem &input, const std::vector<double> &weights,
                       unsigned iterations) {
    Measurement result{input, {}};
    photara::ba::VulkanBearingOptimizer optimizer(result.problem);
    result.summary = optimizer.solve(weights, iterations, 0.0, 0.0);
    optimizer.download(result.problem);
    return result;
}

void print_result(std::string_view backend, const BearingSolveSummary &summary) {
    std::cout << backend << "," << summary.iterations << "," << summary.initial_cost << ","
              << summary.final_cost << "," << summary.seconds << "\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        const unsigned cameras = argc > 1 ? parse_unsigned(argv[1], "cameras") : 96;
        const unsigned points = argc > 2 ? parse_unsigned(argv[2], "points") : 8192;
        const unsigned views = argc > 3 ? parse_unsigned(argv[3], "views") : 8;
        const unsigned iterations = argc > 4 ? parse_unsigned(argv[4], "iterations") : 8;
        const unsigned repeats = argc > 5 ? parse_unsigned(argv[5], "repeats") : 3;
        if (argc > 6) {
            throw std::invalid_argument(
                "Usage: photara_bearing_backend_benchmark [cameras] [points] [views] "
                "[iterations] [repeats]");
        }
        if (!photara::ba::CudaOptimizer::is_available()) {
            throw std::runtime_error("CUDA bearing backend is unavailable");
        }
        if (!photara::ba::VulkanBearingOptimizer::is_available()) {
            throw std::runtime_error("Vulkan bearing backend is unavailable");
        }

        const BearingProblem input = make_problem(cameras, points, views);
        const std::vector<double> weights(input.observations.size(), 1.0);
        std::cout << std::setprecision(12) << "device,"
                  << photara::ba::VulkanBearingOptimizer::device_name() << "\n"
                  << "problem," << cameras << "," << points << "," << input.observations.size()
                  << "," << iterations << "," << repeats << "\n"
                  << "backend,iterations,initial_cost,final_cost,seconds\n";

        // One untimed iteration initializes both driver/runtime paths before
        // measurements.
        (void)run_cuda(input, weights, 1);
        (void)run_vulkan(input, weights, 1);

        double cuda_seconds = 0;
        double vulkan_seconds = 0;
        double worst_difference = 0;
        for (unsigned repeat = 0; repeat < repeats; ++repeat) {
            // Alternate order to reduce temperature and boost-clock bias.
            Measurement cuda;
            Measurement vulkan;
            if ((repeat & 1u) == 0) {
                cuda = run_cuda(input, weights, iterations);
                vulkan = run_vulkan(input, weights, iterations);
            } else {
                vulkan = run_vulkan(input, weights, iterations);
                cuda = run_cuda(input, weights, iterations);
            }
            print_result("cuda", cuda.summary);
            print_result("vulkan", vulkan.summary);
            if (!cuda.summary.usable || !vulkan.summary.usable) {
                throw std::runtime_error("a bearing backend returned an unusable result");
            }
            cuda_seconds += cuda.summary.seconds;
            vulkan_seconds += vulkan.summary.seconds;
            worst_difference = std::max(worst_difference,
                                        maximum_parameter_difference(cuda.problem, vulkan.problem));
        }
        cuda_seconds /= repeats;
        vulkan_seconds /= repeats;
        std::cout << "average,cuda," << cuda_seconds << "\n"
                  << "average,vulkan," << vulkan_seconds << "\n"
                  << "speed_ratio_cuda_over_vulkan," << cuda_seconds / vulkan_seconds << "\n"
                  << "maximum_parameter_difference," << worst_difference << "\n";
        return worst_difference <= 1e-5 ? 0 : 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
