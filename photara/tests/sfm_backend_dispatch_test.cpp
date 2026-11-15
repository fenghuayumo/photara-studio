#include "../src/sfm/solver_dispatch.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace photara;
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
ba::Problem fixture() {
    ba::Problem p;
    p.poses.resize(4);
    p.intrinsics.push_back({500, 500, 320, 240});
    p.pose_intrinsic.assign(4, 0);
    for (unsigned c = 0; c < 4; ++c)
        p.poses[c].cx = .3 * c;
    for (unsigned i = 0; i < 40; ++i) {
        ba::Point3 point{.1 * (int(i % 8) - 4), .1 * (int(i / 8) - 2), 3. + .03 * i};
        p.points.push_back(point);
        for (unsigned c = 0; c < 4; ++c)
            p.observations.push_back(
                c, i, 500 * (point.x - p.poses[c].cx) / point.z + 320,
                500 * point.y / point.z + 240);
    }
    for (unsigned i = 1; i < 40; ++i)
        p.points[i].x += .02;
    return p;
}
class RejectingSolver final : public ba::BearingSolver {
  public:
    ba::BearingSolveSummary solve(const std::vector<double>&, unsigned, double, double) override {
        return {};
    }
    void download(ba::BearingProblem&) const override {
        throw std::logic_error("unusable step downloaded");
    }
};
int main() {
    try {
        sfm::BundleOptions options;
        options.cuda_min_observations = 0;
        ba::OptimizerOptions opt;
        opt.maximum_iterations = 15;
        for (auto preference :
             {sfm::BundleBackendPreference::cpu, sfm::BundleBackendPreference::cuda,
              sfm::BundleBackendPreference::vulkan}) {
            auto problem = fixture();
            const auto initial = ba::evaluate_cost(problem);
            sfm::BundleSummary summary;
            sfm::detail::solve_bundle(problem, opt, options, preference, true, false, summary);
            require(summary.optimizer.usable(), "BA dispatch returned unusable solution");
            require(
                ba::evaluate_cost(problem) < initial * .1,
                "BA dispatch failed to reduce reprojection cost");
            if (preference == sfm::BundleBackendPreference::cpu)
                require(summary.backend == sfm::BundleBackend::cpu, "explicit CPU used a GPU");
            if (preference == sfm::BundleBackendPreference::cuda &&
                !ba::solver_backend(ba::SolverDevice::cuda).built)
                require(
                    summary.backend == sfm::BundleBackend::cpu,
                    "unbuilt CUDA fell through to Vulkan");
        }
        for (auto device : {ba::SolverDevice::cuda, ba::SolverDevice::vulkan}) {
            const auto& backend = ba::solver_backend(device);
            if (!backend.available()) {
                std::cout << backend.name
                          << " bearing numeric checks skipped: backend unavailable\n";
                continue;
            }
            ba::BearingProblem bearings;
            bearings.cameras = {{0., 0., 0.}, {.3, 0., 0.}, {.6, 0., 0.}, {.9, 0., 0.}};
            bearings.anchor = bearings.baseline_first = 0;
            bearings.baseline_second = 3;
            bearings.baseline = .9;
            for (unsigned i = 0; i < 40; ++i) {
                std::array<double, 3> point{
                    .1 * (int(i % 8) - 4), .1 * (int(i / 8) - 2), 3. + .03 * i};
                for (unsigned c = 0; c < 4; ++c) {
                    std::array<double, 3> ray{
                        point[0] - bearings.cameras[c][0], point[1], point[2]};
                    double length = std::sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
                    for (auto& x : ray)
                        x /= length;
                    bearings.observations.push_back({c, i, ray});
                }
                point[0] += .02;
                bearings.points.push_back(point);
            }
            auto anchor = bearings.cameras.front();
            sfm::detail::PositioningSolver selected{backend.make_bearing(bearings), backend.name};
            ba::BearingSolveSummary step;
            require(
                sfm::detail::solve_positioning_pass(
                    selected, bearings, std::vector<double>(bearings.observations.size(), 1.), 20,
                    1e-9, 0, 0, step),
                "bearing adapter failed to solve");
            require(
                step.final_cost < step.initial_cost * .1,
                "bearing adapter failed to reduce angular cost");
            require(bearings.cameras.front() == anchor, "bearing adapter moved fixed anchor");
        }
        bool prepared = false;
        ba::BearingProblem problem;
        sfm::GlobalPositioningOptions positioning;
        for (auto preference :
             {sfm::PositioningBackend::cpu, sfm::PositioningBackend::automatic,
              sfm::PositioningBackend::cuda, sfm::PositioningBackend::vulkan}) {
            positioning.backend = preference;
            auto selected = sfm::detail::make_positioning_solver(
                positioning, 10, 4, [&]() -> const ba::BearingProblem& {
                    prepared = true;
                    return problem;
                });
            require(
                !selected.solver && !prepared, "small positioning problem allocated a GPU solver");
        }
        sfm::detail::PositioningSolver rejected{std::make_unique<RejectingSolver>(), "test"};
        ba::BearingSolveSummary step;
        require(
            !sfm::detail::solve_positioning_pass(rejected, problem, {}, 1, 1e-6, 1, 0, step),
            "unusable bearing step accepted");
        require(!rejected.solver, "failed solver was not released for CPU fallback");
        std::cout << "SfM shared dispatch, backend availability, CPU fallback and BA cost checks "
                     "passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
