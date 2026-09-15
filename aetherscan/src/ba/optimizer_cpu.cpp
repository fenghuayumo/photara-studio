#include "ba/optimizer.hpp"
#include "core/logging.hpp"

#include "ba/linearizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(AETHERSCAN_HAS_OPENMP)
#include <omp.h>
#endif

namespace aetherscan::ba {
namespace {

constexpr std::size_t pose_size = 6;
constexpr std::size_t point_size = 3;
constexpr std::size_t pose_block_size = pose_size * pose_size;
constexpr std::size_t point_block_size = point_size * point_size;
constexpr std::size_t cross_block_size = pose_size * point_size;
constexpr std::size_t dense_intrinsic_group_limit = 4;

LinearizerOptions make_linearizer_options(const OptimizerOptions& options) {
    return LinearizerOptions{
        options.huber_delta,
        options.minimum_depth,
        options.optimize_focal,
        options.optimize_aspect_ratio,
        options.optimize_principal_point,
        options.optimize_distortion};
}

std::size_t count_intrinsic_dof(const OptimizerOptions& options) noexcept {
    return ::aetherscan::ba::intrinsic_dof(make_linearizer_options(options));
}

struct Adjacency {
    std::vector<std::size_t> point_offsets;
    std::vector<std::size_t> point_observations;
    std::vector<std::size_t> camera_offsets;
    std::vector<std::size_t> camera_observations;
};

struct System {
    std::vector<double> camera_hessian;
    std::vector<double> camera_rhs;
    std::vector<double> point_inverse;
    std::vector<double> point_rhs;
    std::vector<double> cross;
    // Grouped intrinsic blocks (empty when dof == 0).
    std::size_t intrinsic_dof{0};
    std::size_t intrinsic_group_count{0};
    std::vector<double> intrinsic_hessian;       // groups x (dof x dof)
    std::vector<double> intrinsic_rhs;           // groups x dof
    std::vector<double> pose_intrinsic_cross;    // obs x (6 x dof)
    std::vector<double> point_intrinsic_cross;   // obs x (3 x dof)
};

struct IntrinsicSchur {
    std::size_t dof{0};
    std::vector<double> S_ii;
    std::vector<double> S_ci;
    std::vector<double> b_i;
};

struct SchurPattern {
    std::vector<std::size_t> row_offsets;
    std::vector<Index> columns;
    std::vector<std::size_t> diagonal_blocks;
    // Full-matrix block index -> packed upper-triangle block index.
    // Lower-triangle entries contain max(); they are materialized only after
    // the thread-local upper triangle has been reduced.
    std::vector<std::size_t> upper_indices;
    std::size_t upper_block_count{0};
};

struct ExplicitSchur {
    std::vector<double> blocks;
};

struct AssemblyWorkspace {
    std::vector<double> local_cameras;
    std::vector<double> local_intrinsics;
    std::vector<double> local_schur;
};

Adjacency build_adjacency(const Problem& problem) {
    Adjacency result;
    result.point_offsets.assign(problem.points.size() + 1, 0);
    result.camera_offsets.assign(problem.poses.size() + 1, 0);
    for (std::size_t i = 0; i < problem.observations.size(); ++i) {
        ++result.point_offsets[problem.observations.point[i] + 1];
        ++result.camera_offsets[problem.observations.camera[i] + 1];
    }
    std::partial_sum(result.point_offsets.begin(), result.point_offsets.end(),
                     result.point_offsets.begin());
    std::partial_sum(result.camera_offsets.begin(), result.camera_offsets.end(),
                     result.camera_offsets.begin());
    result.point_observations.resize(problem.observations.size());
    result.camera_observations.resize(problem.observations.size());
    auto point_cursor = result.point_offsets;
    auto camera_cursor = result.camera_offsets;
    for (std::size_t i = 0; i < problem.observations.size(); ++i) {
        result.point_observations[point_cursor[problem.observations.point[i]]++] = i;
        result.camera_observations[camera_cursor[problem.observations.camera[i]]++] = i;
    }
    return result;
}

SchurPattern build_schur_pattern(const Problem& problem, const Adjacency& adjacency) {
    std::vector<std::vector<Index>> rows(problem.poses.size());
    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera)
        rows[camera].push_back(static_cast<Index>(camera));
    for (std::size_t point = 0; point < problem.points.size(); ++point) {
        for (std::size_t first = adjacency.point_offsets[point];
             first < adjacency.point_offsets[point + 1]; ++first) {
            const Index camera_first = problem.observations.camera[adjacency.point_observations[first]];
            auto& row = rows[camera_first];
            for (std::size_t second = adjacency.point_offsets[point];
                 second < adjacency.point_offsets[point + 1]; ++second)
                row.push_back(problem.observations.camera[adjacency.point_observations[second]]);
        }
    }
    SchurPattern pattern;
    pattern.row_offsets.resize(problem.poses.size() + 1);
    pattern.diagonal_blocks.resize(problem.poses.size());
    for (std::size_t camera = 0; camera < rows.size(); ++camera) {
        auto& row = rows[camera];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        pattern.row_offsets[camera] = pattern.columns.size();
        pattern.columns.insert(pattern.columns.end(), row.begin(), row.end());
        pattern.diagonal_blocks[camera] = pattern.row_offsets[camera] +
            static_cast<std::size_t>(std::lower_bound(row.begin(), row.end(), static_cast<Index>(camera)) - row.begin());
    }
    pattern.row_offsets.back() = pattern.columns.size();
    pattern.upper_indices.assign(
        pattern.columns.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t row = 0; row < problem.poses.size(); ++row) {
        for (std::size_t cursor = pattern.row_offsets[row];
             cursor < pattern.row_offsets[row + 1]; ++cursor) {
            if (pattern.columns[cursor] >= row)
                pattern.upper_indices[cursor] = pattern.upper_block_count++;
        }
    }
    return pattern;
}

std::size_t find_schur_block(
    const SchurPattern& pattern, const std::size_t row, const Index column) {
    const auto begin = pattern.columns.begin() + static_cast<std::ptrdiff_t>(pattern.row_offsets[row]);
    const auto end = pattern.columns.begin() + static_cast<std::ptrdiff_t>(pattern.row_offsets[row + 1]);
    const auto found = std::lower_bound(begin, end, column);
    return pattern.row_offsets[row] + static_cast<std::size_t>(found - begin);
}

void assemble_explicit_schur(
    const Problem& problem, const Adjacency& adjacency, const SchurPattern& pattern,
    const System& system, AssemblyWorkspace& workspace,
    ExplicitSchur& reduced) {
    reduced.blocks.assign(pattern.columns.size() * pose_block_size, 0.0);
    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera)
        std::copy_n(system.camera_hessian.data() + camera * pose_block_size, pose_block_size,
                    reduced.blocks.data() + pattern.diagonal_blocks[camera] * pose_block_size);

    int thread_count = 1;
#if defined(AETHERSCAN_HAS_OPENMP)
    const std::size_t local_values =
        pattern.upper_block_count * pose_block_size;
    const std::size_t values_per_thread =
        std::max<std::size_t>(1, local_values);
    const std::size_t memory_limited_threads =
        std::max<std::size_t>(1, 32'000'000 / values_per_thread);
    thread_count = std::min<int>(
        omp_get_max_threads(), static_cast<int>(memory_limited_threads));
#endif
    // Retain the large thread-local matrices across LM iterations. Point-major
    // traversal is important here: it keeps inverse/cross accesses contiguous.
    workspace.local_schur.assign(
        static_cast<std::size_t>(thread_count) * local_values, 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
    for (std::int64_t point_signed = 0;
         point_signed < static_cast<std::int64_t>(problem.points.size());
         ++point_signed) {
        const auto point = static_cast<std::size_t>(point_signed);
        int thread = 0;
#if defined(AETHERSCAN_HAS_OPENMP)
        thread = omp_get_thread_num();
#endif
        double* destination = workspace.local_schur.data() +
            static_cast<std::size_t>(thread) * local_values;
        const double* inverse =
            system.point_inverse.data() + point * point_block_size;
        for (std::size_t first = adjacency.point_offsets[point];
             first < adjacency.point_offsets[point + 1]; ++first) {
            const std::size_t observation_first =
                adjacency.point_observations[first];
            const Index camera_first =
                problem.observations.camera[observation_first];
            const double* cross_first = system.cross.data() + observation_first * cross_block_size;
            double transformed[cross_block_size]{};
            for (std::size_t row = 0; row < pose_size; ++row)
                for (std::size_t column = 0; column < point_size; ++column)
                    for (std::size_t k = 0; k < point_size; ++k)
                        transformed[row * point_size + column] +=
                            cross_first[row * point_size + k] * inverse[k * point_size + column];
            for (std::size_t second = adjacency.point_offsets[point];
                 second < adjacency.point_offsets[point + 1]; ++second) {
                const std::size_t observation_second = adjacency.point_observations[second];
                const Index camera_second = problem.observations.camera[observation_second];
                // The reduced camera system is symmetric. Assemble only its
                // upper triangle, then materialize the lower triangle below.
                if (camera_second < camera_first) continue;
                const double* cross_second = system.cross.data() + observation_second * cross_block_size;
                const std::size_t full_block = find_schur_block(
                    pattern, camera_first, camera_second);
                double* block = destination +
                    pattern.upper_indices[full_block] * pose_block_size;
                for (std::size_t row = 0; row < pose_size; ++row)
                    for (std::size_t column = 0; column < pose_size; ++column)
                        for (std::size_t k = 0; k < point_size; ++k)
                            block[row * pose_size + column] -=
                                transformed[row * point_size + k] *
                                cross_second[column * point_size + k];
            }
        }
    }

    // Only the upper triangle was evaluated. Reduce it into the persistent
    // result, then materialize the lower triangle by block transpose.
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::int64_t row_signed = 0;
         row_signed < static_cast<std::int64_t>(problem.poses.size());
         ++row_signed) {
        const auto row = static_cast<std::size_t>(row_signed);
        for (std::size_t cursor = pattern.row_offsets[row];
             cursor < pattern.row_offsets[row + 1]; ++cursor) {
            const auto column = static_cast<std::size_t>(pattern.columns[cursor]);
            if (column < row) continue;
            const std::size_t upper = pattern.upper_indices[cursor];
            double* block =
                reduced.blocks.data() + cursor * pose_block_size;
            for (std::size_t value = 0; value < pose_block_size; ++value) {
                double sum = block[value];
                for (int thread = 0; thread < thread_count; ++thread)
                    sum += workspace.local_schur
                        [static_cast<std::size_t>(thread) *
                             local_values +
                         upper * pose_block_size + value];
                block[value] = sum;
            }
        }
    }
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t row_signed = 0;
         row_signed < static_cast<std::int64_t>(problem.poses.size());
         ++row_signed) {
        const auto row = static_cast<std::size_t>(row_signed);
        for (std::size_t cursor = pattern.row_offsets[row];
             cursor < pattern.row_offsets[row + 1]; ++cursor) {
            const auto column = static_cast<std::size_t>(pattern.columns[cursor]);
            if (column >= row) continue;
            const std::size_t upper = find_schur_block(
                pattern, column, static_cast<Index>(row));
            double* lower_block =
                reduced.blocks.data() + cursor * pose_block_size;
            const double* upper_block =
                reduced.blocks.data() + upper * pose_block_size;
            for (std::size_t block_row = 0; block_row < pose_size; ++block_row)
                for (std::size_t block_column = 0;
                     block_column < pose_size; ++block_column)
                    lower_block[block_row * pose_size + block_column] =
                        upper_block[block_column * pose_size + block_row];
        }
    }
}

bool invert_symmetric_3x3(const double* matrix, double* inverse) {
    const double a = matrix[0];
    const double b = matrix[1];
    const double c = matrix[2];
    const double d = matrix[4];
    const double e = matrix[5];
    const double f = matrix[8];
    const double c00 = d * f - e * e;
    const double c01 = c * e - b * f;
    const double c02 = b * e - c * d;
    const double c11 = a * f - c * c;
    const double c12 = b * c - a * e;
    const double c22 = a * d - b * b;
    const double determinant = a * c00 + b * c01 + c * c02;
    if (!(determinant > 1e-30) || !std::isfinite(determinant)) {
        return false;
    }
    const double scale = 1.0 / determinant;
    inverse[0] = c00 * scale; inverse[1] = c01 * scale; inverse[2] = c02 * scale;
    inverse[3] = inverse[1]; inverse[4] = c11 * scale; inverse[5] = c12 * scale;
    inverse[6] = inverse[2]; inverse[7] = inverse[5]; inverse[8] = c22 * scale;
    return true;
}

void multiply3(const double* matrix, const double* vector, double* output) {
    for (std::size_t row = 0; row < point_size; ++row) {
        output[row] = matrix[row * point_size] * vector[0] +
                      matrix[row * point_size + 1] * vector[1] +
                      matrix[row * point_size + 2] * vector[2];
    }
}

bool solve_spd6(const double* matrix, const double* rhs, double* solution) {
    double lower[pose_block_size]{};
    for (std::size_t row = 0; row < pose_size; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = matrix[row * pose_size + column];
            for (std::size_t k = 0; k < column; ++k) {
                value -= lower[row * pose_size + k] * lower[column * pose_size + k];
            }
            if (row == column) {
                if (!(value > 1e-24) || !std::isfinite(value)) {
                    return false;
                }
                lower[row * pose_size + column] = std::sqrt(value);
            } else {
                lower[row * pose_size + column] =
                    value / lower[column * pose_size + column];
            }
        }
    }
    double temporary[pose_size]{};
    for (std::size_t row = 0; row < pose_size; ++row) {
        double value = rhs[row];
        for (std::size_t k = 0; k < row; ++k) {
            value -= lower[row * pose_size + k] * temporary[k];
        }
        temporary[row] = value / lower[row * pose_size + row];
    }
    for (std::size_t reverse = pose_size; reverse-- > 0;) {
        double value = temporary[reverse];
        for (std::size_t k = reverse + 1; k < pose_size; ++k) {
            value -= lower[k * pose_size + reverse] * solution[k];
        }
        solution[reverse] = value / lower[reverse * pose_size + reverse];
    }
    return true;
}

bool solve_spd(
    const double* matrix, const std::size_t size,
    const double* rhs, double* solution) {
    std::vector<double> lower(size * size, 0.0);
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t column = 0; column <= row; ++column) {
            double value = matrix[row * size + column];
            for (std::size_t k = 0; k < column; ++k)
                value -=
                    lower[row * size + k] * lower[column * size + k];
            if (row == column) {
                if (!(value > 1e-24) || !std::isfinite(value))
                    return false;
                lower[row * size + column] = std::sqrt(value);
            } else {
                lower[row * size + column] =
                    value / lower[column * size + column];
            }
        }
    }
    std::vector<double> temporary(size, 0.0);
    for (std::size_t row = 0; row < size; ++row) {
        double value = rhs[row];
        for (std::size_t k = 0; k < row; ++k)
            value -= lower[row * size + k] * temporary[k];
        temporary[row] = value / lower[row * size + row];
    }
    for (std::size_t reverse = size; reverse-- > 0;) {
        double value = temporary[reverse];
        for (std::size_t k = reverse + 1; k < size; ++k)
            value -= lower[k * size + reverse] * solution[k];
        solution[reverse] = value / lower[reverse * size + reverse];
    }
    return true;
}

void assemble_system(
    const Problem& problem,
    const LinearizationOutput& linearization,
    const Adjacency& adjacency,
    const double damping,
    const bool fix_first_point,
    const bool optimize_points,
    const bool optimize_rotations,
    const bool optimize_translations,
    const std::size_t intrinsic_dof,
    AssemblyWorkspace& workspace,
    System& system) {
    const std::size_t camera_count = problem.poses.size();
    const std::size_t point_count = problem.points.size();
    system.camera_hessian.assign(camera_count * pose_block_size, 0.0);
    system.camera_rhs.assign(camera_count * pose_size, 0.0);
    system.point_inverse.assign(point_count * point_block_size, 0.0);
    system.point_rhs.assign(point_count * point_size, 0.0);
    system.cross.assign(problem.observations.size() * cross_block_size, 0.0);
    system.intrinsic_dof = intrinsic_dof;
    system.intrinsic_group_count = problem.intrinsics.size();
    if (intrinsic_dof > 0) {
        system.intrinsic_hessian.assign(
            problem.intrinsics.size() * intrinsic_dof * intrinsic_dof, 0.0);
        system.intrinsic_rhs.assign(
            problem.intrinsics.size() * intrinsic_dof, 0.0);
        system.pose_intrinsic_cross.assign(
            problem.observations.size() * pose_size * intrinsic_dof, 0.0);
        system.point_intrinsic_cross.assign(
            problem.observations.size() * point_size * intrinsic_dof, 0.0);
    } else {
        system.intrinsic_hessian.clear();
        system.intrinsic_rhs.clear();
        system.pose_intrinsic_cross.clear();
        system.point_intrinsic_cross.clear();
    }

    int thread_count = 1;
#if defined(AETHERSCAN_HAS_OPENMP)
    const std::size_t values_per_thread = std::max<std::size_t>(1, camera_count * 42);
    const std::size_t memory_limited_threads = std::max<std::size_t>(1, 32'000'000 / values_per_thread);
    thread_count = std::min<int>(omp_get_max_threads(), static_cast<int>(memory_limited_threads));
#endif
    workspace.local_cameras.assign(
        static_cast<std::size_t>(thread_count) * camera_count * 42, 0.0);
    workspace.local_intrinsics.assign(
        intrinsic_dof > 0
            ? static_cast<std::size_t>(thread_count) *
                  problem.intrinsics.size() *
                  (intrinsic_dof * intrinsic_dof + intrinsic_dof)
            : 0,
        0.0);

#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
    for (std::int64_t point_signed = 0;
         point_signed < static_cast<std::int64_t>(point_count); ++point_signed) {
        const auto point = static_cast<std::size_t>(point_signed);
        int thread = 0;
#if defined(AETHERSCAN_HAS_OPENMP)
        thread = omp_get_thread_num();
#endif
        double* camera_accumulator = workspace.local_cameras.data() +
            static_cast<std::size_t>(thread) * camera_count * 42;
        double* intrinsic_accumulator =
            intrinsic_dof > 0
                ? workspace.local_intrinsics.data() +
                      static_cast<std::size_t>(thread) *
                          problem.intrinsics.size() *
                          (intrinsic_dof * intrinsic_dof + intrinsic_dof)
                : nullptr;
        double point_hessian[point_block_size]{};
        double* point_rhs = system.point_rhs.data() + point * point_size;
        for (std::size_t cursor = adjacency.point_offsets[point];
             cursor < adjacency.point_offsets[point + 1]; ++cursor) {
            const std::size_t observation = adjacency.point_observations[cursor];
            const auto& value = linearization.observations[observation];
            if (!value.valid) {
                continue;
            }
            const std::size_t camera = problem.observations.camera[observation];
            double* camera_hessian = camera_accumulator + camera * 42;
            double* camera_rhs = camera_hessian + pose_block_size;
            double* cross = system.cross.data() + observation * cross_block_size;
            for (std::size_t row = 0; row < pose_size; ++row) {
                for (std::size_t column = 0; column < pose_size; ++column) {
                    camera_hessian[row * pose_size + column] +=
                        value.pose_jacobian[row] * value.pose_jacobian[column] +
                        value.pose_jacobian[pose_size + row] *
                        value.pose_jacobian[pose_size + column];
                }
                camera_rhs[row] -= value.pose_jacobian[row] * value.residual[0] +
                                   value.pose_jacobian[pose_size + row] * value.residual[1];
                for (std::size_t column = 0; column < point_size; ++column) {
                    cross[row * point_size + column] =
                        value.pose_jacobian[row] * value.point_jacobian[column] +
                        value.pose_jacobian[pose_size + row] *
                        value.point_jacobian[point_size + column];
                }
            }
            for (std::size_t row = 0; row < point_size; ++row) {
                point_rhs[row] -= value.point_jacobian[row] * value.residual[0] +
                                  value.point_jacobian[point_size + row] * value.residual[1];
                for (std::size_t column = 0; column < point_size; ++column) {
                    point_hessian[row * point_size + column] +=
                        value.point_jacobian[row] * value.point_jacobian[column] +
                        value.point_jacobian[point_size + row] *
                        value.point_jacobian[point_size + column];
                }
            }
            if (intrinsic_dof > 0) {
                const std::size_t intrinsic_group =
                    problem.intrinsic_index(camera);
                double* group_accumulator =
                    intrinsic_accumulator +
                    intrinsic_group *
                        (intrinsic_dof * intrinsic_dof + intrinsic_dof);
                double* pose_intr =
                    system.pose_intrinsic_cross.data() +
                    observation * pose_size * intrinsic_dof;
                double* point_intr =
                    system.point_intrinsic_cross.data() +
                    observation * point_size * intrinsic_dof;
                for (std::size_t param = 0; param < intrinsic_dof; ++param) {
                    const double j0 = value.intrinsic_jacobian[param];
                    const double j1 =
                        value.intrinsic_jacobian[k_max_intrinsic_params + param];
                    group_accumulator[intrinsic_dof * intrinsic_dof + param] -=
                        j0 * value.residual[0] + j1 * value.residual[1];
                    for (std::size_t other = 0; other < intrinsic_dof; ++other) {
                        const double o0 = value.intrinsic_jacobian[other];
                        const double o1 =
                            value.intrinsic_jacobian[k_max_intrinsic_params + other];
                        group_accumulator[param * intrinsic_dof + other] +=
                            j0 * o0 + j1 * o1;
                    }
                    for (std::size_t row = 0; row < pose_size; ++row) {
                        pose_intr[row * intrinsic_dof + param] =
                            value.pose_jacobian[row] * j0 +
                            value.pose_jacobian[pose_size + row] * j1;
                    }
                    for (std::size_t row = 0; row < point_size; ++row) {
                        point_intr[row * intrinsic_dof + param] =
                            value.point_jacobian[row] * j0 +
                            value.point_jacobian[point_size + row] * j1;
                    }
                }
            }
        }
        for (std::size_t diagonal = 0; diagonal < point_size; ++diagonal) {
            point_hessian[diagonal * point_size + diagonal] +=
                damping * (point_hessian[diagonal * point_size + diagonal] + 1.0);
        }
        double* inverse = system.point_inverse.data() + point * point_block_size;
        if (!optimize_points || (fix_first_point && point == 0) ||
            !invert_symmetric_3x3(point_hessian, inverse)) {
            std::fill(inverse, inverse + point_block_size, 0.0);
            std::fill(point_rhs, point_rhs + point_size, 0.0);
        }
    }

#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(camera_count); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        double* destination_hessian = system.camera_hessian.data() + camera * pose_block_size;
        double* destination_rhs = system.camera_rhs.data() + camera * pose_size;
        for (int thread = 0; thread < thread_count; ++thread) {
            const double* source = workspace.local_cameras.data() +
                static_cast<std::size_t>(thread) * camera_count * 42 + camera * 42;
            for (std::size_t i = 0; i < pose_block_size; ++i) destination_hessian[i] += source[i];
            for (std::size_t i = 0; i < pose_size; ++i) destination_rhs[i] += source[pose_block_size + i];
        }
        for (std::size_t diagonal = 0; diagonal < pose_size; ++diagonal) {
            destination_hessian[diagonal * pose_size + diagonal] +=
                damping * (destination_hessian[diagonal * pose_size + diagonal] + 1.0);
        }
        if (!optimize_rotations) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                for (std::size_t column = 0; column < pose_size; ++column) {
                    destination_hessian[axis * pose_size + column] = 0.0;
                    destination_hessian[column * pose_size + axis] = 0.0;
                }
                destination_hessian[axis * pose_size + axis] = 1.0;
                destination_rhs[axis] = 0.0;
            }
        }
        if (!optimize_translations) {
            for (std::size_t axis = 3; axis < pose_size; ++axis) {
                for (std::size_t column = 0; column < pose_size; ++column) {
                    destination_hessian[axis * pose_size + column] = 0.0;
                    destination_hessian[column * pose_size + axis] = 0.0;
                }
                destination_hessian[axis * pose_size + axis] = 1.0;
                destination_rhs[axis] = 0.0;
            }
        }
    }
    if (intrinsic_dof > 0) {
        const std::size_t group_stride =
            intrinsic_dof * intrinsic_dof + intrinsic_dof;
        for (std::size_t group = 0; group < problem.intrinsics.size(); ++group) {
            double* group_hessian =
                system.intrinsic_hessian.data() +
                group * intrinsic_dof * intrinsic_dof;
            double* group_rhs =
                system.intrinsic_rhs.data() + group * intrinsic_dof;
            for (int thread = 0; thread < thread_count; ++thread) {
                const double* source =
                    workspace.local_intrinsics.data() +
                    (static_cast<std::size_t>(thread) *
                         problem.intrinsics.size() +
                     group) *
                        group_stride;
                for (std::size_t i = 0;
                     i < intrinsic_dof * intrinsic_dof; ++i)
                    group_hessian[i] += source[i];
                for (std::size_t i = 0; i < intrinsic_dof; ++i)
                    group_rhs[i] +=
                        source[intrinsic_dof * intrinsic_dof + i];
            }
            for (std::size_t diagonal = 0; diagonal < intrinsic_dof;
                 ++diagonal) {
                group_hessian[diagonal * intrinsic_dof + diagonal] +=
                    damping *
                    (group_hessian[diagonal * intrinsic_dof + diagonal] +
                     1.0);
            }
            if (problem.is_intrinsic_constant(group)) {
                std::fill(
                    group_hessian,
                    group_hessian + intrinsic_dof * intrinsic_dof, 0.0);
                std::fill(group_rhs, group_rhs + intrinsic_dof, 0.0);
                for (std::size_t diagonal = 0; diagonal < intrinsic_dof;
                     ++diagonal)
                    group_hessian[diagonal * intrinsic_dof + diagonal] = 1.0;
            }
        }
        for (std::size_t observation = 0;
             observation < problem.observations.size(); ++observation) {
            const std::size_t camera =
                problem.observations.camera[observation];
            const std::size_t group = problem.intrinsic_index(camera);
            if (!problem.is_intrinsic_constant(group)) continue;
            double* pose_intr =
                system.pose_intrinsic_cross.data() +
                observation * pose_size * intrinsic_dof;
            double* point_intr =
                system.point_intrinsic_cross.data() +
                observation * point_size * intrinsic_dof;
            std::fill(
                pose_intr, pose_intr + pose_size * intrinsic_dof, 0.0);
            std::fill(
                point_intr, point_intr + point_size * intrinsic_dof, 0.0);
        }
    }
    if (!optimize_rotations) {
        for (std::size_t observation = 0;
             observation < problem.observations.size(); ++observation) {
            double* cross =
                system.cross.data() + observation * cross_block_size;
            std::fill(cross, cross + 3 * point_size, 0.0);
            if (intrinsic_dof > 0) {
                double* pose_intr =
                    system.pose_intrinsic_cross.data() +
                    observation * pose_size * intrinsic_dof;
                std::fill(pose_intr, pose_intr + 3 * intrinsic_dof, 0.0);
            }
        }
    }
    if (!optimize_translations) {
        for (std::size_t observation = 0;
             observation < problem.observations.size(); ++observation) {
            double* cross =
                system.cross.data() + observation * cross_block_size;
            std::fill(
                cross + 3 * point_size,
                cross + pose_size * point_size, 0.0);
            if (intrinsic_dof > 0) {
                double* pose_intr =
                    system.pose_intrinsic_cross.data() +
                    observation * pose_size * intrinsic_dof;
                std::fill(
                    pose_intr + 3 * intrinsic_dof,
                    pose_intr + pose_size * intrinsic_dof, 0.0);
            }
        }
    }
}

void schur_rhs(
    const Problem& problem, const Adjacency& adjacency, const System& system,
    const bool fix_first, std::vector<double>& rhs) {
    rhs = system.camera_rhs;
    std::vector<double> reduced_point_rhs(problem.points.size() * point_size, 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t point_signed = 0;
         point_signed < static_cast<std::int64_t>(problem.points.size()); ++point_signed) {
        const auto point = static_cast<std::size_t>(point_signed);
        multiply3(system.point_inverse.data() + point * point_block_size,
                  system.point_rhs.data() + point * point_size,
                  reduced_point_rhs.data() + point * point_size);
    }
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(problem.poses.size()); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        double* output = rhs.data() + camera * pose_size;
        if (problem.is_pose_constant(camera, fix_first)) {
            std::fill(output, output + pose_size, 0.0);
            continue;
        }
        for (std::size_t cursor = adjacency.camera_offsets[camera];
             cursor < adjacency.camera_offsets[camera + 1]; ++cursor) {
            const std::size_t observation = adjacency.camera_observations[cursor];
            const std::size_t point = problem.observations.point[observation];
            const double* temporary = reduced_point_rhs.data() + point * point_size;
            const double* cross = system.cross.data() + observation * cross_block_size;
            for (std::size_t row = 0; row < pose_size; ++row) {
                for (std::size_t column = 0; column < point_size; ++column) {
                    output[row] -= cross[row * point_size + column] * temporary[column];
                }
            }
        }
    }
}

void schur_multiply(
    const Problem& problem, const Adjacency& adjacency, const System& system,
    const bool fix_first, const std::vector<double>& input, std::vector<double>& output,
    std::vector<double>& point_temporary) {
    output.assign(input.size(), 0.0);
    point_temporary.assign(problem.points.size() * point_size, 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t point_signed = 0;
         point_signed < static_cast<std::int64_t>(problem.points.size()); ++point_signed) {
        const auto point = static_cast<std::size_t>(point_signed);
        double accumulated[point_size]{};
        for (std::size_t cursor = adjacency.point_offsets[point];
             cursor < adjacency.point_offsets[point + 1]; ++cursor) {
            const std::size_t observation = adjacency.point_observations[cursor];
            const std::size_t camera = problem.observations.camera[observation];
            if (problem.is_pose_constant(camera, fix_first)) continue;
            const double* cross = system.cross.data() + observation * cross_block_size;
            const double* camera_input = input.data() + camera * pose_size;
            for (std::size_t column = 0; column < point_size; ++column) {
                for (std::size_t row = 0; row < pose_size; ++row) {
                    accumulated[column] += cross[row * point_size + column] * camera_input[row];
                }
            }
        }
        multiply3(system.point_inverse.data() + point * point_block_size,
                  accumulated, point_temporary.data() + point * point_size);
    }
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(problem.poses.size()); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        double* destination = output.data() + camera * pose_size;
        const double* camera_input = input.data() + camera * pose_size;
        if (problem.is_pose_constant(camera, fix_first)) {
            std::copy(camera_input, camera_input + pose_size, destination);
            continue;
        }
        const double* hessian = system.camera_hessian.data() + camera * pose_block_size;
        for (std::size_t row = 0; row < pose_size; ++row) {
            for (std::size_t column = 0; column < pose_size; ++column) {
                destination[row] += hessian[row * pose_size + column] * camera_input[column];
            }
        }
        for (std::size_t cursor = adjacency.camera_offsets[camera];
             cursor < adjacency.camera_offsets[camera + 1]; ++cursor) {
            const std::size_t observation = adjacency.camera_observations[cursor];
            const std::size_t point = problem.observations.point[observation];
            const double* cross = system.cross.data() + observation * cross_block_size;
            const double* temporary = point_temporary.data() + point * point_size;
            for (std::size_t row = 0; row < pose_size; ++row) {
                for (std::size_t column = 0; column < point_size; ++column) {
                    destination[row] -= cross[row * point_size + column] * temporary[column];
                }
            }
        }
    }
}

void explicit_schur_multiply(
    const Problem& problem, const SchurPattern& pattern,
    const ExplicitSchur& system, const bool fix_first,
    const std::vector<double>& input, std::vector<double>& output) {
    const std::size_t camera_count = pattern.row_offsets.size() - 1;
    output.assign(input.size(), 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(pattern.columns.size() >= 32768)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(camera_count); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        double* destination = output.data() + camera * pose_size;
        if (problem.is_pose_constant(camera, fix_first)) {
            std::copy_n(input.data() + camera * pose_size, pose_size, destination);
            continue;
        }
        for (std::size_t block = pattern.row_offsets[camera];
             block < pattern.row_offsets[camera + 1]; ++block) {
            const std::size_t column_camera = pattern.columns[block];
            if (problem.is_pose_constant(column_camera, fix_first)) continue;
            const double* matrix = system.blocks.data() + block * pose_block_size;
            const double* vector = input.data() + column_camera * pose_size;
            for (std::size_t row = 0; row < pose_size; ++row)
                for (std::size_t column = 0; column < pose_size; ++column)
                    destination[row] += matrix[row * pose_size + column] * vector[column];
        }
    }
}

void precondition_explicit(
    const Problem& problem, const SchurPattern& pattern,
    const ExplicitSchur& system, const bool fix_first,
    const std::vector<double>& residual, std::vector<double>& output) {
    const std::size_t camera_count = pattern.row_offsets.size() - 1;
    output.assign(residual.size(), 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(camera_count >= 512)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(camera_count); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        if (problem.is_pose_constant(camera, fix_first)) continue;
        const double* diagonal = system.blocks.data() + pattern.diagonal_blocks[camera] * pose_block_size;
        if (!solve_spd6(diagonal, residual.data() + camera * pose_size,
                        output.data() + camera * pose_size)) {
            for (std::size_t i = 0; i < pose_size; ++i)
                output[camera * pose_size + i] = residual[camera * pose_size + i] /
                    std::max(diagonal[i * pose_size + i], 1e-12);
        }
    }
}

double dot(const std::vector<double>& first, const std::vector<double>& second) {
    double value = 0.0;
    // PCG executes several reductions per iteration. For small camera
    // systems, repeatedly waking the full OpenMP team dominates the solve.
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for reduction(+:value) schedule(static) if(first.size() >= 8192)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(first.size()); ++i) {
        value += first[static_cast<std::size_t>(i)] * second[static_cast<std::size_t>(i)];
    }
    return value;
}

void precondition(
    const Problem& problem, const System& system, const bool fix_first,
    const std::vector<double>& residual, std::vector<double>& output) {
    const std::size_t camera_count = system.camera_rhs.size() / pose_size;
    output.assign(residual.size(), 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(camera_count >= 512)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(camera_count); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        if (problem.is_pose_constant(camera, fix_first)) continue;
        if (!solve_spd6(system.camera_hessian.data() + camera * pose_block_size,
                        residual.data() + camera * pose_size,
                        output.data() + camera * pose_size)) {
            for (std::size_t i = 0; i < pose_size; ++i) {
                const double diagonal = system.camera_hessian[camera * pose_block_size + i * pose_size + i];
                output[camera * pose_size + i] = residual[camera * pose_size + i] /
                    std::max(diagonal, 1e-12);
            }
        }
    }
}

std::size_t solve_pcg(
    const Problem& problem, const Adjacency& adjacency, const System& system,
    const SchurPattern& pattern, const ExplicitSchur& explicit_system,
    const OptimizerOptions& options, const std::vector<double>& rhs,
    std::vector<double>& solution) {
    solution.assign(rhs.size(), 0.0);
    std::vector<double> residual = rhs;
    std::vector<double> z, direction, product, point_temporary;
    if (explicit_system.blocks.empty())
        precondition(problem, system, options.fix_first_pose, residual, z);
    else
        precondition_explicit(
            problem, pattern, explicit_system, options.fix_first_pose, residual, z);
    direction = z;
    double rz = dot(residual, z);
    const double target = options.pcg_tolerance * options.pcg_tolerance *
                          std::max(dot(rhs, rhs), 1e-30);
    if (!std::isfinite(rz)) return 0;
    if (dot(residual, residual) <= target || std::abs(rz) <= 1e-30)
        return 0;
    std::size_t iteration = 0;
    for (; iteration < options.maximum_pcg_iterations; ++iteration) {
        if (explicit_system.blocks.empty())
            schur_multiply(problem, adjacency, system, options.fix_first_pose,
                           direction, product, point_temporary);
        else
            explicit_schur_multiply(
                problem, pattern, explicit_system, options.fix_first_pose,
                direction, product);
        const double denominator = dot(direction, product);
        if (!(denominator > 1e-30) || !std::isfinite(denominator)) break;
        const double alpha = rz / denominator;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(solution.size() >= 8192)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(solution.size()); ++i) {
            const auto index = static_cast<std::size_t>(i);
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
        }
        if (dot(residual, residual) <= target) {
            ++iteration;
            break;
        }
        if (explicit_system.blocks.empty())
            precondition(problem, system, options.fix_first_pose, residual, z);
        else
            precondition_explicit(
                problem, pattern, explicit_system, options.fix_first_pose, residual, z);
        const double next_rz = dot(residual, z);
        if (!std::isfinite(next_rz)) break;
        if (std::abs(rz) <= 1e-30) break;
        const double beta = next_rz / rz;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(direction.size() >= 8192)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(direction.size()); ++i) {
            const auto index = static_cast<std::size_t>(i);
            direction[index] = z[index] + beta * direction[index];
        }
        rz = next_rz;
    }
    return iteration;
}

void recover_point_step(
    const Problem& problem, const Adjacency& adjacency, const System& system,
    const std::vector<double>& camera_step, const std::vector<double>& intrinsic_step,
    std::vector<double>& point_step) {
    point_step.assign(problem.points.size() * point_size, 0.0);
    const std::size_t block_dof = system.intrinsic_dof;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t point_signed = 0;
         point_signed < static_cast<std::int64_t>(problem.points.size()); ++point_signed) {
        const auto point = static_cast<std::size_t>(point_signed);
        double rhs[point_size];
        std::copy_n(system.point_rhs.data() + point * point_size, point_size, rhs);
        for (std::size_t cursor = adjacency.point_offsets[point];
             cursor < adjacency.point_offsets[point + 1]; ++cursor) {
            const std::size_t observation = adjacency.point_observations[cursor];
            const std::size_t camera = problem.observations.camera[observation];
            const double* cross = system.cross.data() + observation * cross_block_size;
            const double* step = camera_step.data() + camera * pose_size;
            for (std::size_t column = 0; column < point_size; ++column) {
                for (std::size_t row = 0; row < pose_size; ++row) {
                    rhs[column] -= cross[row * point_size + column] * step[row];
                }
            }
            if (block_dof > 0 && !intrinsic_step.empty()) {
                const std::size_t intrinsic_offset =
                    problem.intrinsic_index(camera) * block_dof;
                const double* point_intr =
                    system.point_intrinsic_cross.data() +
                    observation * point_size * block_dof;
                for (std::size_t column = 0; column < point_size; ++column) {
                    for (std::size_t param = 0; param < block_dof; ++param) {
                        rhs[column] -=
                            point_intr[column * block_dof + param] *
                            intrinsic_step[intrinsic_offset + param];
                    }
                }
            }
        }
        multiply3(system.point_inverse.data() + point * point_block_size,
                  rhs, point_step.data() + point * point_size);
    }
}

void build_joint_rhs(
    const Problem& problem, const Adjacency& adjacency,
    const System& system, const bool fix_first,
    std::vector<double>& rhs) {
    schur_rhs(problem, adjacency, system, fix_first, rhs);
    const std::size_t camera_values = problem.poses.size() * pose_size;
    const std::size_t block_dof = system.intrinsic_dof;
    const std::size_t intrinsic_values =
        system.intrinsic_group_count * block_dof;
    rhs.resize(camera_values + intrinsic_values, 0.0);
    std::copy(
        system.intrinsic_rhs.begin(), system.intrinsic_rhs.end(),
        rhs.begin() + static_cast<std::ptrdiff_t>(camera_values));
    for (std::size_t point = 0; point < problem.points.size(); ++point) {
        double reduced_point[point_size]{};
        multiply3(
            system.point_inverse.data() + point * point_block_size,
            system.point_rhs.data() + point * point_size,
            reduced_point);
        for (std::size_t cursor = adjacency.point_offsets[point];
             cursor < adjacency.point_offsets[point + 1]; ++cursor) {
            const std::size_t observation =
                adjacency.point_observations[cursor];
            const std::size_t camera =
                problem.observations.camera[observation];
            const std::size_t offset =
                camera_values +
                problem.intrinsic_index(camera) * block_dof;
            const double* point_intrinsic =
                system.point_intrinsic_cross.data() +
                observation * point_size * block_dof;
            for (std::size_t param = 0; param < block_dof; ++param)
                for (std::size_t row = 0; row < point_size; ++row)
                    rhs[offset + param] -=
                        point_intrinsic[row * block_dof + param] *
                        reduced_point[row];
        }
    }
}

void joint_multiply(
    const Problem& problem, const Adjacency& adjacency,
    const System& system, const bool fix_first,
    const std::vector<double>& input, std::vector<double>& output) {
    const std::size_t camera_values = problem.poses.size() * pose_size;
    const std::size_t block_dof = system.intrinsic_dof;
    const std::size_t intrinsic_values =
        system.intrinsic_group_count * block_dof;
    const std::size_t total_values = camera_values + intrinsic_values;
    output.assign(total_values, 0.0);

    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera) {
        double* destination = output.data() + camera * pose_size;
        const double* source = input.data() + camera * pose_size;
        if (problem.is_pose_constant(camera, fix_first)) {
            std::copy_n(source, pose_size, destination);
            continue;
        }
        const double* hessian =
            system.camera_hessian.data() + camera * pose_block_size;
        for (std::size_t row = 0; row < pose_size; ++row)
            for (std::size_t column = 0; column < pose_size; ++column)
                destination[row] +=
                    hessian[row * pose_size + column] * source[column];
    }
    for (std::size_t group = 0; group < system.intrinsic_group_count; ++group) {
        const double* hessian =
            system.intrinsic_hessian.data() +
            group * block_dof * block_dof;
        const double* source =
            input.data() + camera_values + group * block_dof;
        double* destination =
            output.data() + camera_values + group * block_dof;
        for (std::size_t row = 0; row < block_dof; ++row)
            for (std::size_t column = 0; column < block_dof; ++column)
                destination[row] +=
                    hessian[row * block_dof + column] * source[column];
    }

    int thread_count = 1;
#if defined(AETHERSCAN_HAS_OPENMP)
    const std::size_t memory_limited_threads = std::max<std::size_t>(
        1, 32'000'000 / std::max<std::size_t>(1, total_values));
    thread_count = std::min<int>(
        omp_get_max_threads(), static_cast<int>(memory_limited_threads));
#endif
    std::vector<double> local(
        static_cast<std::size_t>(thread_count) * total_values, 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
    for (std::int64_t point_signed = 0;
         point_signed < static_cast<std::int64_t>(problem.points.size());
         ++point_signed) {
        const std::size_t point = static_cast<std::size_t>(point_signed);
        int thread = 0;
#if defined(AETHERSCAN_HAS_OPENMP)
        thread = omp_get_thread_num();
#endif
        double* destination =
            local.data() + static_cast<std::size_t>(thread) * total_values;
        double point_input[point_size]{};
        for (std::size_t cursor = adjacency.point_offsets[point];
             cursor < adjacency.point_offsets[point + 1]; ++cursor) {
            const std::size_t observation =
                adjacency.point_observations[cursor];
            const std::size_t camera =
                problem.observations.camera[observation];
            const std::size_t group =
                problem.intrinsic_index(camera);
            const double* camera_input =
                input.data() + camera * pose_size;
            const double* intrinsic_input =
                input.data() + camera_values + group * block_dof;
            const double* cross =
                system.cross.data() + observation * cross_block_size;
            const double* pose_intrinsic =
                system.pose_intrinsic_cross.data() +
                observation * pose_size * block_dof;
            const double* point_intrinsic =
                system.point_intrinsic_cross.data() +
                observation * point_size * block_dof;
            if (!problem.is_pose_constant(camera, fix_first)) {
                for (std::size_t column = 0; column < point_size; ++column)
                    for (std::size_t row = 0; row < pose_size; ++row)
                        point_input[column] +=
                            cross[row * point_size + column] *
                            camera_input[row];
                double* camera_destination =
                    destination + camera * pose_size;
                double* intrinsic_destination =
                    destination + camera_values + group * block_dof;
                for (std::size_t row = 0; row < pose_size; ++row) {
                    for (std::size_t param = 0; param < block_dof; ++param) {
                        camera_destination[row] +=
                            pose_intrinsic[row * block_dof + param] *
                            intrinsic_input[param];
                        intrinsic_destination[param] +=
                            pose_intrinsic[row * block_dof + param] *
                            camera_input[row];
                    }
                }
            }
            for (std::size_t row = 0; row < point_size; ++row)
                for (std::size_t param = 0; param < block_dof; ++param)
                    point_input[row] +=
                        point_intrinsic[row * block_dof + param] *
                        intrinsic_input[param];
        }
        double reduced[point_size]{};
        multiply3(
            system.point_inverse.data() + point * point_block_size,
            point_input, reduced);
        for (std::size_t cursor = adjacency.point_offsets[point];
             cursor < adjacency.point_offsets[point + 1]; ++cursor) {
            const std::size_t observation =
                adjacency.point_observations[cursor];
            const std::size_t camera =
                problem.observations.camera[observation];
            const std::size_t group =
                problem.intrinsic_index(camera);
            const double* cross =
                system.cross.data() + observation * cross_block_size;
            const double* point_intrinsic =
                system.point_intrinsic_cross.data() +
                observation * point_size * block_dof;
            if (!problem.is_pose_constant(camera, fix_first)) {
                double* camera_destination =
                    destination + camera * pose_size;
                for (std::size_t row = 0; row < pose_size; ++row)
                    for (std::size_t column = 0; column < point_size; ++column)
                        camera_destination[row] -=
                            cross[row * point_size + column] *
                            reduced[column];
            }
            double* intrinsic_destination =
                destination + camera_values + group * block_dof;
            for (std::size_t param = 0; param < block_dof; ++param)
                for (std::size_t row = 0; row < point_size; ++row)
                    intrinsic_destination[param] -=
                        point_intrinsic[row * block_dof + param] *
                        reduced[row];
        }
    }
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t value_signed = 0;
         value_signed < static_cast<std::int64_t>(total_values);
         ++value_signed) {
        const std::size_t value = static_cast<std::size_t>(value_signed);
        for (int thread = 0; thread < thread_count; ++thread)
            output[value] +=
                local[static_cast<std::size_t>(thread) * total_values + value];
    }
}

void precondition_joint(
    const Problem& problem, const System& system,
    const bool fix_first, const std::vector<double>& residual,
    std::vector<double>& output) {
    const std::size_t camera_values = problem.poses.size() * pose_size;
    const std::size_t block_dof = system.intrinsic_dof;
    output.assign(residual.size(), 0.0);
    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera) {
        if (problem.is_pose_constant(camera, fix_first)) continue;
        const double* hessian =
            system.camera_hessian.data() + camera * pose_block_size;
        const double* rhs = residual.data() + camera * pose_size;
        double* destination = output.data() + camera * pose_size;
        if (!solve_spd6(hessian, rhs, destination)) {
            for (std::size_t i = 0; i < pose_size; ++i)
                destination[i] =
                    rhs[i] /
                    std::max(hessian[i * pose_size + i], 1e-12);
        }
    }
    for (std::size_t group = 0; group < system.intrinsic_group_count; ++group) {
        const double* hessian =
            system.intrinsic_hessian.data() +
            group * block_dof * block_dof;
        const double* rhs =
            residual.data() + camera_values + group * block_dof;
        double* destination =
            output.data() + camera_values + group * block_dof;
        if (!solve_spd(hessian, block_dof, rhs, destination)) {
            for (std::size_t i = 0; i < block_dof; ++i)
                destination[i] =
                    rhs[i] /
                    std::max(hessian[i * block_dof + i], 1e-12);
        }
    }
}

std::size_t solve_joint_pcg(
    const Problem& problem, const Adjacency& adjacency,
    const System& system, const OptimizerOptions& options,
    const std::vector<double>& rhs, std::vector<double>& solution) {
    solution.assign(rhs.size(), 0.0);
    std::vector<double> residual = rhs;
    std::vector<double> z, direction, product;
    precondition_joint(
        problem, system, options.fix_first_pose, residual, z);
    direction = z;
    double rz = dot(residual, z);
    const double target =
        options.pcg_tolerance * options.pcg_tolerance *
        std::max(dot(rhs, rhs), 1e-30);
    if (!std::isfinite(rz)) return 0;
    if (dot(residual, residual) <= target || std::abs(rz) <= 1e-30)
        return 0;
    std::size_t iteration = 0;
    for (; iteration < options.maximum_pcg_iterations; ++iteration) {
        joint_multiply(
            problem, adjacency, system, options.fix_first_pose,
            direction, product);
        const double denominator = dot(direction, product);
        if (!(denominator > 1e-30) || !std::isfinite(denominator)) break;
        const double alpha = rz / denominator;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i = 0;
             i < static_cast<std::int64_t>(solution.size()); ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
        }
        if (dot(residual, residual) <= target) {
            ++iteration;
            break;
        }
        precondition_joint(
            problem, system, options.fix_first_pose, residual, z);
        const double next_rz = dot(residual, z);
        if (!std::isfinite(next_rz)) break;
        if (std::abs(rz) <= 1e-30) break;
        const double beta = next_rz / rz;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i = 0;
             i < static_cast<std::int64_t>(direction.size()); ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            direction[index] = z[index] + beta * direction[index];
        }
        rz = next_rz;
    }
    return iteration;
}

IntrinsicSchur build_intrinsic_schur(
    const Problem& problem, const Adjacency& adjacency,
    const System& system) {
    IntrinsicSchur result;
    const std::size_t block_dof = system.intrinsic_dof;
    result.dof = block_dof * system.intrinsic_group_count;
    if (result.dof == 0) return result;
    const std::size_t dof = result.dof;
    result.S_ii.assign(dof * dof, 0.0);
    result.b_i.assign(dof, 0.0);
    result.S_ci.assign(problem.poses.size() * pose_size * dof, 0.0);

    for (std::size_t group = 0; group < system.intrinsic_group_count; ++group) {
        const std::size_t offset = group * block_dof;
        const double* hessian =
            system.intrinsic_hessian.data() +
            group * block_dof * block_dof;
        const double* rhs = system.intrinsic_rhs.data() + offset;
        for (std::size_t row = 0; row < block_dof; ++row) {
            result.b_i[offset + row] = rhs[row];
            for (std::size_t column = 0; column < block_dof; ++column)
                result.S_ii[(offset + row) * dof + offset + column] =
                    hessian[row * block_dof + column];
        }
    }
    for (std::size_t observation = 0;
         observation < problem.observations.size(); ++observation) {
        const std::size_t camera =
            problem.observations.camera[observation];
        const std::size_t offset =
            problem.intrinsic_index(camera) * block_dof;
        const double* pose_intrinsic =
            system.pose_intrinsic_cross.data() +
            observation * pose_size * block_dof;
        double* destination =
            result.S_ci.data() + camera * pose_size * dof;
        for (std::size_t row = 0; row < pose_size; ++row)
            for (std::size_t param = 0; param < block_dof; ++param)
                destination[row * dof + offset + param] +=
                    pose_intrinsic[row * block_dof + param];
    }

    for (std::size_t point = 0; point < problem.points.size(); ++point) {
        const double* inverse =
            system.point_inverse.data() + point * point_block_size;
        std::vector<double> point_intrinsic(point_size * dof, 0.0);
        for (std::size_t cursor = adjacency.point_offsets[point];
             cursor < adjacency.point_offsets[point + 1]; ++cursor) {
            const std::size_t observation =
                adjacency.point_observations[cursor];
            const std::size_t camera =
                problem.observations.camera[observation];
            const std::size_t offset =
                problem.intrinsic_index(camera) * block_dof;
            const double* source =
                system.point_intrinsic_cross.data() +
                observation * point_size * block_dof;
            for (std::size_t row = 0; row < point_size; ++row)
                for (std::size_t param = 0; param < block_dof; ++param)
                    point_intrinsic[row * dof + offset + param] +=
                        source[row * block_dof + param];
        }
        std::vector<double> inverse_point_intrinsic(
            point_size * dof, 0.0);
        for (std::size_t param = 0; param < dof; ++param) {
            const double column[point_size] = {
                point_intrinsic[param],
                point_intrinsic[dof + param],
                point_intrinsic[2 * dof + param]};
            double transformed[point_size]{};
            multiply3(inverse, column, transformed);
            for (std::size_t row = 0; row < point_size; ++row)
                inverse_point_intrinsic[row * dof + param] =
                    transformed[row];
        }
        for (std::size_t row = 0; row < dof; ++row)
            for (std::size_t column = 0; column < dof; ++column)
                for (std::size_t axis = 0; axis < point_size; ++axis)
                    result.S_ii[row * dof + column] -=
                        point_intrinsic[axis * dof + row] *
                        inverse_point_intrinsic[axis * dof + column];
        double reduced_point[point_size]{};
        multiply3(
            inverse, system.point_rhs.data() + point * point_size,
            reduced_point);
        for (std::size_t param = 0; param < dof; ++param)
            for (std::size_t axis = 0; axis < point_size; ++axis)
                result.b_i[param] -=
                    point_intrinsic[axis * dof + param] *
                    reduced_point[axis];
        for (std::size_t cursor = adjacency.point_offsets[point];
             cursor < adjacency.point_offsets[point + 1]; ++cursor) {
            const std::size_t observation =
                adjacency.point_observations[cursor];
            const std::size_t camera =
                problem.observations.camera[observation];
            const double* cross =
                system.cross.data() + observation * cross_block_size;
            double* camera_intrinsic =
                result.S_ci.data() + camera * pose_size * dof;
            for (std::size_t row = 0; row < pose_size; ++row)
                for (std::size_t param = 0; param < dof; ++param)
                    for (std::size_t axis = 0; axis < point_size; ++axis)
                        camera_intrinsic[row * dof + param] -=
                            cross[row * point_size + axis] *
                            inverse_point_intrinsic[axis * dof + param];
        }
    }
    return result;
}

void apply_intrinsic_step(
    Problem& problem, const std::vector<double>& step, const OptimizerOptions& options) {
    if (step.empty() || problem.intrinsics.empty()) return;
    const std::size_t block_dof = count_intrinsic_dof(options);
    if (step.size() != problem.intrinsics.size() * block_dof)
        throw std::invalid_argument("Intrinsic step does not match group count");
    for (std::size_t group = 0; group < problem.intrinsics.size(); ++group) {
        if (problem.is_intrinsic_constant(group)) continue;
        PinholeIntrinsics& intrinsics = problem.intrinsics[group];
        std::size_t index = group * block_dof;
        if (options.optimize_focal) {
            intrinsics.fx += step[index++];
            if (options.optimize_aspect_ratio)
                intrinsics.fy += step[index++];
            else
                intrinsics.fy += step[index - 1];
            double minimum_fx = 1.0;
            double maximum_fx = std::numeric_limits<double>::infinity();
            double minimum_fy = 1.0;
            double maximum_fy = std::numeric_limits<double>::infinity();
            if (group < problem.initial_intrinsics.size()) {
                const double fx0 =
                    std::max(problem.initial_intrinsics[group].fx, 1.0);
                const double fy0 =
                    std::max(problem.initial_intrinsics[group].fy, 1.0);
                minimum_fx = std::max(1.0, fx0 * options.min_focal_ratio);
                maximum_fx = fx0 * options.max_focal_ratio;
                minimum_fy = std::max(1.0, fy0 * options.min_focal_ratio);
                maximum_fy = fy0 * options.max_focal_ratio;
            }
            intrinsics.fx = std::clamp(intrinsics.fx, minimum_fx, maximum_fx);
            intrinsics.fy = std::clamp(intrinsics.fy, minimum_fy, maximum_fy);
        }
        if (options.optimize_principal_point) {
            intrinsics.cx += step[index++];
            intrinsics.cy += step[index++];
        }
        if (options.optimize_distortion) {
            intrinsics.k1 += step[index++];
            intrinsics.k2 += step[index++];
            intrinsics.p1 += step[index++];
            intrinsics.p2 += step[index++];
        }
    }
}

void normalize_group_focals(Problem& problem) {
    for (PinholeIntrinsics& intrinsics : problem.intrinsics) {
        const double focal = 0.5 * (intrinsics.fx + intrinsics.fy);
        intrinsics.fx = focal;
        intrinsics.fy = focal;
    }
}

double focal_prior_cost(
    const Problem& problem, const OptimizerOptions& options) {
    if (!options.optimize_focal || options.focal_prior_weight <= 0.0 ||
        problem.initial_intrinsics.empty())
        return 0.0;
    const double observation_scale =
        static_cast<double>(std::max<std::size_t>(problem.observations.size(), 1));
    double cost = 0.0;
    for (std::size_t group = 0; group < problem.intrinsics.size(); ++group) {
        if (problem.is_intrinsic_constant(group)) continue;
        if (group >= problem.initial_intrinsics.size()) continue;
        if (options.optimize_aspect_ratio) {
            const double fx0 =
                std::max(problem.initial_intrinsics[group].fx, 1.0);
            const double fy0 =
                std::max(problem.initial_intrinsics[group].fy, 1.0);
            const double relative_fx =
                (problem.intrinsics[group].fx - fx0) / fx0;
            const double relative_fy =
                (problem.intrinsics[group].fy - fy0) / fy0;
            cost += 0.25 * options.focal_prior_weight * observation_scale *
                    (relative_fx * relative_fx + relative_fy * relative_fy);
        } else {
            const double f0 = std::max(
                0.5 * (problem.initial_intrinsics[group].fx +
                       problem.initial_intrinsics[group].fy),
                1.0);
            const double f = 0.5 *
                (problem.intrinsics[group].fx + problem.intrinsics[group].fy);
            const double relative = (f - f0) / f0;
            cost += 0.5 * options.focal_prior_weight * observation_scale *
                    relative * relative;
        }
    }
    return cost;
}

double linearized_total_cost(
    const Problem& problem,
    const LinearizationOutput& linearization,
    const OptimizerOptions& options) {
    if (linearization.observations.size() != problem.observations.size())
        throw std::invalid_argument(
            "Linearization size does not match BA observations");
    double cost = 0.0;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for reduction(+:cost) schedule(static)
#endif
    for (std::int64_t observation_signed = 0;
         observation_signed <
         static_cast<std::int64_t>(problem.observations.size());
         ++observation_signed) {
        const std::size_t observation =
            static_cast<std::size_t>(observation_signed);
        const double observation_weight =
            problem.observations.weight[observation];
        if (observation_weight == 0.0) continue;
        const LinearizedObservation& value =
            linearization.observations[observation];
        if (!value.valid || !(value.robust_weight > 0.0)) {
            cost += observation_weight * 1e12;
            continue;
        }
        const double weighted_squared_norm =
            value.residual[0] * value.residual[0] +
            value.residual[1] * value.residual[1];
        const double raw_norm =
            std::sqrt(weighted_squared_norm / value.robust_weight);
        if (!std::isfinite(raw_norm)) {
            cost += observation_weight * 1e12;
            continue;
        }
        const double robust =
            options.huber_delta > 0.0 && raw_norm > options.huber_delta
                ? options.huber_delta *
                      (raw_norm - 0.5 * options.huber_delta)
                : 0.5 * raw_norm * raw_norm;
        cost += observation_weight * robust;
    }
    return cost + focal_prior_cost(problem, options);
}

void apply_focal_prior(
    System& system, const Problem& problem, const OptimizerOptions& options) {
    if (!options.optimize_focal || options.focal_prior_weight <= 0.0 ||
        system.intrinsic_dof == 0 || problem.initial_intrinsics.empty())
        return;
    const double observation_scale =
        static_cast<double>(std::max<std::size_t>(problem.observations.size(), 1));
    for (std::size_t group = 0; group < problem.intrinsics.size(); ++group) {
        if (problem.is_intrinsic_constant(group)) continue;
        if (group >= problem.initial_intrinsics.size()) continue;
        double* hessian = system.intrinsic_hessian.data() +
                          group * system.intrinsic_dof * system.intrinsic_dof;
        double* rhs =
            system.intrinsic_rhs.data() + group * system.intrinsic_dof;
        if (options.optimize_aspect_ratio) {
            const double fx0 =
                std::max(problem.initial_intrinsics[group].fx, 1.0);
            const double fy0 =
                std::max(problem.initial_intrinsics[group].fy, 1.0);
            const double weight_fx = 0.5 * options.focal_prior_weight *
                                     observation_scale / (fx0 * fx0);
            const double weight_fy = 0.5 * options.focal_prior_weight *
                                     observation_scale / (fy0 * fy0);
            hessian[0] += weight_fx;
            hessian[system.intrinsic_dof + 1] += weight_fy;
            rhs[0] -= weight_fx * (problem.intrinsics[group].fx - fx0);
            rhs[1] -= weight_fy * (problem.intrinsics[group].fy - fy0);
        } else {
            const double f0 = std::max(
                0.5 * (problem.initial_intrinsics[group].fx +
                       problem.initial_intrinsics[group].fy),
                1.0);
            const double f = 0.5 *
                (problem.intrinsics[group].fx + problem.intrinsics[group].fy);
            const double weight =
                options.focal_prior_weight * observation_scale / (f0 * f0);
            hessian[0] += weight;
            // System stores rhs = -J^T r and solves H dx = rhs.
            rhs[0] -= weight * (f - f0);
        }
    }
}

void apply_step(Problem& problem, const std::vector<double>& camera_step,
                const std::vector<double>& point_step, const bool fix_first,
                const bool optimize_rotations,
                const bool optimize_translations) {
    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera) {
        if (problem.is_pose_constant(camera, fix_first)) continue;
        Pose& pose = problem.poses[camera];
        const double* step = camera_step.data() + camera * pose_size;
        if (optimize_rotations) {
            const double angle =
                std::sqrt(step[0] * step[0] + step[1] * step[1] +
                          step[2] * step[2]);
            double dw = 1.0;
            double dx = 0.5 * step[0];
            double dy = 0.5 * step[1];
            double dz = 0.5 * step[2];
            if (angle > 1e-12) {
                dw = std::cos(0.5 * angle);
                const double scale = std::sin(0.5 * angle) / angle;
                dx = scale * step[0];
                dy = scale * step[1];
                dz = scale * step[2];
            }
            const double qw =
                dw * pose.qw - dx * pose.qx - dy * pose.qy - dz * pose.qz;
            const double qx =
                dw * pose.qx + dx * pose.qw + dy * pose.qz - dz * pose.qy;
            const double qy =
                dw * pose.qy - dx * pose.qz + dy * pose.qw + dz * pose.qx;
            const double qz =
                dw * pose.qz + dx * pose.qy - dy * pose.qx + dz * pose.qw;
            const double inverse_norm =
                1.0 / std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            pose.qw = qw * inverse_norm;
            pose.qx = qx * inverse_norm;
            pose.qy = qy * inverse_norm;
            pose.qz = qz * inverse_norm;
        }
        if (optimize_translations) {
            pose.cx += step[3]; pose.cy += step[4]; pose.cz += step[5];
        }
    }
    for (std::size_t point = 0; point < problem.points.size(); ++point) {
        problem.points[point].x += point_step[point * point_size];
        problem.points[point].y += point_step[point * point_size + 1];
        problem.points[point].z += point_step[point * point_size + 2];
    }
}

double step_norm(
    const std::vector<double>& camera, const std::vector<double>& point,
    const std::vector<double>& intrinsics = {}) {
    return std::sqrt(
        dot(camera, camera) + dot(point, point) +
        (intrinsics.empty() ? 0.0 : dot(intrinsics, intrinsics)));
}

}  // namespace

double evaluate_cost(const Problem& problem, const double huber_delta, const double minimum_depth) {
    problem.validate();
    double cost = 0.0;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for reduction(+:cost) schedule(static)
#endif
    for (std::int64_t observation_signed = 0;
         observation_signed < static_cast<std::int64_t>(problem.observations.size());
         ++observation_signed) {
        const auto observation = static_cast<std::size_t>(observation_signed);
        const Pose& pose = problem.poses[problem.observations.camera[observation]];
        const std::size_t camera =
            problem.observations.camera[observation];
        const PinholeIntrinsics& intrinsics =
            problem.intrinsics[problem.intrinsic_index(camera)];
        const Point3& point = problem.points[problem.observations.point[observation]];
        const double dx = point.x - pose.cx, dy = point.y - pose.cy, dz = point.z - pose.cz;
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
        double rx = 0.0, ry = 0.0;
        if (uses_bearing_projection(intrinsics.model)) {
            // No cheirality for a full-sphere camera; the residual is taken in
            // the tangent plane so the seam and the poles stay well behaved.
            const double length2 = px * px + py * py + pz * pz;
            if (!(length2 > minimum_depth * minimum_depth) ||
                !std::isfinite(length2)) {
                cost += problem.observations.weight[observation] * 1e12;
                continue;
            }
            const EquirectTangentBasis basis = equirect_tangent_basis(
                problem.observations.x[observation],
                problem.observations.y[observation], intrinsics.fx,
                intrinsics.fy, intrinsics.cx, intrinsics.cy);
            const EquirectLocalReprojection local =
                equirect_local_reprojection(px, py, pz, basis);
            if (!local.valid) {
                cost += problem.observations.weight[observation] * 1e12;
                continue;
            }
            rx = local.residual_x;
            ry = local.residual_y;
        } else {
            if (pz <= minimum_depth || !std::isfinite(pz)) {
                // Invalidating observations must never look like an improvement.
                cost += problem.observations.weight[observation] * 1e12;
                continue;
            }
            const auto projection = project_camera_plane(intrinsics.model, px/pz, py/pz,
                intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2);
            const double distorted_x = projection.x, distorted_y = projection.y;
            rx = intrinsics.fx * distorted_x + intrinsics.cx - problem.observations.x[observation];
            ry = intrinsics.fy * distorted_y + intrinsics.cy - problem.observations.y[observation];
        }
        const double norm = std::sqrt(rx * rx + ry * ry);
        if (!std::isfinite(norm)) {
            cost += problem.observations.weight[observation] * 1e12;
            continue;
        }
        const double robust = huber_delta > 0.0 && norm > huber_delta
            ? huber_delta * (norm - 0.5 * huber_delta) : 0.5 * norm * norm;
        cost += problem.observations.weight[observation] * robust;
    }
    return cost;
}

bool OptimizerSummary::usable() const noexcept {
    return termination != TerminationReason::numerical_failure && std::isfinite(final_cost);
}

std::string OptimizerSummary::brief_report() const {
    std::ostringstream report;
    report << "BA: " << initial_cost << " -> " << final_cost
           << ", accepted=" << successful_steps
           << ", rejected=" << unsuccessful_steps
           << ", time=" << total_time_ms << " ms"
           << " [linearize=" << linearization_time_ms
           << ", assemble=" << assembly_time_ms
           << ", solve=" << solve_time_ms
           << ", update/cost=" << update_and_cost_time_ms << ']';
    return report.str();
}

OptimizerSummary optimize_cpu(Problem& problem, const OptimizerOptions& options) {
    core::StageScope stage("ba.cpu");
    problem.validate();
    if (options.maximum_iterations == 0 || options.maximum_pcg_iterations == 0 ||
        options.initial_damping <= 0.0 || options.pcg_tolerance <= 0.0) {
        throw std::invalid_argument("Invalid BA optimizer options");
    }
    const std::size_t block_dof = count_intrinsic_dof(options);
    if (options.optimize_focal && !options.optimize_aspect_ratio)
        normalize_group_focals(problem);
    if (problem.initial_intrinsics.empty())
        problem.initial_intrinsics = problem.intrinsics;

    const auto started = std::chrono::steady_clock::now();
    const Adjacency adjacency = build_adjacency(problem);
    const SchurPattern schur_pattern = build_schur_pattern(problem, adjacency);
    const LinearizerOptions linearizer_options = make_linearizer_options(options);
    OptimizerSummary summary;
    LinearizationOutput linearization;
    LinearizationOutput candidate_linearization;
    const EvaluationStats initial_evaluation =
        linearize_cpu(problem, linearization, linearizer_options);
    summary.linearization_time_ms += initial_evaluation.elapsed_ms;
    summary.initial_cost =
        linearized_total_cost(problem, linearization, options);
    summary.final_cost = summary.initial_cost;
    double damping = options.initial_damping;
    // Keep the large observation-sized buffers alive across LM iterations.
    // assign() below clears their contents while retaining capacity.
    System system;
    AssemblyWorkspace assembly_workspace;
    ExplicitSchur explicit_schur;

    for (std::size_t iteration = 0; iteration < options.maximum_iterations; ++iteration) {
        auto stage_started = std::chrono::steady_clock::now();
        assemble_system(
            problem, linearization, adjacency, damping,
            options.fix_first_point, options.optimize_points,
            options.optimize_rotations, options.optimize_translations,
            block_dof, assembly_workspace,
            system);
        apply_focal_prior(system, problem, options);
        const bool use_dense_intrinsic_schur =
            block_dof > 0 &&
            problem.intrinsics.size() <= dense_intrinsic_group_limit;
        if (block_dof == 0 || use_dense_intrinsic_schur) {
            assemble_explicit_schur(
                problem, adjacency, schur_pattern, system,
                assembly_workspace, explicit_schur);
        } else {
            explicit_schur.blocks.clear();
        }
        IntrinsicSchur intrinsic_schur =
            use_dense_intrinsic_schur
                ? build_intrinsic_schur(problem, adjacency, system)
                : IntrinsicSchur{};
        auto stage_stopped = std::chrono::steady_clock::now();
        summary.assembly_time_ms +=
            std::chrono::duration<double, std::milli>(stage_stopped - stage_started).count();
        stage_started = stage_stopped;

        std::vector<double> rhs, camera_step, point_step, intrinsic_step;

        if (block_dof == 0) {
            schur_rhs(
                problem, adjacency, system, options.fix_first_pose, rhs);
            const std::size_t pcg_iterations = solve_pcg(
                problem, adjacency, system, schur_pattern, explicit_schur,
                options, rhs, camera_step);
            recover_point_step(
                problem, adjacency, system, camera_step, intrinsic_step, point_step);
            summary.iterations.push_back(IterationSummary{
                iteration, 0.0, damping, 0.0, pcg_iterations, false});
        } else if (use_dense_intrinsic_schur) {
            const std::size_t dof = intrinsic_schur.dof;
            schur_rhs(
                problem, adjacency, system, options.fix_first_pose, rhs);
            std::vector<double> camera_rhs_solution;
            std::size_t pcg_total = solve_pcg(
                problem, adjacency, system, schur_pattern, explicit_schur,
                options, rhs, camera_rhs_solution);
            std::vector<double> camera_intrinsic_inverse(
                rhs.size() * dof, 0.0);
            for (std::size_t param = 0; param < dof; ++param) {
                std::vector<double> column(rhs.size(), 0.0);
                for (std::size_t camera = 0;
                     camera < problem.poses.size(); ++camera) {
                    if (problem.is_pose_constant(
                            camera, options.fix_first_pose))
                        continue;
                    for (std::size_t row = 0; row < pose_size; ++row)
                        column[camera * pose_size + row] =
                            intrinsic_schur.S_ci
                                [(camera * pose_size + row) * dof + param];
                }
                std::vector<double> solved;
                pcg_total += solve_pcg(
                    problem, adjacency, system, schur_pattern,
                    explicit_schur, options, column, solved);
                for (std::size_t value = 0; value < solved.size(); ++value)
                    camera_intrinsic_inverse[value * dof + param] =
                        solved[value];
            }
            std::vector<double> reduced_hessian = intrinsic_schur.S_ii;
            std::vector<double> reduced_rhs = intrinsic_schur.b_i;
            for (std::size_t camera = 0;
                 camera < problem.poses.size(); ++camera) {
                if (problem.is_pose_constant(
                        camera, options.fix_first_pose))
                    continue;
                for (std::size_t row = 0; row < pose_size; ++row) {
                    const std::size_t camera_row = camera * pose_size + row;
                    for (std::size_t i = 0; i < dof; ++i) {
                        const double cross =
                            intrinsic_schur.S_ci[camera_row * dof + i];
                        reduced_rhs[i] -=
                            cross * camera_rhs_solution[camera_row];
                        for (std::size_t j = 0; j < dof; ++j)
                            reduced_hessian[i * dof + j] -=
                                cross *
                                camera_intrinsic_inverse
                                    [camera_row * dof + j];
                    }
                }
            }
            intrinsic_step.assign(dof, 0.0);
            if (!solve_spd(
                    reduced_hessian.data(), dof, reduced_rhs.data(),
                    intrinsic_step.data()))
                std::fill(
                    intrinsic_step.begin(), intrinsic_step.end(), 0.0);
            std::vector<double> corrected_rhs = rhs;
            for (std::size_t camera = 0;
                 camera < problem.poses.size(); ++camera) {
                if (problem.is_pose_constant(
                        camera, options.fix_first_pose))
                    continue;
                for (std::size_t row = 0; row < pose_size; ++row) {
                    const std::size_t camera_row = camera * pose_size + row;
                    for (std::size_t param = 0; param < dof; ++param)
                        corrected_rhs[camera_row] -=
                            intrinsic_schur.S_ci[camera_row * dof + param] *
                            intrinsic_step[param];
                }
            }
            pcg_total += solve_pcg(
                problem, adjacency, system, schur_pattern, explicit_schur,
                options, corrected_rhs, camera_step);
            recover_point_step(
                problem, adjacency, system, camera_step, intrinsic_step,
                point_step);
            summary.iterations.push_back(IterationSummary{
                iteration, 0.0, damping, 0.0, pcg_total, false});
        } else {
            build_joint_rhs(
                problem, adjacency, system, options.fix_first_pose, rhs);
            std::vector<double> joint_step;
            const std::size_t pcg_iterations = solve_joint_pcg(
                problem, adjacency, system, options, rhs, joint_step);
            const std::size_t camera_values =
                problem.poses.size() * pose_size;
            camera_step.assign(
                joint_step.begin(),
                joint_step.begin() +
                    static_cast<std::ptrdiff_t>(camera_values));
            intrinsic_step.assign(
                joint_step.begin() +
                    static_cast<std::ptrdiff_t>(camera_values),
                joint_step.end());
            recover_point_step(
                problem, adjacency, system, camera_step, intrinsic_step, point_step);
            summary.iterations.push_back(IterationSummary{
                iteration, 0.0, damping, 0.0, pcg_iterations, false});
        }

        const double norm = step_norm(camera_step, point_step, intrinsic_step);
        stage_stopped = std::chrono::steady_clock::now();
        summary.solve_time_ms +=
            std::chrono::duration<double, std::milli>(stage_stopped - stage_started).count();
        stage_started = stage_stopped;
        if (!std::isfinite(norm)) {
            summary.termination = TerminationReason::numerical_failure;
            break;
        }
        const auto old_poses = problem.poses;
        const auto old_points = problem.points;
        const auto old_intrinsics = problem.intrinsics;
        apply_step(
            problem, camera_step, point_step, options.fix_first_pose,
            options.optimize_rotations, options.optimize_translations);
        if (block_dof > 0)
            apply_intrinsic_step(problem, intrinsic_step, options);
        const EvaluationStats candidate_evaluation =
            linearize_cpu(
                problem, candidate_linearization, linearizer_options);
        const double candidate_cost =
            linearized_total_cost(problem, candidate_linearization, options);
        const bool accepted = std::isfinite(candidate_cost) && candidate_cost < summary.final_cost;
        stage_stopped = std::chrono::steady_clock::now();
        summary.linearization_time_ms += candidate_evaluation.elapsed_ms;
        summary.update_and_cost_time_ms += std::max(
            0.0,
            std::chrono::duration<double, std::milli>(
                stage_stopped - stage_started)
                    .count() -
                candidate_evaluation.elapsed_ms);
        summary.iterations.back().cost = accepted ? candidate_cost : summary.final_cost;
        summary.iterations.back().step_norm = norm;
        summary.iterations.back().accepted = accepted;
        core::Logger::instance().debug(
            "BA iteration=", iteration, " cost=", summary.iterations.back().cost,
            " damping=", damping, " step_norm=", norm,
            " pcg_iterations=", summary.iterations.back().pcg_iterations,
            " accepted=", accepted);
        if (accepted) {
            const double previous_cost = summary.final_cost;
            summary.final_cost = candidate_cost;
            std::swap(linearization, candidate_linearization);
            ++summary.successful_steps;
            damping = std::max(options.minimum_damping, damping * 0.333333333333);
            if (norm <= options.step_tolerance ||
                previous_cost - candidate_cost <=
                    options.function_tolerance * std::max(1.0, previous_cost)) {
                summary.termination = TerminationReason::converged;
                break;
            }
        } else {
            problem.poses = old_poses;
            problem.points = old_points;
            problem.intrinsics = old_intrinsics;
            ++summary.unsuccessful_steps;
            damping = std::min(options.maximum_damping, damping * 10.0);
            if (damping >= options.maximum_damping) {
                // Rejected candidates are rolled back, so an earlier accepted
                // state remains usable even if LM later stalls at max damping.
                summary.termination = summary.successful_steps > 0
                    ? TerminationReason::maximum_iterations
                    : TerminationReason::numerical_failure;
                break;
            }
        }
    }
    const auto stopped = std::chrono::steady_clock::now();
    summary.total_time_ms =
        std::chrono::duration<double, std::milli>(stopped - started).count();
    stage.finish(summary.brief_report());
    return summary;
}

}  // namespace aetherscan::ba
