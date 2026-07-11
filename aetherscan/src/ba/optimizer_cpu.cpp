#include "ba/optimizer.hpp"

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
};

struct SchurPattern {
    std::vector<std::size_t> row_offsets;
    std::vector<Index> columns;
    std::vector<std::size_t> diagonal_blocks;
};

struct ExplicitSchur {
    std::vector<double> blocks;
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
    return pattern;
}

std::size_t find_schur_block(
    const SchurPattern& pattern, const std::size_t row, const Index column) {
    const auto begin = pattern.columns.begin() + static_cast<std::ptrdiff_t>(pattern.row_offsets[row]);
    const auto end = pattern.columns.begin() + static_cast<std::ptrdiff_t>(pattern.row_offsets[row + 1]);
    const auto found = std::lower_bound(begin, end, column);
    return pattern.row_offsets[row] + static_cast<std::size_t>(found - begin);
}

ExplicitSchur assemble_explicit_schur(
    const Problem& problem, const Adjacency& adjacency, const SchurPattern& pattern,
    const System& system) {
    ExplicitSchur reduced;
    reduced.blocks.assign(pattern.columns.size() * pose_block_size, 0.0);
    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera)
        std::copy_n(system.camera_hessian.data() + camera * pose_block_size, pose_block_size,
                    reduced.blocks.data() + pattern.diagonal_blocks[camera] * pose_block_size);

    int thread_count = 1;
#if defined(AETHERSCAN_HAS_OPENMP)
    const std::size_t values_per_thread = std::max<std::size_t>(1, reduced.blocks.size());
    const std::size_t memory_limited_threads = std::max<std::size_t>(1, 32'000'000 / values_per_thread);
    thread_count = std::min<int>(omp_get_max_threads(), static_cast<int>(memory_limited_threads));
#endif
    std::vector<double> local(static_cast<std::size_t>(thread_count) * reduced.blocks.size(), 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 64) num_threads(thread_count)
#endif
    for (std::int64_t point_signed = 0;
         point_signed < static_cast<std::int64_t>(problem.points.size()); ++point_signed) {
        const auto point = static_cast<std::size_t>(point_signed);
        int thread = 0;
#if defined(AETHERSCAN_HAS_OPENMP)
        thread = omp_get_thread_num();
#endif
        double* destination = local.data() + static_cast<std::size_t>(thread) * reduced.blocks.size();
        const double* inverse = system.point_inverse.data() + point * point_block_size;
        for (std::size_t first = adjacency.point_offsets[point];
             first < adjacency.point_offsets[point + 1]; ++first) {
            const std::size_t observation_first = adjacency.point_observations[first];
            const Index camera_first = problem.observations.camera[observation_first];
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
                const double* cross_second = system.cross.data() + observation_second * cross_block_size;
                double* block = destination +
                    find_schur_block(pattern, camera_first, camera_second) * pose_block_size;
                for (std::size_t row = 0; row < pose_size; ++row)
                    for (std::size_t column = 0; column < pose_size; ++column)
                        for (std::size_t k = 0; k < point_size; ++k)
                            block[row * pose_size + column] -=
                                transformed[row * point_size + k] *
                                cross_second[column * point_size + k];
            }
        }
    }
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t value_signed = 0;
         value_signed < static_cast<std::int64_t>(reduced.blocks.size()); ++value_signed) {
        const auto value = static_cast<std::size_t>(value_signed);
        double sum = reduced.blocks[value];
        for (int thread = 0; thread < thread_count; ++thread)
            sum += local[static_cast<std::size_t>(thread) * reduced.blocks.size() + value];
        reduced.blocks[value] = sum;
    }
    return reduced;
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

System assemble_system(
    const Problem& problem,
    const LinearizationOutput& linearization,
    const Adjacency& adjacency,
    const double damping,
    const bool fix_first_point) {
    const std::size_t camera_count = problem.poses.size();
    const std::size_t point_count = problem.points.size();
    System system;
    system.camera_hessian.assign(camera_count * pose_block_size, 0.0);
    system.camera_rhs.assign(camera_count * pose_size, 0.0);
    system.point_inverse.assign(point_count * point_block_size, 0.0);
    system.point_rhs.assign(point_count * point_size, 0.0);
    system.cross.assign(problem.observations.size() * cross_block_size, 0.0);

    int thread_count = 1;
#if defined(AETHERSCAN_HAS_OPENMP)
    const std::size_t values_per_thread = std::max<std::size_t>(1, camera_count * 42);
    const std::size_t memory_limited_threads = std::max<std::size_t>(1, 32'000'000 / values_per_thread);
    thread_count = std::min<int>(omp_get_max_threads(), static_cast<int>(memory_limited_threads));
#endif
    std::vector<double> local_cameras(
        static_cast<std::size_t>(thread_count) * camera_count * 42, 0.0);

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
        double* camera_accumulator = local_cameras.data() +
            static_cast<std::size_t>(thread) * camera_count * 42;
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
        }
        for (std::size_t diagonal = 0; diagonal < point_size; ++diagonal) {
            point_hessian[diagonal * point_size + diagonal] +=
                damping * (point_hessian[diagonal * point_size + diagonal] + 1.0);
        }
        double* inverse = system.point_inverse.data() + point * point_block_size;
        if ((fix_first_point && point == 0) ||
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
            const double* source = local_cameras.data() +
                static_cast<std::size_t>(thread) * camera_count * 42 + camera * 42;
            for (std::size_t i = 0; i < pose_block_size; ++i) destination_hessian[i] += source[i];
            for (std::size_t i = 0; i < pose_size; ++i) destination_rhs[i] += source[pose_block_size + i];
        }
        for (std::size_t diagonal = 0; diagonal < pose_size; ++diagonal) {
            destination_hessian[diagonal * pose_size + diagonal] +=
                damping * (destination_hessian[diagonal * pose_size + diagonal] + 1.0);
        }
    }
    return system;
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
        if (fix_first && camera == 0) {
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
            if (fix_first && camera == 0) continue;
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
        if (fix_first && camera == 0) {
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
    const SchurPattern& pattern, const ExplicitSchur& system, const bool fix_first,
    const std::vector<double>& input, std::vector<double>& output) {
    const std::size_t camera_count = pattern.row_offsets.size() - 1;
    output.assign(input.size(), 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(camera_count); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        double* destination = output.data() + camera * pose_size;
        if (fix_first && camera == 0) {
            std::copy_n(input.data(), pose_size, destination);
            continue;
        }
        for (std::size_t block = pattern.row_offsets[camera];
             block < pattern.row_offsets[camera + 1]; ++block) {
            const std::size_t column_camera = pattern.columns[block];
            if (fix_first && column_camera == 0) continue;
            const double* matrix = system.blocks.data() + block * pose_block_size;
            const double* vector = input.data() + column_camera * pose_size;
            for (std::size_t row = 0; row < pose_size; ++row)
                for (std::size_t column = 0; column < pose_size; ++column)
                    destination[row] += matrix[row * pose_size + column] * vector[column];
        }
    }
}

void precondition_explicit(
    const SchurPattern& pattern, const ExplicitSchur& system, const bool fix_first,
    const std::vector<double>& residual, std::vector<double>& output) {
    const std::size_t camera_count = pattern.row_offsets.size() - 1;
    output.assign(residual.size(), 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(camera_count); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        if (fix_first && camera == 0) continue;
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
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for reduction(+:value) schedule(static)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(first.size()); ++i) {
        value += first[static_cast<std::size_t>(i)] * second[static_cast<std::size_t>(i)];
    }
    return value;
}

void precondition(
    const System& system, const bool fix_first,
    const std::vector<double>& residual, std::vector<double>& output) {
    const std::size_t camera_count = system.camera_rhs.size() / pose_size;
    output.assign(residual.size(), 0.0);
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t camera_signed = 0;
         camera_signed < static_cast<std::int64_t>(camera_count); ++camera_signed) {
        const auto camera = static_cast<std::size_t>(camera_signed);
        if (fix_first && camera == 0) continue;
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
        precondition(system, options.fix_first_pose, residual, z);
    else
        precondition_explicit(pattern, explicit_system, options.fix_first_pose, residual, z);
    direction = z;
    double rz = dot(residual, z);
    const double target = options.pcg_tolerance * options.pcg_tolerance *
                          std::max(dot(rhs, rhs), 1e-30);
    if (!std::isfinite(rz)) return 0;
    std::size_t iteration = 0;
    for (; iteration < options.maximum_pcg_iterations; ++iteration) {
        if (explicit_system.blocks.empty())
            schur_multiply(problem, adjacency, system, options.fix_first_pose,
                           direction, product, point_temporary);
        else
            explicit_schur_multiply(
                pattern, explicit_system, options.fix_first_pose, direction, product);
        const double denominator = dot(direction, product);
        if (!(denominator > 1e-30) || !std::isfinite(denominator)) break;
        const double alpha = rz / denominator;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
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
            precondition(system, options.fix_first_pose, residual, z);
        else
            precondition_explicit(pattern, explicit_system, options.fix_first_pose, residual, z);
        const double next_rz = dot(residual, z);
        if (!std::isfinite(next_rz)) break;
        const double beta = next_rz / rz;
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
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
    const std::vector<double>& camera_step, std::vector<double>& point_step) {
    point_step.assign(problem.points.size() * point_size, 0.0);
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
        }
        multiply3(system.point_inverse.data() + point * point_block_size,
                  rhs, point_step.data() + point * point_size);
    }
}

void apply_step(Problem& problem, const std::vector<double>& camera_step,
                const std::vector<double>& point_step, const bool fix_first) {
    for (std::size_t camera = 0; camera < problem.poses.size(); ++camera) {
        if (fix_first && camera == 0) continue;
        Pose& pose = problem.poses[camera];
        const double* step = camera_step.data() + camera * pose_size;
        const double angle = std::sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2]);
        double dw = 1.0, dx = 0.5 * step[0], dy = 0.5 * step[1], dz = 0.5 * step[2];
        if (angle > 1e-12) {
            dw = std::cos(0.5 * angle);
            const double scale = std::sin(0.5 * angle) / angle;
            dx = scale * step[0]; dy = scale * step[1]; dz = scale * step[2];
        }
        const double qw = dw * pose.qw - dx * pose.qx - dy * pose.qy - dz * pose.qz;
        const double qx = dw * pose.qx + dx * pose.qw + dy * pose.qz - dz * pose.qy;
        const double qy = dw * pose.qy - dx * pose.qz + dy * pose.qw + dz * pose.qx;
        const double qz = dw * pose.qz + dx * pose.qy - dy * pose.qx + dz * pose.qw;
        const double inverse_norm = 1.0 / std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        pose.qw = qw * inverse_norm; pose.qx = qx * inverse_norm;
        pose.qy = qy * inverse_norm; pose.qz = qz * inverse_norm;
        pose.cx += step[3]; pose.cy += step[4]; pose.cz += step[5];
    }
    for (std::size_t point = 0; point < problem.points.size(); ++point) {
        problem.points[point].x += point_step[point * point_size];
        problem.points[point].y += point_step[point * point_size + 1];
        problem.points[point].z += point_step[point * point_size + 2];
    }
}

double step_norm(const std::vector<double>& camera, const std::vector<double>& point) {
    return std::sqrt(dot(camera, camera) + dot(point, point));
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
        const PinholeIntrinsics& intrinsics =
            problem.intrinsics[problem.observations.camera[observation]];
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
        if (pz <= minimum_depth || !std::isfinite(pz)) {
            // Invalidating observations must never look like an improvement.
            cost += problem.observations.weight[observation] * 1e12;
            continue;
        }
        const double x = px / pz, y = py / pz, radius2 = x * x + y * y;
        const double radial = 1.0 + intrinsics.k1 * radius2 + intrinsics.k2 * radius2 * radius2;
        const double distorted_x = x * radial + 2.0 * intrinsics.p1 * x * y +
                                   intrinsics.p2 * (radius2 + 2.0 * x * x);
        const double distorted_y = y * radial + intrinsics.p1 * (radius2 + 2.0 * y * y) +
                                   2.0 * intrinsics.p2 * x * y;
        const double rx = intrinsics.fx * distorted_x + intrinsics.cx - problem.observations.x[observation];
        const double ry = intrinsics.fy * distorted_y + intrinsics.cy - problem.observations.y[observation];
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
    problem.validate();
    if (options.maximum_iterations == 0 || options.maximum_pcg_iterations == 0 ||
        options.initial_damping <= 0.0 || options.pcg_tolerance <= 0.0) {
        throw std::invalid_argument("Invalid BA optimizer options");
    }
    const auto started = std::chrono::steady_clock::now();
    const Adjacency adjacency = build_adjacency(problem);
    const SchurPattern schur_pattern = build_schur_pattern(problem, adjacency);
    OptimizerSummary summary;
    summary.initial_cost = evaluate_cost(problem, options.huber_delta, options.minimum_depth);
    summary.final_cost = summary.initial_cost;
    double damping = options.initial_damping;
    LinearizationOutput linearization;

    for (std::size_t iteration = 0; iteration < options.maximum_iterations; ++iteration) {
        auto stage_started = std::chrono::steady_clock::now();
        linearize_cpu(problem, linearization,
                      LinearizerOptions{options.huber_delta, options.minimum_depth});
        auto stage_stopped = std::chrono::steady_clock::now();
        summary.linearization_time_ms +=
            std::chrono::duration<double, std::milli>(stage_stopped - stage_started).count();
        stage_started = stage_stopped;
        System system = assemble_system(
            problem, linearization, adjacency, damping, options.fix_first_point);
        ExplicitSchur explicit_schur = assemble_explicit_schur(
            problem, adjacency, schur_pattern, system);
        stage_stopped = std::chrono::steady_clock::now();
        summary.assembly_time_ms +=
            std::chrono::duration<double, std::milli>(stage_stopped - stage_started).count();
        stage_started = stage_stopped;
        std::vector<double> rhs, camera_step, point_step;
        schur_rhs(problem, adjacency, system, options.fix_first_pose, rhs);
        const std::size_t pcg_iterations = solve_pcg(
            problem, adjacency, system, schur_pattern, explicit_schur,
            options, rhs, camera_step);
        recover_point_step(problem, adjacency, system, camera_step, point_step);
        const double norm = step_norm(camera_step, point_step);
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
        apply_step(problem, camera_step, point_step, options.fix_first_pose);
        const double candidate_cost = evaluate_cost(problem, options.huber_delta, options.minimum_depth);
        const bool accepted = std::isfinite(candidate_cost) && candidate_cost < summary.final_cost;
        stage_stopped = std::chrono::steady_clock::now();
        summary.update_and_cost_time_ms +=
            std::chrono::duration<double, std::milli>(stage_stopped - stage_started).count();
        summary.iterations.push_back(IterationSummary{
            iteration, accepted ? candidate_cost : summary.final_cost,
            damping, norm, pcg_iterations, accepted});
        if (accepted) {
            const double previous_cost = summary.final_cost;
            summary.final_cost = candidate_cost;
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
            ++summary.unsuccessful_steps;
            damping = std::min(options.maximum_damping, damping * 10.0);
            if (damping >= options.maximum_damping) {
                summary.termination = TerminationReason::numerical_failure;
                break;
            }
        }
    }
    const auto stopped = std::chrono::steady_clock::now();
    summary.total_time_ms =
        std::chrono::duration<double, std::milli>(stopped - started).count();
    return summary;
}

}  // namespace aetherscan::ba
