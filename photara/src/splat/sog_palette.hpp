#pragma once
#include "sog_vulkan.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace photara::splat::detail {

// Make every requested codeword useful, including non-power-of-two budgets.
// Splitting a non-singleton cluster at one of its actual members cannot
// increase its within-cluster SSE after recomputing the means.
inline void complete_sog_palette(
    const std::vector<float>& points,
    unsigned n,
    unsigned dims,
    unsigned budget,
    SogPalette& palette) {
    const unsigned k = std::min(n, budget);
    if (palette.labels.size() != n || palette.centroids.size() / dims > k)
        throw std::invalid_argument("Invalid SOG palette completion");
    palette.centroids.resize(std::size_t(k) * dims);
    std::vector<unsigned> counts(k);
    for (auto label : palette.labels) {
        if (label >= k)
            throw std::invalid_argument("Invalid SOG label");
        ++counts[label];
    }
    std::vector<double> sums(std::size_t(k) * dims);
    for (unsigned i = 0; i < n; ++i)
        for (unsigned d = 0; d < dims; ++d)
            sums[std::size_t(palette.labels[i]) * dims + d] += points[std::size_t(i) * dims + d];
    for (unsigned c = 0; c < k; ++c)
        if (counts[c])
            for (unsigned d = 0; d < dims; ++d)
                palette.centroids[std::size_t(c) * dims + d] =
                    float(sums[std::size_t(c) * dims + d] / counts[c]);
    std::vector<double> error(n);
    for (unsigned i = 0; i < n; ++i)
        for (unsigned d = 0; d < dims; ++d) {
            const double delta = double(points[std::size_t(i) * dims + d]) -
                                 palette.centroids[std::size_t(palette.labels[i]) * dims + d];
            error[i] += delta * delta;
        }
    std::vector<unsigned> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) {
        return error[a] != error[b] ? error[a] > error[b] : a < b;
    });
    std::size_t cursor = 0;
    for (unsigned c = 0; c < k; ++c)
        if (!counts[c]) {
            while (cursor < n && counts[palette.labels[order[cursor]]] <= 1)
                ++cursor;
            if (cursor == n)
                throw std::logic_error("Cannot fill SOG codebook budget");
            const unsigned i = order[cursor++], donor = palette.labels[i];
            --counts[donor];
            ++counts[c];
            palette.labels[i] = c;
            for (unsigned d = 0; d < dims; ++d) {
                sums[std::size_t(donor) * dims + d] -= points[std::size_t(i) * dims + d];
                sums[std::size_t(c) * dims + d] = points[std::size_t(i) * dims + d];
            }
        }
    for (unsigned c = 0; c < k; ++c)
        for (unsigned d = 0; d < dims; ++d)
            palette.centroids[std::size_t(c) * dims + d] =
                float(sums[std::size_t(c) * dims + d] / counts[c]);
}

inline void refine_sog_palette_cpu(
    const std::vector<float>& points,
    unsigned n,
    unsigned dims,
    unsigned budget,
    SogPalette& palette) {
    complete_sog_palette(points, n, dims, budget, palette);
    const unsigned k = unsigned(palette.centroids.size() / dims);
    for (unsigned iteration = 0; iteration < 3; ++iteration) {
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t row = 0; row < n; ++row) {
            const auto* p = points.data() + row * dims;
            unsigned selected = palette.labels[row];
            double best = 0;
            for (unsigned d = 0; d < dims; ++d)
                best +=
                    std::pow(double(p[d]) - palette.centroids[std::size_t(selected) * dims + d], 2);
            for (unsigned c = 0; c < k; ++c) {
                double distance = 0;
                for (unsigned d = 0; d < dims && distance < best; ++d)
                    distance +=
                        std::pow(double(p[d]) - palette.centroids[std::size_t(c) * dims + d], 2);
                if (distance < best) {
                    best = distance;
                    selected = c;
                }
            }
            palette.labels[row] = selected;
        }
        complete_sog_palette(points, n, dims, budget, palette);
    }
}
} // namespace photara::splat::detail
