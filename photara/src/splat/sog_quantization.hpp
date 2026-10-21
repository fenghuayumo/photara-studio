#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace photara::splat::detail {

inline std::uint8_t sog_nearest_codebook(const std::vector<float>& book, float value) {
    const auto hi = std::lower_bound(book.begin(), book.end(), value);
    if (hi == book.begin()) return 0;
    if (hi == book.end()) return static_cast<std::uint8_t>(book.size() - 1);
    const auto lo = hi - 1;
    return static_cast<std::uint8_t>((double(value) - *lo <= double(*hi) - value ? lo : hi) - book.begin());
}

// Distribution-aware scalar quantization: adaptive uniform/quantile histogram,
// sqrt(count) weights, then optimal 1D weighted SSE partitioning. Sorted bin
// centers make the DP Monge; divide-and-conquer reduces its quadratic search
// to O(256 * H * log H), with H <= 1024 independent of scene size.
inline std::vector<float> sog_codebook(const std::vector<float>& values, const char* label) {
    if (values.empty()) throw std::runtime_error(std::string("Empty SOG ") + label);
    std::vector<float> sorted(values);
    for (float v : sorted)
        if (!std::isfinite(v)) throw std::runtime_error(std::string("Non-finite SOG ") + label);
    std::sort(sorted.begin(), sorted.end());
    std::vector<float> distinct;
    for (float v : sorted) {
        if (distinct.empty() || v != distinct.back()) distinct.push_back(v);
        if (distinct.size() > 256) break;
    }
    if (distinct.size() <= 256) {
        distinct.resize(256, distinct.back());
        return distinct;
    }

    constexpr unsigned bins = 1024;
    const double low = sorted.front(), range = double(sorted.back()) - low;
    const double iqr = double(sorted[sorted.size() * 3 / 4]) - sorted[sorted.size() / 4];
    const double beta = std::clamp(1.0 - iqr / range, 0.5, 0.999);
    std::array<double, bins> counts{}, sums{};
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        const double value = (double(sorted[i]) - low) / range;
        const auto bin = std::min<unsigned>(bins - 1, static_cast<unsigned>(
            bins * (beta * double(i) / sorted.size() + (1 - beta) * value)));
        counts[bin] += 1;
        sums[bin] += value;
    }
    std::vector<double> centers, weights;
    for (unsigned i = 0; i < bins; ++i) {
        if (counts[i] == 0) continue;
        centers.push_back(sums[i] / counts[i]);
        weights.push_back(std::sqrt(counts[i]));
    }
    const unsigned h = static_cast<unsigned>(centers.size());
    const unsigned k = std::min(256U, h);
    std::vector<double> w(h + 1), wx(h + 1), wxx(h + 1);
    for (unsigned i = 0; i < h; ++i) {
        w[i + 1] = w[i] + weights[i];
        wx[i + 1] = wx[i] + weights[i] * centers[i];
        wxx[i + 1] = wxx[i] + weights[i] * centers[i] * centers[i];
    }
    const auto cost = [&](unsigned a, unsigned b) {
        const double weight = w[b] - w[a], sum = wx[b] - wx[a];
        return std::max(0.0, wxx[b] - wxx[a] - sum * sum / weight);
    };
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> previous(h + 1, inf), current(h + 1, inf);
    std::vector<unsigned> splits((k + 1) * (h + 1));
    previous[0] = 0;
    for (unsigned clusters = 1; clusters <= k; ++clusters) {
        std::fill(current.begin(), current.end(), inf);
        const auto solve = [&](auto&& self, unsigned left, unsigned right,
                               unsigned opt_left, unsigned opt_right) -> void {
            if (left > right) return;
            const unsigned mid = (left + right) / 2;
            unsigned best = opt_left;
            double best_cost = inf;
            for (unsigned split = opt_left; split <= std::min(opt_right, mid - 1); ++split) {
                const double candidate = previous[split] + cost(split, mid);
                if (candidate < best_cost) { best_cost = candidate; best = split; }
            }
            current[mid] = best_cost;
            splits[clusters * (h + 1) + mid] = best;
            if (mid > left) self(self, left, mid - 1, opt_left, best);
            if (mid < right) self(self, mid + 1, right, best, opt_right);
        };
        solve(solve, clusters, h, clusters - 1, h - 1);
        previous.swap(current);
    }
    std::vector<float> book(k);
    unsigned end = h;
    for (unsigned c = k; c > 0; --c) {
        const unsigned begin = splits[c * (h + 1) + end];
        book[c - 1] = static_cast<float>(low + range * (wx[end] - wx[begin]) / (w[end] - w[begin]));
        end = begin;
    }
    std::sort(book.begin(), book.end());
    book.resize(256, book.back());
    return book;
}

} // namespace photara::splat::detail
