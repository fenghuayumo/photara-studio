#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

namespace aetherscan::mvs::maxflow {

// FIFO push-relabel max-flow with unary source/sink capacities. Nodes are
// [0, n). It is iterative (no graph-depth recursion) and therefore remains
// safe for the million-cell visibility graphs produced by global meshing.
class Graph {
public:
    explicit Graph(const std::size_t nodes = 0) { reset(nodes); }

    void reset(const std::size_t nodes) {
        n_ = nodes;
        head_.assign(n_, -1);
        edges_.clear();
        src_.assign(n_, 0.F);
        snk_.assign(n_, 0.F);
        source_side_.assign(n_, 0);
    }

    void add_tweights(const std::size_t node, float source, float sink) {
        const float m = std::min(source, sink);
        source -= m;
        sink -= m;
        src_[node] += source;
        snk_[node] += sink;
    }

    void add_edge(
        const std::size_t i, const std::size_t j, const float cap_ij,
        const float cap_ji) {
        const int a = static_cast<int>(edges_.size());
        edges_.push_back({j, head_[i], a + 1, cap_ij});
        head_[i] = a;
        const int b = static_cast<int>(edges_.size());
        edges_.push_back({i, head_[j], a, cap_ji});
        head_[j] = b;
    }

    float maxflow() {
        constexpr float epsilon = 1e-12F;
        if (n_ == 0) return 0.F;

        const std::size_t original_nodes = n_;
        const std::size_t source = original_nodes;
        const std::size_t sink = original_nodes + 1;
        const std::size_t total_nodes = original_nodes + 2;
        head_.resize(total_nodes, -1);
        for (std::size_t i = 0; i < original_nodes; ++i) {
            if (src_[i] > epsilon) add_edge(source, i, src_[i], 0.F);
            if (snk_[i] > epsilon) add_edge(i, sink, snk_[i], 0.F);
        }

        std::vector<std::size_t> height(total_nodes, 0);
        std::vector<std::size_t> height_count(total_nodes * 2 + 1, 0);
        std::vector<float> excess(total_nodes, 0.F);
        std::vector<int> current = head_;
        std::vector<std::uint8_t> active(total_nodes, 0);
        std::queue<std::size_t> queue;
        height[source] = total_nodes;
        height_count[0] = total_nodes - 1;
        height_count[total_nodes] = 1;

        const auto enqueue = [&](const std::size_t node) {
            if (node == source || node == sink || active[node] ||
                excess[node] <= epsilon || height[node] >= total_nodes * 2)
                return;
            active[node] = 1;
            queue.push(node);
        };

        for (int index = head_[source]; index >= 0;
             index = edges_[static_cast<std::size_t>(index)].next) {
            Edge& edge = edges_[static_cast<std::size_t>(index)];
            const float pushed = edge.cap;
            if (pushed <= epsilon) continue;
            edge.cap = 0.F;
            edges_[static_cast<std::size_t>(edge.rev)].cap += pushed;
            excess[edge.to] += pushed;
            excess[source] -= pushed;
            enqueue(edge.to);
        }

        while (!queue.empty()) {
            const std::size_t node = queue.front();
            queue.pop();
            active[node] = 0;
            while (excess[node] > epsilon) {
                int& edge_index = current[node];
                if (edge_index < 0) {
                    const std::size_t old_height = height[node];
                    std::size_t next_height = total_nodes * 2;
                    for (int index = head_[node]; index >= 0;
                         index = edges_[static_cast<std::size_t>(index)].next) {
                        const Edge& edge = edges_[static_cast<std::size_t>(index)];
                        if (edge.cap > epsilon)
                            next_height = std::min(next_height, height[edge.to] + 1);
                    }
                    --height_count[old_height];
                    height[node] = next_height;
                    ++height_count[next_height];
                    current[node] = head_[node];

                    // Gap relabel: no active path can cross an empty level.
                    if (old_height < total_nodes &&
                        height_count[old_height] == 0) {
                        for (std::size_t i = 0; i < original_nodes; ++i) {
                            if (height[i] <= old_height ||
                                height[i] >= total_nodes)
                                continue;
                            --height_count[height[i]];
                            height[i] = total_nodes + 1;
                            ++height_count[height[i]];
                            current[i] = head_[i];
                        }
                    }
                    if (next_height >= total_nodes * 2) break;
                    continue;
                }

                Edge& edge = edges_[static_cast<std::size_t>(edge_index)];
                if (edge.cap > epsilon && height[node] == height[edge.to] + 1) {
                    const float pushed = std::min(excess[node], edge.cap);
                    edge.cap -= pushed;
                    edges_[static_cast<std::size_t>(edge.rev)].cap += pushed;
                    excess[node] -= pushed;
                    excess[edge.to] += pushed;
                    enqueue(edge.to);
                } else {
                    edge_index = edge.next;
                }
            }
            enqueue(node);
        }

        std::fill(source_side_.begin(), source_side_.end(), 0);
        std::vector<std::uint8_t> reachable(total_nodes, 0);
        reachable[source] = 1;
        queue.push(source);
        while (!queue.empty()) {
            const std::size_t node = queue.front();
            queue.pop();
            for (int index = head_[node]; index >= 0;
                 index = edges_[static_cast<std::size_t>(index)].next) {
                const Edge& edge = edges_[static_cast<std::size_t>(index)];
                if (edge.cap <= epsilon || reachable[edge.to]) continue;
                reachable[edge.to] = 1;
                queue.push(edge.to);
            }
        }
        for (std::size_t i = 0; i < original_nodes; ++i)
            source_side_[i] = reachable[i];
        return excess[sink];
    }

    [[nodiscard]] bool is_source_side(const std::size_t node) const {
        return source_side_[node] != 0;
    }

private:
    struct Edge {
        std::size_t to{};
        int next{-1};
        int rev{-1};
        float cap{0.F};
    };

    std::size_t n_{0};
    std::vector<int> head_;
    std::vector<Edge> edges_;
    std::vector<float> src_;
    std::vector<float> snk_;
    std::vector<std::uint8_t> source_side_;
};

}  // namespace aetherscan::mvs::maxflow
