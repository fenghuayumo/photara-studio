#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

namespace aetherscan::mvs::maxflow {

// Dinic max-flow with unary source/sink capacities. Nodes are [0, n).
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
        float flow = 0.F;
        std::vector<int> level(n_);
        std::vector<int> iter(n_);

        auto bfs = [&]() -> bool {
            std::fill(level.begin(), level.end(), -1);
            std::queue<std::size_t> q;
            for (std::size_t i = 0; i < n_; ++i) {
                if (src_[i] > 1e-12F) {
                    level[i] = 0;
                    q.push(i);
                }
            }
            bool hit = false;
            while (!q.empty()) {
                const std::size_t u = q.front();
                q.pop();
                if (snk_[u] > 1e-12F) hit = true;
                for (int e = head_[u]; e >= 0; e = edges_[static_cast<std::size_t>(e)].next) {
                    const auto& edge = edges_[static_cast<std::size_t>(e)];
                    if (edge.cap <= 1e-12F || level[edge.to] >= 0) continue;
                    level[edge.to] = level[u] + 1;
                    q.push(edge.to);
                }
            }
            return hit;
        };

        const auto dfs = [&](auto&& self, const std::size_t u, const float f) -> float {
            if (snk_[u] > 1e-12F) {
                const float pushed = std::min(f, snk_[u]);
                snk_[u] -= pushed;
                return pushed;
            }
            for (int& ei = iter[u]; ei >= 0; ei = edges_[static_cast<std::size_t>(ei)].next) {
                auto& edge = edges_[static_cast<std::size_t>(ei)];
                if (edge.cap <= 1e-12F || level[edge.to] != level[u] + 1) continue;
                const float pushed = self(self, edge.to, std::min(f, edge.cap));
                if (pushed > 1e-12F) {
                    edge.cap -= pushed;
                    edges_[static_cast<std::size_t>(edge.rev)].cap += pushed;
                    return pushed;
                }
            }
            return 0.F;
        };

        while (bfs()) {
            iter = head_;
            bool any = false;
            for (std::size_t i = 0; i < n_; ++i) {
                while (src_[i] > 1e-12F && level[i] == 0) {
                    const float pushed = dfs(dfs, i, src_[i]);
                    if (pushed <= 1e-12F) break;
                    src_[i] -= pushed;
                    flow += pushed;
                    any = true;
                }
            }
            if (!any) break;
        }

        // Min-cut: nodes that can still reach the sink in the residual graph
        // are sink-side; the rest are source-side.
        std::vector<char> reach_sink(n_, 0);
        std::queue<std::size_t> q;
        for (std::size_t i = 0; i < n_; ++i) {
            if (snk_[i] > 1e-12F) {
                reach_sink[i] = 1;
                q.push(i);
            }
        }
        while (!q.empty()) {
            const std::size_t u = q.front();
            q.pop();
            for (int e = head_[u]; e >= 0; e = edges_[static_cast<std::size_t>(e)].next) {
                const auto& edge = edges_[static_cast<std::size_t>(e)];
                // Reverse residual: edges_[edge.rev] goes to->u with residual cap.
                const auto& rev = edges_[static_cast<std::size_t>(edge.rev)];
                if (rev.cap <= 1e-12F || reach_sink[edge.to]) continue;
                reach_sink[edge.to] = 1;
                q.push(edge.to);
            }
        }
        for (std::size_t i = 0; i < n_; ++i)
            source_side_[i] = reach_sink[i] ? 0 : 1;
        return flow;
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
