#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/graph_traits.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace aetherscan::mvs::maxflow {

// Boykov-Kolmogorov is substantially faster than a generic push-relabel
// implementation on the sparse, locally connected visibility graphs emitted
// by a 3D Delaunay triangulation. Boost's implementation is permissively
// licensed and provides the same source/sink partition required by meshing.
class Graph {
private:
    using Traits = boost::adjacency_list_traits<
        boost::vecS, boost::vecS, boost::directedS>;
    using EdgeDescriptor = Traits::edge_descriptor;

    struct EdgeProperties {
        float capacity{0.F};
        float residual{0.F};
        EdgeDescriptor reverse{};
    };

    using GraphType = boost::adjacency_list<
        boost::vecS, boost::vecS, boost::directedS, boost::no_property,
        EdgeProperties>;
    using VertexDescriptor =
        typename boost::graph_traits<GraphType>::vertex_descriptor;
    using VertexSize = typename boost::graph_traits<GraphType>::vertices_size_type;

public:
    explicit Graph(const std::size_t nodes = 0) { reset(nodes); }

    void reset(const std::size_t nodes) {
        n_ = nodes;
        graph_ = GraphType(nodes + 2);
        source_ = static_cast<VertexDescriptor>(nodes);
        sink_ = static_cast<VertexDescriptor>(nodes + 1);
        src_.assign(nodes, 0.F);
        snk_.assign(nodes, 0.F);
        source_side_.assign(nodes, 0);
        flow_offset_ = 0.F;
        solved_ = false;
    }

    void add_tweights(const std::size_t node, float source, float sink) {
        if (node >= n_ || source < 0.F || sink < 0.F)
            throw std::out_of_range("invalid max-flow terminal capacity");
        const float common = std::min(source, sink);
        flow_offset_ += common;
        src_[node] += source - common;
        snk_[node] += sink - common;
    }

    void add_edge(
        const std::size_t from, const std::size_t to, const float forward,
        const float reverse) {
        if (from >= n_ || to >= n_ || forward < 0.F || reverse < 0.F)
            throw std::out_of_range("invalid max-flow edge capacity");
        add_edge_pair(
            static_cast<VertexDescriptor>(from),
            static_cast<VertexDescriptor>(to), forward, reverse);
    }

    float maxflow() {
        if (solved_)
            throw std::logic_error("max-flow graph can only be solved once");
        solved_ = true;
        if (n_ == 0) return flow_offset_;
        for (std::size_t node = 0; node < n_; ++node) {
            if (src_[node] > 0.F)
                add_edge_pair(
                    source_, static_cast<VertexDescriptor>(node), src_[node],
                    0.F);
            if (snk_[node] > 0.F)
                add_edge_pair(
                    static_cast<VertexDescriptor>(node), sink_, snk_[node],
                    0.F);
        }

        const VertexSize vertices = boost::num_vertices(graph_);
        std::vector<EdgeDescriptor> predecessor(vertices);
        std::vector<boost::default_color_type> color(vertices);
        std::vector<VertexSize> distance(vertices);
        const float flow = boost::boykov_kolmogorov_max_flow(
            graph_, boost::get(&EdgeProperties::capacity, graph_),
            boost::get(&EdgeProperties::residual, graph_),
            boost::get(&EdgeProperties::reverse, graph_), predecessor.data(),
            color.data(), distance.data(), boost::get(boost::vertex_index, graph_),
            source_, sink_);
        for (std::size_t node = 0; node < n_; ++node)
            source_side_[node] =
                color[node] != boost::white_color ? std::uint8_t{1}
                                                  : std::uint8_t{0};
        return flow_offset_ + flow;
    }

    [[nodiscard]] bool is_source_side(const std::size_t node) const {
        return source_side_[node] != 0;
    }

private:
    void add_edge_pair(
        const VertexDescriptor from, const VertexDescriptor to,
        const float forward, const float reverse) {
        const EdgeDescriptor edge = boost::add_edge(from, to, graph_).first;
        const EdgeDescriptor reverse_edge =
            boost::add_edge(to, from, graph_).first;
        graph_[edge].capacity = forward;
        graph_[reverse_edge].capacity = reverse;
        graph_[edge].reverse = reverse_edge;
        graph_[reverse_edge].reverse = edge;
    }

    std::size_t n_{0};
    GraphType graph_;
    VertexDescriptor source_{};
    VertexDescriptor sink_{};
    std::vector<float> src_;
    std::vector<float> snk_;
    std::vector<std::uint8_t> source_side_;
    float flow_offset_{0.F};
    bool solved_{false};
};

}  // namespace aetherscan::mvs::maxflow
