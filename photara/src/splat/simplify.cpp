#include "splat/simplify.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace photara::splat {
namespace {

constexpr float k_logit_eps = 1e-6F;
constexpr float k_scale_eps = 1e-8F;
constexpr float k_appearance = 8.F;

float sigmoid(const float logit) {
    if (logit >= 0.F) {
        const float z = std::exp(-logit);
        return 1.F / (1.F + z);
    }
    const float z = std::exp(std::max(logit, -80.F));
    return z / (1.F + z);
}

float logit(const float opacity) {
    const float o = std::clamp(opacity, k_logit_eps, 1.F - k_logit_eps);
    return std::log(o / (1.F - o));
}

void quat_normalize(float q[4]) {
    const float n = std::sqrt(
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(n > 1e-12F)) {
        q[0] = 1.F;
        q[1] = q[2] = q[3] = 0.F;
        return;
    }
    const float inv = 1.F / n;
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
}

void quat_to_matrix(const float q[4], float r[9]) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    r[0] = 1.F - 2.F * (y * y + z * z);
    r[1] = 2.F * (x * y - w * z);
    r[2] = 2.F * (x * z + w * y);
    r[3] = 2.F * (x * y + w * z);
    r[4] = 1.F - 2.F * (x * x + z * z);
    r[5] = 2.F * (y * z - w * x);
    r[6] = 2.F * (x * z - w * y);
    r[7] = 2.F * (y * z + w * x);
    r[8] = 1.F - 2.F * (x * x + y * y);
}

void matrix_to_quat(const float r[9], float q[4]) {
    const float t = r[0] + r[4] + r[8];
    if (t > 0.F) {
        const float s = std::sqrt(t + 1.F) * 2.F;
        q[0] = 0.25F * s;
        q[1] = (r[7] - r[5]) / s;
        q[2] = (r[2] - r[6]) / s;
        q[3] = (r[3] - r[1]) / s;
    } else if (r[0] > r[4] && r[0] > r[8]) {
        const float s = std::sqrt(1.F + r[0] - r[4] - r[8]) * 2.F;
        q[0] = (r[7] - r[5]) / s;
        q[1] = 0.25F * s;
        q[2] = (r[1] + r[3]) / s;
        q[3] = (r[2] + r[6]) / s;
    } else if (r[4] > r[8]) {
        const float s = std::sqrt(1.F + r[4] - r[0] - r[8]) * 2.F;
        q[0] = (r[2] - r[6]) / s;
        q[1] = (r[1] + r[3]) / s;
        q[2] = 0.25F * s;
        q[3] = (r[5] + r[7]) / s;
    } else {
        const float s = std::sqrt(1.F + r[8] - r[0] - r[4]) * 2.F;
        q[0] = (r[3] - r[1]) / s;
        q[1] = (r[2] + r[6]) / s;
        q[2] = (r[5] + r[7]) / s;
        q[3] = 0.25F * s;
    }
    quat_normalize(q);
    if (q[0] < 0.F) {
        q[0] = -q[0];
        q[1] = -q[1];
        q[2] = -q[2];
        q[3] = -q[3];
    }
}

void covariance(const float q[4], const float s[3], float cov[6]) {
    float r[9];
    quat_to_matrix(q, r);
    const float lx = s[0] * s[0], ly = s[1] * s[1], lz = s[2] * s[2];
    // Σ = R diag(s^2) R^T, stored xx, yy, zz, xy, xz, yz.
    float m[9]{};
    for (int col = 0; col < 3; ++col) {
        const float lam = col == 0 ? lx : col == 1 ? ly : lz;
        for (int row = 0; row < 3; ++row)
            m[row * 3 + col] = r[row * 3 + col] * lam;
    }
    float sxx = 0.F, syy = 0.F, szz = 0.F, sxy = 0.F, sxz = 0.F, syz = 0.F;
    for (int k = 0; k < 3; ++k) {
        sxx += m[k] * r[k];
        syy += m[3 + k] * r[3 + k];
        szz += m[6 + k] * r[6 + k];
        sxy += m[k] * r[3 + k];
        sxz += m[k] * r[6 + k];
        syz += m[3 + k] * r[6 + k];
    }
    cov[0] = sxx;
    cov[1] = syy;
    cov[2] = szz;
    cov[3] = sxy;
    cov[4] = sxz;
    cov[5] = syz;
}

void jacobi_eigen_spd(const float cov[6], float eval[3], float evec[9]) {
    float a[9] = {
        cov[0], cov[3], cov[4],
        cov[3], cov[1], cov[5],
        cov[4], cov[5], cov[2]};
    evec[0] = evec[4] = evec[8] = 1.F;
    evec[1] = evec[2] = evec[3] = evec[5] = evec[6] = evec[7] = 0.F;
    for (int iter = 0; iter < 24; ++iter) {
        const float xy = std::abs(a[1]), xz = std::abs(a[2]), yz = std::abs(a[5]);
        int p = 0, q = 1;
        float apq = xy;
        if (xz > apq) {
            p = 0;
            q = 2;
            apq = xz;
        }
        if (yz > apq) {
            p = 1;
            q = 2;
            apq = yz;
        }
        if (apq < 1e-12F) break;
        const float app = a[p * 3 + p];
        const float aqq = a[q * 3 + q];
        const float tau = (aqq - app) / (2.F * a[p * 3 + q]);
        const float t = std::copysign(
            1.F / (std::abs(tau) + std::sqrt(1.F + tau * tau)), tau);
        const float c = 1.F / std::sqrt(1.F + t * t);
        const float s = t * c;
        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q) continue;
            const float akp = a[k * 3 + p];
            const float akq = a[k * 3 + q];
            a[k * 3 + p] = a[p * 3 + k] = c * akp - s * akq;
            a[k * 3 + q] = a[q * 3 + k] = s * akp + c * akq;
        }
        a[p * 3 + p] = c * c * app - 2.F * s * c * a[p * 3 + q] + s * s * aqq;
        a[q * 3 + q] = s * s * app + 2.F * s * c * a[p * 3 + q] + c * c * aqq;
        a[p * 3 + q] = a[q * 3 + p] = 0.F;
        for (int k = 0; k < 3; ++k) {
            const float vip = evec[k * 3 + p];
            const float viq = evec[k * 3 + q];
            evec[k * 3 + p] = c * vip - s * viq;
            evec[k * 3 + q] = s * vip + c * viq;
        }
    }
    eval[0] = std::max(a[0], k_scale_eps);
    eval[1] = std::max(a[4], k_scale_eps);
    eval[2] = std::max(a[8], k_scale_eps);
}

void covariance_to_qs(const float cov[6], float q[4], float s[3]) {
    float eval[3], evec[9];
    jacobi_eigen_spd(cov, eval, evec);
    if ((evec[0] * (evec[4] * evec[8] - evec[5] * evec[7]) -
         evec[1] * (evec[3] * evec[8] - evec[5] * evec[6]) +
         evec[2] * (evec[3] * evec[7] - evec[4] * evec[6])) < 0.F) {
        evec[2] = -evec[2];
        evec[5] = -evec[5];
        evec[8] = -evec[8];
    }
    matrix_to_quat(evec, q);
    s[0] = std::sqrt(eval[0]);
    s[1] = std::sqrt(eval[1]);
    s[2] = std::sqrt(eval[2]);
}

float kl_gaussian(
    const float mu_p[3], const float cov_p[6],
    const float mu_q[3], const float cov_q[6]) {
    float eval[3], evec[9];
    jacobi_eigen_spd(cov_q, eval, evec);
    float inv[6]{};
    for (int k = 0; k < 3; ++k) {
        const float ik = 1.F / eval[k];
        const float x = evec[k], y = evec[3 + k], z = evec[6 + k];
        inv[0] += ik * x * x;
        inv[1] += ik * y * y;
        inv[2] += ik * z * z;
        inv[3] += ik * x * y;
        inv[4] += ik * x * z;
        inv[5] += ik * y * z;
    }
    const float d0 = mu_p[0] - mu_q[0];
    const float d1 = mu_p[1] - mu_q[1];
    const float d2 = mu_p[2] - mu_q[2];
    const float mahal = d0 * (inv[0] * d0 + inv[3] * d1 + inv[4] * d2) +
        d1 * (inv[3] * d0 + inv[1] * d1 + inv[5] * d2) +
        d2 * (inv[4] * d0 + inv[5] * d1 + inv[2] * d2);
    const float tr = inv[0] * cov_p[0] + inv[1] * cov_p[1] + inv[2] * cov_p[2] +
        2.F * (inv[3] * cov_p[3] + inv[4] * cov_p[4] + inv[5] * cov_p[5]);
    float eval_p[3], evec_p[9];
    jacobi_eigen_spd(cov_p, eval_p, evec_p);
    const float log_det = std::log(eval[0] * eval[1] * eval[2] /
        (eval_p[0] * eval_p[1] * eval_p[2]));
    return 0.5F * (tr + mahal - 3.F + log_det);
}

struct SplatSet {
    std::vector<float> mean;
    std::vector<float> scale;
    std::vector<float> quat;
    std::vector<float> opacity;
    std::vector<float> sh;
    std::vector<float> normal;
    std::vector<float> filter;
    std::size_t n{};
    std::size_t sh_dim{};
    unsigned sh_degree{};
};

float* mean_at(SplatSet& set, const std::size_t i) {
    return set.mean.data() + i * 3;
}
const float* mean_at(const SplatSet& set, const std::size_t i) {
    return set.mean.data() + i * 3;
}

void swap_splat(SplatSet& set, const std::size_t a, const std::size_t b) {
    if (a == b) return;
    for (int k = 0; k < 3; ++k) std::swap(set.mean[a * 3 + k], set.mean[b * 3 + k]);
    for (int k = 0; k < 3; ++k) std::swap(set.scale[a * 3 + k], set.scale[b * 3 + k]);
    for (int k = 0; k < 4; ++k) std::swap(set.quat[a * 4 + k], set.quat[b * 4 + k]);
    std::swap(set.opacity[a], set.opacity[b]);
    for (std::size_t k = 0; k < set.sh_dim; ++k)
        std::swap(set.sh[a * set.sh_dim + k], set.sh[b * set.sh_dim + k]);
    if (!set.normal.empty())
        for (int k = 0; k < 4; ++k)
            std::swap(set.normal[a * 4 + k], set.normal[b * 4 + k]);
    if (!set.filter.empty()) std::swap(set.filter[a], set.filter[b]);
}

void compact_to(SplatSet& set, const std::size_t n) {
    set.n = n;
    set.mean.resize(n * 3);
    set.scale.resize(n * 3);
    set.quat.resize(n * 4);
    set.opacity.resize(n);
    set.sh.resize(n * set.sh_dim);
    if (!set.normal.empty()) set.normal.resize(n * 4);
    if (!set.filter.empty()) set.filter.resize(n);
}

SplatSet from_model(const GaussianModel& model) {
    SplatSet set;
    set.n = model.size();
    if (set.n == 0) return set;
    set.sh_degree = model.sh_degree;
    set.sh_dim = 3;
    if (model.sh.is_valid() && model.sh.shape().rank() >= 2)
        set.sh_dim = model.sh.shape()[1] * 3;
    const auto means = model.means.to_vector();
    const auto logs = model.log_scales.to_vector();
    const auto quats = model.quaternions.to_vector();
    const auto logits = model.opacity_logits.to_vector();
    const auto sh = model.sh.is_valid() ? model.sh.to_vector() : std::vector<float>{};
    set.mean = means;
    set.quat = quats;
    set.scale.resize(set.n * 3);
    set.opacity.resize(set.n);
    set.sh.assign(set.n * set.sh_dim, 0.F);
    for (std::size_t i = 0; i < set.n; ++i) {
        set.scale[i * 3] = std::exp(logs[i * 3]);
        set.scale[i * 3 + 1] = std::exp(logs[i * 3 + 1]);
        set.scale[i * 3 + 2] = std::exp(logs[i * 3 + 2]);
        quat_normalize(&set.quat[i * 4]);
        set.opacity[i] = sigmoid(logits[i]);
        const std::size_t take = std::min(set.sh_dim, sh.size() - i * set.sh_dim);
        if (i * set.sh_dim < sh.size())
            std::copy_n(sh.data() + i * set.sh_dim, take, set.sh.data() + i * set.sh_dim);
    }
    if (model.normal_features.is_valid() &&
        model.normal_features.numel() == set.n * 4)
        set.normal = model.normal_features.to_vector();
    if (model.filter_3d.is_valid() && model.filter_3d.numel() == set.n)
        set.filter = model.filter_3d.to_vector();
    return set;
}

GaussianModel to_model(const SplatSet& set) {
    GaussianModel model;
    model.sh_degree = set.sh_degree;
    if (set.n == 0) return model;
    std::vector<float> logs(set.n * 3), logits(set.n);
    for (std::size_t i = 0; i < set.n; ++i) {
        logs[i * 3] = std::log(std::max(set.scale[i * 3], k_scale_eps));
        logs[i * 3 + 1] = std::log(std::max(set.scale[i * 3 + 1], k_scale_eps));
        logs[i * 3 + 2] = std::log(std::max(set.scale[i * 3 + 2], k_scale_eps));
        logits[i] = logit(set.opacity[i]);
    }
    const auto cpu = tinytensor::Device::CPU;
    model.means = tinytensor::Tensor::from_vector(set.mean, {set.n, 3}, cpu);
    model.log_scales = tinytensor::Tensor::from_vector(logs, {set.n, 3}, cpu);
    model.quaternions = tinytensor::Tensor::from_vector(set.quat, {set.n, 4}, cpu);
    model.opacity_logits = tinytensor::Tensor::from_vector(logits, {set.n, 1}, cpu);
    const std::size_t bases = std::max<std::size_t>(1, set.sh_dim / 3);
    model.sh = tinytensor::Tensor::from_vector(
        set.sh, {set.n, bases, 3}, cpu);
    if (!set.normal.empty())
        model.normal_features = tinytensor::Tensor::from_vector(
            set.normal, {set.n, 4}, cpu);
    if (!set.filter.empty())
        model.filter_3d = tinytensor::Tensor::from_vector(
            set.filter, {set.n, 1}, cpu);
    return model;
}

void prune_opacity(SplatSet& set, const float min_opacity) {
    std::size_t w = 0;
    for (std::size_t i = 0; i < set.n; ++i) {
        if (set.opacity[i] < min_opacity) continue;
        if (w != i) swap_splat(set, w, i);
        ++w;
    }
    compact_to(set, w);
}

struct KdNode {
    std::int32_t left{-1};
    std::int32_t right{-1};
    std::uint32_t index{};
    std::uint8_t axis{};
};

void build_kd(
    std::vector<KdNode>& nodes, std::vector<std::uint32_t>& order,
    const SplatSet& set, const int begin, const int end, const int axis) {
    if (begin >= end) return;
    const int mid = (begin + end) / 2;
    std::nth_element(
        order.begin() + begin, order.begin() + mid, order.begin() + end,
        [&](const std::uint32_t a, const std::uint32_t b) {
            return mean_at(set, a)[axis] < mean_at(set, b)[axis];
        });
    const std::int32_t id = static_cast<std::int32_t>(nodes.size());
    nodes.push_back({});
    nodes[static_cast<std::size_t>(id)].index = order[static_cast<std::size_t>(mid)];
    nodes[static_cast<std::size_t>(id)].axis = static_cast<std::uint8_t>(axis);
    const int next = (axis + 1) % 3;
    if (begin < mid) {
        nodes[static_cast<std::size_t>(id)].left =
            static_cast<std::int32_t>(nodes.size());
        build_kd(nodes, order, set, begin, mid, next);
    }
    if (mid + 1 < end) {
        nodes[static_cast<std::size_t>(id)].right =
            static_cast<std::int32_t>(nodes.size());
        build_kd(nodes, order, set, mid + 1, end, next);
    }
}

void knn_search(
    const std::vector<KdNode>& nodes, const SplatSet& set, const std::size_t query,
    const unsigned k, std::int32_t node,
    std::priority_queue<std::pair<float, std::uint32_t>>& heap) {
    if (node < 0) return;
    const KdNode& cur = nodes[static_cast<std::size_t>(node)];
    const float* q = mean_at(set, query);
    const float* p = mean_at(set, cur.index);
    const float dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
    const float d2 = dx * dx + dy * dy + dz * dz;
    if (cur.index != query) {
        if (heap.size() < k) {
            heap.push({d2, cur.index});
        } else if (d2 < heap.top().first) {
            heap.pop();
            heap.push({d2, cur.index});
        }
    }
    const float delta = q[cur.axis] - p[cur.axis];
    const std::int32_t first = delta <= 0.F ? cur.left : cur.right;
    const std::int32_t second = delta <= 0.F ? cur.right : cur.left;
    knn_search(nodes, set, query, k, first, heap);
    if (heap.size() < k || delta * delta < heap.top().first)
        knn_search(nodes, set, query, k, second, heap);
}

std::vector<std::array<std::uint32_t, 16>> knn_graph(
    const SplatSet& set, const unsigned k) {
    const unsigned kk = std::min(k, 16U);
    std::vector<std::array<std::uint32_t, 16>> graph(set.n);
    for (auto& row : graph) row.fill(~0U);
    if (set.n <= 1) return graph;
    std::vector<std::uint32_t> order(set.n);
    std::iota(order.begin(), order.end(), 0U);
    std::vector<KdNode> nodes;
    nodes.reserve(set.n);
    build_kd(nodes, order, set, 0, static_cast<int>(set.n), 0);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(set.n); ++i) {
        std::priority_queue<std::pair<float, std::uint32_t>> heap;
        knn_search(nodes, set, static_cast<std::size_t>(i), kk, 0, heap);
        std::array<std::uint32_t, 16> row{};
        row.fill(~0U);
        int slot = static_cast<int>(heap.size()) - 1;
        while (!heap.empty() && slot >= 0) {
            row[static_cast<std::size_t>(slot)] = heap.top().second;
            heap.pop();
            --slot;
        }
        graph[static_cast<std::size_t>(i)] = row;
    }
    return graph;
}

float mass_of(const SplatSet& set, const std::size_t i) {
    const float* s = set.scale.data() + i * 3;
    return std::max(set.opacity[i] * s[0] * s[1] * s[2], 1e-12F);
}

void merge_pair(SplatSet& set, const std::size_t a, const std::size_t b) {
    const float m1 = mass_of(set, a);
    const float m2 = mass_of(set, b);
    const float m = m1 + m2;
    const float w1 = m1 / m, w2 = m2 / m;
    float mu[3];
    for (int k = 0; k < 3; ++k)
        mu[k] = w1 * set.mean[a * 3 + k] + w2 * set.mean[b * 3 + k];
    float cov1[6], cov2[6];
    covariance(&set.quat[a * 4], &set.scale[a * 3], cov1);
    covariance(&set.quat[b * 4], &set.scale[b * 3], cov2);
    float d1[3], d2[3];
    for (int k = 0; k < 3; ++k) {
        d1[k] = set.mean[a * 3 + k] - mu[k];
        d2[k] = set.mean[b * 3 + k] - mu[k];
    }
    float cov[6];
    cov[0] = w1 * (cov1[0] + d1[0] * d1[0]) + w2 * (cov2[0] + d2[0] * d2[0]);
    cov[1] = w1 * (cov1[1] + d1[1] * d1[1]) + w2 * (cov2[1] + d2[1] * d2[1]);
    cov[2] = w1 * (cov1[2] + d1[2] * d1[2]) + w2 * (cov2[2] + d2[2] * d2[2]);
    cov[3] = w1 * (cov1[3] + d1[0] * d1[1]) + w2 * (cov2[3] + d2[0] * d2[1]);
    cov[4] = w1 * (cov1[4] + d1[0] * d1[2]) + w2 * (cov2[4] + d2[0] * d2[2]);
    cov[5] = w1 * (cov1[5] + d1[1] * d1[2]) + w2 * (cov2[5] + d2[1] * d2[2]);
    float q[4], s[3];
    covariance_to_qs(cov, q, s);
    // Spread lives in the merged covariance. Opacity stays a mass-weighted
    // blend, not total mass divided by the new volume.
    const float opacity = std::clamp(
        w1 * set.opacity[a] + w2 * set.opacity[b],
        k_logit_eps, 1.F - k_logit_eps);
    for (int k = 0; k < 3; ++k) set.mean[a * 3 + k] = mu[k];
    for (int k = 0; k < 3; ++k) set.scale[a * 3 + k] = s[k];
    for (int k = 0; k < 4; ++k) set.quat[a * 4 + k] = q[k];
    set.opacity[a] = opacity;
    for (std::size_t k = 0; k < set.sh_dim; ++k)
        set.sh[a * set.sh_dim + k] =
            w1 * set.sh[a * set.sh_dim + k] + w2 * set.sh[b * set.sh_dim + k];
    if (!set.normal.empty()) {
        for (int k = 0; k < 4; ++k)
            set.normal[a * 4 + k] =
                w1 * set.normal[a * 4 + k] + w2 * set.normal[b * 4 + k];
        const float nx = set.normal[a * 4], ny = set.normal[a * 4 + 1],
                    nz = set.normal[a * 4 + 2];
        const float nn = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nn > 1e-8F) {
            set.normal[a * 4] = nx / nn;
            set.normal[a * 4 + 1] = ny / nn;
            set.normal[a * 4 + 2] = nz / nn;
        }
    }
    if (!set.filter.empty())
        set.filter[a] = w1 * set.filter[a] + w2 * set.filter[b];
}

float pair_cost(const SplatSet& set, const std::size_t a, const std::size_t b) {
    const float m1 = mass_of(set, a);
    const float m2 = mass_of(set, b);
    const float m = m1 + m2;
    const float w1 = m1 / m, w2 = m2 / m;
    float mu[3];
    for (int k = 0; k < 3; ++k)
        mu[k] = w1 * set.mean[a * 3 + k] + w2 * set.mean[b * 3 + k];
    float cov1[6], cov2[6];
    covariance(&set.quat[a * 4], &set.scale[a * 3], cov1);
    covariance(&set.quat[b * 4], &set.scale[b * 3], cov2);
    float d1[3], d2[3];
    for (int k = 0; k < 3; ++k) {
        d1[k] = set.mean[a * 3 + k] - mu[k];
        d2[k] = set.mean[b * 3 + k] - mu[k];
    }
    float cov[6];
    cov[0] = w1 * (cov1[0] + d1[0] * d1[0]) + w2 * (cov2[0] + d2[0] * d2[0]);
    cov[1] = w1 * (cov1[1] + d1[1] * d1[1]) + w2 * (cov2[1] + d2[1] * d2[1]);
    cov[2] = w1 * (cov1[2] + d1[2] * d1[2]) + w2 * (cov2[2] + d2[2] * d2[2]);
    cov[3] = w1 * (cov1[3] + d1[0] * d1[1]) + w2 * (cov2[3] + d2[0] * d2[1]);
    cov[4] = w1 * (cov1[4] + d1[0] * d1[2]) + w2 * (cov2[4] + d2[0] * d2[2]);
    cov[5] = w1 * (cov1[5] + d1[1] * d1[2]) + w2 * (cov2[5] + d2[1] * d2[2]);
    const float geom = m1 * kl_gaussian(mean_at(set, a), cov1, mu, cov) +
        m2 * kl_gaussian(mean_at(set, b), cov2, mu, cov);
    float app = 0.F;
    const std::size_t dim = std::min<std::size_t>(3, set.sh_dim);
    for (std::size_t k = 0; k < dim; ++k) {
        const float d = set.sh[a * set.sh_dim + k] - set.sh[b * set.sh_dim + k];
        app += d * d;
    }
    return geom + k_appearance * app;
}

struct Edge {
    float cost{};
    std::uint32_t a{};
    std::uint32_t b{};
};

bool cancelled(const SimplifyOptions& options) {
    return options.cancelled && options.cancelled();
}

void report(const SimplifyOptions& options, const float progress) {
    if (options.progress) options.progress(std::clamp(progress, 0.F, 1.F));
}

}  // namespace

GaussianModel simplify_gaussians(
    const GaussianModel& model, const SimplifyOptions& options) {
    if (model.size() == 0) return model;
    if (!(options.keep_ratio > 0.F) || options.keep_ratio > 1.F)
        throw std::invalid_argument("simplify keep_ratio must be in (0, 1]");
    if (options.knn == 0)
        throw std::invalid_argument("simplify knn must be at least 1");
    SplatSet set = from_model(model);
    const std::size_t original = set.n;
    prune_opacity(set, std::max(options.min_opacity, 0.F));
    report(options, 0.08F);
    const std::size_t target = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::llround(
               static_cast<double>(original) *
               static_cast<double>(options.keep_ratio))));
    const unsigned knn = std::min(options.knn, 16U);
    const float cap = std::clamp(options.merge_cap, 0.01F, 0.9F);
    int pass = 0;
    while (set.n > target) {
        if (cancelled(options)) break;
        auto graph = knn_graph(set, knn);
        std::vector<Edge> edges;
        edges.reserve(set.n * knn);
        for (std::size_t i = 0; i < set.n; ++i) {
            for (unsigned n = 0; n < knn; ++n) {
                const std::uint32_t j = graph[i][n];
                if (j == ~0U || j <= i || j >= set.n) continue;
                edges.push_back({pair_cost(set, i, j),
                    static_cast<std::uint32_t>(i), j});
            }
        }
        std::sort(edges.begin(), edges.end(),
            [](const Edge& a, const Edge& b) { return a.cost < b.cost; });
        const std::size_t room = set.n > target ? set.n - target : 0;
        const std::size_t max_merges = std::min(
            room,
            std::max<std::size_t>(
                1, static_cast<std::size_t>(static_cast<float>(set.n) * cap)));
        std::vector<std::uint8_t> used(set.n, 0);
        std::vector<std::pair<std::uint32_t, std::uint32_t>> pairs;
        pairs.reserve(max_merges);
        for (const Edge& edge : edges) {
            if (pairs.size() >= max_merges) break;
            if (used[edge.a] || used[edge.b]) continue;
            used[edge.a] = used[edge.b] = 1;
            pairs.push_back({edge.a, edge.b});
        }
        if (pairs.empty()) break;
        std::vector<std::uint8_t> drop(set.n, 0);
        for (const auto& [a, b] : pairs) {
            merge_pair(set, a, b);
            drop[b] = 1;
        }
        std::size_t w = 0;
        for (std::size_t i = 0; i < set.n; ++i) {
            if (drop[i]) continue;
            if (w != i) swap_splat(set, w, i);
            ++w;
        }
        compact_to(set, w);
        ++pass;
        const float done = original > target
            ? static_cast<float>(original - set.n) /
                  static_cast<float>(original - target)
            : 1.F;
        report(options, 0.1F + 0.85F * std::min(done, 1.F));
        if (pass > 64) break;
    }
    if (options.sh_degree >= 0 &&
        static_cast<unsigned>(options.sh_degree) < set.sh_degree) {
        const std::size_t keep = static_cast<std::size_t>(
            (options.sh_degree + 1) * (options.sh_degree + 1) * 3);
        if (keep < set.sh_dim) {
            std::vector<float> sh(set.n * keep);
            for (std::size_t i = 0; i < set.n; ++i)
                std::copy_n(set.sh.data() + i * set.sh_dim, keep,
                    sh.data() + i * keep);
            set.sh = std::move(sh);
            set.sh_dim = keep;
            set.sh_degree = static_cast<unsigned>(options.sh_degree);
        }
    }
    report(options, 1.F);
    core::Logger::instance().info(
        "splat simplify original=", original, " kept=", set.n,
        " keep_ratio=", options.keep_ratio, " passes=", pass);
    return to_model(set);
}

}  // namespace photara::splat
