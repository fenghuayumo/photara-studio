#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(1, 0)]] RWByteAddressBuffer system;
[[vk::binding(2, 0)]] RWByteAddressBuffer rhs;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(1, 1, 1)]
void main() {
    uint anchor = pc.p0;
    uint first = pc.p1;
    uint second = pc.p2;
    uint width = pc.n * 3u;
    double baseline = unpack(pc.baseline_lo, pc.baseline_hi);
    double u0 = load_double(cameras, second * 3u) - load_double(cameras, first * 3u);
    double u1 = load_double(cameras, second * 3u + 1u) - load_double(cameras, first * 3u + 1u);
    double u2 = load_double(cameras, second * 3u + 2u) - load_double(cameras, first * 3u + 2u);
    double length = sqrt_d(fmax_d(u0 * u0 + u1 * u1 + u2 * u2, 0.0));
    double inv = 1.0 / fmax_d(length, 1.0e-10);
    double direction[3] = {u0 * inv, u1 * inv, u2 * inv};
    [unroll] for (uint row = 0u; row < 3u; ++row) {
        double pull = direction[row] * (length - baseline);
        store_double(rhs, first * 3u + row, load_double(rhs, first * 3u + row) + pull);
        store_double(rhs, second * 3u + row, load_double(rhs, second * 3u + row) - pull);
        [unroll] for (uint col = 0u; col < 3u; ++col) {
            double hessian = direction[row] * direction[col];
            uint ff = (first * 3u + row) * width + first * 3u + col;
            uint ss = (second * 3u + row) * width + second * 3u + col;
            uint fs = (first * 3u + row) * width + second * 3u + col;
            uint sf = (second * 3u + row) * width + first * 3u + col;
            store_double(system, ff, load_double(system, ff) + hessian);
            store_double(system, ss, load_double(system, ss) + hessian);
            store_double(system, fs, load_double(system, fs) - hessian);
            store_double(system, sf, load_double(system, sf) - hessian);
        }
    }
    for (uint row = 0u; row < 3u; ++row) {
        uint fixed = anchor * 3u + row;
        store_double(rhs, fixed, 0.0);
        for (uint col = 0u; col < width; ++col) {
            store_double(system, fixed * width + col, 0.0);
            store_double(system, col * width + fixed, 0.0);
        }
        store_double(system, fixed * width + fixed, 1.0);
    }
}
