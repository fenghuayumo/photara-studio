#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer observations;
[[vk::binding(2, 0)]] RWByteAddressBuffer eliminated;
[[vk::binding(3, 0)]] RWByteAddressBuffer point_rhs;
[[vk::binding(4, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(5, 0)]] RWByteAddressBuffer indices;
[[vk::binding(6, 0)]] RWByteAddressBuffer scale;
[[vk::binding(7, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(8, 0)]] RWByteAddressBuffer system;
[[vk::binding(9, 0)]] RWByteAddressBuffer rhs;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint step = max(pc.groups, 1u) * 256u;
    double lambda = unpack(pc.lambda_lo, pc.lambda_hi);
    uint width = pc.n * 3u;
    double u0 = load_double(cameras, pc.p1 * 3u) - load_double(cameras, pc.p0 * 3u);
    double u1 = load_double(cameras, pc.p1 * 3u + 1u) - load_double(cameras, pc.p0 * 3u + 1u);
    double u2 = load_double(cameras, pc.p1 * 3u + 2u) - load_double(cameras, pc.p0 * 3u + 2u);
    double length2 = fmax_d(u0 * u0 + u1 * u1 + u2 * u2, 1.0e-20);
    double direction[3] = {u0, u1, u2};
    for (uint cam = id.x; cam < pc.n; cam += step) {
        double h0 = 0.0, h1 = 0.0, h2 = 0.0, h3 = 0.0, h4 = 0.0, h5 = 0.0, h6 = 0.0, h7 = 0.0, h8 = 0.0;
        double b0 = 0.0, b1 = 0.0, b2 = 0.0;
        uint begin = load_uint(offsets, cam);
        uint end = load_uint(offsets, cam + 1u);
        for (uint j = begin; j < end; ++j) {
            uint obs = load_uint(indices, j);
            uint track = observations.Load(obs * 32u + 4u);
            uint base = obs * 12u;
            h0 += load_double(lin, base);
            h1 += load_double(lin, base + 1u);
            h2 += load_double(lin, base + 2u);
            h3 += load_double(lin, base + 3u);
            h4 += load_double(lin, base + 4u);
            h5 += load_double(lin, base + 5u);
            h6 += load_double(lin, base + 6u);
            h7 += load_double(lin, base + 7u);
            h8 += load_double(lin, base + 8u);
            double g0 = load_double(lin, base + 9u);
            double g1 = load_double(lin, base + 10u);
            double g2 = load_double(lin, base + 11u);
            double gradient[3] = {g0, g1, g2};
            [unroll] for (uint row = 0u; row < 3u; ++row) {
                double acc = -gradient[row];
                [unroll] for (uint k = 0u; k < 3u; ++k) {
                    acc += load_double(eliminated, obs * 9u + row * 3u + k) *
                           load_double(point_rhs, track * 3u + k);
                }
                if (row == 0u) b0 = acc + b0;
                else if (row == 1u) b1 = acc + b1;
                else b2 = acc + b2;
            }
        }
        bool gauge = cam == pc.p0 || cam == pc.p1;
        double diagonal[3] = {h0, h4, h8};
        double hessian[9] = {h0, h1, h2, h3, h4, h5, h6, h7, h8};
        double gradient_out[3] = {b0, b1, b2};
        [unroll] for (uint row = 0u; row < 3u; ++row) {
            store_double(rhs, cam * 3u + row, gradient_out[row]);
            double pivot = diagonal[row] + (gauge ? direction[row] * direction[row] / length2 : 0.0);
            double column_scale = load_double(scale, cam * 3u + row);
            hessian[row * 3u + row] += lambda * fmin_d(fmax_d(pivot, 1.0e-6 * column_scale), 1.0e32 * column_scale);
            [unroll] for (uint col = 0u; col < 3u; ++col) {
                uint index = (cam * 3u + row) * width + cam * 3u + col;
                store_double(system, index, hessian[row * 3u + col]);
            }
        }
    }
}
