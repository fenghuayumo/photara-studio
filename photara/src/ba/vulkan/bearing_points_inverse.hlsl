#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(2, 0)]] RWByteAddressBuffer indices;
[[vk::binding(3, 0)]] RWByteAddressBuffer scale;
[[vk::binding(4, 0)]] RWByteAddressBuffer inverse;
[[vk::binding(5, 0)]] RWByteAddressBuffer rhs;
[[vk::binding(6, 0)]] RWByteAddressBuffer info;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint step = max(pc.groups, 1u) * 256u;
    double lambda = unpack(pc.lambda_lo, pc.lambda_hi);
    for (uint p = id.x; p < pc.n; p += step) {
        double h0 = 0.0, h1 = 0.0, h2 = 0.0, h3 = 0.0, h4 = 0.0, h5 = 0.0, h6 = 0.0, h7 = 0.0, h8 = 0.0;
        double b0 = 0.0, b1 = 0.0, b2 = 0.0;
        uint begin = load_uint(offsets, p);
        uint end = load_uint(offsets, p + 1u);
        for (uint j = begin; j < end; ++j) {
            uint obs = load_uint(indices, j);
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
            b0 += load_double(lin, base + 9u);
            b1 += load_double(lin, base + 10u);
            b2 += load_double(lin, base + 11u);
        }
        double s0 = load_double(scale, p * 3u);
        double s1 = load_double(scale, p * 3u + 1u);
        double s2 = load_double(scale, p * 3u + 2u);
        h0 += lambda * fmin_d(fmax_d(h0, 1.0e-6 * s0), 1.0e32 * s0);
        h4 += lambda * fmin_d(fmax_d(h4, 1.0e-6 * s1), 1.0e32 * s1);
        h8 += lambda * fmin_d(fmax_d(h8, 1.0e-6 * s2), 1.0e32 * s2);
        double i0, i1, i2, i3, i4, i5, i6, i7, i8;
        if (!invert3(h0, h1, h2, h3, h4, h5, h6, h7, h8, i0, i1, i2, i3, i4, i5, i6, i7, i8)) {
            uint ignored;
            info.InterlockedExchange(4u, 1u, ignored);
            i0 = 0.0; i1 = 0.0; i2 = 0.0; i3 = 0.0; i4 = 0.0; i5 = 0.0; i6 = 0.0; i7 = 0.0; i8 = 0.0;
        }
        store_double(inverse, p * 9u, i0);
        store_double(inverse, p * 9u + 1u, i1);
        store_double(inverse, p * 9u + 2u, i2);
        store_double(inverse, p * 9u + 3u, i3);
        store_double(inverse, p * 9u + 4u, i4);
        store_double(inverse, p * 9u + 5u, i5);
        store_double(inverse, p * 9u + 6u, i6);
        store_double(inverse, p * 9u + 7u, i7);
        store_double(inverse, p * 9u + 8u, i8);
        store_double(rhs, p * 3u, b0);
        store_double(rhs, p * 3u + 1u, b1);
        store_double(rhs, p * 3u + 2u, b2);
    }
}
