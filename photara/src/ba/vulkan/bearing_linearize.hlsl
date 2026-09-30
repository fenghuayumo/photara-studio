#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(1, 0)]] RWByteAddressBuffer points;
[[vk::binding(2, 0)]] RWByteAddressBuffer observations;
[[vk::binding(3, 0)]] RWByteAddressBuffer weights;
[[vk::binding(4, 0)]] RWByteAddressBuffer lin;
[[vk::binding(5, 0)]] RWByteAddressBuffer costs;
[[vk::push_constant]] ConstantBuffer<Push> pc;

struct Obs {
    uint camera;
    uint track;
    double d0;
    double d1;
    double d2;
};

Obs load_obs(uint index) {
    uint base = index * 32u;
    Obs obs;
    obs.camera = observations.Load(base);
    obs.track = observations.Load(base + 4u);
    obs.d0 = asdouble(observations.Load(base + 8u), observations.Load(base + 12u));
    obs.d1 = asdouble(observations.Load(base + 16u), observations.Load(base + 20u));
    obs.d2 = asdouble(observations.Load(base + 24u), observations.Load(base + 28u));
    return obs;
}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint step = max(pc.groups, 1u) * 256u;
    double huber = unpack(pc.huber_lo, pc.huber_hi);
    for (uint i = id.x; i < pc.n; i += step) {
        Obs obs = load_obs(i);
        double ux = load_double(points, obs.track * 3u) - load_double(cameras, obs.camera * 3u);
        double uy = load_double(points, obs.track * 3u + 1u) - load_double(cameras, obs.camera * 3u + 1u);
        double uz = load_double(points, obs.track * 3u + 2u) - load_double(cameras, obs.camera * 3u + 2u);
        double length2 = ux * ux + uy * uy + uz * uz;
        double length = sqrt_d(length2);
        if (!(length > 1.0e-10) || !isfinite_d(length)) {
            store_double(costs, i, asdouble(0u, 0x7FF00000u));
            continue;
        }
        ux /= length;
        uy /= length;
        uz /= length;
        double r0 = obs.d0 - ux;
        double r1 = obs.d1 - uy;
        double r2 = obs.d2 - uz;
        double residual2 = r0 * r0 + r1 * r1 + r2 * r2;
        double residual = sqrt_d(residual2);
        double weight = load_double(weights, i);
        double robust = residual > huber ? huber / residual : 1.0;
        double weighted = weight * robust;
        double cost = weight * (residual > huber ? huber * (residual - 0.5 * huber) : 0.5 * residual2);
        store_double(costs, i, cost);
        if ((pc.flags & 1u) == 0u) continue;

        // P = (I - u u^T) / length. Keep P^T P and P^T (d - u); the shorter
        // identities drop precision on weakly observed directions.
        double projection[9];
        double u_comp[3] = {ux, uy, uz};
        double residual_comp[3] = {r0, r1, r2};
        [unroll] for (uint row = 0u; row < 3u; ++row) {
            [unroll] for (uint col = 0u; col < 3u; ++col) {
                double identity = row == col ? 1.0 : 0.0;
                projection[row * 3u + col] = (identity - u_comp[row] * u_comp[col]) / length;
            }
        }
        [unroll] for (uint row = 0u; row < 3u; ++row) {
            double gradient = 0.0;
            [unroll] for (uint k = 0u; k < 3u; ++k) {
                gradient += projection[k * 3u + row] * residual_comp[k];
            }
            store_double(lin, i * 12u + 9u + row, weighted * gradient);
            [unroll] for (uint col = 0u; col < 3u; ++col) {
                double hessian = 0.0;
                [unroll] for (uint k = 0u; k < 3u; ++k) {
                    hessian += projection[k * 3u + row] * projection[k * 3u + col];
                }
                store_double(lin, i * 12u + row * 3u + col, weighted * hessian);
            }
        }
    }
}
