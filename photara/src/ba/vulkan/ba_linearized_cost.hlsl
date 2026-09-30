#include "ba_common.hlsli"
#include "ba_atomic.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> weights;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> total;
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double shared_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    double huber = unpack(pc.huber_lo, pc.huber_hi);
    uint stride = max(pc.groups, 1u) * 256u;
    double local = 0.0;
    for (uint observation = group_id.x * 256u + group_thread; observation < pc.n;
         observation += stride) {
        double weight = weights[observation];
        if (weight == 0.0) continue;
        if (!lin_valid(lin, observation)) {
            local += weight * 1e12;
            continue;
        }
        double robust = load_lin(lin, observation, 36u);
        if (!(robust > 0.0)) {
            local += weight * 1e12;
            continue;
        }
        double rx = load_lin(lin, observation, 0u);
        double ry = load_lin(lin, observation, 1u);
        double norm = sqrt_d((rx * rx + ry * ry) / robust);
        if (!isfinite_d(norm)) {
            local += weight * 1e12;
            continue;
        }
        double robust_cost = (huber > 0.0 && norm > huber) ? huber * (norm - 0.5 * huber)
                                                           : 0.5 * norm * norm;
        local += weight * robust_cost;
    }
    shared_sum[group_thread] = local;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (group_thread < offset) shared_sum[group_thread] += shared_sum[group_thread + offset];
        GroupMemoryBarrierWithGroupSync();
    }
    if (group_thread == 0u) atomic_add_f64(total[0], 1u, 0u, shared_sum[0]);
}
