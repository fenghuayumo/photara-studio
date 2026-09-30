#include "ba_common.hlsli"
#include "ba_atomic.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> left;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> right;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> result;
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double shared_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint stride = max(pc.groups, 1u) * 256u;
    double sum = 0.0;
    for (uint index = group_id.x * 256u + group_thread; index < pc.n; index += stride)
        sum += left[index] * right[index];
    shared_sum[group_thread] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 128u; offset > 0u; offset >>= 1u) {
        if (group_thread < offset) shared_sum[group_thread] += shared_sum[group_thread + offset];
        GroupMemoryBarrierWithGroupSync();
    }
    if (group_thread == 0u) atomic_add_f64(result[pc.p0], 1u, 0u, shared_sum[0]);
}
