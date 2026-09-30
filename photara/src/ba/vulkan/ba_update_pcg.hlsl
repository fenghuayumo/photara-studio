#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> solution;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> residual;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> direction;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> product;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> scalars;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    double alpha = scalars[4];
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint index = group_id.x * 256u + group_thread; index < pc.n; index += stride) {
        solution[index] += alpha * direction[index];
        residual[index] -= alpha * product[index];
    }
}
