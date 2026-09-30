#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> hessian;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    double damping = unpack(pc.damp_lo, pc.damp_hi);
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint camera = group_id.x * 256u + group_thread; camera < pc.n; camera += stride) {
        [unroll] for (uint axis = 0u; axis < 6u; ++axis) {
            uint index = camera * 36u + axis * 6u + axis;
            double diagonal = hessian[index];
            hessian[index] = diagonal + damping * (diagonal + 1.0);
        }
    }
}
