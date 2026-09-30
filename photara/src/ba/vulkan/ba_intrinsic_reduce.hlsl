#include "ba_common.hlsli"
#include "ba_atomic.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> intrinsic_h;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> pose_intr;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> direction;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> scratch;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> output;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint camera = group_id.x * 256u + group_thread; camera < pc.n; camera += stride) {
        uint group = load_uint(pose_group, camera);
        uint destination = pc.camera_values + group * dof;
        for (uint param = 0u; param < dof; ++param) {
            double value = 0.0;
            [unroll] for (uint row = 0u; row < 6u; ++row)
                value += pose_intr[camera * 6u * dof + row * dof + param] *
                         direction[camera * 6u + row];
            atomic_add_f64(output[destination + param], 1u, 0u, value);
        }
    }
    for (uint group = group_id.x * 256u + group_thread; group < pc.p0; group += stride) {
        uint base = pc.camera_values + group * dof;
        for (uint row = 0u; row < dof; ++row) {
            double value = 0.0;
            for (uint column = 0u; column < dof; ++column)
                value += intrinsic_h[group * dof * dof + row * dof + column] *
                         direction[base + column];
            atomic_add_f64(output[base + row], 1u, 0u, value - scratch[group * dof + row]);
        }
    }
}
