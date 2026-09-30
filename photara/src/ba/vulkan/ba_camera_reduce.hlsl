#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> hessian;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> pose_intr;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> direction;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> scratch;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> output;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    bool fix_first = (pc.flags & kFixPose) != 0u;
    bool opt_rot = (pc.flags & kOptRot) != 0u;
    bool opt_trans = (pc.flags & kOptTrans) != 0u;
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint camera = group_id.x * 256u + group_thread; camera < pc.n; camera += stride) {
        if (fix_first && camera == 0u) {
            [unroll] for (uint row = 0u; row < 6u; ++row)
                output[camera * 6u + row] = direction[camera * 6u + row];
            continue;
        }
        double result[6];
        [unroll] for (uint row = 0u; row < 6u; ++row) {
            double value = 0.0;
            [unroll] for (uint column = 0u; column < 6u; ++column)
                value += hessian[camera * 36u + row * 6u + column] * direction[camera * 6u + column];
            result[row] = value - scratch[camera * 6u + row];
        }
        if (dof > 0u) {
            uint group = load_uint(pose_group, camera);
            for (uint row = 0u; row < 6u; ++row) {
                double value = 0.0;
                for (uint param = 0u; param < dof; ++param)
                    value += pose_intr[camera * 6u * dof + row * dof + param] *
                             direction[pc.camera_values + group * dof + param];
                result[row] += value;
            }
        }
        [unroll] for (uint row = 0u; row < 6u; ++row) {
            bool active = row < 3u ? opt_rot : opt_trans;
            output[camera * 6u + row] = active ? result[row] : direction[camera * 6u + row];
        }
    }
}
