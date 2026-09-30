#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(1, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(2, 0)]] RWByteAddressBuffer constant;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> pose_intr;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> track_intr;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint camera = group_id.x * 256u + group_thread; camera < pc.n; camera += stride) {
        if (load_uint(constant, load_uint(pose_group, camera)) == 0u) continue;
        for (uint element = 0u; element < 6u * dof; ++element)
            pose_intr[camera * 6u * dof + element] = 0.0;
    }
    for (uint observation = group_id.x * 256u + group_thread; observation < pc.p0;
         observation += stride) {
        uint camera = load_uint(cameras, observation);
        if (load_uint(constant, load_uint(pose_group, camera)) == 0u) continue;
        for (uint element = 0u; element < dof * 3u; ++element)
            track_intr[observation * dof * 3u + element] = 0.0;
    }
}
