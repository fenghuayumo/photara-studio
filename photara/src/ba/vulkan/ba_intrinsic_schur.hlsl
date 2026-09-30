#include "ba_common.hlsli"
#include "ba_atomic.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(1, 0)]] RWByteAddressBuffer indices;
[[vk::binding(2, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(3, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> reduced;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> track_intr;
[[vk::binding(6, 0)]] RWStructuredBuffer<double> output;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint track = group_id.x * 256u + group_thread; track < pc.n; track += stride) {
        uint begin = load_uint(offsets, track);
        uint end = load_uint(offsets, track + 1u);
        for (uint cursor = begin; cursor < end; ++cursor) {
            uint observation = load_uint(indices, cursor);
            uint camera = load_uint(cameras, observation);
            uint group = load_uint(pose_group, camera);
            for (uint param = 0u; param < dof; ++param) {
                double value = track_intr[observation * dof * 3u + param] * reduced[track * 3u] +
                               track_intr[observation * dof * 3u + dof + param] * reduced[track * 3u + 1u] +
                               track_intr[observation * dof * 3u + 2u * dof + param] * reduced[track * 3u + 2u];
                atomic_add_f64(output[pc.camera_values + group * dof + param], 1u, 0u, -value);
            }
        }
    }
}
