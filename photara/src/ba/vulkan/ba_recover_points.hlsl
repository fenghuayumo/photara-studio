#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(1, 0)]] RWByteAddressBuffer indices;
[[vk::binding(2, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(3, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> cross;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> track_intr;
[[vk::binding(6, 0)]] RWStructuredBuffer<double> inverse;
[[vk::binding(7, 0)]] RWStructuredBuffer<double> rhs;
[[vk::binding(8, 0)]] RWStructuredBuffer<double> camera_step;
[[vk::binding(9, 0)]] RWStructuredBuffer<double> track_step;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint track = group_id.x * 256u + group_thread; track < pc.n; track += stride) {
        double value0 = rhs[track * 3u];
        double value1 = rhs[track * 3u + 1u];
        double value2 = rhs[track * 3u + 2u];
        uint begin = load_uint(offsets, track);
        uint end = load_uint(offsets, track + 1u);
        for (uint cursor = begin; cursor < end; ++cursor) {
            uint observation = load_uint(indices, cursor);
            uint camera = load_uint(cameras, observation);
            [unroll] for (uint column = 0u; column < 3u; ++column) {
                double sub = 0.0;
                [unroll] for (uint row = 0u; row < 6u; ++row)
                    sub += cross[observation * 18u + row * 3u + column] *
                           camera_step[camera * 6u + row];
                if (column == 0u) value0 -= sub;
                else if (column == 1u) value1 -= sub;
                else value2 -= sub;
            }
            if (dof > 0u) {
                uint group = load_uint(pose_group, camera);
                for (uint param = 0u; param < dof; ++param) {
                    double step = camera_step[pc.camera_values + group * dof + param];
                    value0 -= track_intr[observation * dof * 3u + param] * step;
                    value1 -= track_intr[observation * dof * 3u + dof + param] * step;
                    value2 -= track_intr[observation * dof * 3u + 2u * dof + param] * step;
                }
            }
        }
        uint h = track * 9u;
        track_step[track * 3u] = inverse[h] * value0 + inverse[h + 1u] * value1 + inverse[h + 2u] * value2;
        track_step[track * 3u + 1u] =
            inverse[h + 3u] * value0 + inverse[h + 4u] * value1 + inverse[h + 5u] * value2;
        track_step[track * 3u + 2u] =
            inverse[h + 6u] * value0 + inverse[h + 7u] * value1 + inverse[h + 8u] * value2;
    }
}
