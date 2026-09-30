#include "ba_common.hlsli"
#include "ba_atomic.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer camera_offsets;
[[vk::binding(1, 0)]] RWByteAddressBuffer camera_observations;
[[vk::binding(2, 0)]] RWByteAddressBuffer track_ids;
[[vk::binding(3, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> cross;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> track_intr;
[[vk::binding(6, 0)]] RWStructuredBuffer<double> temporary;
[[vk::binding(7, 0)]] RWStructuredBuffer<double> camera_scratch;
[[vk::binding(8, 0)]] RWStructuredBuffer<double> intrinsic_scratch;
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double camera_partial[6 * 256];
groupshared double intrinsic_partial[8 * 256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint dof = pc.dof;
    for (uint camera = group_id.x; camera < pc.n; camera += max(pc.groups, 1u)) {
        double camera_sum[6];
        double intrinsic_sum[8];
        [unroll] for (uint row = 0u; row < 6u; ++row) camera_sum[row] = 0.0;
        [unroll] for (uint row = 0u; row < 8u; ++row) intrinsic_sum[row] = 0.0;

        uint begin = load_uint(camera_offsets, camera);
        uint end = load_uint(camera_offsets, camera + 1u);
        for (uint entry = begin + group_thread; entry < end; entry += 256u) {
            uint observation = load_uint(camera_observations, entry);
            uint track = load_uint(track_ids, observation);
            double x = temporary[track * 3u];
            double y = temporary[track * 3u + 1u];
            double z = temporary[track * 3u + 2u];
            [unroll] for (uint row = 0u; row < 6u; ++row) {
                camera_sum[row] +=
                    cross[observation * 18u + row * 3u] * x +
                    cross[observation * 18u + row * 3u + 1u] * y +
                    cross[observation * 18u + row * 3u + 2u] * z;
            }
            for (uint row = 0u; row < dof; ++row) {
                intrinsic_sum[row] +=
                    track_intr[observation * dof * 3u + row] * x +
                    track_intr[observation * dof * 3u + dof + row] * y +
                    track_intr[observation * dof * 3u + 2u * dof + row] * z;
            }
        }

        [unroll] for (uint row = 0u; row < 6u; ++row)
            camera_partial[row * 256u + group_thread] = camera_sum[row];
        [unroll] for (uint row = 0u; row < 8u; ++row)
            intrinsic_partial[row * 256u + group_thread] = intrinsic_sum[row];
        GroupMemoryBarrierWithGroupSync();

        for (uint offset = 128u; offset > 0u; offset >>= 1u) {
            if (group_thread < offset) {
                [unroll] for (uint row = 0u; row < 6u; ++row)
                    camera_partial[row * 256u + group_thread] +=
                        camera_partial[row * 256u + group_thread + offset];
                [unroll] for (uint row = 0u; row < 8u; ++row)
                    intrinsic_partial[row * 256u + group_thread] +=
                        intrinsic_partial[row * 256u + group_thread + offset];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (group_thread == 0u) {
            [unroll] for (uint row = 0u; row < 6u; ++row)
                camera_scratch[camera * 6u + row] = camera_partial[row * 256u];
            if (dof > 0u) {
                uint group = load_uint(pose_group, camera);
                for (uint row = 0u; row < dof; ++row)
                    atomic_add_f64(intrinsic_scratch[group * dof + row], 1u, 0u,
                                   intrinsic_partial[row * 256u]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
