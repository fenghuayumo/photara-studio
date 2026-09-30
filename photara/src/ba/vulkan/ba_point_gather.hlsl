#include "ba_common.hlsli"
#include "ba_atomic.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(1, 0)]] RWByteAddressBuffer track_ids;
[[vk::binding(2, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> cross;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> track_intr;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> direction;
[[vk::binding(6, 0)]] RWStructuredBuffer<double> scratch;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    bool fix_first = (pc.flags & kFixPose) != 0u;
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint observation = group_id.x * 256u + group_thread; observation < pc.n;
         observation += stride) {
        uint camera = load_uint(cameras, observation);
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        if (!(fix_first && camera == 0u)) {
            [unroll] for (uint row = 0u; row < 6u; ++row) {
                double value = direction[camera * 6u + row];
                sum0 += cross[observation * 18u + row * 3u] * value;
                sum1 += cross[observation * 18u + row * 3u + 1u] * value;
                sum2 += cross[observation * 18u + row * 3u + 2u] * value;
            }
        }
        if (dof > 0u) {
            uint group = load_uint(pose_group, camera);
            for (uint param = 0u; param < dof; ++param) {
                double value = direction[pc.camera_values + group * dof + param];
                sum0 += track_intr[observation * dof * 3u + param] * value;
                sum1 += track_intr[observation * dof * 3u + dof + param] * value;
                sum2 += track_intr[observation * dof * 3u + 2u * dof + param] * value;
            }
        }
        uint track = load_uint(track_ids, observation);
        atomic_add_f64(scratch[track * 3u], 1u, 0u, sum0);
        atomic_add_f64(scratch[track * 3u + 1u], 1u, 0u, sum1);
        atomic_add_f64(scratch[track * 3u + 2u], 1u, 0u, sum2);
    }
}
